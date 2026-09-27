"""真实 P0 runner 的无权重 stand-in 测试。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from can.v2.pretrained_moe_p0 import P0Error
from can.v2.pretrained_moe_p0.capability_eval import evaluate_fixture, generate_one
from can.v2.pretrained_moe_p0.deepseek_adapter import DeepSeekHostAdapter
from can.v2.pretrained_moe_p0.fixture import validate_fixture
from can.v2.pretrained_moe_p0.qwen_adapter import QwenHostAdapter
from can.v2.pretrained_moe_p0.real_artifacts import (
    write_preflight_artifact,
    write_run_artifacts,
)
from can.v2.pretrained_moe_p0.real_loader import (
    LoadedRealHost,
    _device_facts,
    _infer_architecture,
    _packed_expert_records,
    _quantization_config,
)
from can.v2.pretrained_moe_p0.real_runner import RealP0Runner
from can.v2.pretrained_moe_p0.real_types import (
    ArchitectureMap,
    ExpertCall,
    GenerationRecord,
    GenerationRequest,
    PreflightResult,
    ProbeResult,
    RealRunResult,
    ResourceSample,
    validate_reassembly_indices,
)
from can.v2.pretrained_moe_p0.registry import validate_registry
from can.v2.pretrained_moe_p0.resource_probe import (
    candidate_load_smoke,
    infrastructure_preflight,
    validate_resource_gates,
)
from can.v2.pretrained_moe_p0.snapshot import (
    _declared_snapshot_size,
    _existing_snapshot_bytes,
    freeze_snapshot,
    inventory_snapshot,
    load_snapshot_manifest,
    manifest_payload,
    validate_download_budget,
    verify_read_only,
    verify_snapshot_against_manifest,
    write_snapshot_manifest,
)
from can.v2.pretrained_moe_p0.structure_probe import (
    inspect_real_structure,
    make_mixed_probe_request,
    validate_probe_result,
)


def _case(case_id: str, group: str) -> dict:
    """构造测试 fixture case 并绑定内容摘要。"""
    item = {
        "case_id": case_id,
        "group": group,
        "system_text": "Answer with only the requested text. Do not explain.",
        "user_text": f"Identifier: CODE-{case_id}. Return it.",
        "expected_text": f"CODE-{case_id}",
        "metric": "strict_em" if group == "format_copy" else "normalized_em",
        "max_new_tokens": 24,
    }
    if group != "format_copy":
        item.update(
            fact_source="stated_in_prompt", rationale="事实在 prompt 中明确给出。"
        )
    body = dict(item)
    item["content_sha256"] = hashlib.sha256(
        json.dumps(
            body, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return item


def _fixture() -> tuple:
    """返回三条可解析 fixture。"""
    payload = {
        "schema_version": 1,
        "fixture_id": "p0-moe-public-fixture-v1",
        "cases": [
            _case(f"{group}-{index:02d}", group)
            for group in ("format_copy", "single_hop", "two_hop")
            for index in range(8)
        ],
    }
    return validate_fixture(payload)


def _candidate_payload() -> dict:
    """构造 C1/C2 两个正式 passed 候选。"""
    candidates = []
    for order in (1, 2):
        candidates.append(
            {
                "candidate_id": f"C{order}",
                "repository_id": f"org/model-{order}",
                "attempt_order": order,
                "requested_revision": "main",
                "resolved_commit_sha": chr(96 + order) * 40,
                "profile": {
                    "profile_id": f"c{order}-bnb-nf4-bf16-v1",
                    "dtype": "bf16",
                    "quantization_config": {
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": "bfloat16",
                    },
                    "allow_remote_code": order == 2,
                },
                "expected_architecture_family": (
                    "qwen2_moe" if order == 1 else "deepseek"
                ),
                "expected_moe_variant": "native_shared_and_routed",
                "max_snapshot_bytes": 1024,
                "license_review_status": "approved",
                "metadata_source_sha256": "b" * 64,
                "p0a_status": "passed",
                "p0a_decision_sha256": "c" * 64,
                "p0a_failure_codes": [],
            }
        )
    return {
        "schema_version": 1,
        "registry_id": "p0-moe-host-v1",
        "candidates": candidates,
    }


def _architecture(**changes) -> ArchitectureMap:
    """构造可按字段替换的真实结构摘要。"""
    base = ArchitectureMap(
        "deepseek",
        "Model",
        ("block",),
        ("block.gate",),
        ("block.shared",),
        ("block.experts",),
        2,
        "eager",
        True,
        True,
        3,
    )
    return replace(base, **changes)


class _Tokenizer:
    """提供最小 chat template/decode 行为的 tokenizer。"""

    eos_token_id = 99

    def apply_chat_template(
        self, messages, add_generation_prompt, tokenize, return_tensors
    ):
        """把消息转换为单个虚拟 prompt token。"""
        assert add_generation_prompt and tokenize and return_tensors == "pt"
        return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    def decode(self, values, skip_special_tokens=False):
        """把测试 token 映射到固定答案文本。"""
        if values and values[0] == 7:
            return (
                "CODE-format_copy-00<|im_end|>"
                if 99 in values
                else "CODE-format_copy-00"
            )
        return "bad"


class _Model:
    """返回固定 continuation 的最小生成模型。"""

    def generate(self, **kwargs):
        """保留 prompt 并追加正确答案标记和 EOS。"""
        assert kwargs["do_sample"] is False
        return [[1, 2, 7, 99]]


def test_generate_one_enforces_greedy_and_safe_decode() -> None:
    """验证 P0-B 单样本生成使用冻结参数并记录 stop reason。"""
    request = GenerationRequest(
        "format_copy-00", "s", "u", "CODE-format_copy-00", "strict_em", 24, False
    )
    record = generate_one(_Tokenizer(), _Model(), request)
    assert record.matched is True
    assert record.stop_reason == "eos"
    assert record.continuation_tokens == (7, 99)


def test_generate_one_rejects_control_character() -> None:
    """控制字符必须进入稳定失败，而不是被静默清理。"""

    class BadTokenizer(_Tokenizer):
        def decode(self, values, skip_special_tokens=False):
            return "\x00"

    request = GenerationRequest("c", "s", "u", "x", "strict_em", 24, False)
    with pytest.raises(P0Error, match="generation_invalid_control_token"):
        generate_one(BadTokenizer(), _Model(), request)


def test_evaluate_fixture_records_failure_without_fallback() -> None:
    """能力执行失败时不得静默计为成功。"""

    class Adapter:
        def baseline_generate(self, request):
            """始终返回错误文本，用于验证门槛失败。"""
            raise P0Error("generation_failed", "injected")

    result = evaluate_fixture(Adapter(), _fixture())
    assert result["status"] == "failed"
    assert result["errors"]


def test_mixed_probe_indices_and_zero_call_contract() -> None:
    """验证 shared-only 行零 routed call 且索引重组保持。"""
    request = make_mixed_probe_request(batch=2, sequence=2, experts=5, top_k=4)
    result = ProbeResult(
        output_shape=(2, 2, 8),
        selected_ids=((), (), (0, 1, 2, 3), (0, 1, 2, 3)),
        expert_calls=tuple(
            ExpertCall(f"E{expert}", 1, 0, 2, "routed") for expert in range(4)
        ),
        original_indices=(0, 1, 2, 3),
        reassembled_indices=(0, 1, 2, 3),
        kv_bound=True,
    )
    validate_probe_result(request, result, 5)

    bad = ProbeResult(
        output_shape=result.output_shape,
        selected_ids=result.selected_ids,
        expert_calls=result.expert_calls + (ExpertCall("E4", 0, 0, 0, "routed"),),
        original_indices=result.original_indices,
        reassembled_indices=result.reassembled_indices,
        kv_bound=True,
    )
    with pytest.raises(P0Error, match="routed_call_not_zero"):
        validate_probe_result(request, bad, 5)


def test_reassembly_rejects_duplicate_or_reordered_indices() -> None:
    """索引重复、乱序或重组变化必须拒绝。"""
    with pytest.raises(ValueError):
        validate_reassembly_indices((0, 0), (0, 1), 2)
    with pytest.raises(ValueError):
        validate_reassembly_indices((1, 0), (1, 0), 2)
    with pytest.raises(ValueError):
        validate_reassembly_indices((0, 1), (1, 0), 2)


def test_snapshot_inventory_and_manifest_payload(tmp_path: Path) -> None:
    """snapshot 盘点只接受普通文件并计算摘要。"""
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    files = inventory_snapshot(root)
    assert files[0].path == "config.json"
    assert files[0].sha256 == hashlib.sha256(b"{}").hexdigest()
    assert not verify_read_only(root, files)
    payload = manifest_payload(
        SimpleNamespace(
            candidate_id="C1", repository_id="org/model", resolved_commit_sha="a" * 40
        ),
        root,
        files,
    )
    assert payload["total_bytes"] == 2


def test_preflight_is_explicitly_non_formal_and_non_overwriting(tmp_path: Path) -> None:
    """preflight 产物必须带 non_formal，且输出目录不可覆盖。"""
    result = PreflightResult(
        "infrastructure",
        True,
        "failed",
        {"network": False},
        ("offline_isolation_unavailable",),
    )
    output = tmp_path / "preflight"
    write_preflight_artifact(output, result)
    summary = json.loads(
        (output / "preflight_summary.json").read_text(encoding="utf-8")
    )
    assert summary["non_formal"] is True
    with pytest.raises(P0Error, match="artifact_exists"):
        write_preflight_artifact(output, result)


def test_infrastructure_preflight_requires_explicit_isolation_marker(
    monkeypatch,
) -> None:
    """仅设置离线变量不应伪造网络隔离通过。"""
    monkeypatch.delenv("CAN_NETWORK_ISOLATION_VERIFIED", raising=False)
    result = infrastructure_preflight()
    assert result.non_formal is True
    assert result.status == "failed"
    assert "network_isolation" in result.failure_codes


def test_deepseek_adapter_executes_only_allowed_real_experts() -> None:
    """C2 stand-in 必须调用实际 expert module，并跳过 shared-only routed 路径。"""
    import torch

    class Expert(torch.nn.Module):
        """记录调用次数的最小实际 expert。"""

        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)
            self.calls = 0

        def forward(self, hidden):
            """执行真实矩阵乘法并累积调用次数。"""
            self.calls += 1
            return self.linear(hidden)

    class Gate(torch.nn.Module):
        """提供 DeepSeek gate 所需冻结属性。"""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(3, 4))
            self.scoring_func = "softmax"
            self.norm_topk_prob = True

    class DeepseekMoE(torch.nn.Module):
        """模拟归档源码的 ModuleList/shared_experts 结构。"""

        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.num_experts_per_tok = 2
            self.gate = Gate()
            self.experts = torch.nn.ModuleList([Expert(), Expert(), Expert()])
            self.shared_experts = Expert()

        def moe_infer(self):
            """仅用于标识真实 C2 block 接口。"""

        def forward(self, hidden):
            """执行与归档源码等价的 eval top-k 与 shared 合并。"""
            flat = hidden.reshape(-1, 4)
            scores = torch.nn.functional.linear(flat, self.gate.weight).softmax(-1)
            weights, indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            routed = torch.zeros_like(flat)
            for row in range(flat.shape[0]):
                for position, expert_index in enumerate(indices[row].tolist()):
                    routed[row : row + 1].add_(
                        self.experts[expert_index](flat[row : row + 1])
                        * weights[row, position]
                    )
            return (routed + self.shared_experts(flat)).reshape_as(hidden)

    class Model(torch.nn.Module):
        """容纳一个真实可枚举的 MoE block。"""

        def __init__(self) -> None:
            super().__init__()
            self.block = DeepseekMoE()

    model = Model()
    architecture = ArchitectureMap(
        "deepseek",
        "Model",
        ("block",),
        ("block.gate",),
        ("block.shared_experts",),
        ("block.experts",),
        2,
        "eager",
        False,
        False,
        3,
    )
    loaded = SimpleNamespace(model=model, architecture=architecture)
    adapter = DeepSeekHostAdapter(loaded)
    inspected = adapter.inspect_architecture()
    assert inspected.supports_execution_counter is True
    all_allowed = adapter.observe_all_allowed()
    assert all_allowed["output_exact"] is True
    assert all_allowed["routed_actual_calls"] > 0
    for expert in model.block.experts:
        expert.calls = 0
    model.block.shared_experts.calls = 0
    request = make_mixed_probe_request(batch=2, sequence=2, experts=3, top_k=2)
    with adapter.install_probe():
        result = adapter.run_moe_probe(request)
    validate_probe_result(request, result, 3)
    assert sum(expert.calls for expert in model.block.experts) == 4
    assert model.block.shared_experts.calls == 2


def test_qwen_adapter_rejects_selected_ids_as_execution_evidence() -> None:
    """C1 packed container 不得把 router selection 当成逐 expert matmul 证据。"""
    loaded = SimpleNamespace(
        architecture=_architecture(architecture_family="qwen2_moe"),
        model=SimpleNamespace(),
    )
    adapter = QwenHostAdapter(loaded)
    assert adapter.inspect_architecture().supports_execution_counter is False
    request = make_mixed_probe_request(batch=2, sequence=1, experts=3, top_k=2)
    with adapter.install_probe():
        with pytest.raises(P0Error, match="expert_call_unobservable"):
            adapter.run_moe_probe(request)


def test_structure_probe_records_real_ledgers_and_failures() -> None:
    """P0-C 成功必须同时包含 all-allowed、mixed 和 KV 证据。"""
    request = make_mixed_probe_request(batch=2, sequence=2, experts=3, top_k=2)
    result = ProbeResult(
        (2, 2, 4),
        ((), (), (0, 1), (0, 1)),
        (
            ExpertCall("shared", 0, 0, 0, "shared"),
            ExpertCall("E0", 1, 0, 2, "routed"),
            ExpertCall("E1", 1, 0, 2, "routed"),
        ),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        True,
    )

    class Adapter:
        """提供完整 P0-C 证据的 stand-in adapter。"""

        def state_digest(self):
            return "stable"

        def observe_all_allowed(self):
            return {
                "output_exact": True,
                "routed_actual_calls": 1,
                "shared_actual_calls": 1,
            }

        @contextmanager
        def install_probe(self):
            yield

        def run_moe_probe(self, incoming):
            assert incoming.allowed_mask == request.allowed_mask
            return result

        def uninstall_probe(self):
            return None

        def verify_kv_semantics(self):
            return True

    report = inspect_real_structure(Adapter(), _architecture())
    assert report["passed"] is True
    assert report["probe_result"] == result

    report = inspect_real_structure(
        Adapter(), _architecture(supports_execution_counter=False)
    )
    assert report["passed"] is False
    assert "expert_call_unobservable" in report["failure_codes"]


def test_resource_hard_gates() -> None:
    """P0-D 必须拒绝超时、缺样本、显存过量、余量不足和 offload。"""
    good = ResourceSample(
        "complete", 1.0, 1, 1024, 1024, 2 * 1024**3, 16 * 1024**3, False, False
    )
    validate_resource_gates(1.0, 2.0, (good,))
    with pytest.raises(P0Error, match="load_timeout"):
        validate_resource_gates(1201.0, 2.0, (good,))
    with pytest.raises(P0Error, match="candidate_timeout"):
        validate_resource_gates(1.0, 5401.0, (good,))
    with pytest.raises(P0Error, match="resource_evidence_missing"):
        validate_resource_gates(1.0, 2.0, ())
    with pytest.raises(P0Error, match="offload_detected"):
        validate_resource_gates(1.0, 2.0, (replace(good, cpu_offload_detected=True),))
    with pytest.raises(P0Error, match="gpu_memory_limit_exceeded"):
        validate_resource_gates(
            1.0, 2.0, (replace(good, cuda_reserved_bytes=16 * 1024**3),)
        )
    with pytest.raises(P0Error, match="gpu_memory_headroom_insufficient"):
        validate_resource_gates(1.0, 2.0, (replace(good, gpu_free_bytes=1),))
    bf16_sample = replace(
        good,
        cuda_reserved_bytes=30 * 1024**3,
        gpu_free_bytes=9 * 1024**3,
    )
    validate_resource_gates(
        1.0,
        2.0,
        (bf16_sample,),
        max_reserved_bytes=36 * 1024**3,
        min_free_bytes=8 * 1024**3,
    )
    with pytest.raises(P0Error, match="gpu_memory_limit_exceeded"):
        validate_resource_gates(
            1.0,
            2.0,
            (bf16_sample,),
            max_reserved_bytes=29 * 1024**3,
            min_free_bytes=8 * 1024**3,
        )


def test_candidate_smoke_reports_quantization_and_observability() -> None:
    """候选 smoke 必须区分 NF4、offload 和实际执行可观测性。"""
    records = (
        {
            "is_params4bit": True,
            "quant_type": "nf4",
            "storage_bytes": 8,
        },
    )
    loaded = SimpleNamespace(
        packed_expert_records=records,
        parameter_devices=("cuda:0",),
        cpu_offload_detected=False,
        disk_offload_detected=False,
        architecture=_architecture(),
        load_seconds=1.0,
        parameter_dtypes=("torch.uint8",),
        quantization_class="nf4",
        cuda_peak_allocated=1,
        cuda_peak_reserved=2,
    )
    assert candidate_load_smoke(loaded).status == "passed"
    failed = candidate_load_smoke(
        loaded,
        _architecture(supports_execution_counter=False),
    )
    assert "expert_call_unobservable" in failed.failure_codes
    loaded.packed_expert_records = ({"is_params4bit": False, "quant_type": None},)
    assert "packed_expert_not_nf4" in candidate_load_smoke(loaded).failure_codes


def test_candidate_smoke_accepts_registered_bf16_experts() -> None:
    """BF16 profile 只接受 CUDA 上未量化的 BF16 expert 参数。"""
    candidate = validate_registry(_candidate_payload())[0]
    candidate = replace(
        candidate,
        profile=replace(
            candidate.profile,
            profile_id="c1-bf16-v1",
            quantization_config={},
        ),
    )
    loaded = SimpleNamespace(
        candidate=candidate,
        packed_expert_records=(
            {
                "is_params4bit": False,
                "quant_type": None,
                "dtype": "torch.bfloat16",
                "device": "cuda:0",
            },
        ),
        parameter_devices=("cuda:0",),
        cpu_offload_detected=False,
        disk_offload_detected=False,
        architecture=_architecture(),
        load_seconds=1.0,
        parameter_dtypes=("torch.bfloat16",),
        quantization_class="none",
        cuda_peak_allocated=1,
        cuda_peak_reserved=2,
    )
    assert candidate_load_smoke(loaded).status == "passed"
    loaded.packed_expert_records = (
        {
            "is_params4bit": False,
            "quant_type": None,
            "dtype": "torch.float16",
            "device": "cuda:0",
        },
    )
    assert "packed_expert_not_bf16" in candidate_load_smoke(loaded).failure_codes


def test_registry_profile_resource_budget_is_optional_and_strict() -> None:
    """旧 registry 使用历史门槛，新 profile 可登记独立资源门槛。"""
    payload = _candidate_payload()
    parsed = validate_registry(payload)
    assert parsed[0].profile.max_reserved_bytes == int(14.5 * 1024**3)
    payload["candidates"][0]["profile"].update(
        {"max_reserved_bytes": 36 * 1024**3, "min_free_bytes": 8 * 1024**3}
    )
    parsed = validate_registry(payload)
    assert parsed[0].profile.max_reserved_bytes == 36 * 1024**3
    assert parsed[0].profile.min_free_bytes == 8 * 1024**3


def test_loader_helpers_inspect_modules_devices_and_nf4() -> None:
    """loader 的结构、device 和 packed 参数盘点可在 CPU stand-in 上验证。"""
    import torch

    class Params4bit(torch.nn.Parameter):
        """提供 quant_state 的最小四比特参数类型。"""

    class Experts(torch.nn.Module):
        """暴露一个 packed expert 参数。"""

        def __init__(self):
            super().__init__()
            parameter = Params4bit(torch.zeros(2, 2), requires_grad=False)
            parameter.quant_type = "nf4"
            self.weight = parameter

    class Gate(torch.nn.Module):
        """暴露 top-k router 属性。"""

        def __init__(self):
            super().__init__()
            self.top_k = 2

    class FakeMoe(torch.nn.Module):
        """组合 shared/routed/router 结构。"""

        def __init__(self):
            super().__init__()
            self.gate = Gate()
            self.experts = Experts()
            self.shared_expert = torch.nn.Linear(2, 2)

    class Model(torch.nn.Module):
        """最小模型配置和模块树。"""

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_experts_per_tok=2, num_experts=3)
            self.block = FakeMoe()

    model = Model()
    candidate = validate_registry(_candidate_payload())[0]
    architecture = _infer_architecture(candidate, model)
    assert architecture.native_top_k == 2
    assert architecture.routed_expert_count == 3
    assert architecture.shared_paths
    assert _packed_expert_records(model)[0]["is_params4bit"] is True
    devices, dtypes, cpu, disk = _device_facts(model)
    assert devices == ("cpu",) and cpu is True and disk is False

    class Bits:
        """捕获冻结 NF4 构造参数。"""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    assert _quantization_config(candidate, Bits).kwargs["bnb_4bit_quant_type"] == "nf4"
    assert (
        _quantization_config(
            replace(
                candidate, profile=replace(candidate.profile, quantization_config={})
            ),
            Bits,
        )
        is None
    )


def test_snapshot_manifest_strict_load_verify_and_drift(
    tmp_path: Path, monkeypatch
) -> None:
    """snapshot manifest 必须绑定 registry identity 和完整 inventory。"""
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    candidate = validate_registry(_candidate_payload())[0]
    files = inventory_snapshot(root)
    payload = manifest_payload(candidate, root, files, True, True)
    path = root / "snapshot_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_snapshot_manifest(path)
    monkeypatch.setattr(
        "can.v2.pretrained_moe_p0.snapshot.verify_read_only", lambda root, files: True
    )
    verified = verify_snapshot_against_manifest(candidate, root, loaded)
    assert verified.total_bytes == 2
    (root / "config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(P0Error, match="snapshot_digest_mismatch"):
        verify_snapshot_against_manifest(candidate, root, loaded)


def test_formal_artifacts_write_ledgers_and_summary_sidecar(tmp_path: Path) -> None:
    """正式 artifact 必须包含调用 ledger 和可复核 summary sidecar。"""
    candidate = validate_registry(_candidate_payload())[0]
    result = RealRunResult(
        "C1",
        candidate.profile.profile_id,
        "failed",
        "passed",
        "passed",
        "failed",
        "not_run",
        ("expert_call_unobservable",),
        router_ledger=({"global_row": 0, "selected_ids": [0]},),
        expert_calls=(ExpertCall("E0", 0, 0, 0, "routed"),),
    )
    output = tmp_path / "run"
    hashes = write_run_artifacts(output, candidate, result, {"schema_version": 1})
    assert (output / "expert_calls.jsonl").read_text(encoding="utf-8")
    assert (output / "summary.json.sha256").read_text(
        encoding="ascii"
    ).strip() == hashes["summary_sha256"]


def test_prior_candidate_failure_requires_matching_sidecar(tmp_path: Path) -> None:
    """C2 只能消费 C1 的正式 failed summary 和匹配摘要。"""
    candidates = validate_registry(_candidate_payload())
    runner = RealP0Runner(candidates)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"candidate_id": "C1", "status": "failed", "failure_codes": ["x"]}),
        encoding="utf-8",
    )
    sidecar = tmp_path / "summary.json.sha256"
    sidecar.write_text(
        hashlib.sha256(summary.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    runner._validate_prior_failure(candidates[1], summary)
    sidecar.write_text("0" * 64, encoding="ascii")
    with pytest.raises(P0Error, match="snapshot_digest_mismatch"):
        runner._validate_prior_failure(candidates[1], summary)


def test_cross_process_p0b_requires_three_exact_signatures() -> None:
    """三次新进程结果必须逐 token、stop reason 和评分精确一致。"""
    record = GenerationRecord("c0", "x", "x", "x", True, (1,), (2,), "eos", False)
    capability = {
        "status": "passed",
        "groups": {"format": {"correct": 8, "total": 8}},
        "record_objects": (record,),
    }
    signature = RealP0Runner._p0b_signature(capability)
    RealP0Runner._validate_cross_process_p0b(
        capability, (signature, signature, signature)
    )
    with pytest.raises(P0Error, match="cross_process_evidence_missing"):
        RealP0Runner._validate_cross_process_p0b(capability, None)
    with pytest.raises(P0Error, match="cross_process_determinism_failed"):
        RealP0Runner._validate_cross_process_p0b(
            capability, (signature, signature, {"status": "failed"})
        )


def test_generic_adapter_real_kv_prefill_decode_and_negative_zero_call() -> None:
    """通用 KV probe 必须执行 prefill/decode，并在错误 identity 时零调用。"""
    import torch

    from can.v2.pretrained_moe_p0.real_adapter import TransformersHostAdapter

    class Tokenizer:
        """返回固定 tensor 编码。"""

        def __call__(self, text, return_tensors):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class Moe(torch.nn.Module):
        """产生实际 hook 调用的最小 MoE module。"""

        def forward(self, hidden):
            return hidden

    class Model(torch.nn.Module):
        """返回 logits 和 cache 的最小 causal host。"""

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.moe = Moe()

        def forward(
            self, input_ids, attention_mask=None, past_key_values=None, **kwargs
        ):
            hidden = self.moe(torch.ones(input_ids.shape[0], input_ids.shape[1], 2))
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
            return SimpleNamespace(logits=logits, past_key_values=(hidden,))

    loaded = SimpleNamespace(
        tokenizer=Tokenizer(),
        model=Model(),
        architecture=_architecture(moe_layers=("moe",)),
    )
    adapter = TransformersHostAdapter(loaded)
    assert adapter.verify_kv_semantics() is True


def test_formal_runner_writes_failure_for_infrastructure_gate(
    tmp_path: Path, monkeypatch
) -> None:
    """正式 run 在 snapshot 前失败也必须生成不可覆盖 summary。"""
    candidates = validate_registry(_candidate_payload())
    runner = RealP0Runner(candidates)
    monkeypatch.setattr(
        "can.v2.pretrained_moe_p0.real_runner.infrastructure_preflight",
        lambda root: PreflightResult(
            "infrastructure",
            True,
            "failed",
            {"network": False},
            ("network_isolation",),
        ),
    )
    output = tmp_path / "run"
    result = runner.run_formal_candidate("C1", tmp_path / "missing", tuple(), output)
    assert result.status == "failed"
    assert "offline_isolation_unavailable" in result.failure_codes
    assert (output / "summary.json.sha256").is_file()


def test_formal_runner_success_collects_generation_and_call_ledgers(
    tmp_path: Path, monkeypatch
) -> None:
    """正式成功路径必须汇总 P0-B records、P0-C ledgers 和 P0-D 资源。"""
    import can.v2.pretrained_moe_p0.real_runner as runner_module

    candidates = validate_registry(_candidate_payload())
    runner = RealP0Runner(candidates)
    architecture = _architecture(architecture_family="qwen2_moe")
    loaded = SimpleNamespace(
        model=SimpleNamespace(), architecture=architecture, load_seconds=1.0
    )
    record = GenerationRecord("c0", "x", "x", "x", True, (1,), (2,), "eos", False)
    capability = {
        "status": "passed",
        "groups": {"format": {"correct": 8, "total": 8}},
        "thresholds": {"format": 7},
        "errors": [],
        "record_objects": (record,),
        "in_process_run_count": 6,
        "in_process_difference_count": 0,
        "cache_difference_count": 0,
    }
    signature = RealP0Runner._p0b_signature(capability)
    probe = ProbeResult(
        (2, 1, 4),
        ((), (0, 1)),
        (ExpertCall("E0", 1, 0, 1, "routed"),),
        (0, 1),
        (0, 1),
        True,
    )
    good_sample = ResourceSample(
        "complete",
        1.0,
        1,
        1,
        1,
        2 * 1024**3,
        16 * 1024**3,
        False,
        False,
    )
    monkeypatch.setattr(
        runner_module,
        "infrastructure_preflight",
        lambda root: PreflightResult("infrastructure", True, "passed", {"x": True}),
    )
    monkeypatch.setattr(runner_module, "load_snapshot_manifest", lambda path: {})
    monkeypatch.setattr(
        runner_module,
        "verify_snapshot_against_manifest",
        lambda candidate, root, manifest: SimpleNamespace(read_only_verified=True),
    )
    monkeypatch.setattr(
        runner_module, "load_transformers_host", lambda candidate, root: loaded
    )
    monkeypatch.setattr(
        runner_module, "evaluate_fixture", lambda adapter, cases: capability
    )
    monkeypatch.setattr(
        runner_module,
        "inspect_real_structure",
        lambda adapter, arch: {
            "passed": True,
            "failure_codes": (),
            "gates": {
                "stable_moe_boundary": True,
                "mask_before_dispatch": True,
                "expert_zero_call_observable": True,
                "native_shared_branch": True,
                "mixed_batch_indices": True,
                "kv_identity_binding": True,
                "state_dict_stable": True,
            },
            "all_allowed": {
                "selected_ids": ((0, 1),),
                "routed_actual_calls": 2,
                "shared_actual_calls": 1,
            },
            "probe_result": probe,
        },
    )
    monkeypatch.setattr(runner_module, "sample_resource", lambda *args: good_sample)
    monkeypatch.setattr(
        runner_module,
        "validate_resource_gates",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        QwenHostAdapter, "inspect_architecture", lambda self: architecture
    )
    output = tmp_path / "run"
    result = runner.run_formal_candidate(
        "C1",
        tmp_path,
        tuple(),
        output,
        cross_process_p0b=(signature, signature, signature),
    )
    assert result.status == "passed"
    assert result.generations == (record,)
    assert result.expert_calls == probe.expert_calls
    assert (output / "router_ledger.jsonl").read_text(encoding="utf-8")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["exit_code"] == 0
    assert summary["p0b_metrics"]["thresholds"]["format"] == 7
    assert len(summary["p0c_gates"]) == 7
    assert summary["p0c_metrics"]["routed_actual_calls"] == 3
    assert summary["determinism"]["new_process_run_count"] == 3
    assert summary["resource_metrics"]["max_cuda_reserved_bytes"] == 1


def test_p0b_error_diagnostics_preserve_codes_without_sensitive_text() -> None:
    """P0-B 摘要应保存错误码计数和有限定位样本，不保存异常消息。"""
    capability = {
        "errors": [
            {
                "case_id": "format-00",
                "use_cache": "False",
                "repeat": "0",
                "code": "chat_template_rejected",
                "message": "secret prompt must not persist",
            },
            {
                "case_id": "format-01",
                "use_cache": "True",
                "repeat": "1",
                "code": "chat_template_rejected",
            },
            {
                "case_id": "single-00",
                "use_cache": "False",
                "repeat": "0",
                "code": "generation_failed",
            },
        ]
    }
    counts = RealP0Runner._p0b_error_code_counts(capability)
    examples = RealP0Runner._p0b_error_examples(capability)
    assert counts == {
        "chat_template_rejected": 2,
        "generation_failed": 1,
    }
    assert len(examples) == 3
    assert all(
        set(item) == {"case_id", "use_cache", "repeat", "code"} for item in examples
    )


def test_structure_false_boundary_always_has_stable_failure_code() -> None:
    """MoE 边界缺失时不得返回 false gate 加空失败码。"""

    class Adapter:
        """提供不会改变状态的最小失败 adapter。"""

        def state_digest(self):
            return "stable"

        def observe_all_allowed(self):
            raise P0Error("host_architecture_unresolved", "no boundary")

        @contextmanager
        def install_probe(self):
            yield

        def uninstall_probe(self):
            return None

    report = inspect_real_structure(
        Adapter(),
        _architecture(moe_layers=(), router_paths=(), routed_paths=()),
    )
    assert report["passed"] is False
    assert "host_architecture_unresolved" in report["failure_codes"]
    assert "batch_index_not_preserved" in report["failure_codes"]


def test_state_digest_detects_in_place_parameter_write() -> None:
    """state digest 必须借助 tensor version 发现原地权重写入。"""
    import torch

    from can.v2.pretrained_moe_p0.real_adapter import TransformersHostAdapter

    model = torch.nn.Linear(2, 2, bias=False)
    loaded = SimpleNamespace(model=model, architecture=_architecture())
    adapter = TransformersHostAdapter(loaded)
    before = adapter.state_digest()
    with torch.no_grad():
        model.weight.add_(1)
    assert adapter.state_digest() != before


def test_deepseek_probe_restores_mode_and_disables_grad() -> None:
    """C2 mixed probe 必须无梯度执行，并恢复原 block 训练模式。"""
    import torch

    grad_states = []

    class Expert(torch.nn.Module):
        """记录 forward 时梯度开关的最小 expert。"""

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(4))

        def forward(self, hidden):
            grad_states.append(torch.is_grad_enabled())
            return hidden @ self.weight

    class Gate(torch.nn.Module):
        """提供 softmax gate 属性。"""

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(3, 4))
            self.scoring_func = "softmax"
            self.norm_topk_prob = True

    class DeepseekMoE(torch.nn.Module):
        """提供 C2 probe 所需的实际 ModuleList。"""

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.num_experts_per_tok = 2
            self.gate = Gate()
            self.experts = torch.nn.ModuleList([Expert(), Expert(), Expert()])
            self.shared_experts = Expert()

        def moe_infer(self):
            """标识 DeepSeek MoE block。"""

    class Model(torch.nn.Module):
        """容纳一个 DeepSeek block。"""

        def __init__(self):
            super().__init__()
            self.block = DeepseekMoE()

    model = Model()
    model.train()
    adapter = DeepSeekHostAdapter(
        SimpleNamespace(model=model, architecture=_architecture())
    )
    request = make_mixed_probe_request(batch=2, sequence=1, experts=3, top_k=2)
    with adapter.install_probe():
        adapter.run_moe_probe(request)
    assert model.block.training is True
    assert grad_states and not any(grad_states)


def test_snapshot_budget_and_readonly_freeze(tmp_path: Path, monkeypatch) -> None:
    """下载前预算、P0 总预算与只读冻结均应 fail closed。"""
    candidate = validate_registry(_candidate_payload())[0]
    root = tmp_path / "snapshots" / "C1"
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    manifest = write_snapshot_manifest(
        candidate, root, root / "snapshot_manifest.json", freeze=True
    )
    assert manifest.read_only_verified is True
    assert verify_read_only(root, manifest.files)
    assert _existing_snapshot_bytes(root.parent) == 2

    second = root.parent / "C2"
    monkeypatch.setattr(
        "can.v2.pretrained_moe_p0.snapshot.shutil.disk_usage",
        lambda path: SimpleNamespace(free=4096),
    )
    validate_download_budget(candidate, second, 1)
    with pytest.raises(P0Error, match="snapshot_size_invalid"):
        validate_download_budget(candidate, second, candidate.max_snapshot_bytes + 1)
    monkeypatch.setattr(
        "can.v2.pretrained_moe_p0.snapshot._existing_snapshot_bytes",
        lambda parent: 80 * 1024**3,
    )
    with pytest.raises(P0Error, match="snapshot_total_budget_exceeded"):
        validate_download_budget(candidate, second, 1)


def test_declared_snapshot_size_requires_complete_fixed_revision(monkeypatch) -> None:
    """Hub 声明大小必须完整且绑定 registry 的 resolved commit。"""
    candidate = validate_registry(_candidate_payload())[0]

    class Api:
        """返回可控 Hub metadata。"""

        def model_info(self, repo_id, revision, files_metadata):
            return SimpleNamespace(
                sha=candidate.resolved_commit_sha,
                siblings=(SimpleNamespace(size=10), SimpleNamespace(size=20)),
            )

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    assert _declared_snapshot_size(candidate) == 30

    class BadApi(Api):
        """返回缺失文件大小的 Hub metadata。"""

        def model_info(self, repo_id, revision, files_metadata):
            return SimpleNamespace(
                sha=candidate.resolved_commit_sha,
                siblings=(SimpleNamespace(size=None),),
            )

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=BadApi))
    with pytest.raises(P0Error, match="snapshot_metadata_incomplete"):
        _declared_snapshot_size(candidate)


def test_formal_runner_rejects_expired_controller_budget(
    tmp_path: Path,
) -> None:
    """正式 runner 在任何加载前拒绝超过候选与 P0 总预算的调用。"""
    runner = RealP0Runner(validate_registry(_candidate_payload()))
    result = runner.run_formal_candidate(
        "C1",
        tmp_path / "missing",
        tuple(),
        tmp_path / "run",
        controller_elapsed_seconds=4 * 60 * 60 + 1,
    )
    assert result.status == "failed"
    assert result.failure_codes == ("p0_total_timeout",)
    assert result.exit_code == 2


def test_worker_failures_do_not_persist_stderr(monkeypatch, tmp_path: Path) -> None:
    """worker 崩溃证据不得保存可能含 token 的 stderr。"""
    from scripts.run_p0_moe_real import _collect_p0b_workers

    args = SimpleNamespace(
        registry=tmp_path / "registry.json",
        fixture=tmp_path / "fixture.json",
        candidate_id="C1",
        snapshot_root=tmp_path / "snapshot",
        output_root=tmp_path / "output",
        prior_candidate_summary=None,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr="SECRET_TOKEN=must-not-persist"
        ),
    )
    results = _collect_p0b_workers(args, time.monotonic())
    assert len(results) == 3
    assert all(item["worker_error"] == "worker_failed" for item in results)
    assert "SECRET_TOKEN" not in json.dumps(results)
