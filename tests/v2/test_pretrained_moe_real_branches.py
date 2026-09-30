"""P0-MoE.14 真实宿主 runner 的纯 CPU 分支与失败语义测试。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from can.v2.pretrained_moe_p0.capability_eval import (
    _input_ids,
    _safe_decode,
    _tensor_to_list,
    evaluate_fixture,
    generate_one,
)
from can.v2.pretrained_moe_p0.deepseek_adapter import DeepSeekHostAdapter
from can.v2.pretrained_moe_p0.fixture import validate_fixture
from can.v2.pretrained_moe_p0.qwen_adapter import QwenHostAdapter
from can.v2.pretrained_moe_p0.real_adapter import TransformersHostAdapter
from can.v2.pretrained_moe_p0.real_loader import (
    _device_facts,
    _infer_architecture,
    _module_path_records,
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
    ProbeRequest,
    ProbeResult,
    ResourceSample,
    SnapshotFile,
    tensor_index_contract,
    validate_reassembly_indices,
)
from can.v2.pretrained_moe_p0.registry import validate_registry
from can.v2.pretrained_moe_p0.resource_probe import (
    candidate_load_smoke,
    infrastructure_preflight,
    sample_resource,
)
from can.v2.pretrained_moe_p0.snapshot import (
    _declared_snapshot_size,
    download_snapshot,
    inventory_snapshot,
    load_snapshot_manifest,
    validate_download_budget,
    verify_snapshot_against_manifest,
    write_snapshot_manifest,
)
from can.v2.pretrained_moe_p0.structure_probe import (
    allowed_mask_for_mixed,
    inspect_real_structure,
    make_mixed_probe_request,
    validate_probe_result,
)
from can.v2.pretrained_moe_p0.types import P0Error


def _candidate_payload(status: str = "passed") -> dict[str, Any]:
    """构造含 C1/C2 的最小严格 registry。"""

    candidates = []
    for order in (1, 2):
        failures = [] if status == "passed" else ["human_review_required"]
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
                "license_review_status": (
                    "approved" if status == "passed" else "machine_detected"
                ),
                "metadata_source_sha256": "b" * 64,
                "p0a_status": status,
                "p0a_decision_sha256": "c" * 64,
                "p0a_failure_codes": failures,
            }
        )
    return {
        "schema_version": 1,
        "registry_id": "p0-moe-host-v1",
        "candidates": candidates,
    }


def _architecture(**changes: Any) -> ArchitectureMap:
    """构造可按字段替换的结构摘要。"""

    return replace(
        ArchitectureMap(
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
        ),
        **changes,
    )


def _request() -> GenerationRequest:
    """返回固定生成请求。"""

    return GenerationRequest("case", "system", "user", "ok", "strict_em", 4, True)


@pytest.mark.parametrize("value", [None, [1, "x"], "123"])
def test_tensor_to_list_rejects_invalid_values(value: Any) -> None:
    """token 序列必须为严格整数列表。"""

    with pytest.raises(P0Error, match="generation_tensor_invalid"):
        _tensor_to_list(value)


def test_generation_helpers_cover_templates_tensors_and_failures() -> None:
    """生成 helper 应覆盖 chat template、tensor 和长度失败分支。"""

    import torch

    with pytest.raises(P0Error, match="chat_template_missing"):
        _input_ids(SimpleNamespace(), _request())
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: torch.tensor([[1, 2]]),
        decode=lambda ids, **kwargs: "ok",
        eos_token_id=None,
    )
    assert _tensor_to_list(torch.tensor([[1, 2]])) == [1, 2]
    model = SimpleNamespace(generate=lambda **kwargs: torch.tensor([[1]]))
    with pytest.raises(P0Error, match="输出短于 prompt"):
        generate_one(tokenizer, model, _request())
    long_tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: list(range(257)),
        decode=tokenizer.decode,
        eos_token_id=None,
    )
    with pytest.raises(P0Error, match="prompt_too_long"):
        generate_one(long_tokenizer, model, _request())
    broken = SimpleNamespace(
        decode=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError())
    )
    with pytest.raises(P0Error, match="generation_decode_failed"):
        _safe_decode(broken, (1,))


def test_generation_helpers_classify_input_output_and_length_failures() -> None:
    """生成失败必须区分输入结构、输出结构和输出长度。"""

    import torch

    request = _request()
    invalid_input = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: {"input_ids": {"bad": True}},
        decode=lambda ids, **kwargs: "ok",
        eos_token_id=None,
    )
    with pytest.raises(P0Error) as input_error:
        generate_one(
            invalid_input, SimpleNamespace(generate=lambda **kwargs: None), request
        )
    assert input_error.value.code == "input_tensor_invalid"
    assert input_error.value.details["value"]["type"] == "dict"

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: {
            "input_ids": torch.tensor([[1, 2]])
        },
        decode=lambda ids, **kwargs: "ok",
        eos_token_id=None,
    )
    with pytest.raises(P0Error) as output_error:
        generate_one(
            tokenizer, SimpleNamespace(generate=lambda **kwargs: {"bad": True}), request
        )
    assert output_error.value.code == "output_tensor_invalid"

    with pytest.raises(P0Error) as length_error:
        generate_one(
            tokenizer,
            SimpleNamespace(generate=lambda **kwargs: torch.tensor([[1]])),
            request,
        )
    assert length_error.value.code == "output_shorter_than_prompt"
    assert length_error.value.details == {
        "prompt_length": 2,
        "output_length": 1,
        "output": {"type": "Tensor", "shape": [1, 1], "ndim": 2},
    }


def test_generate_one_excludes_eos_from_answer_text_but_keeps_audit_tokens() -> None:
    """EOS 不得进入答案匹配，但完整 continuation 必须保留。"""

    class EosTokenizer:
        """返回可观察 EOS 标记的最小 tokenizer。"""

        eos_token_id = 9

        def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
            """返回固定 prompt。"""
            return {"input_ids": [[1]], "attention_mask": [[1]]}

        def decode(self, values: Any, **kwargs: Any) -> str:
            """模拟 tokenizer 将 EOS 解码为控制标记。"""
            return "answer<|im_end|>" if 9 in values else "answer"

    model = SimpleNamespace(generate=lambda **kwargs: [[1, 7, 9, 8]])
    request = GenerationRequest("eos", "s", "u", "answer", "strict_em", 4, False)
    record = generate_one(EosTokenizer(), model, request)
    assert record.matched is True
    assert record.generated_text == "answer"
    assert record.continuation_tokens == (7, 9, 8)
    assert record.stop_reason == "eos"


def test_evaluate_fixture_keeps_cache_difference_as_diagnostic() -> None:
    """P0-B 以 canonical cache 模式验收，跨模式差异仅作诊断。"""

    cases = []
    for group in ("format", "single", "two"):
        for index in range(8):
            cases.append(
                SimpleNamespace(
                    case_id=f"{group}-{index}",
                    system_text="s",
                    user_text="u",
                    expected_text="ok",
                    metric="strict_em",
                    max_new_tokens=4,
                )
            )

    class Adapter:
        """根据 cache 模式返回可控 token 签名。"""

        different = False

        def baseline_generate(self, request: GenerationRequest) -> GenerationRecord:
            token = 2 if self.different and request.use_cache else 1
            return GenerationRecord(
                request.case_id,
                "ok",
                "ok",
                "ok",
                True,
                (0,),
                (token,),
                "eos",
                request.use_cache,
            )

    adapter = Adapter()
    result = evaluate_fixture(adapter, cases)
    assert result["status"] == "passed"
    assert result["cache_difference_count"] == 0
    adapter.different = True
    result = evaluate_fixture(adapter, cases)
    assert result["status"] == "passed"
    assert result["cache_difference_count"] == 1
    assert result["canonical_use_cache"] is True
    assert result["diagnostic_deterministic"] is True


@pytest.mark.parametrize(
    ("probe_request", "code"),
    [
        (ProbeRequest((0, 1, 1), (), 2, (), ()), "probe_fixture_invalid"),
        (
            ProbeRequest(
                (2, 1, 1), (((False,),), ((False,),)), 1, ((True,), (True,)), ("a",)
            ),
            "request_binding_invalid",
        ),
        (
            ProbeRequest((2, 1, 1), (((False,),),), 1, ((True,), (True,)), ("a", "b")),
            "mask_shape_invalid",
        ),
        (
            ProbeRequest(
                (2, 1, 1), (((True,),), ((True,),)), 2, ((True,), (True,)), ("a", "b")
            ),
            "topk_changed",
        ),
    ],
)
def test_deepseek_request_validation_fail_closed(
    probe_request: ProbeRequest, code: str
) -> None:
    """C2 probe 的结构和绑定错误应有稳定失败码。"""

    with pytest.raises(P0Error, match=code):
        DeepSeekHostAdapter._validate_request(probe_request, 1, 1)


def test_deepseek_rejects_missing_block_and_non_softmax() -> None:
    """C2 必须拒绝缺失 MoE block 与未知 gate 评分函数。"""

    import torch

    adapter = DeepSeekHostAdapter(
        SimpleNamespace(model=torch.nn.Linear(2, 2), architecture=_architecture())
    )
    with pytest.raises(P0Error, match="host_architecture_unresolved"):
        adapter.inspect_architecture()
    block = SimpleNamespace(
        gate=SimpleNamespace(weight=torch.zeros(2, 2), scoring_func="sigmoid"),
        num_experts_per_tok=1,
    )
    with pytest.raises(P0Error, match="host_interface_not_controllable"):
        DeepSeekHostAdapter._masked_routing(
            block, torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool)
        )


def test_probe_context_bridge_and_reentry_contract() -> None:
    """通用 probe bridge 应拒绝未安装/重入/非法返回并执行卸载。"""

    removed = []
    request = make_mixed_probe_request(batch=2, sequence=1, experts=3, top_k=2)
    valid = ProbeResult((2, 1, 8), ((), (0, 1)), (), (0, 1), (0, 1), True)
    model = SimpleNamespace(
        can_p0_moe_probe=True,
        run_can_p0_moe_probe=lambda incoming: valid,
        remove_can_p0_moe_probe=lambda: removed.append(True),
    )
    adapter = TransformersHostAdapter(
        SimpleNamespace(model=model, architecture=_architecture())
    )
    with pytest.raises(P0Error, match="probe_not_installed"):
        adapter.run_moe_probe(request)
    with adapter.install_probe():
        assert adapter.run_moe_probe(request) is valid
        with pytest.raises(P0Error, match="probe_reentry_rejected"):
            with adapter.install_probe():
                pass
    assert removed == [True]
    model.run_can_p0_moe_probe = lambda incoming: object()
    with adapter.install_probe():
        with pytest.raises(P0Error, match="probe_result_invalid"):
            adapter.run_moe_probe(request)
    model.can_p0_moe_probe = False
    with adapter.install_probe():
        with pytest.raises(P0Error, match="host_interface_not_controllable"):
            adapter.run_moe_probe(request)
    with pytest.raises(P0Error, match="expert_call_unobservable"):
        adapter.observe_all_allowed()


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"selected_ids": ((0,), (0, 1))}, "routed_call_not_zero"),
        ({"selected_ids": ((), (0,))}, "topk_changed"),
        ({"selected_ids": ((), (1, 3))}, "forbidden_expert_selected"),
        ({"selected_ids": ((),)}, "probe_result_invalid"),
        ({"kv_bound": False}, "kv_semantics_unsupported"),
    ],
)
def test_probe_result_failure_matrix(change: dict[str, Any], code: str) -> None:
    """selected、zero-call、top-k 与 KV 错误应逐项拒绝。"""

    request = make_mixed_probe_request(batch=2, sequence=1, experts=3, top_k=2)
    result = ProbeResult((2, 1, 8), ((), (0, 1)), (), (0, 1), (0, 1), True)
    result = replace(result, **change)
    with pytest.raises(P0Error, match=code):
        validate_probe_result(request, result, 3)


def test_probe_fixture_and_reassembly_input_validation() -> None:
    """mixed fixture 和重组索引应拒绝非法维度与类型。"""

    with pytest.raises(P0Error, match="mask_shape_invalid"):
        allowed_mask_for_mixed(2, 1, 1, 2)
    with pytest.raises(P0Error, match="probe_fixture_invalid"):
        make_mixed_probe_request(batch=3)
    with pytest.raises(ValueError, match="valid_count"):
        validate_reassembly_indices((), (), -1)
    with pytest.raises(ValueError, match="非负整数"):
        validate_reassembly_indices((-1,), (-1,), 1)


def test_snapshot_manifest_strict_failure_matrix(tmp_path: Path) -> None:
    """snapshot manifest 应拒绝缺失、重复、非法 JSON 和错误 schema。"""

    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_snapshot_manifest(missing)
    cases = {
        "duplicate.json": '{"schema_version":1,"schema_version":1}',
        "invalid.json": "{",
        "version.json": '{"schema_version":2}',
        "fields.json": '{"schema_version":1}',
    }
    for name, raw in cases.items():
        path = tmp_path / name
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(P0Error):
            load_snapshot_manifest(path)


def test_snapshot_inventory_and_budget_failure_matrix(
    tmp_path: Path, monkeypatch
) -> None:
    """snapshot 路径、覆盖、磁盘和绑定错误都应 fail closed。"""

    candidate = validate_registry(_candidate_payload())[0]
    with pytest.raises(P0Error, match="snapshot_missing"):
        inventory_snapshot(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(P0Error, match="snapshot_empty"):
        inventory_snapshot(empty)
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    output = root / "snapshot_manifest.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(P0Error, match="artifact_exists"):
        write_snapshot_manifest(candidate, root, output)
    monkeypatch.setattr(
        "can.v2.pretrained_moe_p0.snapshot.shutil.disk_usage",
        lambda path: SimpleNamespace(free=0),
    )
    with pytest.raises(P0Error, match="snapshot_disk_space_insufficient"):
        validate_download_budget(candidate, tmp_path / "budget" / "new", 1)


def test_snapshot_verification_binding_and_readonly_failures(
    tmp_path: Path, monkeypatch
) -> None:
    """snapshot 二次验证应绑定候选、仓库、revision、只读和总大小。"""

    candidate = validate_registry(_candidate_payload())[0]
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    files = inventory_snapshot(root)
    manifest = {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "repository_id": candidate.repository_id,
        "resolved_commit_sha": candidate.resolved_commit_sha,
        "root": str(root),
        "files": [item.__dict__ for item in files],
        "total_bytes": 2,
        "read_only_verified": True,
        "offline_verified": True,
    }
    for field, value, code in (
        ("candidate_id", "bad", "snapshot_binding_mismatch"),
        ("repository_id", "bad", "snapshot_binding_mismatch"),
        ("resolved_commit_sha", "0" * 40, "revision_not_immutable"),
    ):
        changed = dict(manifest)
        changed[field] = value
        with pytest.raises(P0Error, match=code):
            verify_snapshot_against_manifest(candidate, root, changed)
    monkeypatch.setattr(
        "can.v2.pretrained_moe_p0.snapshot.verify_read_only", lambda *args: False
    )
    with pytest.raises(P0Error, match="snapshot_not_read_only"):
        verify_snapshot_against_manifest(candidate, root, manifest)


def test_declared_snapshot_revision_and_download_preconditions(
    tmp_path: Path, monkeypatch
) -> None:
    """下载入口应拒绝 revision 漂移、非 passed 与已有目录。"""

    candidate = validate_registry(_candidate_payload())[0]

    class Api:
        """返回漂移 revision。"""

        def model_info(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(sha="0" * 40, siblings=(SimpleNamespace(size=1),))

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    with pytest.raises(P0Error, match="revision_not_immutable"):
        _declared_snapshot_size(candidate)
    provisional = validate_registry(_candidate_payload("provisional"))[0]
    with pytest.raises(P0Error, match="p0a_rejected"):
        download_snapshot(provisional, tmp_path / "new")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(P0Error, match="artifact_exists"):
        download_snapshot(candidate, existing)


def test_resource_preflight_and_sampling(monkeypatch, tmp_path: Path) -> None:
    """基础设施 preflight 与资源采样应记录环境、CUDA 和 offload。"""

    for name in (
        "CAN_NETWORK_ISOLATION_VERIFIED",
        "CAN_READONLY_SNAPSHOT_VERIFIED",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        monkeypatch.setenv(name, "1")
    result = infrastructure_preflight(tmp_path)
    assert result.details["snapshot_root"] == str(tmp_path.resolve())
    assert result.checks["snapshot_root_exists"] is True
    sample = sample_resource(
        "stage",
        0.0,
        SimpleNamespace(hf_device_map={"layer.0": "cpu", "layer.1": "disk"}),
    )
    assert sample.cpu_offload_detected and sample.disk_offload_detected


def test_candidate_smoke_defaults_to_loaded_architecture() -> None:
    """candidate smoke 未显式给 architecture 时应使用加载结果。"""

    loaded = SimpleNamespace(
        packed_expert_records=(),
        architecture=_architecture(native_top_k=0, supports_execution_counter=False),
        parameter_devices=("cpu",),
        cpu_offload_detected=True,
        disk_offload_detected=True,
        load_seconds=1.0,
        parameter_dtypes=("float32",),
        quantization_class="none",
        cuda_peak_allocated=None,
        cuda_peak_reserved=None,
    )
    result = candidate_load_smoke(loaded)
    assert set(result.failure_codes) == {
        "packed_expert_not_nf4",
        "offload_detected",
        "expert_call_unobservable",
        "native_top_k_unresolved",
    }


def test_loader_helper_failure_and_architecture_branches() -> None:
    """loader helper 应拒绝无模块树并解析 module/config 的不同提示。"""

    import torch

    candidate = validate_registry(_candidate_payload())[0]
    with pytest.raises(P0Error, match="host_architecture_unresolved"):
        _module_path_records(SimpleNamespace())

    class Marker(torch.nn.Module):
        """暴露 expert id、mask bridge 和 backend。"""

        def __init__(self) -> None:
            super().__init__()
            self.expert_id = 1
            self.use_experts_implementation = "native"

        def set_allowed_mask(self, mask: Any) -> None:
            """提供可识别的 pre-dispatch mask API。"""

    class Model(torch.nn.Module):
        """提供 config fallback 字段。"""

        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(num_selected_experts=2, n_routed_experts=3)
            self.moe = Marker()
            self.sharedexpert = torch.nn.Linear(1, 1)

    architecture = _infer_architecture(candidate, Model())
    assert architecture.backend_id == "native"
    assert architecture.supports_row_mask is True
    assert architecture.supports_execution_counter is True
    assert architecture.native_top_k == 2
    assert architecture.routed_expert_count == 3


def test_quantization_and_device_helper_failure_branches() -> None:
    """量化 profile 和 device inventory 应拒绝未知配置并识别 meta。"""

    import torch

    candidate = validate_registry(_candidate_payload())[0]
    with pytest.raises(P0Error, match="optional_dependency_missing"):
        _quantization_config(candidate, None)
    bad = replace(
        candidate,
        profile=replace(
            candidate.profile,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "fp4",
                "bnb_4bit_compute_dtype": "float16",
            },
        ),
    )
    with pytest.raises(P0Error, match="quantization_profile_unverifiable"):
        _quantization_config(bad, object())

    class Model:
        """返回 CPU 与 meta tensor。"""

        def parameters(self):
            return (torch.ones(1),)

        def buffers(self):
            return (torch.empty(1, device="meta"),)

    devices, _, cpu, disk = _device_facts(Model())
    assert devices == ("cpu", "meta") and cpu and disk
    assert _packed_expert_records(torch.nn.Linear(1, 1)) == ()


def test_runner_selection_preflights_and_worker_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """runner 应覆盖候选选择、两类 preflight 和 P0-B worker 适配器。"""

    import can.v2.pretrained_moe_p0.real_runner as module

    candidates = validate_registry(_candidate_payload())
    with pytest.raises(P0Error, match="registry_order_invalid"):
        RealP0Runner(())
    runner = RealP0Runner(candidates)
    with pytest.raises(P0Error, match="candidate_unknown"):
        runner._candidate(candidates, "missing")
    provisional = validate_registry(_candidate_payload("provisional"))
    with pytest.raises(P0Error, match="p0a_rejected"):
        runner._candidate(provisional, "C1")

    preflight = PreflightResult("infrastructure", True, "passed", {"ok": True})
    monkeypatch.setattr(
        module, "infrastructure_preflight", lambda root=None, **kwargs: preflight
    )
    monkeypatch.setattr(module, "write_preflight_artifact", lambda *args: None)
    assert runner.run_infrastructure_preflight(tmp_path) is preflight

    architecture = _architecture()
    loaded = SimpleNamespace(model=SimpleNamespace(), architecture=architecture)
    monkeypatch.setattr(module, "load_snapshot_manifest", lambda path: {})
    monkeypatch.setattr(module, "verify_snapshot_against_manifest", lambda *args: None)
    monkeypatch.setattr(module, "load_transformers_host", lambda *args: loaded)
    smoke = PreflightResult("candidate", True, "passed", {"ok": True})
    monkeypatch.setattr(module, "candidate_load_smoke", lambda *args: smoke)
    monkeypatch.setattr(
        DeepSeekHostAdapter, "inspect_architecture", lambda self: architecture
    )
    assert runner.run_candidate_preflight("C2", tmp_path, tmp_path / "out") is smoke

    record = GenerationRecord("c", "x", "x", "x", True, (), (1,), "eos", True)
    capability = {"status": "passed", "groups": {}, "record_objects": (record,)}
    monkeypatch.setattr(module, "evaluate_fixture", lambda *args: capability)
    signature = runner.run_p0b_worker("C1", tmp_path, ())
    assert signature["signatures"][0]["continuation_tokens"] == [1]
    monkeypatch.setattr(
        module,
        "infrastructure_preflight",
        lambda root=None: PreflightResult(
            "infrastructure", True, "failed", {"network": False}
        ),
    )
    with pytest.raises(P0Error, match="offline_isolation_unavailable"):
        runner.run_p0b_worker("C1", tmp_path, ())


def test_prior_failure_missing_sidecar_and_invalid_payload(tmp_path: Path) -> None:
    """C2 前序失败摘要必须有 sidecar 且内容代表正式失败。"""

    candidates = validate_registry(_candidate_payload())
    runner = RealP0Runner(candidates)
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    with pytest.raises(P0Error, match="prior_candidate_failure_missing"):
        runner._validate_prior_failure(candidates[1], summary)
    sidecar = tmp_path / "summary.json.sha256"
    sidecar.write_text(
        hashlib.sha256(summary.read_bytes()).hexdigest(), encoding="ascii"
    )
    with pytest.raises(P0Error, match="prior_candidate_failure_invalid"):
        runner._validate_prior_failure(candidates[1], summary)


def test_structure_failure_gates_and_state_mutation() -> None:
    """P0-C 应同时报告 all-allowed、结构能力和状态篡改失败。"""

    class Adapter:
        """返回失败 all-allowed 并在探查后改变摘要。"""

        calls = 0

        def state_digest(self):
            self.calls += 1
            return str(self.calls)

        def observe_all_allowed(self):
            return {
                "output_exact": False,
                "routed_actual_calls": 0,
                "shared_actual_calls": 0,
            }

        @contextmanager
        def install_probe(self):
            yield

        def uninstall_probe(self):
            return None

    architecture = _architecture(
        native_top_k=0,
        supports_row_mask=False,
        supports_execution_counter=False,
        shared_paths=(),
    )
    report = inspect_real_structure(Adapter(), architecture)
    assert report["passed"] is False
    assert "native_top_k_unresolved" in report["failure_codes"]
    assert "native_shared_expert_missing" in report["failure_codes"]
    assert "host_interface_not_controllable" in report["failure_codes"]


def test_qwen_and_generic_adapter_uninstalled_paths() -> None:
    """候选 adapter 未安装 probe 时必须直接拒绝，baseline 应委托生成器。"""

    import torch

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: torch.tensor([[1]]),
        decode=lambda ids, **kwargs: "ok",
        eos_token_id=None,
    )
    model = SimpleNamespace(generate=lambda **kwargs: torch.tensor([[1, 2]]))
    loaded = SimpleNamespace(
        tokenizer=tokenizer, model=model, architecture=_architecture()
    )
    generic = TransformersHostAdapter(loaded)
    assert generic.baseline_generate(_request()).generated_text == "ok"
    qwen = QwenHostAdapter(loaded)
    with pytest.raises(P0Error, match="probe_not_installed"):
        qwen.run_moe_probe(make_mixed_probe_request())


def test_cuda_peak_reset_selects_device_before_reset() -> None:
    """验证真实 loader 先选中设备，再重置 CUDA 峰值统计。"""

    from can.v2.pretrained_moe_p0.real_loader import _reset_cuda_peak_memory

    events = []

    class FakeCuda:
        """记录 CUDA 设备调用顺序的最小替身。"""

        def set_device(self, device_index: int) -> None:
            """记录显式设备选择。"""
            events.append(("set_device", device_index))

        def reset_peak_memory_stats(self, device_index: int) -> None:
            """记录峰值统计重置。"""
            events.append(("reset_peak_memory_stats", device_index))

    _reset_cuda_peak_memory(SimpleNamespace(cuda=FakeCuda()), 0)
    assert events == [("set_device", 0), ("reset_peak_memory_stats", 0)]


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("not_callable", "tokenizer 不支持"),
        ("encode_error", "KV probe 编码失败"),
        ("missing_ids", "缺少 input_ids"),
        ("missing_cache", "prefill 未返回"),
        ("no_prefill_call", "prefill 未经过"),
        ("missing_decode_cache", "decode 未返回"),
        ("no_decode_call", "decode 未经过"),
        ("runtime_error", "真实 KV prefill/decode 失败"),
    ],
)
def test_generic_kv_failure_matrix(mode: str, message: str) -> None:
    """KV probe 的编码、prefill、decode 和调用证据缺失应稳定拒绝。"""

    import torch

    class Moe(torch.nn.Module):
        """提供可计数的 MoE 边界。"""

        def forward(self, hidden: Any) -> Any:
            return hidden

    class Model(torch.nn.Module):
        """根据 mode 构造不同 KV 失败。"""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.moe = Moe()
            self.calls = 0

        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if mode == "runtime_error":
                raise RuntimeError("boom")
            should_call = mode != "no_prefill_call" and not (
                mode == "no_decode_call" and self.calls > 1
            )
            if should_call:
                self.moe(torch.ones(input_ids.shape[0], input_ids.shape[1], 2))
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
            cache = (logits,) if mode != "missing_cache" else None
            if mode == "missing_decode_cache" and self.calls > 1:
                cache = None
            return SimpleNamespace(logits=logits, past_key_values=cache)

    if mode == "not_callable":
        tokenizer: Any = object()
    elif mode == "encode_error":
        tokenizer = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError())
    elif mode == "missing_ids":
        tokenizer = lambda *args, **kwargs: {}
    else:
        tokenizer = lambda *args, **kwargs: {"input_ids": torch.tensor([[1, 2]])}
    adapter = TransformersHostAdapter(
        SimpleNamespace(
            tokenizer=tokenizer,
            model=Model(),
            architecture=_architecture(moe_layers=("moe",)),
        )
    )
    with pytest.raises(P0Error, match=message):
        adapter.verify_kv_semantics()


def test_deepseek_remaining_validation_branches() -> None:
    """C2 probe 应拒绝 sequence/expert mask 与允许数量错误。"""

    base = make_mixed_probe_request(batch=2, sequence=1, experts=3, top_k=2)
    cases = (
        replace(base, allowed_mask=(base.allowed_mask[0], ())),
        replace(
            base,
            allowed_mask=(((False, False),), ((True, True),)),
        ),
        replace(
            base,
            allowed_mask=(((False, False, False),), ((True, False, False),)),
        ),
    )
    for incoming in cases:
        with pytest.raises(P0Error):
            DeepSeekHostAdapter._validate_request(incoming, 3, 2)


def test_deepseek_single_topk_skips_normalization() -> None:
    """top-k=1 时应保持原权重，不执行多 expert 归一化分支。"""

    import torch

    block = SimpleNamespace(
        gate=SimpleNamespace(
            weight=torch.zeros(2, 2), scoring_func="softmax", norm_topk_prob=True
        ),
        num_experts_per_tok=1,
    )
    indices, weights = DeepSeekHostAdapter._masked_routing(
        block, torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool)
    )
    assert indices.shape == weights.shape == (1, 1)


def test_probe_result_index_and_call_failure_matrix() -> None:
    """P0-C 应拒绝 mask shape、索引覆盖与实际 forbidden 调用。"""

    request = make_mixed_probe_request(batch=2, sequence=1, experts=3, top_k=2)
    valid = ProbeResult((2, 1, 8), ((), (0, 1)), (), (0, 1), (0, 1), True)
    malformed = replace(request, allowed_mask=(request.allowed_mask[0],))
    with pytest.raises(P0Error, match="mask_shape_invalid"):
        validate_probe_result(malformed, valid, 3)
    with pytest.raises(P0Error, match="batch_index_not_preserved"):
        validate_probe_result(
            request,
            replace(valid, original_indices=(0, 2), reassembled_indices=(0, 2)),
            3,
        )
    out_of_range = replace(
        valid,
        expert_calls=(ExpertCall("E0", 2, 0, 2, "routed"),),
    )
    with pytest.raises(P0Error, match="expert_call_index_invalid"):
        validate_probe_result(request, out_of_range, 3)
    shared_routed = replace(
        valid,
        expert_calls=(ExpertCall("E0", 0, 0, 0, "routed"),),
    )
    with pytest.raises(P0Error, match="routed_call_not_zero"):
        validate_probe_result(request, shared_routed, 3)
    forbidden = replace(
        valid,
        expert_calls=(ExpertCall("E9", 1, 0, 1, "routed"),),
    )
    with pytest.raises(P0Error, match="expert_call_index_invalid"):
        validate_probe_result(request, forbidden, 3)


def test_structure_rejects_unresolved_expert_count() -> None:
    """实际 expert 数少于 native top-k 时必须拒绝。"""

    class Adapter:
        """提供成功 all-allowed，但 expert 数量不足。"""

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

        def uninstall_probe(self):
            return None

    report = inspect_real_structure(
        Adapter(), _architecture(native_top_k=3, routed_expert_count=2)
    )
    assert "host_architecture_unresolved" in report["failure_codes"]


def test_snapshot_remaining_schema_and_size_failures(
    tmp_path: Path, monkeypatch
) -> None:
    """snapshot 应覆盖声明类型、manifest 状态类型和冻结后复核失败。"""

    candidate = validate_registry(_candidate_payload())[0]
    with pytest.raises(P0Error, match="snapshot_metadata_incomplete"):
        validate_download_budget(candidate, tmp_path / "new", 0)
    with pytest.raises(P0Error, match="snapshot_missing"):
        from can.v2.pretrained_moe_p0.snapshot import freeze_snapshot

        freeze_snapshot(tmp_path / "missing")

    root = tmp_path / "root"
    root.mkdir()
    (root / "subdir").mkdir()
    (root / "subdir" / "config.json").write_text("{}", encoding="utf-8")
    assert inventory_snapshot(root)[0].path == "subdir/config.json"
    manifest_path = tmp_path / "manifest.json"
    base = {
        "schema_version": 1,
        "candidate_id": "C1",
        "repository_id": "org/model-1",
        "resolved_commit_sha": "a" * 40,
        "root": str(root),
        "files": [],
        "total_bytes": 2,
        "read_only_verified": True,
        "offline_verified": True,
    }
    for field, value in (
        ("total_bytes", 0),
        ("read_only_verified", 1),
        ("offline_verified", 1),
    ):
        changed = dict(base)
        changed[field] = value
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(P0Error, match="artifact_schema_mismatch"):
            load_snapshot_manifest(manifest_path)
    files = inventory_snapshot(root)
    manifest = dict(base)
    manifest["files"] = [item.__dict__ for item in files]
    manifest["total_bytes"] = 999
    monkeypatch.setattr(
        "can.v2.pretrained_moe_p0.snapshot.verify_read_only", lambda *args: True
    )
    with pytest.raises(P0Error, match="snapshot_size_invalid"):
        verify_snapshot_against_manifest(candidate, root, manifest)


def test_snapshot_freeze_recheck_failure(tmp_path: Path, monkeypatch) -> None:
    """freeze 后仍可写时不得返回成功 manifest。"""

    candidate = validate_registry(_candidate_payload())[0]
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "can.v2.pretrained_moe_p0.snapshot.freeze_snapshot", lambda root: None
    )
    monkeypatch.setattr(
        "can.v2.pretrained_moe_p0.snapshot.verify_read_only", lambda *args: False
    )
    with pytest.raises(P0Error, match="snapshot_not_read_only"):
        write_snapshot_manifest(
            candidate, root, root / "snapshot_manifest.json", freeze=True
        )


def test_resource_cuda_sampling_and_runtime_error(monkeypatch) -> None:
    """资源采样应记录 CUDA 值，并在 CUDA 查询失败时保守留空。"""

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 1)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (3, 4))
    sample = sample_resource("cuda", 0.0)
    assert (sample.cuda_allocated_bytes, sample.cuda_reserved_bytes) == (1, 2)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    assert sample_resource("error", 0.0).cuda_allocated_bytes is None


def test_runner_additional_failure_and_generic_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """正式 runner 应覆盖候选时限、只读失败、能力失败与无 probe 结构失败。"""

    import can.v2.pretrained_moe_p0.real_runner as module

    candidates = validate_registry(_candidate_payload())
    runner = RealP0Runner(candidates)
    timed = runner.run_formal_candidate(
        "C1",
        tmp_path,
        (),
        tmp_path / "timed",
        controller_elapsed_seconds=91 * 60,
    )
    assert "candidate_timeout" in timed.failure_codes

    monkeypatch.setattr(
        module,
        "infrastructure_preflight",
        lambda root: PreflightResult("infrastructure", True, "passed", {}),
    )
    monkeypatch.setattr(module, "load_snapshot_manifest", lambda path: {})
    monkeypatch.setattr(
        module,
        "verify_snapshot_against_manifest",
        lambda *args: SimpleNamespace(read_only_verified=False),
    )
    readonly = runner.run_formal_candidate("C1", tmp_path, (), tmp_path / "readonly")
    assert "snapshot_not_read_only" in readonly.failure_codes

    architecture = _architecture(architecture_family="qwen2_moe")
    loaded = SimpleNamespace(
        model=SimpleNamespace(), architecture=architecture, load_seconds=1.0
    )
    monkeypatch.setattr(
        module,
        "verify_snapshot_against_manifest",
        lambda *args: SimpleNamespace(read_only_verified=True),
    )
    monkeypatch.setattr(module, "load_transformers_host", lambda *args: loaded)
    monkeypatch.setattr(
        QwenHostAdapter, "inspect_architecture", lambda self: architecture
    )
    monkeypatch.setattr(
        module,
        "sample_resource",
        lambda *args: ResourceSample("x", 0.0, None, 1, 1, 2**31, 2**34, False, False),
    )
    monkeypatch.setattr(
        module,
        "evaluate_fixture",
        lambda *args: {"status": "failed", "record_objects": (), "errors": []},
    )
    failed = runner.run_formal_candidate("C1", tmp_path, (), tmp_path / "capability")
    assert failed.p0b == "failed"
    assert "capability_below_threshold" in failed.failure_codes


def test_runner_signature_and_prior_missing_summary() -> None:
    """P0-B 签名和 C2 前序摘要缺失必须拒绝。"""

    runner = RealP0Runner(validate_registry(_candidate_payload()))
    with pytest.raises(P0Error, match="cross_process_evidence_invalid"):
        runner._p0b_signature({"record_objects": []})
    with pytest.raises(P0Error, match="prior_candidate_failure_missing"):
        runner._validate_prior_failure(runner.candidates[1], None)


def test_tensor_index_contract_and_count_mismatch() -> None:
    """全局行索引应拒绝非正维度与错误有效行数量。"""

    assert tensor_index_contract(2, 2) == (0, 1, 2, 3)
    for args in ((0, 1), (1, 0), (True, 1)):
        with pytest.raises(ValueError):
            tensor_index_contract(*args)
    with pytest.raises(ValueError, match="有效行数量"):
        validate_reassembly_indices((0,), (0,), 2)
