"""P0-A 元数据盘点、人工审阅和 registry 生成测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from can.v2.pretrained_moe_p0 import P0Error, load_registry
from can.v2.pretrained_moe_p0 import p0a as p0a_module
from can.v2.pretrained_moe_p0.p0a import (
    P0A_CANDIDATES,
    finalize_p0a_registry,
    prepare_p0a_reviews,
)


def _write_json(path: Path, payload: object) -> None:
    """写入测试用 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _input_root(tmp_path: Path) -> Path:
    """构造三个固定候选的小型 Hub metadata 和文件目录。"""

    root = tmp_path / "input"
    configs = {
        "C1": {
            "model_type": "qwen2_moe",
            "architectures": ["Qwen2MoeForCausalLM"],
            "num_experts": 60,
            "num_experts_per_tok": 4,
            "shared_expert_intermediate_size": 5632,
        },
        "C2": {
            "model_type": "deepseek",
            "architectures": ["DeepseekForCausalLM"],
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_experts_per_tok": 6,
        },
        "C3": {
            "model_type": "granitemoe",
            "architectures": ["GraniteMoeForCausalLM"],
            "num_local_experts": 32,
            "num_experts_per_tok": 8,
        },
    }
    for index, preset in enumerate(P0A_CANDIDATES, 1):
        directory = root / preset.metadata_directory
        _write_json(directory / "config.json", configs[preset.candidate_id])
        (directory / "README.md").write_text(
            f"# {preset.candidate_id}\nlicense: apache-2.0\n", encoding="utf-8"
        )
        (directory / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
        if preset.candidate_id == "C2":
            (directory / "modeling_deepseek.py").write_text(
                "class DeepseekMoE:\n    pass\n", encoding="utf-8"
            )
        siblings = [
            {"rfilename": "config.json", "size": 100, "blob_id": None, "lfs": None},
            {
                "rfilename": "model-00001-of-00002.safetensors",
                "size": 1024 * index,
                "blob_id": None,
                "lfs": None,
            },
        ]
        metadata = {
            "candidate_id": preset.candidate_id,
            "repository_id": preset.repository_id,
            "requested_revision": "main",
            "resolved_commit_sha": f"{index:x}" * 40,
            "pipeline_tag": "text-generation",
            "library_name": "transformers",
            "tags": ["license:apache-2.0"],
            "siblings": siblings,
        }
        _write_json(root / preset.metadata_filename, metadata)
        if preset.candidate_id != "C2":
            (
                root / f"{preset.candidate_id.lower()}-builtin-source-review.txt"
            ).write_text(
                "Frozen built-in Transformers implementation review.\n",
                encoding="utf-8",
            )
    return root


def _complete_reviews(prepared: Path, reject_c3: bool = True) -> None:
    """填写测试用人工审阅结论，保留生成器提供的证据引用。"""

    for preset in P0A_CANDIDATES:
        candidate = prepared / preset.candidate_id
        source_path = candidate / "source_review.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source.update(
            {
                "native_shared_expert": preset.candidate_id != "C3",
                "native_routed_experts": True,
                "shared_always_executes": preset.candidate_id != "C3",
                "mask_before_dispatch_static_feasibility": preset.candidate_id != "C3",
                "expert_call_observability_static_feasibility": preset.candidate_id
                != "C3",
                "reviewer": "test-reviewer",
                "reviewed_at_utc": "2026-09-19T00:00:00Z",
                "status": (
                    "rejected"
                    if reject_c3 and preset.candidate_id == "C3"
                    else "approved"
                ),
                "rationale": "测试固定源码审阅结论。",
            }
        )
        _write_json(source_path, source)

        license_path = candidate / "license_review.json"
        license_review = json.loads(license_path.read_text(encoding="utf-8"))
        license_review.update(
            {
                "declared_license": "apache-2.0",
                "reviewer": "test-reviewer",
                "reviewed_at_utc": "2026-09-19T00:00:00Z",
                "status": "approved",
                "rationale": "测试许可证审阅结论。",
            }
        )
        _write_json(license_path, license_review)

        remote_path = candidate / "remote_code_review.json"
        remote = json.loads(remote_path.read_text(encoding="utf-8"))
        remote.update(
            {
                "reviewer": "test-reviewer",
                "reviewed_at_utc": "2026-09-19T00:00:00Z",
                "status": "approved" if preset.allow_remote_code else "not_required",
                "rationale": "测试 remote-code 审阅结论。",
            }
        )
        _write_json(remote_path, remote)


def test_prepare_and_finalize_generate_bound_registry(tmp_path: Path) -> None:
    """验证盘点、审阅、决策、registry 与摘要 sidecar 的完整闭环。"""

    input_root = _input_root(tmp_path)
    prepared = tmp_path / "prepared"
    summary = prepare_p0a_reviews(input_root, prepared)
    assert summary["status"] == "review_required"
    assert (prepared / "C1" / "source_review.json").is_file()
    assert (prepared / "C2" / "remote_code_review.json").is_file()

    _complete_reviews(prepared)
    decisions = tmp_path / "decisions"
    registry = tmp_path / "candidate_registry.json"
    final = finalize_p0a_registry(input_root, prepared, decisions, registry)
    assert final["status"] == "complete"
    assert final["candidates"][0]["status"] == "passed"
    assert final["candidates"][2]["status"] == "rejected"
    assert "native_shared_expert_missing" in final["candidates"][2]["failure_codes"]

    candidates = load_registry(registry)
    assert [candidate.p0a_status for candidate in candidates] == [
        "passed",
        "passed",
        "rejected",
    ]
    for candidate in candidates:
        decision_path = (
            decisions / f"{candidate.candidate_id.lower()}_p0a_decision.json"
        )
        assert (
            candidate.p0a_decision_sha256
            == hashlib.sha256(decision_path.read_bytes()).hexdigest()
        )
    sidecar = registry.with_suffix(".json.sha256").read_text(encoding="ascii")
    assert hashlib.sha256(registry.read_bytes()).hexdigest() in sidecar


def test_finalize_rejects_pending_and_digest_drift(tmp_path: Path) -> None:
    """验证未完成人工审阅和证据漂移均失败关闭。"""

    input_root = _input_root(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_p0a_reviews(input_root, prepared)
    with pytest.raises(P0Error, match="p0a_review_incomplete"):
        finalize_p0a_registry(
            input_root,
            prepared,
            tmp_path / "decisions-pending",
            tmp_path / "pending.json",
        )

    _complete_reviews(prepared)
    (input_root / "c1-files" / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(P0Error, match="snapshot_digest_mismatch"):
        finalize_p0a_registry(
            input_root,
            prepared,
            tmp_path / "decisions-drift",
            tmp_path / "drift.json",
        )


def test_prepare_rejects_revision_weights_and_overwrite(tmp_path: Path) -> None:
    """验证可变 revision、误下载权重和输出覆盖均被拒绝。"""

    input_root = _input_root(tmp_path)
    metadata = input_root / P0A_CANDIDATES[0].metadata_filename
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["resolved_commit_sha"] = "main"
    _write_json(metadata, payload)
    with pytest.raises(P0Error, match="revision_not_immutable"):
        prepare_p0a_reviews(input_root, tmp_path / "bad-revision")

    input_root = _input_root(tmp_path / "weights")
    (input_root / "c1-files" / "model.safetensors").write_bytes(b"not-a-real-weight")
    prepared = tmp_path / "prepared-weights"
    prepare_p0a_reviews(input_root, prepared)
    _complete_reviews(prepared)
    final = finalize_p0a_registry(
        input_root,
        prepared,
        tmp_path / "decisions-weights",
        tmp_path / "weights.json",
    )
    assert "unexpected_weight_file" in final["candidates"][0]["failure_codes"]
    with pytest.raises(P0Error, match="artifact_exists"):
        prepare_p0a_reviews(input_root, prepared)


def test_finalize_rejects_remote_profile_and_existing_outputs(tmp_path: Path) -> None:
    """验证 remote-code profile 篡改与正式产物覆盖被拒绝。"""

    input_root = _input_root(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_p0a_reviews(input_root, prepared)
    _complete_reviews(prepared)
    remote_path = prepared / "C2" / "remote_code_review.json"
    remote = json.loads(remote_path.read_text(encoding="utf-8"))
    remote["allow_remote_code"] = False
    _write_json(remote_path, remote)
    with pytest.raises(P0Error, match="remote_code_rejected"):
        finalize_p0a_registry(
            input_root,
            prepared,
            tmp_path / "decisions-bad-remote",
            tmp_path / "bad-remote.json",
        )

    _complete_reviews(prepared)
    registry = tmp_path / "exists.json"
    registry.write_text("old", encoding="utf-8")
    with pytest.raises(P0Error, match="artifact_exists"):
        finalize_p0a_registry(
            input_root, prepared, tmp_path / "decisions-exists", registry
        )


@pytest.mark.parametrize(
    "raw,code",
    [
        ('{"x":1,"x":2}', "artifact_duplicate_field"),
        ("not-json", "artifact_json_invalid"),
        ('{"x":NaN}', "artifact_json_invalid"),
    ],
)
def test_strict_json_rejects_malformed_content(
    tmp_path: Path, raw: str, code: str
) -> None:
    """覆盖重复字段、语法错误和非有限 JSON 常量。"""

    path = tmp_path / "value.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(P0Error, match=code):
        p0a_module.load_strict_json(path)
    with pytest.raises(FileNotFoundError):
        p0a_module.load_strict_json(tmp_path / "missing.json")


@pytest.mark.parametrize("value", ["", "../escape", "/absolute", "bad\\path", 1])
def test_safe_relative_path_rejects_escape(value: object) -> None:
    """验证 artifact 路径不能逃逸根目录或混用 Windows 分隔符。"""

    with pytest.raises(P0Error, match="artifact_schema_mismatch"):
        p0a_module._safe_relative_path(value, "test")


def test_low_level_helpers_reject_invalid_inputs(tmp_path: Path) -> None:
    """覆盖 canonical JSON、文件摘要、空目录和非 UTF-8 源码分支。"""

    with pytest.raises(P0Error, match="artifact_json_invalid"):
        p0a_module.canonical_json_bytes({"bad": object()})
    with pytest.raises(P0Error, match="artifact_validation_failed"):
        p0a_module.sha256_file(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(P0Error, match="metadata_unresolved"):
        p0a_module._inventory(empty)
    with pytest.raises(P0Error, match="metadata_unresolved"):
        p0a_module._inventory(tmp_path / "missing-dir")
    source = tmp_path / "source"
    source.mkdir()
    bad = source / "bad.py"
    bad.write_bytes(b"\xff")
    inventory = [
        {
            "path": "bad.py",
            "size_bytes": 1,
            "sha256": hashlib.sha256(b"\xff").hexdigest(),
        }
    ]
    with pytest.raises(P0Error, match="remote_code_rejected"):
        p0a_module._scan_python_files(source, inventory)


@pytest.mark.parametrize(
    "mutator,code",
    [
        (lambda p: p.update(extra=True), "artifact_schema_mismatch"),
        (lambda p: p.update(candidate_id="OTHER"), "artifact_validation_failed"),
        (lambda p: p.update(requested_revision=""), "revision_not_immutable"),
        (lambda p: p.update(resolved_commit_sha="main"), "revision_not_immutable"),
        (lambda p: p.update(siblings=[]), "metadata_unresolved"),
        (
            lambda p: p.update(siblings=[{"rfilename": "config.json"}]),
            "artifact_schema_mismatch",
        ),
        (
            lambda p: p.update(siblings=[{"rfilename": "../bad", "size": 1}]),
            "artifact_schema_mismatch",
        ),
        (
            lambda p: p.update(siblings=[{"rfilename": "config.json", "size": -1}]),
            "artifact_schema_mismatch",
        ),
    ],
)
def test_hub_metadata_rejects_invalid_variants(mutator, code: str) -> None:
    """覆盖 Hub 元数据身份、revision、清单和大小的严格校验。"""

    preset = P0A_CANDIDATES[0]
    payload = {
        "candidate_id": preset.candidate_id,
        "repository_id": preset.repository_id,
        "requested_revision": "main",
        "resolved_commit_sha": "a" * 40,
        "siblings": [{"rfilename": "config.json", "size": 1}],
    }
    mutator(payload)
    with pytest.raises(P0Error, match=code):
        p0a_module._validate_hub_metadata(payload, preset)


def _mutate_prepared_manifest(tmp_path: Path, mutator) -> tuple[Path, Path, Path]:
    """准备完整审阅，并篡改指定候选的 metadata manifest。"""

    input_root = _input_root(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_p0a_reviews(input_root, prepared)
    _complete_reviews(prepared)
    path = prepared / "C1" / "metadata_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutator(manifest)
    _write_json(path, manifest)
    return input_root, prepared, tmp_path / "decisions"


@pytest.mark.parametrize(
    "mutator,code",
    [
        (lambda p: p.update(attempt_order=9), "artifact_validation_failed"),
        (
            lambda p: p.update(resolved_commit_sha="f" * 40),
            "artifact_validation_failed",
        ),
        (
            lambda p: p["hub_metadata"].update(file_count=99),
            "snapshot_digest_mismatch",
        ),
        (
            lambda p: p["local_metadata"].update(inventory_sha256="f" * 64),
            "snapshot_digest_mismatch",
        ),
        (
            lambda p: p["architecture"].update(model_type="dense"),
            "snapshot_digest_mismatch",
        ),
        (
            lambda p: p["remote_code_scan"].update(findings=[{"bad": True}]),
            "snapshot_digest_mismatch",
        ),
    ],
)
def test_finalize_recomputes_prepared_manifest(
    tmp_path: Path, mutator, code: str
) -> None:
    """验证 finalize 不信任可编辑的 prepared manifest 自描述字段。"""

    input_root, prepared, decisions = _mutate_prepared_manifest(tmp_path, mutator)
    with pytest.raises(P0Error, match=code):
        finalize_p0a_registry(
            input_root, prepared, decisions, tmp_path / "registry.json"
        )


def _prepared_reviews(tmp_path: Path) -> tuple[Path, Path]:
    """返回已完成的测试审阅目录。"""

    input_root = _input_root(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_p0a_reviews(input_root, prepared)
    _complete_reviews(prepared)
    return input_root, prepared


@pytest.mark.parametrize(
    "filename,mutator,code",
    [
        (
            "source_review.json",
            lambda p: p.update(candidate_id="OTHER"),
            "artifact_validation_failed",
        ),
        (
            "source_review.json",
            lambda p: p.update(reviewer=""),
            "p0a_review_incomplete",
        ),
        (
            "source_review.json",
            lambda p: p.update(reviewed_at_utc="local-time"),
            "p0a_review_incomplete",
        ),
        (
            "source_review.json",
            lambda p: p.update(rationale=""),
            "p0a_review_incomplete",
        ),
        (
            "source_review.json",
            lambda p: p.update(status="pending"),
            "p0a_review_incomplete",
        ),
        (
            "source_review.json",
            lambda p: p.update(native_shared_expert=1),
            "artifact_schema_mismatch",
        ),
        (
            "source_review.json",
            lambda p: p.update(evidence_files=[]),
            "p0a_review_incomplete",
        ),
        (
            "source_review.json",
            lambda p: p.update(evidence_files="bad"),
            "artifact_schema_mismatch",
        ),
        (
            "source_review.json",
            lambda p: p["evidence_files"][0].update(sha256="bad"),
            "artifact_schema_mismatch",
        ),
        (
            "license_review.json",
            lambda p: p.update(status="pending"),
            "p0a_review_incomplete",
        ),
        (
            "license_review.json",
            lambda p: p.update(declared_license=1),
            "artifact_schema_mismatch",
        ),
        (
            "license_review.json",
            lambda p: p.update(usage_restrictions=[1]),
            "artifact_schema_mismatch",
        ),
        (
            "license_review.json",
            lambda p: p.update(license_files=[]),
            "license_unresolved",
        ),
        (
            "remote_code_review.json",
            lambda p: p.update(status="pending"),
            "p0a_review_incomplete",
        ),
    ],
)
def test_finalize_rejects_incomplete_review_variants(
    tmp_path: Path, filename: str, mutator, code: str
) -> None:
    """覆盖三类人工审阅中缺失、类型错误和未完成状态。"""

    input_root, prepared = _prepared_reviews(tmp_path)
    path = prepared / "C1" / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    _write_json(path, payload)
    with pytest.raises(P0Error, match=code):
        finalize_p0a_registry(
            input_root,
            prepared,
            tmp_path / "decisions",
            tmp_path / "registry.json",
        )


def test_remote_code_findings_require_individual_disposition(tmp_path: Path) -> None:
    """验证 remote-code 扫描命中必须逐条说明接受或拒绝。"""

    input_root = _input_root(tmp_path)
    source = input_root / "c2-files" / "modeling_deepseek.py"
    source.write_text(
        "import requests\nrequests.get('https://example.invalid')\n", encoding="utf-8"
    )
    prepared = tmp_path / "prepared"
    prepare_p0a_reviews(input_root, prepared)
    _complete_reviews(prepared)
    remote_path = prepared / "C2" / "remote_code_review.json"
    remote = json.loads(remote_path.read_text(encoding="utf-8"))
    assert remote["dangerous_call_findings"]
    _write_json(remote_path, remote)
    with pytest.raises(P0Error, match="p0a_review_incomplete"):
        finalize_p0a_registry(
            input_root,
            prepared,
            tmp_path / "pending-decisions",
            tmp_path / "pending.json",
        )

    findings = remote["dangerous_call_findings"]
    remote["dangerous_call_findings"] = []
    _write_json(remote_path, remote)
    with pytest.raises(P0Error, match="remote_code_rejected"):
        finalize_p0a_registry(
            input_root,
            prepared,
            tmp_path / "deleted-findings",
            tmp_path / "deleted-findings.json",
        )

    remote["dangerous_call_findings"] = findings
    for finding in remote["dangerous_call_findings"]:
        finding["review_disposition"] = "accepted"
        finding["rationale"] = "仅为冻结源码中的显式网络调用；本候选后续必须离线加载。"
    _write_json(remote_path, remote)
    final = finalize_p0a_registry(
        input_root,
        prepared,
        tmp_path / "accepted-decisions",
        tmp_path / "accepted.json",
    )
    assert final["candidates"][1]["status"] == "passed"


def test_source_rejection_and_remote_inventory_deletion_fail_closed(
    tmp_path: Path,
) -> None:
    """验证源码明确拒绝会产生失败，且 remote-code 清单不能删项绕过。"""

    input_root, prepared = _prepared_reviews(tmp_path)
    source_path = prepared / "C1" / "source_review.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["status"] = "rejected"
    source["rationale"] = "人工确认该候选接口不可控。"
    _write_json(source_path, source)
    final = finalize_p0a_registry(
        input_root,
        prepared,
        tmp_path / "rejected-decisions",
        tmp_path / "rejected.json",
    )
    assert "source_review_rejected" in final["candidates"][0]["failure_codes"]

    input_root, prepared = _prepared_reviews(tmp_path / "deleted")
    remote_path = prepared / "C2" / "remote_code_review.json"
    remote = json.loads(remote_path.read_text(encoding="utf-8"))
    remote["reviewed_python_files"] = []
    _write_json(remote_path, remote)
    with pytest.raises(P0Error, match="remote_code_rejected"):
        finalize_p0a_registry(
            input_root,
            prepared,
            tmp_path / "deleted-decisions",
            tmp_path / "deleted.json",
        )
