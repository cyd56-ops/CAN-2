"""I1 G1-b adapter 与 M2 集成专项测试。"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

from can.v2.auth_expert_moe_m2 import (
    P1_POLICY,
    P2_POLICY,
    M2AuthExpert,
    M2Context,
    M2RouteCoordinator,
    M2ScopeRegistry,
    M2VerifierProtocol,
)
from can.v2.modint_verifier_g1a import G1AConfig, generate_parameters, parameter_digest
from can.v2.modint_verifier_g1b import derive_integer_bounds
from can.v2.modint_verifier_g1b.manifest import build_g1b_manifest
from can.v2.modint_verifier_m2_i1 import (
    FP32_EXACT_INTEGER_LIMIT,
    I1_EXECUTION_CONFIG_ID,
    G1BVerifierAdapter,
    I1Error,
)
from can.v2.modint_verifier_m2_i1 import adapter as adapter_module
from can.v2.modint_verifier_m2_i1 import (
    build_i1_manifest,
    load_i1_manifest,
    run_i1_artifacts,
    validate_i1_call_ledger,
    validate_i1_fixture,
    validate_i1_manifest,
    validate_i1_summary,
)
from can.v2.pretrained_gate.types import EvidenceReason, RouteKind, VerificationEvidence


def _materials():
    """创建与 G1-b tiny contract 相同的确定性参数和 manifest。"""
    config = G1AConfig(q=17, n=3, m=4, tau=2, master_seed=20260916)
    parameters, secret, _ = generate_parameters(config)
    provenance = {
        "git_commit": "a" * 40,
        "dirty": False,
        "python": "3.11",
        "torch": "2",
    }
    manifest = build_g1b_manifest(
        parameter_digest(parameters),
        derive_integer_bounds(parameters),
        "b" * 64,
        provenance,
    )
    return parameters, torch.from_numpy(secret.copy()).reshape(1, -1), manifest


def _adapter() -> G1BVerifierAdapter:
    """构造绑定固定 tiny manifest 的 adapter。"""
    parameters, _, manifest = _materials()
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return G1BVerifierAdapter(parameters, manifest, digest)


def _invalid(valid: torch.Tensor, adapter: G1BVerifierAdapter) -> torch.Tensor:
    """确定性搜索一个关系失败但格式合法的 credential。"""
    q = adapter.verifier.parameters_ref.config.q
    for delta in range(1, q):
        candidate = valid.clone()
        candidate[0, 0] = (int(candidate[0, 0]) + delta) % q
        if not bool(adapter.verifier(candidate).evidence.accepted[0]):
            return candidate
    raise AssertionError("tiny fixture 未找到 relation failure")


def test_adapter_satisfies_protocol_and_maps_valid_failure() -> None:
    """验证 adapter protocol、valid/failure 和整数诊断映射。"""
    adapter = _adapter()
    _, valid, _ = _materials()
    invalid = _invalid(valid, adapter)
    evidence = adapter(torch.cat((valid, invalid), dim=0).contiguous())
    assert isinstance(adapter, M2VerifierProtocol)
    assert evidence.verified.tolist() == [True, False]
    assert evidence.reason_code.tolist() == [
        int(EvidenceReason.SUCCESS),
        int(EvidenceReason.RELATION_FAILED),
    ]
    assert adapter.validate_evidence(evidence) == 2
    assert adapter.max_abs_residual_upper_bound == 8
    assert adapter.fp32_lossless_verified is True


def test_adapter_rejects_noncanonical_input() -> None:
    """拒绝 FP32、非 Tensor、非连续、错误 shape 和越界整数。"""
    adapter = _adapter()
    with pytest.raises(I1Error, match="I1_INPUT_TYPE"):
        adapter([[0, 0, 0]])  # type: ignore[arg-type]
    with pytest.raises(I1Error, match="I1_INPUT_DTYPE"):
        adapter(torch.zeros((1, 3), dtype=torch.float32))
    with pytest.raises(I1Error, match="I1_INPUT_LAYOUT"):
        adapter(torch.zeros((3, 2), dtype=torch.int64).transpose(0, 1))
    with pytest.raises(ValueError, match="G1B_INPUT_SHAPE"):
        adapter(torch.zeros((1, 2), dtype=torch.int64))
    with pytest.raises(ValueError, match="G1B_INPUT_DOMAIN"):
        adapter(torch.tensor([[17, 0, 0]], dtype=torch.int64))


def test_adapter_binds_manifest_bytes_and_parameters(tmp_path: Path) -> None:
    """验证 manifest bytes 摘要、参数摘要和 bounds 绑定。"""
    parameters, _, manifest = _materials()
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    adapter = G1BVerifierAdapter.from_manifest(parameters, path, digest)
    assert adapter.g1b_manifest_sha256 == digest
    with pytest.raises(ValueError):
        G1BVerifierAdapter.from_manifest(parameters, path, "0" * 64)
    bad = dict(manifest)
    bad["source_parameter_sha256"] = "0" * 64
    bad_digest = hashlib.sha256(
        json.dumps(bad, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(I1Error, match="I1_PARAMETER_DIGEST_MISMATCH"):
        G1BVerifierAdapter(parameters, bad, bad_digest)
    bad = dict(manifest)
    bad["integer_bounds"] = dict(manifest["integer_bounds"])
    bad["integer_bounds"]["max_accumulator_abs"] += 1
    bad_digest = hashlib.sha256(
        json.dumps(bad, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(I1Error, match="I1_INTEGER_BOUNDS_MISMATCH"):
        G1BVerifierAdapter(parameters, bad, bad_digest)


def test_adapter_rejects_unknown_reason_and_result_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未知 reason、protocol、parameter 和 backend 漂移必须 fail closed。"""
    adapter = _adapter()
    _, valid, _ = _materials()
    original = adapter.verifier(valid)
    object.__setattr__(original.evidence, "reason_code", ("UNKNOWN",))
    monkeypatch.setattr(adapter.verifier, "forward", lambda value: original)
    with pytest.raises(I1Error, match="I1_UNKNOWN_REASON"):
        adapter(valid)

    adapter = _adapter()
    original = adapter.verifier(valid)
    object.__setattr__(original.evidence, "protocol_id", "wrong")
    monkeypatch.setattr(adapter.verifier, "forward", lambda value: original)
    with pytest.raises(I1Error, match="I1_PROTOCOL_MISMATCH"):
        adapter(valid)

    adapter = _adapter()
    original = adapter.verifier(valid)
    object.__setattr__(original.evidence, "parameter_digest", "0" * 64)
    monkeypatch.setattr(adapter.verifier, "forward", lambda value: original)
    with pytest.raises(I1Error, match="I1_PARAMETER_DIGEST_MISMATCH"):
        adapter(valid)

    adapter = _adapter()
    original = adapter.verifier(valid)
    object.__setattr__(original, "backend_id", "wrong")
    monkeypatch.setattr(adapter.verifier, "forward", lambda value: original)
    with pytest.raises(I1Error, match="I1_BACKEND_MISMATCH"):
        adapter(valid)


def test_adapter_detects_evidence_tampering_and_foreign_source() -> None:
    """验证私有 seal、签发登记和 Tensor 原地篡改检查。"""
    adapter = _adapter()
    _, valid, _ = _materials()
    evidence = adapter(valid)
    evidence.verified[0] = False
    with pytest.raises(I1Error, match="I1_EVIDENCE_TAMPERED"):
        adapter.validate_evidence(evidence)
    foreign = _adapter()(valid)
    with pytest.raises(I1Error, match="I1_EVIDENCE_SOURCE"):
        adapter.validate_evidence(foreign)


def test_adapter_rejects_lossy_fp32_diagnostic_range() -> None:
    """当 q//2 超过 FP32 连续整数域时拒绝当前 evidence schema。"""
    q = 2 * FP32_EXACT_INTEGER_LIMIT + 3
    config = G1AConfig(q=q, n=1, m=1, tau=1, master_seed=1)
    parameters, _, _ = generate_parameters(config)
    manifest = build_g1b_manifest(
        parameter_digest(parameters),
        derive_integer_bounds(parameters),
        "b" * 64,
        {"git_commit": "a" * 40, "dirty": False, "python": "3", "torch": "2"},
    )
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(I1Error, match="I1_FP32_DIAGNOSTIC_LOSSY"):
        G1BVerifierAdapter(parameters, manifest, digest)


@pytest.mark.parametrize(
    ("policy", "failed_route"),
    ((P1_POLICY, RouteKind.DENY), (P2_POLICY, RouteKind.PUBLIC)),
)
def test_adapter_integrates_with_m2_routes(
    policy: str, failed_route: RouteKind
) -> None:
    """验证相同 scope 下 valid/failure 的 P1/P2 固定路由。"""
    adapter = _adapter()
    _, valid, _ = _materials()
    invalid = _invalid(valid, adapter)
    context = M2Context(("valid", "failure"), I1_EXECUTION_CONFIG_ID)
    auth = M2AuthExpert(adapter, I1_EXECUTION_CONFIG_ID)
    coordinator = M2RouteCoordinator(auth, I1_EXECUTION_CONFIG_ID, policy)
    registry = M2ScopeRegistry(coordinator, {"paired": "advanced"})
    assignment = registry.assignment("paired", context)
    bound = auth.verify(torch.cat((valid, invalid), dim=0), context, assignment)
    route = coordinator.commit(bound, context)
    assert route.routes == (RouteKind.PROTECTED, failed_route)


def _i1_manifest(policy: str = P1_POLICY):
    """构造一个严格有效的 I1 manifest。"""
    parameters, _, _ = _materials()
    return build_i1_manifest(
        policy=policy,
        parameters=parameters,
        m2_manifest_sha256="1" * 64,
        m2_assignment_sha256="2" * 64,
        g1b_backend_manifest_sha256="3" * 64,
        scope_schema_sha256="4" * 64,
        provenance={
            "git_commit": "a" * 40,
            "dirty": False,
            "python": "3.11",
            "torch": "2",
            "numpy": "2",
        },
    )


def test_i1_manifest_roundtrip_and_negative_paths(tmp_path: Path) -> None:
    """覆盖 manifest bytes 摘要、重复字段和冻结配置篡改。"""
    manifest = _i1_manifest()
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert load_i1_manifest(path, digest) == manifest
    with pytest.raises(I1Error, match="I1_MANIFEST_DIGEST_MISMATCH"):
        load_i1_manifest(path, "0" * 64)

    mutations = []
    for key, value in (
        ("policy", "bad"),
        ("g1b_backend_id", "bad"),
        ("norm", "l2"),
        ("max_abs_residual_upper_bound", 999),
        ("fp32_exact_integer_limit", 1),
        ("fp32_lossless_verified", False),
        ("router", {}),
        ("model", {}),
    ):
        item = dict(manifest)
        item[key] = value
        mutations.append(item)
    item = dict(manifest)
    item["integer_bounds"] = {"q": 17}
    mutations.append(item)
    item = dict(manifest)
    item["provenance"] = {"dirty": False}
    mutations.append(item)
    for payload in mutations:
        with pytest.raises((I1Error, ValueError, TypeError)):
            validate_i1_manifest(payload)

    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(I1Error, match="I1_MANIFEST_JSON"):
        load_i1_manifest(path)


def test_i1_runner_artifacts_and_schema_failures(tmp_path: Path) -> None:
    """运行正式双 policy runner，并覆盖 fixture/ledger/summary 负向 schema。"""
    repository = Path(__file__).resolve().parents[2]
    m2_run_id = "run-20260915-04"
    for policy in (P1_POLICY, P2_POLICY):
        source = repository / "results" / "m2-multi-expert-v1" / policy / m2_run_id
        target = tmp_path / "results" / "m2-multi-expert-v1" / policy / m2_run_id
        target.mkdir(parents=True)
        shutil.copy2(source / "manifest.json", target / "manifest.json")
    coverage_report = tmp_path / "coverage.json"
    coverage_report.write_text("{}", encoding="utf-8")
    result = run_i1_artifacts(
        repository_root=tmp_path,
        g1b_dir=repository / "results" / "g1b-modint-v1" / "run-20260916-02",
        m2_run_id=m2_run_id,
        run_id="run-test-01",
        coverage_report=coverage_report,
        statement_coverage=100.0,
        branch_coverage=100.0,
    )
    assert result["status"] == "complete"
    output = (
        tmp_path / "results" / "i1-g1b-m2-integration-v1" / P1_POLICY / "run-test-01"
    )
    fixture = json.loads((output / "fixture.json").read_text(encoding="utf-8"))
    ledger = json.loads((output / "call_ledger.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    validate_i1_fixture(fixture)
    validate_i1_call_ledger(ledger)
    validate_i1_summary(summary)
    assert summary["determinism"] == {"repeat_checked": True, "difference_count": 0}
    with pytest.raises(FileExistsError):
        run_i1_artifacts(
            repository_root=tmp_path,
            g1b_dir=repository / "results" / "g1b-modint-v1" / "run-20260916-02",
            m2_run_id=m2_run_id,
            run_id="run-test-01",
            coverage_report=coverage_report,
            statement_coverage=100.0,
            branch_coverage=100.0,
        )

    bad_fixture = dict(fixture)
    bad_fixture["cases"] = fixture["cases"][:-1]
    with pytest.raises(I1Error, match="I1_FIXTURE_BOUNDARY"):
        validate_i1_fixture(bad_fixture)
    bad_fixture = json.loads(json.dumps(fixture))
    bad_fixture["cases"][0]["scale_factor"] = 1
    with pytest.raises(I1Error, match="I1_FIXTURE_CASE"):
        validate_i1_fixture(bad_fixture)
    bad_ledger = json.loads(json.dumps(ledger))
    bad_ledger["events"][0]["sequence"] = 10
    with pytest.raises(I1Error, match="I1_LEDGER_EVENT"):
        validate_i1_call_ledger(bad_ledger)
    bad_summary = dict(summary)
    bad_summary.pop("failure")
    with pytest.raises(I1Error, match="I1_SUMMARY_SCHEMA"):
        validate_i1_summary(bad_summary)


def _issued_evidence(
    adapter: G1BVerifierAdapter,
    verified: torch.Tensor,
    error_norm: torch.Tensor,
    reason_code: torch.Tensor,
    *,
    seal: Optional[object] = None,
) -> VerificationEvidence:
    """构造登记到 adapter 的受控负向 evidence。"""
    actual_seal = adapter._evidence_seal if seal is None else seal
    tag = (
        "0" * 64
        if any(
            value.device.type == "meta" for value in (verified, error_norm, reason_code)
        )
        else adapter_module._evidence_tag(
            verified, error_norm, reason_code, actual_seal
        )
    )
    evidence = VerificationEvidence(
        verified,
        error_norm,
        reason_code,
        actual_seal,
        tag,
    )
    adapter._issued[id(evidence)] = evidence
    return evidence


def test_adapter_constructor_and_result_negative_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """覆盖 adapter 构造、device、结果类型和 residual 上界分支。"""
    parameters, valid, manifest = _materials()
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(I1Error, match="I1_PARAMETER_TYPE"):
        G1BVerifierAdapter(object(), manifest, digest)  # type: ignore[arg-type]
    with pytest.raises(I1Error, match="I1_MANIFEST_TYPE"):
        G1BVerifierAdapter(parameters, [], digest)  # type: ignore[arg-type]
    with pytest.raises(I1Error, match="I1_DIGEST_SCHEMA"):
        G1BVerifierAdapter(parameters, manifest, "BAD")
    with pytest.raises(I1Error, match="I1_MANIFEST_DIGEST_MISMATCH"):
        G1BVerifierAdapter(parameters, manifest, "c" * 64)

    adapter = _adapter()
    with pytest.raises(I1Error, match="I1_INPUT_DEVICE"):
        adapter(torch.empty((1, 3), dtype=torch.int64, device="meta"))
    monkeypatch.setattr(adapter.verifier, "forward", lambda value: object())
    with pytest.raises(I1Error, match="I1_RESULT_TYPE"):
        adapter(valid)

    adapter = _adapter()
    result = adapter.verifier(valid)
    object.__setattr__(
        result.evidence,
        "max_abs_residual",
        (adapter.max_abs_residual_upper_bound + 1,),
    )
    monkeypatch.setattr(adapter.verifier, "forward", lambda value: result)
    with pytest.raises(I1Error, match="I1_RESIDUAL_RANGE"):
        adapter(valid)

    adapter = _adapter()
    original_validate = adapter_module.validate_g1b_manifest
    monkeypatch.setattr(
        adapter_module,
        "validate_g1b_manifest",
        lambda value: {**original_validate(value), "backend_id": "wrong"},
    )
    with pytest.raises(I1Error, match="I1_BACKEND_MISMATCH"):
        G1BVerifierAdapter(parameters, manifest, digest)


def test_adapter_evidence_validation_negative_branches() -> None:
    """覆盖 evidence shape、dtype、device、reason 和数值语义拒绝。"""
    adapter = _adapter()
    with pytest.raises(I1Error, match="I1_EVIDENCE_TYPE"):
        adapter.validate_evidence(object())  # type: ignore[arg-type]
    foreign_seal = object()
    sealed = _issued_evidence(
        adapter,
        torch.tensor([True]),
        torch.tensor([0.0]),
        torch.tensor([0]),
        seal=foreign_seal,
    )
    with pytest.raises(I1Error, match="I1_EVIDENCE_SOURCE"):
        adapter.validate_evidence(sealed)

    cases = [
        (
            torch.tensor([[True]]),
            torch.tensor([0.0]),
            torch.tensor([0]),
            "I1_EVIDENCE_SHAPE",
        ),
        (
            torch.tensor([True, False]),
            torch.tensor([0.0]),
            torch.tensor([0]),
            "I1_EVIDENCE_SHAPE",
        ),
        (
            torch.tensor([], dtype=torch.bool),
            torch.tensor([], dtype=torch.float32),
            torch.tensor([], dtype=torch.long),
            "I1_EVIDENCE_SHAPE",
        ),
        (
            torch.tensor([1], dtype=torch.long),
            torch.tensor([0.0]),
            torch.tensor([0]),
            "I1_EVIDENCE_DTYPE",
        ),
        (
            torch.tensor([True]),
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0]),
            "I1_EVIDENCE_DTYPE",
        ),
        (
            torch.tensor([True]),
            torch.tensor([0.0]),
            torch.tensor([0], dtype=torch.int32),
            "I1_EVIDENCE_DTYPE",
        ),
        (
            torch.tensor([True]),
            torch.empty((1,), device="meta"),
            torch.tensor([0]),
            "I1_EVIDENCE_DEVICE",
        ),
        (
            torch.tensor([False]),
            torch.tensor([0.0]),
            torch.tensor([99]),
            "I1_EVIDENCE_REASON",
        ),
        (
            torch.tensor([False]),
            torch.tensor([0.0]),
            torch.tensor([0]),
            "I1_EVIDENCE_SEMANTICS",
        ),
        (
            torch.tensor([True]),
            torch.tensor([float("nan")]),
            torch.tensor([0]),
            "I1_EVIDENCE_NUMERIC",
        ),
        (
            torch.tensor([True]),
            torch.tensor([-1.0]),
            torch.tensor([0]),
            "I1_EVIDENCE_NUMERIC",
        ),
        (
            torch.tensor([True]),
            torch.tensor([99.0]),
            torch.tensor([0]),
            "I1_EVIDENCE_NUMERIC",
        ),
    ]
    for verified, error_norm, reason_code, expected in cases:
        evidence = _issued_evidence(adapter, verified, error_norm, reason_code)
        with pytest.raises(I1Error, match=expected):
            adapter.validate_evidence(evidence)


def _valid_fixture_payload() -> dict:
    """构造覆盖全部 boundary role 的最小 fixture。"""
    descriptor = {
        "encoding": "sha256-only",
        "sha256": "a" * 64,
        "dtype": "torch.int64",
        "shape": [1, 3],
    }
    cases = []
    roles = (
        "nominal_valid",
        "nominal_failure",
        "exact_accept_boundary",
        "first_reject_boundary",
        "format_dtype",
        "format_rank",
        "format_shape",
        "domain_below_zero",
        "domain_at_q",
    )
    for index, role in enumerate(roles):
        cases.append(
            {
                "case_id": f"case-{index}",
                "expected_relation_class": (
                    "valid"
                    if index == 0
                    else "format_error" if index >= 4 else "failure"
                ),
                "boundary_role": role,
                "a0_credential": dict(descriptor),
                "a0_expected_reason": "reason",
                "g1b_credential": dict(descriptor),
                "g1b_expected_reason": "reason",
                "rationale": "test",
                "expected_scope": "standard",
                "request_ids": [f"request-{index}"],
                "expected_routes": ["protected" if index == 0 else "error"],
            }
        )
    return {
        "schema_version": 1,
        "execution_config_id": I1_EXECUTION_CONFIG_ID,
        "protocol_id": "i1-g1b-m2-v1",
        "policy": P1_POLICY,
        "cases": cases,
    }


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        (lambda payload: payload.update(policy="bad"), "I1_FIXTURE_SCHEMA"),
        (lambda payload: payload.update(cases=[]), "I1_FIXTURE_SCHEMA"),
        (lambda payload: payload["cases"][0].update(case_id=""), "I1_FIXTURE_CASE"),
        (
            lambda payload: payload["cases"][0].update(expected_relation_class="bad"),
            "I1_FIXTURE_CASE",
        ),
        (
            lambda payload: payload["cases"][0].update(boundary_role="bad"),
            "I1_FIXTURE_BOUNDARY",
        ),
        (lambda payload: payload["cases"][0].update(rationale=""), "I1_FIXTURE_CASE"),
        (
            lambda payload: payload["cases"][0].update(expected_scope="root"),
            "I1_FIXTURE_CASE",
        ),
        (
            lambda payload: payload["cases"][0].update(request_ids=[""]),
            "I1_FIXTURE_CASE",
        ),
        (
            lambda payload: payload["cases"][0].update(expected_routes=["fallback"]),
            "I1_FIXTURE_CASE",
        ),
        (
            lambda payload: payload["cases"][1].update(case_id="case-0"),
            "I1_FIXTURE_CASE",
        ),
        (
            lambda payload: payload["cases"][0].update(a0_credential={}),
            "I1_FIXTURE_CREDENTIAL",
        ),
        (
            lambda payload: payload["cases"][0]["a0_credential"].update(sha256="bad"),
            "I1_FIXTURE_CREDENTIAL",
        ),
        (
            lambda payload: payload["cases"][0]["a0_credential"].update(shape=[-1]),
            "I1_FIXTURE_CREDENTIAL",
        ),
    ),
)
def test_fixture_negative_branches(mutation, code: str) -> None:
    """逐项覆盖 fixture 的 fail-closed schema 分支。"""
    payload = _valid_fixture_payload()
    mutation(payload)
    with pytest.raises(I1Error, match=code):
        validate_i1_fixture(payload)


def _valid_ledger_payload() -> dict:
    """构造最小合法 I1 ledger。"""
    return {
        "schema_version": 1,
        "run_id": "run",
        "execution_config_id": I1_EXECUTION_CONFIG_ID,
        "policy": P1_POLICY,
        "events": [
            {
                "sequence": 0,
                "implementation": "g1b-i1",
                "case_id": "case",
                "expert_id": "E0",
                "kind": "shared",
                "batch_indices": [0],
                "count": 1,
                "verifier_backend_id": "g1b-torch-int64-cpu-v1",
                "verifier_evidence_digest": "a" * 64,
                "route_digest": "b" * 64,
                "zero_call_reason": None,
            }
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.update(policy="bad"),
        lambda payload: payload.update(events="bad"),
        lambda payload: payload["events"][0].update(implementation="bad"),
        lambda payload: payload["events"][0].update(batch_indices=[-1]),
        lambda payload: payload["events"][0].update(route_digest="bad"),
        lambda payload: payload["events"][0].update(
            count=0, batch_indices=[], zero_call_reason=None
        ),
        lambda payload: payload["events"][0].update(
            batch_indices=[], zero_call_reason="bad"
        ),
    ),
)
def test_ledger_negative_branches(mutation) -> None:
    """逐项覆盖 ledger 的 fail-closed schema 分支。"""
    payload = _valid_ledger_payload()
    mutation(payload)
    with pytest.raises(I1Error):
        validate_i1_call_ledger(payload)


def _valid_summary_payload() -> dict:
    """构造最小合法 I1 summary。"""
    return {
        "schema_version": 1,
        "run_id": "run",
        "execution_config_id": I1_EXECUTION_CONFIG_ID,
        "protocol_id": "i1-g1b-m2-v1",
        "policy": P1_POLICY,
        "status": "complete",
        "exit_code": 0,
        "case_count": 1,
        "mathematical_difference_count": 0,
        "adapter_difference_count": 0,
        "route_difference_count": 0,
        "unauthorized_routed_calls": 0,
        "negative_case_count": 0,
        "determinism": {},
        "coverage": {},
        "artifacts": {},
        "provenance": {},
        "failure": None,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.update(protocol_id="bad"),
        lambda payload: payload.update(status="unknown"),
        lambda payload: payload.update(case_count=-1),
        lambda payload: payload.update(coverage=[]),
        lambda payload: payload.update(failure="bad"),
    ),
)
def test_summary_negative_branches(mutation) -> None:
    """逐项覆盖 summary 的 fail-closed schema 分支。"""
    payload = _valid_summary_payload()
    mutation(payload)
    with pytest.raises(I1Error, match="I1_SUMMARY_SCHEMA"):
        validate_i1_summary(payload)
