"""执行 I1 双 policy CPU 集成并生成可审计交付物。"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..auth_expert_moe_m2 import (
    P1_POLICY,
    P2_POLICY,
    M2AuthExpert,
    M2CallLedger,
    M2Config,
    M2Context,
    M2MoE,
    M2RouteCoordinator,
    M2ScopeRegistry,
)
from ..auth_expert_moe_m2.manifest import load_m2_manifest
from ..modint_verifier_g1a import (
    G1AConfig,
    G1AParameters,
    parameter_digest,
    verify_batch,
)
from ..modint_verifier_g1b import verify_tensor_batch
from ..modint_verifier_g1b.manifest import load_g1b_manifest
from ..pretrained_gate.authorization import FixedRelationVerifier
from .adapter import G1BVerifierAdapter
from .artifacts import (
    validate_i1_call_ledger,
    validate_i1_fixture,
    validate_i1_summary,
)
from .manifest import build_i1_manifest, load_i1_manifest
from .types import I1_EXECUTION_CONFIG_ID, I1_PROTOCOL_ID, I1Error

_SCOPES = [
    {"scope_id": "standard", "parent_scope_ids": [], "expert_ids": ["E1"]},
    {
        "scope_id": "advanced",
        "parent_scope_ids": ["standard"],
        "expert_ids": ["E1", "E2"],
    },
    {
        "scope_id": "privileged",
        "parent_scope_ids": ["advanced"],
        "expert_ids": ["E1", "E2", "E3"],
    },
]


def _sha256_bytes(value: bytes) -> str:
    """计算 bytes 的 SHA-256。"""
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    """计算文件原始 bytes 的 SHA-256。"""
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: object) -> bytes:
    """使用稳定 JSON 编码结构化对象。"""
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _git_commit(root: Path) -> str:
    """读取 Git commit，失败时返回全零未知值。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40


def _git_dirty(root: Path) -> bool:
    """读取工作树状态，无法确认时 fail closed 为 dirty。"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _descriptor(value: Tensor) -> Dict[str, Any]:
    """生成不泄露 raw credential 的 Tensor 摘要描述符。"""
    array = value.detach().cpu().contiguous().numpy()
    return {
        "encoding": "sha256-only",
        "sha256": _sha256_bytes(array.tobytes()),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def _digest_evidence(
    verified: Sequence[bool], reasons: Sequence[object], errors: Sequence[object]
) -> str:
    """计算不含私有 seal 的 evidence 诊断摘要。"""
    return _sha256_bytes(
        _canonical_json(
            {
                "verified": [bool(value) for value in verified],
                "reason": [str(value) for value in reasons],
                "error": [float(value) for value in errors],
            }
        )
    )


def _digest_route(routes: Sequence[object], allowed: Sequence[Sequence[str]]) -> str:
    """计算 route 与 allowed experts 的规范摘要。"""
    return _sha256_bytes(
        _canonical_json(
            {
                "routes": [getattr(value, "value", str(value)) for value in routes],
                "allowed": [list(value) for value in allowed],
            }
        )
    )


def _load_parameters(
    g1b_dir: Path, tau: int
) -> Tuple[G1AParameters, Dict[str, Any], str]:
    """从已验收 G1-b 交付物重建并绑定公开参数。"""
    manifest_path = g1b_dir / "manifest.json"
    manifest_sha = _sha256_file(manifest_path)
    manifest = load_g1b_manifest(manifest_path, manifest_sha)
    bounds = manifest["integer_bounds"]
    config = G1AConfig(
        q=bounds["q"],
        n=bounds["n"],
        m=bounds["m"],
        tau=tau,
        protocol_id="g1a-modint-v1",
        execution_config_id="g1a-reference-cpu-v1",
        master_seed=0,
    )
    parameters = G1AParameters(
        A=np.load(g1b_dir / "A.npy", allow_pickle=False),
        b=np.load(g1b_dir / "b.npy", allow_pickle=False),
        config=config,
    )
    if parameter_digest(parameters) != manifest["source_parameter_sha256"]:
        raise I1Error("I1_PARAMETER_DIGEST_MISMATCH", "G1-b A/b/tau 与 manifest 不匹配")
    return parameters, manifest, manifest_sha


def _boundary_credentials(parameters: G1AParameters) -> Dict[str, Tensor]:
    """穷举 tiny 域，选择 nominal 与 tau/tau+1 边界 credential。"""
    config = parameters.config
    candidate_count = config.q**config.n
    if candidate_count > 1_000_000:
        raise I1Error("I1_FIXTURE_DOMAIN_TOO_LARGE", "I1 runner 只允许 tiny 穷举域")
    batch = torch.tensor(
        list(itertools.product(range(config.q), repeat=config.n)), dtype=torch.int64
    )
    result = verify_tensor_batch(batch, parameters)
    maxima = result.evidence.max_abs_residual

    def select(predicate: object, name: str) -> Tensor:
        """按 max residual 谓词选择第一行 canonical credential。"""
        for index, value in enumerate(maxima):
            if bool(predicate(value)):  # type: ignore[operator]
                return batch[index : index + 1].contiguous()
        raise I1Error("I1_FIXTURE_BOUNDARY_MISSING", f"未找到 {name} credential")

    return {
        "nominal_valid": select(lambda value: value < config.tau, "nominal valid"),
        "nominal_failure": select(
            lambda value: value > config.tau + 1, "nominal failure"
        ),
        "exact_accept_boundary": select(lambda value: value == config.tau, "tau"),
        "first_reject_boundary": select(lambda value: value == config.tau + 1, "tau+1"),
    }


def _case_specs(parameters: G1AParameters, policy: str) -> List[Dict[str, Any]]:
    """构造覆盖 relation、scope 和 parser 边界的配对 case。"""
    boundaries = _boundary_credentials(parameters)
    threshold = np.float32(0.1)
    below = np.nextafter(threshold, np.float32(0.0), dtype=np.float32)
    route_failed = "deny" if policy == P1_POLICY else "public"
    cases: List[Dict[str, Any]] = [
        {
            "case_id": "nominal-valid-standard",
            "expected_relation_class": "valid",
            "boundary_role": "nominal_valid",
            "a0": torch.zeros((1, 2), dtype=torch.float32),
            "a0_reason": "SUCCESS",
            "g1b": boundaries["nominal_valid"],
            "g1b_reason": "RELATION_WITHIN_BOUND",
            "scope": "standard",
            "route": "protected",
            "rationale": "两种关系均为非边界合法输入。",
        },
        {
            "case_id": "nominal-valid-advanced",
            "expected_relation_class": "valid",
            "boundary_role": "nominal_valid",
            "a0": torch.zeros((1, 2), dtype=torch.float32),
            "a0_reason": "SUCCESS",
            "g1b": boundaries["nominal_valid"],
            "g1b_reason": "RELATION_WITHIN_BOUND",
            "scope": "advanced",
            "route": "protected",
            "rationale": "同一 valid 抽象类覆盖 advanced scope。",
        },
        {
            "case_id": "exact-accept-privileged",
            "expected_relation_class": "valid",
            "boundary_role": "exact_accept_boundary",
            "a0": torch.tensor([[float(below), 0.0]], dtype=torch.float32),
            "a0_reason": "SUCCESS",
            "g1b": boundaries["exact_accept_boundary"],
            "g1b_reason": "RELATION_WITHIN_BOUND",
            "scope": "privileged",
            "route": "protected",
            "rationale": "A0 使用阈值下一个 FP32，G1-b 使用 linf=tau。",
        },
        {
            "case_id": "nominal-failure",
            "expected_relation_class": "failure",
            "boundary_role": "nominal_failure",
            "a0": torch.ones((1, 2), dtype=torch.float32),
            "a0_reason": "RELATION_FAILED",
            "g1b": boundaries["nominal_failure"],
            "g1b_reason": "RELATION_OUT_OF_BOUND",
            "scope": "standard",
            "route": route_failed,
            "rationale": "两种关系均为非边界关系失败。",
        },
        {
            "case_id": "first-reject-boundary",
            "expected_relation_class": "failure",
            "boundary_role": "first_reject_boundary",
            "a0": torch.tensor([[float(threshold), 0.0]], dtype=torch.float32),
            "a0_reason": "RELATION_FAILED",
            "g1b": boundaries["first_reject_boundary"],
            "g1b_reason": "RELATION_OUT_OF_BOUND",
            "scope": "advanced",
            "route": route_failed,
            "rationale": "A0 严格小于边界拒绝，G1-b 使用 linf=tau+1。",
        },
    ]
    invalid_specs = (
        (
            "format-dtype",
            "format_dtype",
            torch.zeros((1, 2), dtype=torch.float64),
            torch.zeros((1, parameters.config.n), dtype=torch.float32),
            "G1B_INPUT_DTYPE",
        ),
        (
            "format-rank",
            "format_rank",
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(parameters.config.n, dtype=torch.int64),
            "G1B_INPUT_SHAPE",
        ),
        (
            "format-shape",
            "format_shape",
            torch.zeros((1, 3), dtype=torch.float32),
            torch.zeros((1, parameters.config.n - 1), dtype=torch.int64),
            "G1B_INPUT_SHAPE",
        ),
        (
            "domain-below-zero",
            "domain_below_zero",
            torch.zeros((1, 2), dtype=torch.float64),
            torch.tensor([[-1] + [0] * (parameters.config.n - 1)], dtype=torch.int64),
            "G1B_INPUT_DOMAIN",
        ),
        (
            "domain-at-q",
            "domain_at_q",
            torch.zeros((1, 2), dtype=torch.float64),
            torch.tensor(
                [[parameters.config.q] + [0] * (parameters.config.n - 1)],
                dtype=torch.int64,
            ),
            "G1B_INPUT_DOMAIN",
        ),
    )
    for case_id, role, a0_value, g1b_value, reason in invalid_specs:
        cases.append(
            {
                "case_id": case_id,
                "expected_relation_class": "format_error",
                "boundary_role": role,
                "a0": a0_value,
                "a0_reason": "A0_STRUCTURE_ERROR",
                "g1b": g1b_value,
                "g1b_reason": reason,
                "scope": None,
                "route": "error",
                "rationale": "两侧均在关系判定前按各自 canonical parser 拒绝。",
            }
        )
    return cases


def _fixture_payload(cases: Sequence[Mapping[str, Any]], policy: str) -> Dict[str, Any]:
    """将内存 case 转为不含 raw credential 的 fixture。"""
    records = []
    for case in cases:
        request_id = f"request-{case['case_id']}"
        records.append(
            {
                "case_id": case["case_id"],
                "expected_relation_class": case["expected_relation_class"],
                "boundary_role": case["boundary_role"],
                "a0_credential": _descriptor(case["a0"]),
                "a0_expected_reason": case["a0_reason"],
                "g1b_credential": _descriptor(case["g1b"]),
                "g1b_expected_reason": case["g1b_reason"],
                "rationale": case["rationale"],
                "expected_scope": case["scope"],
                "request_ids": [request_id],
                "expected_routes": [case["route"]],
            }
        )
    payload = {
        "schema_version": 1,
        "execution_config_id": I1_EXECUTION_CONFIG_ID,
        "protocol_id": I1_PROTOCOL_ID,
        "policy": policy,
        "cases": records,
    }
    validate_i1_fixture(payload)
    return payload


def _execute_policy(
    *,
    policy: str,
    cases: Sequence[Mapping[str, Any]],
    parameters: G1AParameters,
    adapter: G1BVerifierAdapter,
    run_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    """执行 A0-M2/G1b-I1 配对 case 并返回台账和差分计数。"""
    a0 = FixedRelationVerifier(
        np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.1
    )
    assignments = {str(case["case_id"]): case["scope"] for case in cases}
    a0_auth = M2AuthExpert(a0, I1_EXECUTION_CONFIG_ID)
    g1_auth = M2AuthExpert(adapter, I1_EXECUTION_CONFIG_ID)
    a0_coord = M2RouteCoordinator(a0_auth, I1_EXECUTION_CONFIG_ID, policy)
    g1_coord = M2RouteCoordinator(g1_auth, I1_EXECUTION_CONFIG_ID, policy)
    a0_registry = M2ScopeRegistry(a0_coord, assignments)
    g1_registry = M2ScopeRegistry(g1_coord, assignments)
    config = M2Config(execution_config_id=I1_EXECUTION_CONFIG_ID)
    a0_model = M2MoE(a0_coord, config=config, registry=a0_registry)
    g1_model = M2MoE(g1_coord, config=config, registry=g1_registry)
    hidden = torch.arange(32, dtype=torch.float32).reshape(1, 2, 16) / 32.0
    events: List[Dict[str, Any]] = []
    outputs: List[np.ndarray] = []
    math_diff = adapter_diff = route_diff = unauthorized = negative_count = 0

    for case in cases:
        case_id = str(case["case_id"])
        context = M2Context((f"request-{case_id}",), I1_EXECUTION_CONFIG_ID)
        if case["expected_relation_class"] == "format_error":
            negative_count += 1
            errors = []
            for implementation, auth, credential in (
                ("a0-m2", a0_auth, case["a0"]),
                ("g1b-i1", g1_auth, case["g1b"]),
            ):
                assignment = (
                    a0_registry if implementation == "a0-m2" else g1_registry
                ).assignment(case_id, context)
                try:
                    auth.verify(credential, context, assignment)
                except (TypeError, ValueError, RuntimeError) as exc:
                    errors.append(type(exc).__name__)
                else:
                    route_diff += 1
                events.append(
                    {
                        "sequence": len(events),
                        "implementation": implementation,
                        "case_id": case_id,
                        "expert_id": "ALL",
                        "kind": "none",
                        "batch_indices": [],
                        "count": 0,
                        "verifier_backend_id": (
                            "toy-real-fp32-v1"
                            if implementation == "a0-m2"
                            else "g1b-torch-int64-cpu-v1"
                        ),
                        "verifier_evidence_digest": _sha256_bytes(
                            _canonical_json(errors[-1:])
                        ),
                        "route_digest": _sha256_bytes(b"no-route"),
                        "zero_call_reason": "format_error",
                    }
                )
            if len(errors) != 2:
                route_diff += 1
            continue

        a0_assignment = a0_registry.assignment(case_id, context)
        g1_assignment = g1_registry.assignment(case_id, context)
        a0_bound = a0_auth.verify(case["a0"], context, a0_assignment)
        g1_bound = g1_auth.verify(case["g1b"], context, g1_assignment)
        a0_route = a0_coord.commit(a0_bound, context)
        g1_route = g1_coord.commit(g1_bound, context)
        expected_route = str(case["route"])
        if [value.value for value in a0_route.routes] != [expected_route] or [
            value.value for value in g1_route.routes
        ] != [expected_route]:
            route_diff += 1
        if (
            a0_route.routes != g1_route.routes
            or a0_route.allowed_experts != g1_route.allowed_experts
        ):
            route_diff += 1

        graph = adapter.verifier(case["g1b"])
        host = verify_batch(case["g1b"].tolist(), parameters)
        math_diff += int(
            graph.evidence != host.evidence or graph.residuals != host.residuals
        )
        adapter_diff += int(
            g1_bound.evidence.verified.tolist() != list(graph.evidence.accepted)
            or g1_bound.evidence.error_norm.tolist()
            != [float(value) for value in graph.evidence.max_abs_residual]
        )

        a0_ledger = M2CallLedger(run_id, I1_EXECUTION_CONFIG_ID, policy)
        g1_ledger = M2CallLedger(run_id, I1_EXECUTION_CONFIG_ID, policy)
        a0_output = a0_model(hidden, a0_route, context, a0_ledger, case_id)
        g1_output = g1_model(hidden, g1_route, context, g1_ledger, case_id)
        outputs.append(a0_output.output.detach().cpu().numpy())
        if (
            not torch.equal(a0_output.output, g1_output.output)
            or a0_output.selection != g1_output.selection
        ):
            route_diff += 1
        for implementation, ledger, bound, route, backend in (
            ("a0-m2", a0_ledger, a0_bound, a0_route, "toy-real-fp32-v1"),
            ("g1b-i1", g1_ledger, g1_bound, g1_route, "g1b-torch-int64-cpu-v1"),
        ):
            evidence_digest = _digest_evidence(
                bound.evidence.verified.tolist(),
                bound.evidence.reason_code.tolist(),
                bound.evidence.error_norm.tolist(),
            )
            route_digest = _digest_route(route.routes, route.allowed_experts)
            for event in ledger.events:
                events.append(
                    {
                        "sequence": len(events),
                        "implementation": implementation,
                        "case_id": case_id,
                        "expert_id": event.expert_id,
                        "kind": event.kind,
                        "batch_indices": list(event.batch_indices),
                        "count": event.count,
                        "verifier_backend_id": backend,
                        "verifier_evidence_digest": evidence_digest,
                        "route_digest": route_digest,
                        "zero_call_reason": None,
                    }
                )
            routed = [event for event in ledger.events if event.kind == "routed"]
            if expected_route != "protected" and routed:
                unauthorized += len(routed)
            if expected_route != "protected":
                events.append(
                    {
                        "sequence": len(events),
                        "implementation": implementation,
                        "case_id": case_id,
                        "expert_id": "ROUTED",
                        "kind": "routed",
                        "batch_indices": [],
                        "count": 0,
                        "verifier_backend_id": backend,
                        "verifier_evidence_digest": evidence_digest,
                        "route_digest": route_digest,
                        "zero_call_reason": expected_route,
                    }
                )

    ledger_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "execution_config_id": I1_EXECUTION_CONFIG_ID,
        "policy": policy,
        "events": events,
    }
    metrics = {
        "mathematical_difference_count": math_diff,
        "adapter_difference_count": adapter_diff,
        "route_difference_count": route_diff,
        "unauthorized_routed_calls": unauthorized,
        "negative_case_count": negative_count,
    }
    output_array = np.concatenate(outputs, axis=0).astype("<f4", copy=False)
    return ledger_payload, metrics, output_array


def run_i1_artifacts(
    *,
    repository_root: Path,
    g1b_dir: Path,
    m2_run_id: str,
    run_id: str,
    coverage_report: Path,
    statement_coverage: float,
    branch_coverage: float,
    tau: int = 2,
) -> Dict[str, Any]:
    """生成 P1/P2 I1 结果目录并返回总状态。"""
    root = Path(repository_root)
    report_path = Path(coverage_report)
    if not report_path.is_file():
        raise FileNotFoundError("I1 coverage report 不存在")
    for name, value in (
        ("statement_coverage", statement_coverage),
        ("branch_coverage", branch_coverage),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 100.0
        ):
            raise I1Error("I1_COVERAGE_VALUE", f"{name} 必须是 [0,100] 有限数")
    coverage_passed = statement_coverage >= 95.0 and branch_coverage >= 90.0
    coverage_bytes = report_path.read_bytes()
    parameters, _, g1b_manifest_sha = _load_parameters(Path(g1b_dir), tau)
    adapter = G1BVerifierAdapter.from_manifest(
        parameters, Path(g1b_dir) / "manifest.json", g1b_manifest_sha
    )
    cases_by_policy = {
        policy: _case_specs(parameters, policy) for policy in (P1_POLICY, P2_POLICY)
    }
    result_root = root / "results" / I1_EXECUTION_CONFIG_ID
    targets = {
        policy: result_root / policy / run_id for policy in (P1_POLICY, P2_POLICY)
    }
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("I1 run 已存在，禁止覆盖")
    provenance = {
        "git_commit": _git_commit(root),
        "dirty": _git_dirty(root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    overall = {"status": "complete", "run_id": run_id, "policies": {}}
    scope_sha = _sha256_bytes(_canonical_json(_SCOPES))

    for policy in (P1_POLICY, P2_POLICY):
        output_dir = targets[policy]
        output_dir.mkdir(parents=True, exist_ok=False)
        m2_dir = root / "results" / "m2-multi-expert-v1" / policy / m2_run_id
        m2_manifest_path = m2_dir / "manifest.json"
        m2_manifest_sha = _sha256_file(m2_manifest_path)
        load_m2_manifest(m2_manifest_path, m2_manifest_sha)
        fixture = _fixture_payload(cases_by_policy[policy], policy)
        fixture_bytes = _canonical_json(fixture)
        assignment_sha = _sha256_bytes(
            _canonical_json(
                [
                    {
                        "case_id": case["case_id"],
                        "scope_id": case["scope"],
                        "policy": policy,
                    }
                    for case in cases_by_policy[policy]
                ]
            )
        )
        manifest = build_i1_manifest(
            policy=policy,
            parameters=parameters,
            m2_manifest_sha256=m2_manifest_sha,
            m2_assignment_sha256=assignment_sha,
            g1b_backend_manifest_sha256=g1b_manifest_sha,
            scope_schema_sha256=scope_sha,
            provenance=provenance,
        )
        manifest_bytes = _canonical_json(manifest)
        (output_dir / "manifest.json").write_bytes(manifest_bytes)
        load_i1_manifest(output_dir / "manifest.json", _sha256_bytes(manifest_bytes))
        (output_dir / "fixture.json").write_bytes(fixture_bytes)

        ledger, metrics, reference = _execute_policy(
            policy=policy,
            cases=cases_by_policy[policy],
            parameters=parameters,
            adapter=adapter,
            run_id=run_id,
        )
        repeat_ledger, repeat_metrics, repeat_reference = _execute_policy(
            policy=policy,
            cases=cases_by_policy[policy],
            parameters=parameters,
            adapter=adapter,
            run_id=run_id,
        )
        determinism_difference = int(
            ledger != repeat_ledger
            or metrics != repeat_metrics
            or not np.array_equal(reference, repeat_reference)
        )
        validate_i1_call_ledger(ledger)
        (output_dir / "call_ledger.json").write_bytes(_canonical_json(ledger))
        np.save(output_dir / "reference_outputs.npy", reference)
        (output_dir / "coverage_i1.json").write_bytes(coverage_bytes)
        difference_count = (
            metrics["mathematical_difference_count"]
            + metrics["adapter_difference_count"]
            + metrics["route_difference_count"]
            + metrics["unauthorized_routed_calls"]
            + determinism_difference
            + int(not coverage_passed)
        )
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "execution_config_id": I1_EXECUTION_CONFIG_ID,
            "protocol_id": I1_PROTOCOL_ID,
            "policy": policy,
            "status": "complete" if difference_count == 0 else "failed",
            "exit_code": 0 if difference_count == 0 else 1,
            "case_count": len(cases_by_policy[policy]),
            **metrics,
            "determinism": {
                "repeat_checked": True,
                "difference_count": determinism_difference,
            },
            "coverage": {
                "statement": {"status": "measured", "value": float(statement_coverage)},
                "branch": {"status": "measured", "value": float(branch_coverage)},
                "report": {
                    "path": "coverage_i1.json",
                    "sha256": _sha256_bytes(coverage_bytes),
                },
            },
            "artifacts": {
                "manifest_sha256": _sha256_file(output_dir / "manifest.json"),
                "fixture_sha256": _sha256_file(output_dir / "fixture.json"),
                "ledger_sha256": _sha256_file(output_dir / "call_ledger.json"),
                "reference_outputs_sha256": _sha256_file(
                    output_dir / "reference_outputs.npy"
                ),
                "coverage_sha256": _sha256_file(output_dir / "coverage_i1.json"),
            },
            "provenance": provenance,
            "failure": (
                None
                if difference_count == 0
                else {"code": "I1_DIFFERENCE", "retryable": False}
            ),
        }
        validate_i1_summary(summary)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        overall["policies"][policy] = summary["status"]
        if summary["status"] != "complete":
            overall["status"] = "failed"
    return overall
