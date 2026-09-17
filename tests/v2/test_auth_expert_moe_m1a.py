"""M1a tiny-MoE contract 的确定性专项测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.can.v2.auth_expert_moe import (
    P1_POLICY,
    P2_POLICY,
    CallLedger,
    FixedConstrainedRouter,
    M1AAuthExpert,
    M1aAuthorizationError,
    M1AConfig,
    M1AContext,
    M1aMoE,
    M1ARouteCoordinator,
    M1AScopeRegistry,
    M1ASelection,
)
from src.can.v2.auth_expert_moe.artifacts import (
    load_m1a_fixture,
    validate_call_ledger,
    validate_m1a_summary,
)
from src.can.v2.auth_expert_moe.manifest import load_m1a_manifest
from src.can.v2.pretrained_gate.authorization import FixedRelationVerifier
from src.can.v2.pretrained_gate.types import RouteKind


def _manifest_payload():
    """返回可独立变异的合法 manifest payload。"""

    sha = "0" * 64
    return {
        "schema_version": 1,
        "execution_config_id": "m1a-tiny-moe-v1",
        "protocol_id": "m1a-contract-v1",
        "policy": P1_POLICY,
        "expert_specs": [
            {
                "expert_id": "E0",
                "kind": "shared",
                "capability_level": "general",
                "scope_ids": [],
                "architecture_revision": "v1",
                "weights_sha256": sha,
                "train_data_scope": "public",
                "max_context_length": 8,
                "enabled": True,
            },
            {
                "expert_id": "E1",
                "kind": "protected",
                "capability_level": "protected",
                "scope_ids": ["protected.default"],
                "architecture_revision": "v1",
                "weights_sha256": sha,
                "train_data_scope": "protected",
                "max_context_length": 8,
                "enabled": True,
            },
        ],
        "alpha": 1.0,
        "top_k": 1,
        "router": {"type": "fixed", "normalize_eps": 1e-12},
        "model": {"d_model": 16, "dtype": "float32", "device": "cpu"},
        "fixture": {"inputs_sha256": sha, "reference_outputs_sha256": sha},
        "provenance": {"git_commit": "0" * 40},
    }


def _parts(policy: str = P1_POLICY):
    """构造固定 verifier、AuthExpert、Coordinator 和上下文。"""

    verifier = FixedRelationVerifier(
        np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.1
    )
    auth = M1AAuthExpert(verifier, "m1a-tiny-moe-v1")
    coordinator = M1ARouteCoordinator(auth, "m1a-tiny-moe-v1", policy)
    context = M1AContext(("r0", "r1", "r2"), "m1a-tiny-moe-v1")
    credential = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    evidence = auth.verify(credential, context)
    return auth, coordinator, context, evidence


def test_config_is_frozen():
    """固定配置拒绝 top-k、alpha 和 epsilon 漂移。"""

    assert M1AConfig().top_k == 1
    with pytest.raises(ValueError):
        M1AConfig(top_k=2)
    with pytest.raises(ValueError):
        M1AConfig(alpha=0.5)


def test_context_rejects_duplicate_ids():
    """请求 ID 必须唯一。"""

    with pytest.raises(ValueError):
        M1AContext(("x", "x"), "m1a-tiny-moe-v1")


def test_auth_expert_only_produces_bound_evidence():
    """AuthExpert 返回绑定 evidence，不能伪造 context。"""

    auth, _, context, evidence = _parts()
    auth.validate(evidence, context)
    with pytest.raises(M1aAuthorizationError):
        auth.validate(evidence, M1AContext(("other", "x", "y"), "m1a-tiny-moe-v1"))


def test_p1_route_valid_and_failure_deny():
    """P1 valid 为 protected，关系失败为 deny。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    assert [item.value for item in route.routes] == ["protected", "deny", "protected"]
    assert route.allowed_experts == (("E1",), (), ("E1",))


def test_p2_relation_failure_is_explicit_public_e0():
    """P2 仅将 canonical relation failure 分类为 PUBLIC/E0。"""

    _, coordinator, context, evidence = _parts(P2_POLICY)
    route = coordinator.commit(evidence, context)
    assert [item.value for item in route.routes] == ["protected", "public", "protected"]
    assert route.allowed_experts == (("E1",), ("E0",), ("E1",))


def test_route_validation_rejects_foreign_context():
    """route 必须绑定完整有序 request IDs。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    with pytest.raises(M1aAuthorizationError):
        coordinator.validate(route, ("r1", "r0", "r2"))


def test_router_masks_out_of_scope():
    """固定 Router 不允许从空 mask 选择 routed expert。"""

    router = FixedConstrainedRouter()
    result = router.select(
        torch.tensor([[0.0, 100.0], [0.0, 1.0]]),
        torch.tensor([[True, False], [False, True]]),
    )
    assert result.values == ("E0", "E1")


def test_router_rejects_shape_and_dtype():
    """Router 严格拒绝错误 shape/dtype。"""

    router = FixedConstrainedRouter()
    with pytest.raises(ValueError):
        router.select(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        router.select(torch.zeros(2, 2), torch.zeros(2, 2))


def test_moe_shared_runs_for_every_row_and_routed_is_sparse():
    """mixed batch 中 E0 覆盖全部原始索引，E1 只覆盖授权行。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    model = M1aMoE(coordinator)
    ledger = CallLedger("run", "m1a-tiny-moe-v1", P1_POLICY)
    result = model(torch.ones(3, 4, 16), route, context, ledger)
    assert result.routed_indices.tolist() == [0, 2]
    assert result.denied_indices.tolist() == [1]
    assert ledger.events[0].expert_id == "E0" and ledger.events[0].batch_indices == (
        0,
        1,
        2,
    )
    assert ledger.events[1].expert_id == "E1" and ledger.events[1].batch_indices == (
        0,
        2,
    )


def test_moe_p2_public_has_no_routed_call():
    """P2 PUBLIC/E0 仍只使用已经执行的 shared 分支。"""

    _, coordinator, context, evidence = _parts(P2_POLICY)
    route = coordinator.commit(evidence, context)
    ledger = CallLedger("run", "m1a-tiny-moe-v1", P2_POLICY)
    result = M1aMoE(coordinator)(torch.ones(3, 2, 16), route, context, ledger)
    assert result.selection.values == ("E1", "E0", "E1")
    assert len(ledger.events) == 2
    assert ledger.events[-1].batch_indices == (0, 2)


def test_scope_restriction_removes_e1_without_calling_it():
    """scope 收窄为空后，原 protected 行只能得到 shared 输出。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    registry = M1AScopeRegistry(coordinator)
    full = registry.resolve(route, context)
    narrowed = registry.restrict(full, ((), (), ()))
    model = M1aMoE(coordinator, registry=registry)
    ledger = CallLedger("run", "m1a-tiny-moe-v1", P1_POLICY)
    result = model(torch.ones(3, 2, 16), route, context, ledger, view=narrowed)
    assert result.selection.values == (None, None, None)
    assert result.routed_indices.numel() == 0
    assert len(ledger.events) == 1 and torch.equal(result.output, result.shared_output)


def test_scope_restriction_cannot_expand_deny_row():
    """调用方不能把 DENY 行的空 scope 扩大为 E1。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    registry = M1AScopeRegistry(coordinator)
    view = registry.resolve(route, context)
    with pytest.raises(M1aAuthorizationError):
        registry.restrict(view, (("E1",), ("E1",), ("E1",)))


def test_moe_output_is_finite_and_shape_preserved():
    """组合输出保持 shape、dtype 且全部有限。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    output = M1aMoE(coordinator)(torch.ones(3, 1, 16), route, context).output
    assert output.shape == (3, 1, 16)
    assert output.dtype == torch.float32 and torch.isfinite(output).all()


def test_moe_rejects_external_route_and_context_mismatch():
    """MoE 不接受未登记 route 或错误 context。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    model = M1aMoE(coordinator)
    with pytest.raises(M1aAuthorizationError):
        model(
            torch.ones(3, 1, 16), route, M1AContext(("x", "y", "z"), "m1a-tiny-moe-v1")
        )


def test_moe_rejects_empty_and_nonfinite_hidden():
    """hidden 空 batch 和非有限值 fail closed。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    model = M1aMoE(coordinator)
    with pytest.raises(ValueError):
        model(torch.ones(0, 1, 16), route, M1AContext((), "m1a-tiny-moe-v1"))
    with pytest.raises(ValueError):
        model(torch.full((3, 1, 16), float("nan")), route, context)


def test_ledger_sequence_and_no_tensor_payload():
    """台账 sequence 单调且只含索引/计数。"""

    ledger = CallLedger("r", "m1a-tiny-moe-v1", P1_POLICY)
    ledger.record("c", "E0", "shared", (0, 1))
    data = ledger.to_dict()
    assert data["events"][0]["sequence"] == 0
    assert data["events"][0]["count"] == 1
    assert "hidden" not in json.dumps(data)


def test_manifest_digest_and_strict_schema(tmp_path: Path):
    """manifest 使用完整 bytes 摘要并拒绝未知字段。"""

    payload = _manifest_payload()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_m1a_manifest(path, digest)["policy"] == P1_POLICY
    payload["unexpected"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_m1a_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_manifest_rejects_wrong_digest(tmp_path: Path):
    """错误 manifest 摘要必须拒绝。"""

    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_m1a_manifest(path, "0" * 64)


def test_ledger_validator_rejects_non_forward_count():
    """台账 count 必须表示一次真实 forward。"""

    payload = {
        "schema_version": 1,
        "run_id": "r",
        "execution_config_id": "m1a-tiny-moe-v1",
        "policy": P1_POLICY,
        "events": [
            {
                "sequence": 0,
                "case_id": "c",
                "stage": "forward",
                "expert_id": "E0",
                "kind": "shared",
                "batch_indices": [0],
                "count": 2,
            }
        ],
    }
    with pytest.raises(ValueError):
        validate_call_ledger(payload)


def test_summary_validator_requires_stable_root_schema():
    """summary 缺少字段或使用未知状态时拒绝。"""

    payload = {
        "schema_version": 1,
        "run_id": "r",
        "execution_config_id": "m1a-tiny-moe-v1",
        "protocol_id": "m1a-contract-v1",
        "policy": P1_POLICY,
        "status": "not_run",
        "exit_code": 0,
        "started_at": "x",
        "finished_at": "x",
        "cases": [],
        "metrics": {},
        "determinism": {},
        "coverage": {},
        "artifacts": {},
        "provenance": {},
        "failure": None,
    }
    validate_m1a_summary(payload)
    payload["status"] = "unknown"
    with pytest.raises(ValueError):
        validate_m1a_summary(payload)


def test_fixture_loader_validates_npy_files(tmp_path: Path):
    """fixture loader 校验输入和 reference 的原始文件摘要。"""

    inputs = np.arange(32, dtype="<f4").reshape(2, 1, 16)
    reference = inputs + 1
    np.save(tmp_path / "inputs.npy", inputs, allow_pickle=False)
    np.save(tmp_path / "reference_outputs.npy", reference, allow_pickle=False)
    payload = {
        "schema_version": 1,
        "seeds": {"e0_init": 1, "e1_init": 2, "router_init": 3, "input_init": 4},
        "input": {
            "path": "inputs.npy",
            "dtype": "<f4",
            "shape": [2, 1, 16],
            "sha256": hashlib.sha256(
                (tmp_path / "inputs.npy").read_bytes()
            ).hexdigest(),
        },
        "cases": [],
        "reference": {
            "path": "reference_outputs.npy",
            "format": "npy-v1",
            "dtype": "<f4",
            "shape": [2, 1, 16],
            "sha256": hashlib.sha256(
                (tmp_path / "reference_outputs.npy").read_bytes()
            ).hexdigest(),
            "atol": 1e-6,
            "rtol": 1e-5,
        },
        "expected": [],
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_m1a_fixture(path)["seeds"]["e0_init"] == 1
    payload["input"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_m1a_fixture(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"execution_config_id": ""},
        {"d_model": 0},
        {"dtype": torch.long},
        {"normalize_eps": 1e-6},
    ],
)
def test_config_rejects_invalid_fields(kwargs):
    """配置逐字段拒绝非法值。"""

    with pytest.raises((TypeError, ValueError)):
        M1AConfig(**kwargs)


def test_experts_reject_invalid_inputs_and_remain_frozen():
    """Expert 参数冻结，输入 shape/宽度/有限性/身份均严格检查。"""

    _, coordinator, _, _ = _parts()
    expert = M1aMoE(coordinator).shared
    assert all(not parameter.requires_grad for parameter in expert.parameters())
    for hidden, ids in [
        (torch.ones(2, 16), ("a", "b")),
        (torch.ones(2, 1, 8), ("a", "b")),
        (torch.full((2, 1, 16), float("inf")), ("a", "b")),
        (torch.ones(2, 1, 16), ("a",)),
    ]:
        with pytest.raises(ValueError):
            expert(hidden, ids)
    assert expert(torch.ones(1, 1, 16), ("a",)).shape == (1, 1, 16)


def test_constructor_and_authentication_negative_paths():
    """认证与协调器构造拒绝类型、配置和 policy 混淆。"""

    verifier = FixedRelationVerifier(
        np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.1
    )
    with pytest.raises(TypeError):
        M1AAuthExpert(object(), "x")
    with pytest.raises(ValueError):
        M1AAuthExpert(verifier, "")
    auth = M1AAuthExpert(verifier, "x")
    with pytest.raises(TypeError):
        M1ARouteCoordinator(object(), "x")
    with pytest.raises(ValueError):
        M1ARouteCoordinator(auth, "x", "bad")
    with pytest.raises(M1aAuthorizationError):
        M1ARouteCoordinator(auth, "y")
    with pytest.raises(M1aAuthorizationError):
        auth.verify(torch.zeros(1, 2), M1AContext(("r",), "y"))


def test_router_zero_mask_and_integer_logits_fail_closed():
    """Router 空 mask 返回 None，整数 logits 被拒绝。"""

    router = FixedConstrainedRouter()
    assert router.select(
        torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool)
    ).values == (None,)
    with pytest.raises(ValueError):
        router.select(
            torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.bool)
        )


def test_scope_registry_rejects_foreign_and_malformed_views():
    """scope registry 拒绝错误类型、batch、重复集合和跨 registry view。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    with pytest.raises(TypeError):
        M1AScopeRegistry(object())
    first = M1AScopeRegistry(coordinator)
    second = M1AScopeRegistry(coordinator)
    view = first.resolve(route, context)
    with pytest.raises(ValueError):
        first.restrict(view, ((),))
    with pytest.raises(ValueError):
        first.restrict(view, (("E1", "E1"), (), ("E1",)))
    with pytest.raises(M1aAuthorizationError):
        second.validate(view, route, context)


def test_moe_rejects_bad_constructor_and_hidden_shape():
    """MoE 拒绝错误协调器和 hidden shape/dtype。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    with pytest.raises(M1aAuthorizationError):
        M1aMoE(object())
    model = M1aMoE(coordinator)
    for hidden in (
        torch.ones(3, 16),
        torch.ones(3, 1, 8),
        torch.ones(2, 1, 16),
        torch.ones(3, 1, 16, dtype=torch.float64),
    ):
        with pytest.raises(ValueError):
            model(hidden, route, context)


def test_call_ledger_rejects_empty_and_invalid_indices():
    """调用台账拒绝空、负数和类型混淆索引。"""

    ledger = CallLedger("r", "x", P1_POLICY)
    for indices in ((), (-1,), (True,)):
        with pytest.raises(ValueError):
            ledger.record("c", "E0", "shared", indices)


def test_manifest_all_validation_failures(tmp_path: Path):
    """manifest 的每个严格字段分支均 fail closed。"""

    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_m1a_manifest(missing, "0" * 64)
    valid_path = tmp_path / "manifest.json"
    valid_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    with pytest.raises(ValueError):
        load_m1a_manifest(valid_path, "BAD")

    mutations = []
    for mutate in (
        lambda p: p.update(schema_version=2),
        lambda p: p.update(policy="bad"),
        lambda p: p.update(protocol_id="bad"),
        lambda p: p.update(alpha=0.5),
        lambda p: p.update(router={"type": "fixed"}),
        lambda p: p["router"].update(normalize_eps=1e-6),
        lambda p: p.update(model={"d_model": 16}),
        lambda p: p["model"].update(device="cuda"),
        lambda p: p.update(provenance={"git_commit": "0" * 40, "extra": 1}),
        lambda p: p["provenance"].update(git_commit="bad"),
        lambda p: p.update(expert_specs=[]),
        lambda p: p["expert_specs"][0].update(extra=1),
        lambda p: p["expert_specs"][0].update(weights_sha256="bad"),
        lambda p: p["expert_specs"][0].update(expert_id="E2"),
        lambda p: p["expert_specs"][0].update(enabled=1),
        lambda p: p["expert_specs"][0].update(max_context_length=0),
        lambda p: p.update(fixture={"inputs_sha256": "0" * 64}),
        lambda p: p["fixture"].update(inputs_sha256="bad"),
    ):
        payload = _manifest_payload()
        mutate(payload)
        mutations.append(payload)
    for index, payload in enumerate(mutations):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_m1a_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_m1a_manifest(duplicate, hashlib.sha256(duplicate.read_bytes()).hexdigest())
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_m1a_manifest(nonfinite, hashlib.sha256(nonfinite.read_bytes()).hexdigest())


def test_artifact_validation_negative_paths(tmp_path: Path):
    """fixture、ledger 和 summary 的结构错误均被拒绝。"""

    with pytest.raises(FileNotFoundError):
        load_m1a_fixture(tmp_path / "missing.json")
    path = tmp_path / "fixture.json"
    for raw in ('{"a":1,"a":2}', "{}"):
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError):
            load_m1a_fixture(path)

    base = {
        "schema_version": 1,
        "seeds": {"e0_init": 1, "e1_init": 2, "router_init": 3, "input_init": 4},
        "input": {"path": "inputs.npy", "sha256": "0" * 64},
        "cases": [],
        "reference": {
            "path": "reference_outputs.npy",
            "format": "npy-v1",
            "sha256": "0" * 64,
        },
        "expected": [],
    }
    for mutate in (
        lambda p: p.update(seeds={}),
        lambda p: p["seeds"].update(e0_init=-1),
        lambda p: p.update(input={"path": "bad.npy", "sha256": "0" * 64}),
        lambda p: p["input"].update(sha256="bad"),
        lambda p: p.update(cases={}),
    ):
        payload = json.loads(json.dumps(base))
        mutate(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_m1a_fixture(path)

    for ledger in (
        {},
        {
            "schema_version": 2,
            "run_id": "",
            "execution_config_id": "x",
            "policy": "p",
            "events": [],
        },
        {
            "schema_version": 1,
            "run_id": "r",
            "execution_config_id": "x",
            "policy": "p",
            "events": {},
        },
        {
            "schema_version": 1,
            "run_id": "r",
            "execution_config_id": "x",
            "policy": "p",
            "events": [
                {
                    "sequence": 0,
                    "case_id": "c",
                    "stage": "forward",
                    "expert_id": "E0",
                    "kind": "shared",
                    "batch_indices": [],
                    "count": 1,
                }
            ],
        },
    ):
        with pytest.raises(ValueError):
            validate_call_ledger(ledger)
    with pytest.raises(ValueError):
        validate_m1a_summary({})
    bad_summary = {
        "schema_version": 1,
        "run_id": "r",
        "execution_config_id": "x",
        "protocol_id": "p",
        "policy": P1_POLICY,
        "status": "failed",
        "exit_code": 1,
        "started_at": "x",
        "finished_at": "x",
        "cases": [],
        "metrics": {},
        "determinism": {},
        "coverage": {},
        "artifacts": {},
        "provenance": {},
        "failure": {"code": "x"},
    }
    with pytest.raises(ValueError):
        validate_m1a_summary(bad_summary)


def test_fixture_loader_rejects_tolerance_digest_endian_shape_and_expected(
    tmp_path: Path,
):
    """fixture 的 reference 容差、摘要、字节序、shape 和 expected 类型均校验。"""

    inputs = np.ones((1, 1, 16), dtype="<f4")
    reference = np.ones((1, 1, 16), dtype="<f4")
    np.save(tmp_path / "inputs.npy", inputs, allow_pickle=False)
    np.save(tmp_path / "reference_outputs.npy", reference, allow_pickle=False)
    payload = {
        "schema_version": 1,
        "seeds": {"e0_init": 1, "e1_init": 2, "router_init": 3, "input_init": 4},
        "input": {
            "path": "inputs.npy",
            "format": "npy-v1",
            "dtype": "<f4",
            "shape": [1, 1, 16],
            "sha256": hashlib.sha256(
                (tmp_path / "inputs.npy").read_bytes()
            ).hexdigest(),
        },
        "cases": [],
        "reference": {
            "path": "reference_outputs.npy",
            "format": "npy-v1",
            "dtype": "<f4",
            "shape": [1, 1, 16],
            "sha256": hashlib.sha256(
                (tmp_path / "reference_outputs.npy").read_bytes()
            ).hexdigest(),
            "atol": 1e-6,
            "rtol": 1e-5,
        },
        "expected": [],
    }
    path = tmp_path / "fixture.json"
    for mutate in (
        lambda p: p["reference"].update(atol=1e-3),
        lambda p: p["input"].update(sha256="bad"),
        lambda p: p["reference"].update(shape=[1, 1, 8]),
        lambda p: p.update(expected={}),
    ):
        candidate = json.loads(json.dumps(payload))
        mutate(candidate)
        path.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(ValueError):
            load_m1a_fixture(path)
    big_endian = np.asarray(reference, dtype=">f4")
    np.save(tmp_path / "reference_outputs.npy", big_endian, allow_pickle=False)
    payload["reference"]["sha256"] = hashlib.sha256(
        (tmp_path / "reference_outputs.npy").read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_m1a_fixture(path)


def test_route_and_selection_tamper_paths_are_rejected():
    """route seal、batch、allowed 集合和 Router selection 篡改均在 E1 前拒绝。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    from dataclasses import replace

    for forged in (
        replace(route, _seal=object()),
        replace(route, policy_id=P2_POLICY),
        replace(route, routes=(RouteKind.PROTECTED,)),
        replace(route, allowed_experts=(("E0",), (), ("E1",))),
    ):
        with pytest.raises(M1aAuthorizationError):
            coordinator.validate(forged, context.request_ids)
    with pytest.raises(ValueError):
        M1ASelection(("E2",))


def test_scope_context_mismatch_and_bad_router_are_rejected():
    """已登记 view 不能跨 context，错误 Router 输出不能触发 E1。"""

    _, coordinator, context, evidence = _parts()
    route = coordinator.commit(evidence, context)
    registry = M1AScopeRegistry(coordinator)
    view = registry.resolve(route, context)
    with pytest.raises(M1aAuthorizationError):
        registry.validate(view, route, M1AContext(("x", "y", "z"), "m1a-tiny-moe-v1"))

    class BadRouter(FixedConstrainedRouter):
        """返回与可信 route 不一致的测试 Router。"""

        def select(self, logits, allowed_mask):
            """始终伪造 E0 selection。"""
            return M1ASelection(("E0",) * logits.shape[0])

    model = M1aMoE(coordinator, router=BadRouter(), registry=registry)
    with pytest.raises(M1aAuthorizationError):
        model(torch.ones(3, 1, 16), route, context, view=view)


def test_router_and_coordinator_defensive_branches():
    """覆盖 credential batch、foreign evidence、配置和 route tamper 分支。"""

    verifier = FixedRelationVerifier(
        np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.1
    )
    auth = M1AAuthExpert(verifier, "m1a-tiny-moe-v1")
    context = M1AContext(("r", "s"), "m1a-tiny-moe-v1")
    with pytest.raises(ValueError):
        auth.verify(torch.zeros(1, 2), context)
    foreign = M1AAuthExpert(verifier, "m1a-tiny-moe-v1")
    evidence = foreign.verify(torch.zeros(2, 2), context)
    with pytest.raises(M1aAuthorizationError):
        auth.validate(evidence, context)
    coordinator = M1ARouteCoordinator(auth, "m1a-tiny-moe-v1")
    own = auth.verify(torch.zeros(2, 2), context)
    route = coordinator.commit(own, context)
    object.__setattr__(context, "execution_config_id", "tampered")
    with pytest.raises(M1aAuthorizationError):
        coordinator.commit(own, context)
    object.__setattr__(context, "execution_config_id", "m1a-tiny-moe-v1")
    for field, value in (
        ("_seal", object()),
        ("policy_id", P2_POLICY),
        ("routes", (RouteKind.PROTECTED,)),
        ("allowed_experts", (("E0",), ())),
    ):
        forged = route
        original = getattr(route, field)
        object.__setattr__(forged, field, value)
        with pytest.raises((M1aAuthorizationError, ValueError)):
            coordinator.validate(forged, context.request_ids)
        # 恢复为不可变对象的原值，后续分支继续使用同一登记对象。
        object.__setattr__(forged, field, original)
