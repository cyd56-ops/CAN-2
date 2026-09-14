"""G0 固定预训练宿主 Gate 的 CPU 契约测试。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.pretrained_gate import (
    P1_POLICY_ID,
    BatchExecutionError,
    CacheRegistry,
    CacheStateError,
    CallLedger,
    FixedRelationVerifier,
    GatedHostAdapter,
    HostSpec,
    PrefixState,
    ProtectedDispatcher,
    RouteCoordinator,
    RouteKind,
    TinyDecoderHost,
    TinyKVDecoderHost,
    TrustedRequestContext,
    file_sha256,
    load_manifest,
    verify_manifest_sha256,
)
from src.can.v2.pretrained_gate.types import EvidenceReason, _CommittedRoute


@pytest.fixture
def auth_fixture():
    """构造确定性的 FP32 verifier、secret 和 mixed credential。"""

    np.random.seed(20260910)
    params = LWEParams(n=8, m=16)
    matrix, secret, vector = generate_keypair(params, np.random.default_rng(20260910))
    verifier = FixedRelationVerifier(matrix, vector, params.error_threshold)
    invalid = np.full(params.n, 10.0, dtype=np.float32)
    credentials = torch.from_numpy(np.stack([secret, invalid, secret]))
    context = TrustedRequestContext(("request-a", "request-b", "request-c"), "tiny-v1")
    coordinator = RouteCoordinator(verifier, "tiny-v1")
    return verifier, coordinator, credentials, context


def test_fixed_verifier_is_hard_and_reference_aligned(auth_fixture):
    """verifier 在 train/eval 下都应产生同一 hard evidence。"""

    verifier, coordinator, credentials, context = auth_fixture
    residual = credentials.numpy() @ verifier.A.numpy().T - verifier.b.numpy()
    reference = np.linalg.norm(residual, axis=1) < verifier.error_threshold
    decisions = []
    for mode in (True, False):
        verifier.train(mode)
        evidence = verifier(credentials)
        assert evidence.verified.tolist() == reference.tolist()
        decisions.append(coordinator.commit(evidence, context).routes)
    assert decisions == [(RouteKind.PROTECTED, RouteKind.DENY, RouteKind.PROTECTED)] * 2


def test_fixed_verifier_uses_strict_threshold_boundary():
    """误差范数等于阈值时必须拒绝，不能使用小于等于判据。"""

    verifier = FixedRelationVerifier(
        np.ones((1, 1), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        1.0,
    )
    evidence = verifier(torch.tensor([[0.0], [1.0]], dtype=torch.float32))
    assert evidence.verified.tolist() == [True, False]
    assert evidence.reason_code.tolist() == [
        int(EvidenceReason.SUCCESS),
        int(EvidenceReason.RELATION_FAILED),
    ]


def test_fixed_verifier_rejects_finite_numerical_overflow():
    """有限输入导致矩阵运算溢出时应逐行数值拒绝。"""

    maximum = np.finfo(np.float32).max
    verifier = FixedRelationVerifier(
        np.asarray([[maximum]], dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        1.0,
    )
    evidence = verifier(torch.tensor([[maximum]], dtype=torch.float32))
    assert evidence.verified.tolist() == [False]
    assert evidence.reason_code.tolist() == [int(EvidenceReason.NUMERICAL_FAILURE)]
    assert torch.isinf(evidence.error_norm).all()


def test_verifier_rejects_structure_but_rejects_nonfinite_rows(auth_fixture):
    """结构错误整批失败，非有限行生成逐行拒绝 evidence。"""

    verifier, _, _, _ = auth_fixture
    with pytest.raises(ValueError):
        verifier(torch.empty((0, verifier.n), dtype=torch.float32))
    with pytest.raises(TypeError):
        verifier(torch.ones((1, verifier.n), dtype=torch.float64))
    credential = torch.zeros((2, verifier.n), dtype=torch.float32)
    credential[1, 0] = float("nan")
    evidence = verifier(credential)
    assert evidence.reason_code.tolist()[-1] == 2
    assert not bool(evidence.verified[-1])


def test_evidence_source_and_content_are_bound(auth_fixture):
    """跨 verifier 或原地改写 evidence 必须被协调器拒绝。"""

    verifier, coordinator, credentials, context = auth_fixture
    evidence = verifier(credentials)
    other = FixedRelationVerifier(
        verifier.A.detach().numpy(),
        verifier.b.detach().numpy(),
        verifier.error_threshold,
    )
    with pytest.raises(PermissionError):
        RouteCoordinator(other, "tiny-v1").commit(evidence, context)
    evidence.verified[0] = False
    with pytest.raises(PermissionError):
        coordinator.commit(evidence, context)


def test_route_requires_full_ordered_context(auth_fixture):
    """换序、旧请求和配置变化都不能复用已提交 route。"""

    verifier, coordinator, credentials, context = auth_fixture
    committed = coordinator.commit(verifier(credentials), context)
    with pytest.raises(PermissionError):
        coordinator.validate_committed_route(
            committed,
            TrustedRequestContext(("request-b", "request-a", "request-c"), "tiny-v1"),
        )
    with pytest.raises(PermissionError):
        coordinator.validate_committed_route(
            committed, TrustedRequestContext(context.request_ids, "other-config")
        )


def test_forged_route_and_public_upgrade_are_rejected(auth_fixture):
    """外部伪造 seal 或尝试把 P1 路由升级为 PUBLIC 时必须拒绝。"""

    verifier, coordinator, credentials, context = auth_fixture
    committed = coordinator.commit(verifier(credentials), context)
    forged = replace(committed, _coordinator_seal=object())
    with pytest.raises(PermissionError, match="协调器"):
        coordinator.validate_committed_route(forged, context)
    public = _CommittedRoute(
        routes=(RouteKind.PUBLIC,) * 3,
        policy_id=P1_POLICY_ID,
        request_ids=context.request_ids,
        execution_config_id=context.execution_config_id,
        _coordinator_seal=committed._coordinator_seal,
    )
    with pytest.raises(PermissionError, match="PUBLIC"):
        coordinator.validate_committed_route(public, context)


def test_dispatcher_mixed_rows_and_zero_call_for_deny(auth_fixture):
    """mixed batch 仅调用 protected 行，且索引覆盖原顺序。"""

    verifier, coordinator, credentials, context = auth_fixture
    committed = coordinator.commit(verifier(credentials), context)
    hidden = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    calls = []

    def protected_fn(sub_batch, request_ids):
        """记录实际 protected 子批并返回其恒等值。"""

        calls.append((sub_batch.clone(), request_ids))
        return sub_batch + 1.0

    result = ProtectedDispatcher(coordinator)(hidden, committed, context, protected_fn)
    assert result.protected_indices.tolist() == [0, 2]
    assert result.denied_indices.tolist() == [1]
    assert calls[0][1] == ("request-a", "request-c")
    assert torch.equal(calls[0][0], hidden[[0, 2]])


@pytest.mark.parametrize("valid_rows", [(), (0,), (0, 1, 2)])
def test_dispatcher_covers_all_valid_all_deny_and_tail_batches(valid_rows):
    """全合法、全拒绝和单行尾批均应保持互斥且完整的索引覆盖。"""

    matrix = np.eye(1, dtype=np.float32)
    verifier = FixedRelationVerifier(matrix, np.zeros(1, dtype=np.float32), 0.5)
    batch_size = 1 if valid_rows == (0,) else 3
    values = torch.ones((batch_size, 1), dtype=torch.float32)
    if valid_rows:
        values[list(valid_rows)] = 0.0
    context = TrustedRequestContext(
        tuple(f"request-{index}" for index in range(batch_size)), "tiny-v1"
    )
    coordinator = RouteCoordinator(verifier, "tiny-v1")
    committed = coordinator.commit(verifier(values), context)
    hidden = torch.arange(batch_size, dtype=torch.float32)[:, None]
    calls = []

    def protected_fn(sub_batch, request_ids):
        """记录该 fixture 的实际合法子批。"""

        calls.append(request_ids)
        return sub_batch

    result = ProtectedDispatcher(coordinator)(hidden, committed, context, protected_fn)
    protected = set(result.protected_indices.tolist())
    denied = set(result.denied_indices.tolist())
    assert protected.isdisjoint(denied)
    assert protected | denied == set(range(batch_size))
    assert bool(calls) is bool(valid_rows)


def test_dispatcher_preserves_protected_gradient_and_denied_zero(auth_fixture):
    """合法 hidden 梯度应等于恒等 reference，拒绝行梯度为零。"""

    verifier, coordinator, credentials, context = auth_fixture
    committed = coordinator.commit(verifier(credentials), context)
    hidden = torch.randn((3, 2, 4), generator=torch.Generator().manual_seed(7))
    hidden.requires_grad_(True)
    result = ProtectedDispatcher(coordinator)(
        hidden, committed, context, lambda value, _: value
    )
    assert result.protected_output is not None
    result.protected_output.sum().backward()
    assert torch.equal(hidden.grad[0], torch.ones_like(hidden.grad[0]))
    assert torch.count_nonzero(hidden.grad[1]).item() == 0
    assert torch.equal(hidden.grad[2], torch.ones_like(hidden.grad[2]))


def test_gated_tiny_host_runs_prefix_for_all_and_suffix_for_valid(auth_fixture):
    """图中 Gate 应位于 prefix 后，并把真实 suffix 限制到合法请求。"""

    verifier, coordinator, credentials, context = auth_fixture
    ledger = CallLedger(context)
    torch.manual_seed(20260910)
    host = TinyDecoderHost(call_observer=ledger.record)
    adapter = GatedHostAdapter(host, verifier, coordinator)
    input_ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    result = adapter(input_ids, mask, credentials, context)
    assert result.protected_indices.tolist() == [0, 2]
    assert ledger.count("embedding", "request-b") == 1
    assert ledger.count("block.0", "request-b") == 1
    assert ledger.count("block.2", "request-b") == 0
    assert ledger.count("norm", "request-b") == 0
    assert ledger.count("lm_head", "request-b") == 0
    assert ledger.records[-1].request_ids == ("request-a", "request-c")


def test_route_is_independent_of_business_tokens_and_parent_mode(auth_fixture):
    """固定 credential 的 hard route 不得随业务 token 或父 adapter 模式变化。"""

    verifier, coordinator, credentials, context = auth_fixture
    torch.manual_seed(11)
    host = TinyDecoderHost()
    adapter = GatedHostAdapter(host, verifier, coordinator)
    mask = torch.ones((3, 3), dtype=torch.bool)
    indices = []
    for mode, tokens in (
        (True, torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        (False, torch.tensor([[9, 8, 7], [6, 5, 4], [3, 2, 1]])),
    ):
        adapter.train(mode)
        output = adapter(tokens, mask, credentials, context)
        indices.append(
            (output.protected_indices.tolist(), output.denied_indices.tolist())
        )
    assert indices == [([0, 2], [1]), ([0, 2], [1])]


def test_gated_tiny_host_rejects_structure_before_prefix(auth_fixture):
    """credential 结构错误必须在任何宿主 prefix 调用前整批失败。"""

    verifier, coordinator, _, context = auth_fixture
    ledger = CallLedger(context)
    host = TinyDecoderHost(call_observer=ledger.record)
    adapter = GatedHostAdapter(host, verifier, coordinator)
    input_ids = torch.ones((3, 2), dtype=torch.long)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    with pytest.raises(TypeError):
        adapter(
            input_ids,
            mask,
            torch.ones((3, verifier.n), dtype=torch.float64),
            context,
        )
    assert ledger.records == ()


def test_tiny_host_rejects_same_size_wrong_row_identity():
    """相同 batch 大小下的行身份错配不能靠 shape 检查蒙混通过。"""

    host = TinyDecoderHost()
    input_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    state = host.prefix(input_ids, mask, ("a", "b"))
    with pytest.raises(PermissionError, match="request IDs"):
        host.select_prefix_state(
            state, torch.tensor([0, 1], dtype=torch.long), ("b", "a")
        )


def test_dispatcher_execution_error_is_batch_atomic(auth_fixture):
    """suffix 意外异常必须转为脱敏整批失败。"""

    verifier, coordinator, credentials, context = auth_fixture
    committed = coordinator.commit(verifier(credentials), context)
    hidden = torch.zeros((3, 2, 4))

    def explode(_, __):
        """注入受控执行错误。"""

        raise RuntimeError("secret diagnostic")

    with pytest.raises(BatchExecutionError, match="protected_execution") as error:
        ProtectedDispatcher(coordinator)(hidden, committed, context, explode)
    assert error.value.cause_type == "RuntimeError"
    assert "secret diagnostic" not in str(error.value)


def test_cache_binding_must_fail_before_incremental_compute():
    """cache 绑定必须核对请求、route、配置和位置。"""

    registry = CacheRegistry("tiny-v1", "model-hash", "p1", 2)
    handle = registry.create_protected("request-a", 4, 4, 4)
    result = registry.preflight_batch((handle,), ("request-a",), (4,), (4,))
    assert result[0].request_id == "request-a"
    with pytest.raises(CacheStateError):
        registry.preflight_batch((handle,), ("request-b",), (4,), (4,))


def test_cache_registry_rejects_cross_registry_and_finished_handle():
    """cache 句柄不能跨配置 registry 使用，也不能在结束后继续。"""

    first = CacheRegistry("tiny-v1", "model-hash", "p1", 2)
    second = CacheRegistry("tiny-v1", "model-hash", "p1", 2)
    handle = first.create_protected("request-a", 2, 3, 3)
    with pytest.raises(CacheStateError, match="registry"):
        second.preflight_batch((handle,), ("request-a",), (2,), (3,))
    first.finish(handle)
    with pytest.raises(CacheStateError, match="inactive_request"):
        first.preflight_batch((handle,), ("request-a",), (2,), (3,))


def test_cache_batch_preflight_is_all_or_nothing():
    """任一行元数据错误时，完整 batch 预检必须在计算前失败。"""

    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 2)
    first = registry.create_protected("a", 2, 2, 2)
    second = registry.create_protected("b", 2, 2, 2)
    registry._replace_for_test(second, position=3)
    with pytest.raises(CacheStateError, match="position"):
        registry.preflight_batch((first, second), ("a", "b"), (2, 2), (2, 2))


def test_tiny_kv_incremental_matches_full_with_right_padding():
    """增量真实 K/V logits 应在有效位置与完整 causal reference 一致。"""

    torch.manual_seed(20260910)
    host = TinyKVDecoderHost(vocab_size=32, d_model=8, num_layers=3)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False]], dtype=torch.bool
    )
    full = host.forward_full(input_ids, mask)
    incremental, keys, values = host.forward_incremental(input_ids, mask)
    torch.testing.assert_close(full[mask], incremental[mask], atol=1e-6, rtol=1e-5)
    assert len(keys) == len(values) == 3
    assert all(value.shape == (2, 4, 8) for value in keys + values)


def test_real_kv_registry_reassembles_rows_and_preserves_order():
    """registry 应在完整预检后按请求顺序重组每层真实 K/V。"""

    torch.manual_seed(7)
    host = TinyKVDecoderHost(num_layers=2)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    _, keys, values = host.forward_incremental(input_ids, mask)
    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 1, 2)
    masks = ((True, True, True), (True, True, False))
    handles = tuple(
        registry.create_protected_kv(
            f"request-{row}",
            tuple(value[row : row + 1] for value in keys),
            tuple(value[row : row + 1] for value in values),
            masks[row],
            position=sum(masks[row]),
        )
        for row in range(2)
    )
    combined_keys, combined_values = registry.preflight_kv_batch(
        handles,
        ("request-0", "request-1"),
        masks,
        (3, 2),
    )
    for expected, actual in zip(keys, combined_keys):
        assert torch.equal(expected, actual)
    for expected, actual in zip(values, combined_values):
        assert torch.equal(expected, actual)


def test_real_kv_registry_binds_exact_mask_not_only_token_count():
    """有效 token 数相同但位置不同的 mask 仍必须被拒绝。"""

    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 1, 2)
    tensors = tuple(torch.zeros((1, 4, 3)) for _ in range(2))
    handle = registry.create_protected_kv(
        "request-a", tensors, tensors, (True, True, True, False), 3
    )
    with pytest.raises(CacheStateError, match="valid_mask"):
        registry.preflight_kv_batch(
            (handle,),
            ("request-a",),
            ((True, True, False, True),),
            (3,),
        )


@pytest.mark.parametrize("fault", ["layer_count", "kv_batch", "physical_length"])
def test_real_kv_registry_rejects_tensor_structure_faults(fault):
    """层数、每请求 batch 维或物理长度篡改必须在读取前失败。"""

    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 1, 2)
    tensors = tuple(torch.zeros((1, 3, 4)) for _ in range(2))
    handle = registry.create_protected_kv(
        "request-a", tensors, tensors, (True, True, True), 3
    )
    if fault == "layer_count":
        bad = tensors[:1]
    elif fault == "kv_batch":
        bad = tuple(torch.zeros((2, 3, 4)) for _ in range(2))
    else:
        bad = tuple(torch.zeros((1, 2, 4)) for _ in range(2))
    registry._replace_kv_for_test(handle, bad, bad)
    with pytest.raises(CacheStateError, match=fault):
        registry.preflight_kv_batch(
            (handle,), ("request-a",), ((True, True, True),), (3,)
        )


def test_host_spec_and_prefix_state_are_strict():
    """tiny host 的结构状态必须包含 mask、position 和完整 batch 身份。"""

    spec = HostSpec("tiny", "local", "hash", 4, 2, True, True, "torch", "float32")
    assert spec.cut_layer == 2
    hidden = torch.zeros((2, 3, 4))
    mask = torch.ones((2, 3), dtype=torch.bool)
    positions = torch.arange(3).repeat(2, 1)
    state = PrefixState(hidden, mask, positions, ("a", "b"), host_state={"cache": None})
    assert state.host_state == {"cache": None}
    with pytest.raises(ValueError):
        PrefixState(hidden, mask, positions, ("a", "a"))


def test_manifest_rejects_unknown_or_duplicate_fields(tmp_path: Path):
    """manifest schema、重复字段和自身摘要必须严格处理。"""

    path = tmp_path / "manifest.json"
    payload = {
        "schema_version": 1,
        "protocol_id": "g0-v1",
        "execution_config_id": "tiny-v1",
        "model": {"revision": "local"},
        "environment": {"python": "3.11"},
        "inputs": {"sha256": "a" * 64},
        "tolerance": {"dtype": "float32", "atol": 1e-5, "rtol": 1e-4},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_manifest(path)["protocol_id"] == "g0-v1"
    digest = file_sha256(path)
    verify_manifest_sha256(path, digest)
    with pytest.raises(ValueError):
        verify_manifest_sha256(path, "0" * 64)
    unknown = dict(payload)
    unknown["extra"] = True
    path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize(
    ("section", "mutation"),
    [
        ("model", {"revision": "local", "unknown": True}),
        ("environment", {"python": "3.11", "unknown": True}),
        ("inputs", {"sha256": "not-a-digest"}),
        ("tolerance", {"dtype": "float16", "atol": 1e-5, "rtol": 1e-4}),
        ("tolerance", {"dtype": "float32", "atol": -1.0, "rtol": 1e-4}),
    ],
)
def test_manifest_rejects_invalid_nested_schema(tmp_path, section, mutation):
    """嵌套未知字段、非法摘要、dtype 和容差必须 fail closed。"""

    payload = {
        "schema_version": 1,
        "protocol_id": "g0-v1",
        "execution_config_id": "tiny-v1",
        "model": {"revision": "local"},
        "environment": {"python": "3.11"},
        "inputs": {"sha256": "a" * 64},
        "tolerance": {"dtype": "float32", "atol": 1e-5, "rtol": 1e-4},
    }
    payload[section] = mutation
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_manifest(path)


def test_manifest_rejects_nested_duplicate_and_nonfinite_value(tmp_path):
    """嵌套重复键与 Python JSON 扩展的 NaN 均不得被接受。"""

    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema_version":1,"protocol_id":"g0-v1",'
        '"execution_config_id":"tiny-v1","model":{"revision":"a",'
        '"revision":"b"},"environment":{"python":"3.11"},'
        f'"inputs":{{"sha256":"{"a" * 64}"}},'
        '"tolerance":{"dtype":"float32","atol":NaN,"rtol":0.0001}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_manifest(path)


def test_package_does_not_export_private_committed_route():
    """进程内授权类型不得出现在包级公开 API 清单。"""

    import src.can.v2.pretrained_gate as package

    assert "_CommittedRoute" not in package.__all__
