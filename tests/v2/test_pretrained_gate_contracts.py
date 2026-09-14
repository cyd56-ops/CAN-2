"""G0 固定 Gate 的 fail-closed 输入与来源绑定测试。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.can.v2.pretrained_gate import (
    P1_POLICY_ID,
    BatchExecutionError,
    CacheHandle,
    CacheRegistry,
    CacheStateError,
    CallLedger,
    FixedRelationVerifier,
    GatedHostAdapter,
    HostSpec,
    InstrumentedModuleCall,
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
from src.can.v2.pretrained_gate.authorization import _evidence_integrity_tag
from src.can.v2.pretrained_gate.cache import bind_cache, validate_cache_binding
from src.can.v2.pretrained_gate.types import (
    DispatchResult,
    EvidenceReason,
    VerificationEvidence,
    _CommittedRoute,
)


def _verifier() -> FixedRelationVerifier:
    """返回便于构造精确结果的二维固定 verifier。"""

    return FixedRelationVerifier(
        np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.5
    )


def _valid_manifest() -> dict[str, object]:
    """返回严格 G0 manifest v1 fixture。"""

    return {
        "schema_version": 1,
        "protocol_id": "g0-v1",
        "execution_config_id": "tiny-v1",
        "model": {"revision": "local"},
        "environment": {"python": "3.11"},
        "inputs": {"sha256": "a" * 64},
        "tolerance": {"dtype": "float32", "atol": 1e-5, "rtol": 1e-4},
    }


@pytest.mark.parametrize(
    ("matrix", "vector", "threshold", "profile", "error"),
    [
        ([], np.zeros(1, np.float32), 1.0, "p", TypeError),
        (np.zeros((1, 1), np.float32), [], 1.0, "p", TypeError),
        (np.zeros((1, 2), np.float32), np.zeros(2, np.float32), 1.0, "p", ValueError),
        (
            np.full((1, 1), np.nan, np.float32),
            np.zeros(1, np.float32),
            1.0,
            "p",
            ValueError,
        ),
        (np.zeros((1, 1), np.float32), np.zeros(1, np.float32), True, "p", TypeError),
        (np.zeros((1, 1), np.float32), np.zeros(1, np.float32), 0.0, "p", ValueError),
        (np.zeros((1, 1), np.float32), np.zeros(1, np.float32), 1.0, "", ValueError),
    ],
)
def test_verifier_constructor_rejects_invalid_configuration(
    matrix, vector, threshold, profile, error
):
    """固定关系参数的类型、shape、有限性与标识必须严格验证。"""

    with pytest.raises(error):
        FixedRelationVerifier(matrix, vector, threshold, profile)


@pytest.mark.parametrize(
    ("credential", "error"),
    [
        ("bad", TypeError),
        (torch.zeros(2), ValueError),
        (torch.zeros((0, 2)), ValueError),
        (torch.zeros((1, 2), dtype=torch.float64), TypeError),
        (torch.empty((1, 2), dtype=torch.float32, device="meta"), ValueError),
    ],
)
def test_verifier_rejects_invalid_credential_structure(credential, error):
    """credential 不得隐式广播、转换 dtype 或跨 device。"""

    with pytest.raises(error):
        _verifier().validate_credential_structure(credential)


def _retag(
    verifier: FixedRelationVerifier, evidence: VerificationEvidence, **changes: object
) -> VerificationEvidence:
    """为受信负向测试重新计算修改后 evidence 的内部完整性标签。"""

    modified = replace(evidence, **changes)
    return replace(
        modified,
        _integrity_tag=_evidence_integrity_tag(
            modified.verified,
            modified.error_norm,
            modified.reason_code,
            verifier._evidence_seal,
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "not_evidence",
        "wrong_rank",
        "length_mismatch",
        "verified_dtype",
        "norm_dtype",
        "reason_dtype",
        "unknown_reason",
        "semantic_mismatch",
        "success_nonfinite",
    ],
)
def test_evidence_validation_rejects_malformed_or_inconsistent_fields(mutation):
    """协调器前的 evidence 校验必须覆盖结构、dtype 和语义一致性。"""

    verifier = _verifier()
    evidence = verifier(torch.zeros((2, 2), dtype=torch.float32))
    if mutation == "not_evidence":
        value = object()
    elif mutation == "wrong_rank":
        value = _retag(verifier, evidence, verified=evidence.verified[:, None])
    elif mutation == "length_mismatch":
        value = _retag(verifier, evidence, error_norm=evidence.error_norm[:1])
    elif mutation == "verified_dtype":
        value = _retag(verifier, evidence, verified=evidence.verified.long())
    elif mutation == "norm_dtype":
        value = _retag(verifier, evidence, error_norm=evidence.error_norm.double())
    elif mutation == "reason_dtype":
        value = _retag(verifier, evidence, reason_code=evidence.reason_code.int())
    elif mutation == "unknown_reason":
        value = _retag(verifier, evidence, reason_code=torch.tensor([99, 99]))
    elif mutation == "semantic_mismatch":
        value = _retag(verifier, evidence, verified=torch.tensor([False, True]))
    else:
        value = _retag(
            verifier,
            evidence,
            error_norm=torch.tensor([float("inf"), 0.0]),
        )
    with pytest.raises((TypeError, ValueError)):
        verifier.validate_evidence(value)


@pytest.mark.parametrize(
    "fault",
    [
        "bad_verifier",
        "bad_policy",
        "bad_config_type",
        "bad_context",
        "context_config",
        "context_size",
    ],
)
def test_coordinator_rejects_invalid_sources_and_context(fault):
    """协调器仅接受固定 policy、绑定 verifier 和完整受信上下文。"""

    verifier = _verifier()
    if fault == "bad_verifier":
        with pytest.raises(TypeError):
            RouteCoordinator(object(), "tiny-v1")
        return
    if fault == "bad_policy":
        with pytest.raises(ValueError):
            RouteCoordinator(verifier, "tiny-v1", "weaker")
        return
    if fault == "bad_config_type":
        with pytest.raises(TypeError):
            RouteCoordinator(verifier, 1)
        return
    coordinator = RouteCoordinator(verifier, "tiny-v1")
    evidence = verifier(torch.zeros((1, 2)))
    if fault == "bad_context":
        context = object()
    elif fault == "context_config":
        context = TrustedRequestContext(("a",), "other")
    else:
        context = TrustedRequestContext(("a", "b"), "tiny-v1")
    with pytest.raises((TypeError, ValueError, PermissionError)):
        coordinator.commit(evidence, context)


@pytest.mark.parametrize(
    "fault",
    [
        "bad_type",
        "context_type",
        "policy",
        "route_config",
        "context_config",
        "empty",
        "unknown",
    ],
)
def test_committed_route_validation_rejects_each_binding_fault(fault):
    """已提交 route 的来源、policy、配置、大小和枚举均需复核。"""

    verifier = _verifier()
    coordinator = RouteCoordinator(verifier, "tiny-v1")
    context = TrustedRequestContext(("a",), "tiny-v1")
    committed = coordinator.commit(verifier(torch.zeros((1, 2))), context)
    if fault == "bad_type":
        route, checked_context = object(), context
    elif fault == "context_type":
        route, checked_context = committed, object()
    elif fault == "policy":
        route, checked_context = replace(committed, policy_id="other"), context
    elif fault == "route_config":
        route = replace(committed, execution_config_id="other")
        checked_context = context
    elif fault == "context_config":
        route = committed
        checked_context = TrustedRequestContext(("a",), "other")
    elif fault == "empty":
        route, checked_context = replace(committed, routes=()), context
    else:
        route = replace(committed, routes=("protected",))
        checked_context = context
    with pytest.raises((TypeError, ValueError, PermissionError)):
        coordinator.validate_committed_route(route, checked_context)


def test_adapter_constructor_and_forward_validate_all_boundaries():
    """adapter 应拒绝错误组件、上下文、batch 及篡改后的 prefix 身份。"""

    verifier = _verifier()
    coordinator = RouteCoordinator(verifier, "tiny-v1")
    host = TinyDecoderHost()
    with pytest.raises(TypeError):
        GatedHostAdapter(object(), verifier, coordinator)
    with pytest.raises(TypeError):
        GatedHostAdapter(host, object(), coordinator)
    with pytest.raises(TypeError):
        GatedHostAdapter(host, verifier, object())
    with pytest.raises(PermissionError):
        GatedHostAdapter(host, verifier, RouteCoordinator(_verifier(), "tiny-v1"))
    adapter = GatedHostAdapter(host, verifier, coordinator)
    ids = torch.ones((1, 2), dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    credential = torch.zeros((1, 2))
    with pytest.raises(TypeError):
        adapter(ids, mask, credential, object())
    with pytest.raises(ValueError):
        adapter(
            ids.repeat(2, 1),
            mask.repeat(2, 1),
            credential,
            TrustedRequestContext(("a", "b"), "tiny-v1"),
        )


@pytest.mark.parametrize(
    "fault", ["coordinator", "hidden", "dtype", "callable", "batch"]
)
def test_dispatcher_rejects_invalid_boundaries(fault):
    """dispatcher 应在 protected 调用前拒绝组件、hidden 和 batch 错误。"""

    verifier = _verifier()
    coordinator = RouteCoordinator(verifier, "tiny-v1")
    if fault == "coordinator":
        with pytest.raises(TypeError):
            ProtectedDispatcher(object())
        return
    context = TrustedRequestContext(("a",), "tiny-v1")
    route = coordinator.commit(verifier(torch.zeros((1, 2))), context)
    hidden = torch.zeros((1, 2))
    callback = lambda value, _: value
    if fault == "hidden":
        hidden = object()
    elif fault == "dtype":
        hidden = torch.zeros((1, 2), dtype=torch.long)
    elif fault == "callable":
        callback = object()
    elif fault == "batch":
        hidden = torch.zeros((2, 2))
    with pytest.raises((TypeError, ValueError)):
        ProtectedDispatcher(coordinator)(hidden, route, context, callback)


def test_batch_execution_error_requires_stable_stage():
    """脱敏执行异常必须具有非空稳定阶段。"""

    with pytest.raises(ValueError):
        BatchExecutionError("", RuntimeError("secret"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"execution_config_id": ""},
        {"model_sha256": ""},
        {"policy_id": ""},
        {"cut_layer": -1},
        {"layer_count": 0},
    ],
)
def test_cache_registry_constructor_rejects_invalid_configuration(kwargs):
    """cache registry 的配置、policy、cut 和层数不得缺失或混淆。"""

    defaults = dict(
        execution_config_id="tiny-v1",
        model_sha256="model-hash",
        policy_id=P1_POLICY_ID,
        cut_layer=1,
        layer_count=2,
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        CacheRegistry(**defaults)


@pytest.mark.parametrize(
    "args",
    [
        ("", 1, 1, 1),
        ("a", -1, 1, 1),
        ("a", 2, 1, 1),
    ],
)
def test_cache_creation_rejects_invalid_identity_and_lengths(args):
    """metadata-only cache 创建同样必须拒绝空身份和非法长度。"""

    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 1)
    with pytest.raises(ValueError):
        registry.create_protected(*args)


@pytest.mark.parametrize(
    "fault",
    ["mask_type", "mask_value", "keys_type", "tensor_type", "shape", "dtype", "device"],
)
def test_real_kv_creation_rejects_invalid_tensor_contract(fault):
    """真实 K/V 创建必须验证 mask、容器、Tensor、shape、dtype 和 device。"""

    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 1, 2)
    keys = tuple(torch.zeros((1, 2, 3)) for _ in range(2))
    values = keys
    mask = (True, True)
    if fault == "mask_type":
        mask = [True, True]
    elif fault == "mask_value":
        mask = (True, 1)
    elif fault == "keys_type":
        keys = list(keys)
    elif fault == "tensor_type":
        keys = (object(), keys[1])
    elif fault == "shape":
        values = (torch.zeros((1, 2, 4)), values[1])
    elif fault == "dtype":
        values = tuple(value.long() for value in values)
    elif fault == "device":
        values = (torch.empty((1, 2, 3), device="meta"), values[1])
    with pytest.raises((TypeError, ValueError, CacheStateError)):
        registry.create_protected_kv("a", keys, values, mask, 2)


@pytest.mark.parametrize(
    "fault",
    ["empty", "size", "duplicate", "handle_type", "handle_id", "unknown"],
)
def test_cache_batch_preflight_rejects_invalid_batch_contract(fault):
    """metadata batch 预检应拒绝空批、错长、重复和伪造句柄。"""

    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 1)
    handle = registry.create_protected("a", 1, 1, 1)
    handles, ids, lengths, positions = (handle,), ("a",), (1,), (1,)
    if fault == "empty":
        handles = ()
    elif fault == "size":
        ids = ("a", "b")
    elif fault == "duplicate":
        handles, ids, lengths, positions = (handle, handle), ("a", "a"), (1, 1), (1, 1)
    elif fault == "handle_type":
        handles = (object(),)
    elif fault == "handle_id":
        handles = (replace(handle, request_id="b"),)
    elif fault == "unknown":
        handles = (CacheHandle("a", object(), handle._registry_seal),)
    with pytest.raises((TypeError, CacheStateError)):
        registry.preflight_batch(handles, ids, lengths, positions)


def test_cache_binding_validator_rejects_tampered_metadata():
    """受信 binding 的 route、活动状态、长度、mask 和位置错误均应拒绝。"""

    binding = bind_cache(
        "a",
        "tiny-v1",
        "model-hash",
        P1_POLICY_ID,
        "protected",
        1,
        2,
        2,
        (True, True),
        2,
        object(),
    )
    defaults = dict(
        request_id="a",
        execution_config_id="tiny-v1",
        model_sha256="model-hash",
        policy_id=P1_POLICY_ID,
        route="protected",
        cut_layer=1,
        expected_processed_valid_tokens=2,
        expected_position=2,
        expected_valid_mask=(True, True),
    )
    mutations = [
        {"route": "deny"},
        {"active": False},
        {"physical_kv_length": 1},
        {"processed_valid_tokens": -1},
        {"valid_mask": (True,)},
        {"valid_mask": (True, 1)},
        {"valid_mask": (True, False)},
    ]
    for mutation in mutations:
        with pytest.raises(CacheStateError):
            validate_cache_binding(replace(binding, **mutation), **defaults)


def test_cache_low_level_errors_reject_empty_reason_and_wrong_binding_type():
    """cache 错误码和 binding 类型必须在入口处明确验证。"""

    with pytest.raises(ValueError):
        CacheStateError("")
    with pytest.raises(TypeError):
        validate_cache_binding(
            object(),
            request_id="a",
            execution_config_id="tiny-v1",
            model_sha256="model-hash",
            policy_id=P1_POLICY_ID,
            route="protected",
            cut_layer=1,
            expected_processed_valid_tokens=1,
            expected_position=1,
        )


def test_real_kv_preflight_rejects_batch_and_identity_faults():
    """真实 K/V 批预检必须拒绝容器、大小、重复身份、mask 和句柄错位。"""

    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 1, 2)
    tensors = tuple(torch.zeros((1, 2, 3)) for _ in range(2))
    first = registry.create_protected_kv("a", tensors, tensors, (True, True), 2)
    second = registry.create_protected_kv("b", tensors, tensors, (True, True), 2)
    cases = [
        ([first], ("a",), ((True, True),), (2,)),
        ((), (), (), ()),
        ((first,), ("a", "b"), ((True, True),), (2,)),
        ((first, second), ("a", "a"), ((True, True), (True, True)), (2, 2)),
        ((first,), ("a",), ([True, True],), (2,)),
        ((replace(first, request_id="b"),), ("a",), ((True, True),), (2,)),
    ]
    for handles, ids, masks, positions in cases:
        with pytest.raises((TypeError, CacheStateError)):
            registry.preflight_kv_batch(handles, ids, masks, positions)


def test_cache_rejects_binding_source_and_cross_layer_device_faults():
    """registry 必须拒绝被替换的 binding seal 与跨层 device K/V。"""

    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 1, 2)
    tensors = tuple(torch.zeros((1, 2, 3)) for _ in range(2))
    handle = registry.create_protected_kv("a", tensors, tensors, (True, True), 2)
    registry._replace_for_test(handle, _registry_seal=object())
    with pytest.raises(CacheStateError, match="binding_source"):
        registry.preflight_kv_batch((handle,), ("a",), ((True, True),), (2,))

    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 1, 2)
    handle = registry.create_protected_kv("a", tensors, tensors, (True, True), 2)
    mixed_devices = (tensors[0], torch.empty((1, 2, 3), device="meta"))
    registry._replace_kv_for_test(handle, mixed_devices, mixed_devices)
    with pytest.raises(CacheStateError, match="cross_layer_device"):
        registry.preflight_kv_batch((handle,), ("a",), ((True, True),), (2,))


def test_cache_clear_invalidates_all_handles():
    """生成生命周期清理后所有旧句柄都必须失效。"""

    registry = CacheRegistry("tiny-v1", "model-hash", P1_POLICY_ID, 1)
    handle = registry.create_protected("a", 1, 1, 1)
    registry.clear()
    with pytest.raises(CacheStateError, match="unknown_handle"):
        registry.preflight_batch((handle,), ("a",), (1,), (1,))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_type": ""},
        {"num_hidden_layers": 1},
        {"cut_layer": 0},
        {"cut_layer": 2},
        {"tied_embeddings": 1},
    ],
)
def test_host_spec_rejects_invalid_structure(kwargs):
    """宿主规格必须明确模型身份、完整 block 边界和 bool capability。"""

    defaults = dict(
        model_type="tiny",
        model_revision="local",
        model_sha256="hash",
        num_hidden_layers=2,
        cut_layer=1,
        tied_embeddings=False,
        supports_kv_cache=True,
        attention_backend="torch",
        dtype="float32",
    )
    defaults.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        HostSpec(**defaults)


@pytest.mark.parametrize(
    "fault", ["hidden", "hidden_dtype", "mask", "position", "shape", "ids", "device"]
)
def test_prefix_state_rejects_invalid_structure(fault):
    """prefix 状态必须在 hidden、mask、position、身份和 device 上完整对齐。"""

    hidden = torch.zeros((1, 2, 3))
    mask = torch.ones((1, 2), dtype=torch.bool)
    positions = torch.arange(2).reshape(1, 2)
    ids = ("a",)
    if fault == "hidden":
        hidden = torch.zeros((1, 2))
    elif fault == "hidden_dtype":
        hidden = hidden.long()
    elif fault == "mask":
        mask = mask.long()
    elif fault == "position":
        positions = positions.int()
    elif fault == "shape":
        mask = torch.ones((1, 1), dtype=torch.bool)
    elif fault == "ids":
        ids = ("a", "b")
    else:
        positions = torch.empty((1, 2), dtype=torch.long, device="meta")
    with pytest.raises((TypeError, ValueError)):
        PrefixState(hidden, mask, positions, ids)


@pytest.mark.parametrize("fault", ["config", "cut_type", "cut_range", "observer"])
def test_tiny_host_constructor_rejects_invalid_values(fault):
    """离线宿主构造参数不得隐式接受 bool、非法 cut 或不可调用观测器。"""

    kwargs = {}
    if fault == "config":
        kwargs["d_model"] = 0
    elif fault == "cut_type":
        kwargs["cut_layer"] = True
    elif fault == "cut_range":
        kwargs["cut_layer"] = 4
    else:
        kwargs["call_observer"] = object()
    with pytest.raises((TypeError, ValueError)):
        TinyDecoderHost(**kwargs)


@pytest.mark.parametrize(
    "fault", ["ids", "dtype", "mask_shape", "mask_dtype", "select_type", "select_dtype"]
)
def test_tiny_host_methods_reject_invalid_inputs(fault):
    """tiny host 的逻辑输入和稀疏索引必须严格验证。"""

    host = TinyDecoderHost()
    ids = torch.ones((1, 2), dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    request_ids = ("a",)
    if fault == "ids":
        ids = ids.float()
    elif fault == "dtype":
        ids = torch.ones(2, dtype=torch.long)
    elif fault == "mask_shape":
        mask = torch.ones((1, 1), dtype=torch.bool)
    elif fault == "mask_dtype":
        mask = mask.long()
    elif fault in {"select_type", "select_dtype"}:
        state = host.prefix(ids, mask, request_ids)
        indices = (
            object() if fault == "select_type" else torch.tensor([0], dtype=torch.int32)
        )
        with pytest.raises(TypeError):
            host.select_prefix_state(state, indices, request_ids)
        return
    with pytest.raises((TypeError, ValueError)):
        host.prefix(ids, mask, request_ids)


@pytest.mark.parametrize(
    "fault",
    [
        "config",
        "rank",
        "dtype",
        "mask_shape",
        "mask_dtype",
        "empty",
        "long",
        "first",
        "padding",
    ],
)
def test_tiny_kv_host_rejects_invalid_inputs(fault):
    """增量 KV fixture 应强制二维 token、Bool mask、非空和右 padding。"""

    if fault == "config":
        with pytest.raises(ValueError):
            TinyKVDecoderHost(num_layers=1)
        return
    host = TinyKVDecoderHost(max_length=4)
    ids = torch.ones((1, 2), dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    if fault == "rank":
        ids = ids[0]
    elif fault == "dtype":
        ids = ids.float()
    elif fault == "mask_shape":
        mask = torch.ones((1, 1), dtype=torch.bool)
    elif fault == "mask_dtype":
        mask = mask.long()
    elif fault == "empty":
        ids, mask = ids[:, :0], mask[:, :0]
    elif fault == "long":
        ids = torch.ones((1, 5), dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
    elif fault == "first":
        mask[0, 0] = False
    elif fault == "padding":
        ids = torch.ones((1, 3), dtype=torch.long)
        mask = torch.tensor([[True, False, True]])
    with pytest.raises((TypeError, ValueError)):
        host.forward_full(ids, mask)


def test_call_ledger_and_instrumented_call_validate_inputs():
    """调用台账必须拒绝未知身份、重复行和非法组件字段。"""

    context = TrustedRequestContext(("a", "b"), "tiny-v1")
    with pytest.raises(TypeError):
        CallLedger(object())
    ledger = CallLedger(context)
    for args in [
        ("", "stage", ("a",)),
        ("component", "", ("a",)),
        ("c", "s", ()),
        ("c", "s", ("a", "a")),
    ]:
        with pytest.raises(ValueError):
            ledger.record(*args)
    with pytest.raises(PermissionError):
        ledger.record("c", "s", ("outside",))
    ledger.record("c", "s", ("a",))
    assert ledger.count("c") == ledger.count("c", "a") == 1
    with pytest.raises(ValueError):
        ledger.count("")
    with pytest.raises(ValueError):
        ledger.count("c", "")
    with pytest.raises(TypeError):
        InstrumentedModuleCall(object(), "c")
    with pytest.raises(ValueError):
        InstrumentedModuleCall(ledger, "")
    wrapper = InstrumentedModuleCall(ledger, "wrapped")
    wrapper.record("protected", ("b",))
    assert ledger.count("wrapped", "b") == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": True},
        {"protocol_id": ""},
        {"model": []},
        {"model": {"revision": 1}},
        {"model": {"revision": ""}},
        {"tolerance": {"dtype": "float32", "atol": True, "rtol": 1e-4}},
        {"tolerance": {"dtype": "float32", "atol": float("inf"), "rtol": 1e-4}},
    ],
)
def test_manifest_rejects_remaining_type_and_value_faults(tmp_path, mutation):
    """manifest 的版本、标识、嵌套对象和数值类型必须严格验证。"""

    payload = _valid_manifest()
    payload.update(mutation)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_manifest(path)


def test_manifest_file_helpers_reject_missing_path_and_bad_digest(tmp_path: Path):
    """文件摘要入口必须拒绝不存在路径和非十六进制信任根。"""

    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        file_sha256(missing)
    with pytest.raises(FileNotFoundError):
        load_manifest(missing)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_manifest_sha256(path, "z" * 64)


@pytest.mark.parametrize(
    "args",
    [
        ((), "tiny-v1"),
        (("",), "tiny-v1"),
        (("a", "a"), "tiny-v1"),
        (("a",), ""),
    ],
)
def test_trusted_request_context_rejects_invalid_identity(args):
    """请求上下文必须具有非空、有序、唯一的服务端身份。"""

    with pytest.raises((TypeError, ValueError)):
        TrustedRequestContext(*args)


@pytest.mark.parametrize("indices", [torch.tensor([0.0]), torch.tensor([[0]])])
def test_dispatch_result_rejects_invalid_index_tensors(indices):
    """调度结果索引必须是一维 LongTensor。"""

    with pytest.raises((TypeError, ValueError)):
        DispatchResult(None, indices, torch.tensor([], dtype=torch.long), ("a",))
