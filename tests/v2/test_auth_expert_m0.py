"""M0 contract 的最小正负测试。"""

from __future__ import annotations

import hashlib
import json
import threading

import numpy as np
import pytest
import torch

from src.can.v2.auth_expert import (
    AuthExpert,
    ExpertKind,
    ExpertSelection,
    M0Dispatcher,
    M0ExecutionConfig,
    M0ExpertSpec,
    M0ScopeSpec,
    M0StateError,
    M0TinyHostBridge,
    ScopeCoordinator,
    ScopeRegistry,
)
from src.can.v2.auth_expert.manifest import file_sha256, load_m0_manifest
from src.can.v2.pretrained_gate import (
    BatchExecutionError,
    CacheStateError,
    FixedRelationVerifier,
    TinyDecoderHost,
    TinyKVDecoderHost,
)
from src.can.v2.pretrained_gate.host import PrefixState


def _valid_manifest_payload():
    """返回可通过 M0 manifest 解析的最小固定 payload。"""

    return {
        "schema_version": 1,
        "protocol_id": "m0-contract-v1",
        "execution_config_id": "m0-test-v1",
        "policy_id": "p1-protected-or-deny-v1",
        "verifier": {
            "backend": "a0-fixed-relation",
            "profile_id": "toy-real-fp32-v1",
            "n": 2,
            "m": 2,
            "threshold": 0.1,
            "dtype": "float32",
            "relation_sha256": "b" * 64,
        },
        "host": {
            "model_type": "tiny_decoder",
            "model_revision": "test",
            "model_sha256": "c" * 64,
            "num_hidden_layers": 4,
            "cut_layer": 2,
            "dtype": "float32",
            "attention_backend": "eager",
            "cache_mode": "none",
        },
        "experts": [
            {"expert_id": "E0", "kind": "public", "enabled": False},
            {"expert_id": "E1", "kind": "protected", "enabled": True},
        ],
        "scopes": [{"scope_id": "protected.default", "expert_ids": ["E1"]}],
        "limits": {"max_batch_size": 8, "max_sequence_length": 8},
        "fixture": {
            "seed": 1,
            "inputs_sha256": "d" * 64,
            "credential_fixture_sha256": "e" * 64,
        },
        "provenance": {
            "git_commit": "a" * 40,
            "dirty": False,
            "source_tree_sha256": "f" * 64,
            "python_version": "3.11",
            "torch_version": "2",
            "numpy_version": "2",
            "device": "cpu",
        },
    }


def _components(cache_mode="none", model_type="tiny_decoder"):
    """构造确定性的 M0 verifier、配置、宿主和认证链。"""

    matrix = np.eye(2, dtype=np.float32)
    secret = np.zeros(2, dtype=np.float32)
    verifier = FixedRelationVerifier(matrix, secret, 0.1)
    config = M0ExecutionConfig(
        execution_config_id="m0-test-v1",
        protocol_id="m0-contract-v1",
        policy_id="p1-protected-or-deny-v1",
        verifier_profile="toy-real-fp32-v1",
        model_type=model_type,
        model_revision="test-host-v1",
        model_sha256="m" * 64,
        relation_sha256="r" * 64,
        num_hidden_layers=4,
        cut_layer=2,
        cache_mode=cache_mode,
        max_batch_size=8,
        max_sequence_length=8,
        experts=(
            M0ExpertSpec("E0", ExpertKind.PUBLIC, False),
            M0ExpertSpec("E1", ExpertKind.PROTECTED, True),
        ),
        scopes=(M0ScopeSpec("protected.default", ("E1",)),),
    )
    auth = AuthExpert(verifier, config)
    coordinator = ScopeCoordinator(auth, config)
    registry = ScopeRegistry(coordinator, config)
    if model_type == "tiny_kv_decoder":
        host = TinyKVDecoderHost(vocab_size=8, d_model=4, num_layers=4, max_length=8)
        bridge = M0TinyHostBridge(host, cut_layer=2)
    else:
        host = TinyDecoderHost(vocab_size=8, d_model=4, num_layers=4, cut_layer=2)
        bridge = M0TinyHostBridge(host)
    return config, auth, coordinator, registry, bridge


def test_m0_session_prefill_mixed_and_denied_rows_have_zero_protected_call():
    """混合 batch 保留原始索引，DENY 行不进入 protected suffix。"""

    config, auth, coordinator, registry, bridge = _components()
    from src.can.v2.auth_expert.runtime import M0Session

    session = M0Session(
        bridge,
        auth,
        coordinator,
        registry,
        config,
        id_factory=iter(["s", "a"]).__next__,
    )
    ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    credential = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
    result = session.prefill(ids, mask, credential)
    assert result.protected_indices.tolist() == [0]
    assert result.denied_indices.tolist() == [1]
    assert result.protected_output is not None
    assert session.state.value == "active"
    second = session.decode_step(torch.tensor([[5], [6]], dtype=torch.long))
    assert second.protected_indices.tolist() == [0]
    assert second.denied_indices.tolist() == [1]


def test_scope_restrict_cannot_expand_and_mask_is_a_copy():
    """scope 收窄只能减少 E1，外部修改 mask 不影响可信 view。"""

    _, auth, coordinator, registry, _ = _components()
    context = __import__(
        "src.can.v2.auth_expert.types", fromlist=["RequestContext"]
    ).RequestContext(("a",), "m0-test-v1", "s", "a")
    evidence = auth.verify(torch.zeros((1, 2)), context)
    route = coordinator.commit(evidence, context)
    view = registry.resolve(route, context)
    mask = view.mask()
    mask[0, 0] = True
    assert not view.mask()[0, 0]
    empty = registry.restrict(view, ((),))
    assert empty.expert_ids == ((),)
    with pytest.raises(PermissionError):
        registry.restrict(empty, (("E1",),))


def test_dispatcher_rejects_wrong_selection_before_callback():
    """越界 selection 在真实 protected callback 前失败。"""

    _, auth, coordinator, registry, bridge = _components()
    context = __import__(
        "src.can.v2.auth_expert.types", fromlist=["RequestContext"]
    ).RequestContext(("a",), "m0-test-v1", "s", "a")
    evidence = auth.verify(torch.zeros((1, 2)), context)
    route = coordinator.commit(evidence, context)
    view = registry.resolve(route, context)
    state = bridge.prefix_full(
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        ("a",),
    )
    called = []
    dispatcher = M0Dispatcher(
        coordinator,
        registry,
        lambda value, ids: called.append(ids) or bridge.suffix_full(value, ids),
    )
    with pytest.raises(PermissionError):
        dispatcher.execute(state, route, view, ExpertSelection((None,)), context)
    assert called == []


def test_session_close_and_failed_prefill_clear_authorization():
    """close 和执行失败后均不可继续使用旧 session。"""

    config, auth, coordinator, registry, bridge = _components()
    from src.can.v2.auth_expert.runtime import M0Session

    session = M0Session(
        bridge,
        auth,
        coordinator,
        registry,
        config,
        id_factory=iter(["s", "a"]).__next__,
    )
    session.close()
    with pytest.raises(M0StateError):
        session.prefill(
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.bool),
            torch.zeros((1, 2)),
        )


def test_prefill_rejects_credential_structure_before_prefix_call():
    """credential 请求级结构错误必须在 host prefix 前 fail closed。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components()
    calls = []
    original = bridge.prefix_full

    def counted_prefix(*args, **kwargs):
        """记录 prefix 是否在结构预检前被调用。"""

        calls.append(True)
        return original(*args, **kwargs)

    bridge.prefix_full = counted_prefix
    session = M0Session(bridge, auth, coordinator, registry, config)
    with pytest.raises(ValueError):
        session.prefill(
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.bool),
            torch.zeros((1, 3)),
        )
    assert calls == []
    assert session.state.value == "failed"


def test_session_close_revokes_evidence_route_and_scope_view():
    """session 关闭后，外部持有的旧授权对象也必须失效。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components()
    session = M0Session(bridge, auth, coordinator, registry, config)
    session.prefill(
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        torch.zeros((1, 2)),
    )
    context = session._context
    evidence = session._evidence
    route = session._route
    view = session._view
    assert context is not None and evidence is not None
    assert route is not None and view is not None
    session.close()
    with pytest.raises(PermissionError):
        auth.validate_evidence(evidence, context)
    with pytest.raises(PermissionError):
        coordinator.validate(route, context)
    assert not registry.is_registered(view)


def test_m0_revocation_and_bridge_error_boundaries():
    """覆盖撤销 API、host bridge 错误分支和 KV 能力声明边界。"""

    from src.can.v2.auth_expert.runtime import M0Session
    from src.can.v2.auth_expert.types import BoundEvidence, RequestContext

    config, auth, coordinator, registry, bridge = _components()
    context = RequestContext(("a",), config.execution_config_id, "s", "a")
    evidence = auth.verify(torch.zeros((1, 2)), context)
    auth.revoke(evidence)
    with pytest.raises(PermissionError):
        auth.validate_evidence(evidence, context)
    with pytest.raises(TypeError):
        auth.revoke(object())
    foreign = BoundEvidence(evidence.evidence, context, object(), object())
    with pytest.raises(PermissionError):
        auth.revoke(foreign)

    route = coordinator.commit(auth.verify(torch.zeros((1, 2)), context), context)
    view = registry.resolve(route, context)
    registry.revoke(route)
    with pytest.raises((TypeError, PermissionError)):
        coordinator.revoke(object())
    assert not registry.is_registered(view)
    assert not registry.is_registered(object())

    state = bridge.prefix_full(
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        ("a",),
    )
    with pytest.raises(PermissionError):
        bridge.suffix_full(state, ("b",))
    with pytest.raises(ValueError):
        bridge.suffix_step(state, ("b",), (), ())
    assert bridge.suffix_with_cache(state, ("a",))[0].shape[0] == 1

    kv_config, kv_auth, kv_coordinator, kv_registry, _ = _components(
        cache_mode="kv", model_type="tiny_decoder"
    )
    session = M0Session(
        bridge,
        kv_auth,
        kv_coordinator,
        kv_registry,
        kv_config,
    )
    with pytest.raises(M0StateError):
        session.prefill(
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.bool),
            torch.zeros((1, 2)),
        )


def test_m0_manifest_uses_same_bytes_for_digest_and_rejects_unknown_nested_fields(
    tmp_path,
):
    """manifest 必须先校验同一份 bytes 摘要，并拒绝嵌套未知字段。"""

    payload = {
        "schema_version": 1,
        "protocol_id": "m0-contract-v1",
        "execution_config_id": "m0-test-v1",
        "policy_id": "p1-protected-or-deny-v1",
        "verifier": {
            "backend": "a0-fixed-relation",
            "profile_id": "toy-real-fp32-v1",
            "n": 2,
            "m": 2,
            "threshold": 0.1,
            "dtype": "float32",
            "relation_sha256": "b" * 64,
        },
        "host": {
            "model_type": "tiny_decoder",
            "model_revision": "test",
            "model_sha256": "c" * 64,
            "num_hidden_layers": 4,
            "cut_layer": 2,
            "dtype": "float32",
            "attention_backend": "eager",
            "cache_mode": "none",
        },
        "experts": [
            {"expert_id": "E0", "kind": "public", "enabled": False},
            {"expert_id": "E1", "kind": "protected", "enabled": True},
        ],
        "scopes": [{"scope_id": "protected.default", "expert_ids": ["E1"]}],
        "limits": {"max_batch_size": 8, "max_sequence_length": 8},
        "fixture": {
            "seed": 1,
            "inputs_sha256": "d" * 64,
            "credential_fixture_sha256": "e" * 64,
        },
        "provenance": {
            "git_commit": "a" * 40,
            "dirty": False,
            "source_tree_sha256": "f" * 64,
            "python_version": "3.11",
            "torch_version": "2",
            "numpy_version": "2",
            "device": "cpu",
        },
    }
    path = tmp_path / "m0.json"
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    config = load_m0_manifest(path, hashlib.sha256(raw).hexdigest())
    assert config.execution_config_id == "m0-test-v1"
    payload["host"]["unexpected"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_m0_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_tiny_kv_bridge_reuses_existing_block_kv_operators():
    """KV bridge 的 prefix/suffix 分段结果应与原始完整 causal 结果一致。"""

    host = TinyKVDecoderHost(vocab_size=8, d_model=4, num_layers=4, max_length=8)
    from src.can.v2.auth_expert.host_bridge import M0TinyHostBridge

    bridge = M0TinyHostBridge(host, cut_layer=2)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    state = bridge.prefix_full(ids, mask, ("a",))
    split = bridge.suffix_full(state, ("a",))
    full = host.forward_full(ids, mask)
    assert torch.allclose(split, full, atol=1e-5, rtol=1e-4)


def test_m0_session_kv_prefill_and_decode_replace_handles_atomically():
    """KV session 应在续步前预检旧句柄，并成功后替换为新句柄。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components(
        cache_mode="kv", model_type="tiny_kv_decoder"
    )
    host = bridge.host
    session = M0Session(
        bridge,
        auth,
        coordinator,
        registry,
        config,
        id_factory=iter(["s", "a"]).__next__,
    )
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    result = session.prefill(ids, mask, torch.zeros((1, 2), dtype=torch.float32))
    assert result.protected_output is not None
    old_handle = session._cache_handles[0]
    first = result.protected_output
    step = session.decode_step(torch.tensor([[3]], dtype=torch.long))
    assert step.protected_output is not None
    expected = host.forward_full(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.ones((1, 3), dtype=torch.bool),
    )
    assert torch.allclose(
        step.protected_output[:, -1:], expected[:, -1:], atol=1e-5, rtol=1e-4
    )
    assert session.state.value == "active"
    assert first.shape[-1] == expected.shape[-1]
    with pytest.raises(CacheStateError):
        session._cache_registry.read_kv(old_handle)


def test_concurrent_call_is_rejected_without_disturbing_active_session():
    """活动 prefill 期间的第二调用必须被拒绝且不改变原 session。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components(
        cache_mode="kv", model_type="tiny_kv_decoder"
    )
    entered = threading.Event()
    release = threading.Event()
    original = bridge.prefix_full
    protected_calls = []
    original_suffix = bridge.suffix_with_cache

    def blocked_prefix(*args, **kwargs):
        """暂停真实 prefix，以制造可控并发窗口。"""

        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    def counted_suffix(*args, **kwargs):
        """记录唯一正常请求的 protected suffix 调用。"""

        protected_calls.append(True)
        return original_suffix(*args, **kwargs)

    bridge.prefix_full = blocked_prefix
    bridge.suffix_with_cache = counted_suffix
    session = M0Session(
        bridge,
        auth,
        coordinator,
        registry,
        config,
        id_factory=iter(["s", "a"]).__next__,
    )
    error = []

    def run_prefill():
        """在线程中执行首个请求。"""

        try:
            session.prefill(
                torch.ones((1, 2), dtype=torch.long),
                torch.ones((1, 2), dtype=torch.bool),
                torch.zeros((1, 2)),
            )
        except BaseException as exc:
            error.append(exc)

    worker = threading.Thread(target=run_prefill)
    worker.start()
    assert entered.wait(timeout=5)
    with pytest.raises(M0StateError, match="concurrent_use"):
        session.prefill(
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.bool),
            torch.zeros((1, 2)),
        )
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive() and error == []
    # 并发拒绝只能影响第二调用；原 session、KV 句柄和真实调用计数必须保持完整。
    assert session.state.value == "active"
    assert session._cache_registry is not None
    assert session._cache_handles is not None
    assert all(handle is not None for handle in session._cache_handles)
    assert protected_calls == [True]


def test_kv_decode_suffix_failure_clears_all_handles():
    """KV decode 的 protected suffix 中途异常必须原子清理旧句柄。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components(
        cache_mode="kv", model_type="tiny_kv_decoder"
    )
    session = M0Session(
        bridge,
        auth,
        coordinator,
        registry,
        config,
        id_factory=iter(["s", "a"]).__next__,
    )
    session.prefill(
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        torch.zeros((1, 2), dtype=torch.float32),
    )
    cache_registry = session._cache_registry
    old_handle = session._cache_handles[0]
    original_step = bridge.suffix_step

    def failing_step(*args, **kwargs):
        """在读取旧 cache 后模拟 protected suffix 执行异常。"""

        raise RuntimeError("injected suffix failure")

    bridge.suffix_step = failing_step
    with pytest.raises(BatchExecutionError, match="protected_execution"):
        session.decode_step(torch.tensor([[3]], dtype=torch.long))

    assert session.state.value == "failed"
    assert session._cache_registry is None
    assert session._cache_handles is None
    with pytest.raises(CacheStateError):
        cache_registry.read_kv(old_handle)
    bridge.suffix_step = original_step


def test_kv_preflight_second_row_failure_rolls_back_first_row():
    """KV batch 预检首行通过、次行失败时必须清理整批句柄。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components(
        cache_mode="kv", model_type="tiny_kv_decoder"
    )
    session = M0Session(
        bridge,
        auth,
        coordinator,
        registry,
        config,
        id_factory=iter(["s", "a"]).__next__,
    )
    ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    mask = torch.ones((2, 2), dtype=torch.bool)
    session.prefill(ids, mask, torch.zeros((2, 2), dtype=torch.float32))
    cache_registry = session._cache_registry
    first_handle, second_handle = session._cache_handles
    # 让第一行完整通过后，在第二行制造 request binding 错误，模拟部分成功预检。
    cache_registry._replace_for_test(second_handle, request_id="foreign")
    with pytest.raises(CacheStateError):
        session.decode_step(torch.tensor([[5], [6]], dtype=torch.long))

    assert session.state.value == "failed"
    assert session._cache_registry is None
    assert session._cache_handles is None
    with pytest.raises(CacheStateError):
        cache_registry.read_kv(first_handle)
    with pytest.raises(CacheStateError):
        cache_registry.read_kv(second_handle)


def test_session_rejects_invalid_decode_shape_and_close_is_terminal():
    """decode 输入形状错误应失败，close 后 session 不得再次执行。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components()
    session = M0Session(
        bridge,
        auth,
        coordinator,
        registry,
        config,
        id_factory=iter(["s", "a"]).__next__,
    )
    session.prefill(
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        torch.zeros((1, 2)),
    )
    with pytest.raises(ValueError):
        session.decode_step(torch.ones((2, 1), dtype=torch.long))
    assert session.state.value == "failed"
    with pytest.raises(M0StateError):
        session.decode_step(torch.ones((1, 1), dtype=torch.long))


def test_session_rejects_empty_and_oversized_prefill():
    """空 batch 与超过最大长度的 prefill 必须 fail closed。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components()
    empty = M0Session(bridge, auth, coordinator, registry, config)
    with pytest.raises(ValueError):
        empty.prefill(
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.bool),
            torch.empty((0, 2)),
        )
    oversized = M0Session(bridge, auth, coordinator, registry, config)
    ids = torch.ones((1, config.max_sequence_length + 1), dtype=torch.long)
    with pytest.raises(ValueError):
        oversized.prefill(
            ids, torch.ones_like(ids, dtype=torch.bool), torch.zeros((1, 2))
        )


@pytest.mark.parametrize(
    "input_ids,mask,credential,error",
    [
        (
            torch.ones((1, 2), dtype=torch.int32),
            torch.ones((1, 2), dtype=torch.bool),
            torch.zeros((1, 2)),
            TypeError,
        ),
        (
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.float32),
            torch.zeros((1, 2)),
            ValueError,
        ),
        (
            torch.ones((1, 2), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.bool),
            object(),
            TypeError,
        ),
    ],
)
def test_prefill_rejects_input_types_before_host_execution(
    input_ids, mask, credential, error
):
    """prefill 在 prefix 前拒绝 token、mask 和 credential 类型错误。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components()
    session = M0Session(bridge, auth, coordinator, registry, config)
    with pytest.raises(error):
        session.prefill(input_ids, mask, credential)
    assert session.state.value == "failed"


def test_auth_and_coordinator_reject_foreign_or_replayed_evidence():
    """evidence 必须来自当前 AuthExpert 且只能提交一次。"""

    config, auth, coordinator, _, _ = _components()
    context = __import__(
        "src.can.v2.auth_expert.types", fromlist=["RequestContext"]
    ).RequestContext(("a",), config.execution_config_id, "s", "a")
    evidence = auth.verify(torch.zeros((1, 2)), context)
    coordinator.commit(evidence, context)
    with pytest.raises(PermissionError):
        coordinator.commit(evidence, context)
    other_config, other_auth, _, _, _ = _components()
    other_context = __import__(
        "src.can.v2.auth_expert.types", fromlist=["RequestContext"]
    ).RequestContext(("a",), other_config.execution_config_id, "s", "a")
    with pytest.raises(PermissionError):
        other_auth.validate_evidence(evidence, other_context)


def test_scope_and_dispatch_reject_foreign_objects_and_bad_state():
    """scope、dispatcher 对来源、类型和 batch 形状执行 fail closed。"""

    config, auth, coordinator, registry, bridge = _components()
    context = __import__(
        "src.can.v2.auth_expert.types", fromlist=["RequestContext"]
    ).RequestContext(("a",), config.execution_config_id, "s", "a")
    route_evidence = auth.verify(torch.zeros((1, 2)), context)
    route = coordinator.commit(route_evidence, context)
    view = registry.resolve(route, context)
    with pytest.raises(ValueError):
        registry.restrict(view, [("E1",)])
    with pytest.raises(PermissionError):
        registry.validate_mask(view, torch.ones((1, 2), dtype=torch.bool))
    dispatcher = M0Dispatcher(
        coordinator, registry, lambda state, ids: bridge.suffix_full(state, ids)
    )
    with pytest.raises(TypeError):
        dispatcher.execute(object(), route, view, ExpertSelection(("E1",)), context)
    state = bridge.prefix_full(
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        ("wrong",),
    )
    with pytest.raises(ValueError):
        dispatcher.execute(state, route, view, ExpertSelection(("E1",)), context)


@pytest.mark.parametrize("bad", ["E0", 3, ("E1", "E1")])
def test_selection_and_scope_reject_invalid_values(bad):
    """M0 selection 与 scope 请求拒绝非法类型或重复项。"""

    config, auth, coordinator, registry, _ = _components()
    context = __import__(
        "src.can.v2.auth_expert.types", fromlist=["RequestContext"]
    ).RequestContext(("a",), config.execution_config_id, "s", "a")
    route_evidence_2 = auth.verify(torch.zeros((1, 2)), context)
    route = coordinator.commit(route_evidence_2, context)
    view = registry.resolve(route, context)
    if not isinstance(bad, tuple):
        with pytest.raises((TypeError, ValueError)):
            ExpertSelection((bad,))
    else:
        with pytest.raises(ValueError):
            registry.restrict(view, (bad,))


def test_m0_types_reject_invalid_context_and_config_values():
    """核心不可变类型拒绝空标识、重复 request 和非法 step。"""

    from src.can.v2.auth_expert.types import M0ExpertSpec, M0ScopeSpec, RequestContext

    with pytest.raises(ValueError):
        M0ExpertSpec("", ExpertKind.PUBLIC, False)
    with pytest.raises(TypeError):
        M0ExpertSpec("E", "public", False)
    with pytest.raises(ValueError):
        M0ScopeSpec("s", ("E", "E"))
    with pytest.raises(ValueError):
        RequestContext(("a", "a"), "cfg", "s", "a")
    with pytest.raises(ValueError):
        RequestContext(("a",), "cfg", "s", "a", -1)


def test_manifest_rejects_bad_digest_and_non_object(tmp_path):
    """manifest 摘要错误或根类型错误必须在解析前拒绝。"""

    path = tmp_path / "bad.json"
    path.write_bytes(b"[]")
    with pytest.raises(ValueError):
        load_m0_manifest(path, "0" * 64)
    with pytest.raises(ValueError):
        load_m0_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_auth_constructor_and_verify_validate_types_and_profile():
    """认证入口拒绝错误 verifier、配置、context 和 credential 结构。"""

    config, auth, _, _, _ = _components()
    from src.can.v2.auth_expert.types import RequestContext

    with pytest.raises(TypeError):
        AuthExpert(object(), config)
    with pytest.raises(TypeError):
        AuthExpert(auth.verifier, object())
    bad_verifier = FixedRelationVerifier(
        np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.1
    )
    bad_verifier.profile_id = "other"  # type: ignore[assignment]
    with pytest.raises(PermissionError):
        AuthExpert(bad_verifier, config)
    context = RequestContext(("a",), config.execution_config_id, "s", "a")
    with pytest.raises(TypeError):
        auth.verify(torch.zeros((1, 2)), object())
    with pytest.raises((TypeError, ValueError)):
        auth.verify(torch.zeros((2, 2)), context)


def test_coordinator_rejects_foreign_context_and_route_mutation():
    """协调器拒绝跨 execution config、伪造 route 和长度错配。"""

    config, auth, coordinator, registry, _ = _components()
    from src.can.v2.auth_expert.types import CommittedRoute, RequestContext

    context = RequestContext(("a",), config.execution_config_id, "s", "a")
    route_evidence_3 = auth.verify(torch.zeros((1, 2)), context)
    route = coordinator.commit(route_evidence_3, context)
    with pytest.raises(PermissionError):
        coordinator.validate(
            route, RequestContext(("b",), config.execution_config_id, "s", "a")
        )
    with pytest.raises(PermissionError):
        coordinator.validate(
            CommittedRoute(
                route.routes,
                route.allowed_experts,
                route.context,
                route.policy_id,
                object(),
                object(),
            ),
            context,
        )
    with pytest.raises(PermissionError):
        coordinator.validate(
            CommittedRoute(
                (route.routes[0],),
                (route.allowed_experts[0],),
                route.context,
                route.policy_id,
                coordinator._seal,
                object(),
            ),
            RequestContext(("a", "b"), config.execution_config_id, "s", "a"),
        )
    assert registry.resolve(route, context).expert_ids == (("E1",),)


def test_dispatch_rejects_bad_output_and_deny_selection():
    """protected 回调的非有限输出和 DENY 行的选择均必须失败。"""

    config, auth, coordinator, registry, bridge = _components()
    from src.can.v2.auth_expert.types import RequestContext

    context = RequestContext(("a", "b"), config.execution_config_id, "s", "a")
    route = coordinator.commit(
        auth.verify(torch.tensor([[0.0, 0.0], [2.0, 2.0]]), context), context
    )
    view = registry.resolve(route, context)
    state = bridge.prefix_full(
        torch.ones((2, 2), dtype=torch.long),
        torch.ones((2, 2), dtype=torch.bool),
        context.request_ids,
    )
    bad = M0Dispatcher(
        coordinator, registry, lambda _state, _ids: torch.tensor([[float("nan")]])
    )
    with pytest.raises(Exception):
        bad.execute(state, route, view, ExpertSelection(("E1", None)), context)
    with pytest.raises(PermissionError):
        M0Dispatcher(
            coordinator, registry, lambda value, ids: bridge.suffix_full(value, ids)
        ).execute(state, route, view, ExpertSelection(("E1", "E1")), context)


def test_tiny_bridge_rejects_invalid_host_and_request_binding():
    """Tiny bridge 对宿主类型、cut 边界和 request ID 绑定执行校验。"""

    with pytest.raises(TypeError):
        M0TinyHostBridge(object())
    with pytest.raises(ValueError):
        M0TinyHostBridge(
            TinyDecoderHost(vocab_size=8, d_model=4, num_layers=4, cut_layer=2),
            cut_layer=0,
        )
    kv_host = TinyKVDecoderHost(vocab_size=8, d_model=4, num_layers=4, max_length=8)
    with pytest.raises(ValueError):
        M0TinyHostBridge(kv_host)
    bridge = M0TinyHostBridge(kv_host, cut_layer=2)
    state = bridge.prefix_full(
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        ("a",),
    )
    with pytest.raises(PermissionError):
        bridge.suffix_full(state, ("b",))
    with pytest.raises(ValueError):
        bridge.suffix_step(state, ("a",), (torch.zeros((1, 1, 1)),), ())


def test_kv_mixed_batch_and_right_padding_preserve_indices():
    """KV mixed batch 支持右 padding，且只为 protected 原始行建立 cache。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components(
        cache_mode="kv", model_type="tiny_kv_decoder"
    )
    session = M0Session(
        bridge,
        auth,
        coordinator,
        registry,
        config,
        id_factory=iter(["s", "a"]).__next__,
    )
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    credential = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    result = session.prefill(ids, mask, credential)
    assert result.protected_indices.tolist() == [0]
    assert result.denied_indices.tolist() == [1]
    assert (
        session._cache_handles is not None
        and session._cache_handles[0] is not None
        and session._cache_handles[1] is None
    )
    # TinyKV fixture 只接受尾部 padding；追加 token 后不能在中间重新激活 padding 行。
    assert session.state.value == "active"


def test_kv_preflight_metadata_failure_is_fail_closed():
    """cache request 元数据被篡改时，续步在 prefix 前拒绝并清理 session。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components(
        cache_mode="kv", model_type="tiny_kv_decoder"
    )
    session = M0Session(
        bridge,
        auth,
        coordinator,
        registry,
        config,
        id_factory=iter(["s", "a"]).__next__,
    )
    session.prefill(
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        torch.zeros((1, 2)),
    )
    handle = session._cache_handles[0]
    session._cache_registry._replace_for_test(handle, request_id="forged")
    with pytest.raises(CacheStateError):
        session.decode_step(torch.tensor([[3]], dtype=torch.long))
    assert session.state.value == "failed" and session._cache_registry is None


def test_session_constructor_rejects_mismatched_components():
    """session 不允许混用不同 execution config 的认证链组件。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components()
    from dataclasses import replace

    other_config = replace(config, execution_config_id="other-config")
    with pytest.raises(M0StateError):
        M0Session(bridge, auth, coordinator, registry, other_config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.pop("host"),
        lambda p: p.__setitem__("schema_version", True),
        lambda p: p["verifier"].__setitem__("threshold", float("inf")),
        lambda p: p["host"].__setitem__("cache_mode", "bad"),
        lambda p: p["experts"][0].__setitem__("enabled", "false"),
        lambda p: p["fixture"].__setitem__("seed", True),
        lambda p: p["provenance"].__setitem__("dirty", "false"),
    ],
)
def test_manifest_rejects_invalid_field_values(tmp_path, mutation):
    """manifest 各层字段的缺失、类型、范围和有限性错误均拒绝。"""

    payload = _valid_manifest_payload()
    mutation(payload)
    path = tmp_path / "invalid.json"
    raw = json.dumps(payload, allow_nan=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    with pytest.raises((ValueError, TypeError, KeyError)):
        load_m0_manifest(path, hashlib.sha256(raw).hexdigest())


def test_manifest_rejects_duplicate_json_field_and_nan(tmp_path):
    """JSON 重复字段和 NaN 常量在摘要匹配后仍必须拒绝。"""

    duplicate = b'{"schema_version":1,"schema_version":1}'
    path = tmp_path / "duplicate.json"
    path.write_bytes(duplicate)
    with pytest.raises(ValueError):
        load_m0_manifest(path, hashlib.sha256(duplicate).hexdigest())
    payload = _valid_manifest_payload()
    payload["verifier"]["threshold"] = "NaN"
    raw = json.dumps(payload, separators=(",", ":")).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_m0_manifest(path, hashlib.sha256(raw).hexdigest())


def test_kv_corrupt_tensor_is_rejected_before_next_step():
    """KV 张量 shape 损坏时，续步失败并清理整个 session。"""

    from src.can.v2.auth_expert.runtime import M0Session

    config, auth, coordinator, registry, bridge = _components(
        cache_mode="kv", model_type="tiny_kv_decoder"
    )
    session = M0Session(
        bridge,
        auth,
        coordinator,
        registry,
        config,
        id_factory=iter(["s", "a"]).__next__,
    )
    session.prefill(
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        torch.zeros((1, 2)),
    )
    handle = session._cache_handles[0]
    keys, values = session._cache_registry.read_kv(handle)
    session._cache_registry._replace_kv_for_test(handle, (keys[0][:, :, :-1],), values)
    with pytest.raises(CacheStateError):
        session.decode_step(torch.tensor([[3]], dtype=torch.long))
    assert session.state.value == "failed"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.__setitem__("protocol_id", "bad"),
        lambda p: p["verifier"].__setitem__("backend", "bad"),
        lambda p: p["verifier"].__setitem__("n", 0),
        lambda p: p["verifier"].__setitem__("m", 1),
        lambda p: p["verifier"].__setitem__("relation_sha256", "x"),
        lambda p: p["host"].__setitem__("model_sha256", "x"),
        lambda p: p["host"].__setitem__("num_hidden_layers", 0),
        lambda p: p.__setitem__("experts", []),
        lambda p: p["experts"].__setitem__(0, {"expert_id": "E0"}),
        lambda p: p.__setitem__("scopes", []),
        lambda p: p["fixture"].__setitem__("inputs_sha256", "x"),
        lambda p: p["provenance"].__setitem__("git_commit", "x"),
        lambda p: p["provenance"].__setitem__("device", "cuda"),
    ],
)
def test_manifest_rejects_remaining_contract_mutations(tmp_path, mutation):
    """M0 manifest 其余协议、摘要、目录和 provenance 分支必须拒绝。"""

    payload = _valid_manifest_payload()
    mutation(payload)
    path = tmp_path / "invalid-contract.json"
    raw = json.dumps(payload, separators=(",", ":")).encode()
    path.write_bytes(raw)
    with pytest.raises((ValueError, TypeError, KeyError)):
        load_m0_manifest(path, hashlib.sha256(raw).hexdigest())


def test_manifest_file_io_and_json_root_fail_closed(tmp_path):
    """manifest 文件不存在、摘要参数错误、编码或根类型错误时均快速失败。"""

    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        file_sha256(missing)
    with pytest.raises(FileNotFoundError):
        load_m0_manifest(missing, "a" * 64)

    path = tmp_path / "bad.json"
    path.write_bytes(b"[]")
    with pytest.raises(ValueError):
        load_m0_manifest(path, "not-a-digest")
    with pytest.raises(ValueError):
        load_m0_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert file_sha256(path) == hashlib.sha256(b"[]").hexdigest()

    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        load_m0_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())

    path.write_bytes(b"[]")
    with pytest.raises(ValueError):
        load_m0_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_manifest_rejects_top_level_size_and_nested_scalar_types(tmp_path):
    """manifest 根字段和 verifier/host/limits 非 object 类型不能绕过严格解析。"""

    for key in ("verifier", "host", "limits", "fixture", "provenance"):
        payload = _valid_manifest_payload()
        payload[key] = []
        path = tmp_path / f"{key}.json"
        raw = json.dumps(payload, separators=(",", ":")).encode()
        path.write_bytes(raw)
        with pytest.raises((ValueError, TypeError, KeyError)):
            load_m0_manifest(path, hashlib.sha256(raw).hexdigest())


def test_request_context_and_dispatch_result_reject_malformed_identity():
    """M0 内部身份和 dispatch 索引结果拒绝空值、重复值及不完整覆盖。"""

    from src.can.v2.auth_expert.types import M0DispatchResult, RequestContext

    with pytest.raises(ValueError):
        RequestContext((), "m0", "s", "a")
    with pytest.raises(TypeError):
        RequestContext((1,), "m0", "s", "a")
    with pytest.raises(ValueError):
        RequestContext(("a", "a"), "m0", "s", "a")
    with pytest.raises(ValueError):
        RequestContext(("a",), "m0", "s", "a", -1)

    config, auth, coordinator, registry, _ = _components()
    context = RequestContext(("a",), config.execution_config_id, "s", "a")
    route_evidence_1 = auth.verify(torch.zeros((1, 2)), context)
    route = coordinator.commit(route_evidence_1, context)
    with pytest.raises((TypeError, ValueError)):
        M0DispatchResult(
            None, torch.tensor(0), torch.tensor([], dtype=torch.long), ("a",), route
        )
    with pytest.raises(ValueError):
        M0DispatchResult(None, torch.tensor([0]), torch.tensor([0]), ("a",), route)


def test_manifest_file_and_digest_guards(tmp_path):
    """manifest 文件不存在、摘要格式错误和超大文件均 fail closed。"""

    from src.can.v2.auth_expert.manifest import file_sha256

    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        file_sha256(missing)
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_m0_manifest(path, "bad")
    with pytest.raises(ValueError):
        load_m0_manifest(path, "0" * 64)
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError):
        load_m0_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: M0ScopeSpec("", ("E1",)),
        lambda: M0ScopeSpec("scope", []),
        lambda: M0ScopeSpec("scope", (1,)),
        lambda: M0ScopeSpec("scope", ("E1", "E1")),
    ],
)
def test_m0_scope_spec_rejects_invalid_identity(factory):
    """scope 类型、空值和重复专家标识必须在构造时拒绝。"""

    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "field,value,exc_type",
    [
        ("execution_config_id", "", ValueError),
        ("protocol_id", "other", ValueError),
        ("policy_id", "other", ValueError),
        ("cache_mode", "bad", ValueError),
        ("num_hidden_layers", True, TypeError),
        ("cut_layer", 0, ValueError),
        ("max_batch_size", 0, ValueError),
        ("experts", (), ValueError),
        (
            "experts",
            (
                M0ExpertSpec("E0", ExpertKind.PUBLIC, True),
                M0ExpertSpec("E1", ExpertKind.PROTECTED, True),
            ),
            ValueError,
        ),
        (
            "experts",
            (
                M0ExpertSpec("E0", ExpertKind.PUBLIC, False),
                M0ExpertSpec("E1", ExpertKind.PROTECTED, False),
            ),
            ValueError,
        ),
        ("scopes", (), ValueError),
    ],
)
def test_m0_execution_config_rejects_each_fixed_contract_boundary(
    field, value, exc_type
):
    """M0 execution config 的固定协议、资源和目录边界逐项 fail closed。"""

    from dataclasses import replace

    config, *_ = _components()
    with pytest.raises(exc_type):
        replace(config, **{field: value})


def test_m0_types_and_authentication_cover_binding_failures():
    """覆盖剩余身份、evidence、route 登记和 allowed expert 防篡改分支。"""

    from dataclasses import replace

    from src.can.v2.auth_expert.types import BoundEvidence, RequestContext

    config, auth, coordinator, _, _ = _components()
    context = RequestContext(("a",), config.execution_config_id, "s", "a")
    evidence = auth.verify(torch.zeros((1, 2)), context)
    with pytest.raises(TypeError):
        auth.validate_evidence(object(), context)
    object.__setattr__(evidence, "_auth_seal", object())
    with pytest.raises(PermissionError):
        auth.validate_evidence(evidence, context)

    with pytest.raises(TypeError):
        __import__(
            "src.can.v2.auth_expert.authentication", fromlist=["ScopeCoordinator"]
        ).ScopeCoordinator(object(), config)
    with pytest.raises(PermissionError):
        __import__(
            "src.can.v2.auth_expert.authentication", fromlist=["ScopeCoordinator"]
        ).ScopeCoordinator(auth, replace(config, execution_config_id="other"))

    # 新建独立 route 以分别测试上下文配置、登记、batch 和 allowed mask 检查。
    evidence = auth.verify(torch.zeros((1, 2)), context)
    other_context = RequestContext(("a",), "other", "s", "a")
    with pytest.raises(PermissionError):
        coordinator.commit(
            auth.verify(torch.zeros((1, 2)), other_context), other_context
        )
    route_evidence_2 = auth.verify(torch.zeros((1, 2)), context)
    route = coordinator.commit(route_evidence_2, context)
    del coordinator._committed[id(route)]
    with pytest.raises(PermissionError):
        coordinator.validate(route, context)

    route_evidence_3 = auth.verify(torch.zeros((1, 2)), context)
    route = coordinator.commit(route_evidence_3, context)
    object.__setattr__(route, "routes", ())
    with pytest.raises(ValueError):
        coordinator.validate(route, context)

    route_evidence_4 = auth.verify(torch.zeros((1, 2)), context)
    route = coordinator.commit(route_evidence_4, context)
    object.__setattr__(route, "allowed_experts", (("forged",),))
    with pytest.raises(PermissionError):
        coordinator.validate(route, context)

    with pytest.raises(TypeError):
        ExpertSelection([])
    with pytest.raises(ValueError):
        __import__(
            "src.can.v2.auth_expert.types", fromlist=["M0DispatchResult"]
        ).M0DispatchResult(
            None, torch.tensor([0]), torch.tensor([], dtype=torch.long), (), route
        )


def test_dispatcher_rejects_bad_view_selection_state_and_outputs():
    """Dispatcher 在回调前拒绝 view/selection/state 错误，并包装执行输出异常。"""

    from src.can.v2.auth_expert.types import RequestContext

    config, auth, coordinator, registry, bridge = _components()
    context = RequestContext(("a",), config.execution_config_id, "s", "a")
    route = coordinator.commit(auth.verify(torch.zeros((1, 2)), context), context)
    view = registry.resolve(route, context)
    state = bridge.prefix_full(
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        ("a",),
    )
    dispatcher = M0Dispatcher(
        coordinator, registry, lambda value, ids: bridge.suffix_full(value, ids)
    )
    with pytest.raises(PermissionError):
        dispatcher.execute(state, route, object(), ExpertSelection(("E1",)), context)
    with pytest.raises(ValueError):
        dispatcher.execute(state, route, view, ExpertSelection(()), context)
    bad_state = bridge.prefix_full(
        torch.ones((2, 2), dtype=torch.long),
        torch.ones((2, 2), dtype=torch.bool),
        ("a", "b"),
    )
    with pytest.raises(ValueError):
        dispatcher.execute(bad_state, route, view, ExpertSelection(("E1",)), context)

    denied_context = RequestContext(("a",), config.execution_config_id, "s2", "a2")
    denied_route = coordinator.commit(
        auth.verify(torch.full((1, 2), 2.0), denied_context), denied_context
    )
    denied_view = registry.resolve(denied_route, denied_context)
    object.__setattr__(denied_view, "expert_ids", (("E1",),))
    denied_state = bridge.prefix_full(
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2), dtype=torch.bool),
        ("a",),
    )
    with pytest.raises(PermissionError):
        dispatcher.execute(
            denied_state,
            denied_route,
            denied_view,
            ExpertSelection(("E1",)),
            denied_context,
        )

    with pytest.raises(BatchExecutionError):
        M0Dispatcher(
            coordinator,
            registry,
            lambda _state, _ids: (_ for _ in ()).throw(RuntimeError("boom")),
        ).execute(state, route, view, ExpertSelection(("E1",)), context)
    with pytest.raises(BatchExecutionError):
        M0Dispatcher(
            coordinator, registry, lambda _state, _ids: torch.full((1, 1), float("nan"))
        ).execute(state, route, view, ExpertSelection(("E1",)), context)


def test_auth_and_coordinator_reject_invalid_context_config_and_route_shape():
    """认证链拒绝不匹配 execution config、非法 evidence 和 route 形状。"""

    config, auth, coordinator, _, _ = _components()
    from src.can.v2.auth_expert.types import BoundEvidence, RequestContext

    context = RequestContext(("a",), config.execution_config_id, "s", "a")
    with pytest.raises(PermissionError):
        coordinator.commit(
            BoundEvidence(
                auth.verifier(torch.zeros((1, 2))), context, auth._seal, object()
            ),
            context,
        )
    with pytest.raises((TypeError, PermissionError)):
        coordinator.validate("route", context)
    with pytest.raises((TypeError, ValueError)):
        auth.validate_input(torch.zeros((1, 2)), "1")
    with pytest.raises(PermissionError):
        coordinator.commit(
            auth.verify(torch.zeros((1, 2)), context),
            RequestContext(("a",), "other", "s", "a"),
        )


def test_scope_registry_and_mask_validation_fail_closed():
    """scope registry 只接受绑定协调器，Router mask 必须逐项等于可信副本。"""

    config, auth, coordinator, registry, _ = _components()
    from src.can.v2.auth_expert.types import RequestContext

    with pytest.raises(TypeError):
        ScopeRegistry(object(), config)
    context = RequestContext(("a",), config.execution_config_id, "s", "a")
    route = coordinator.commit(auth.verify(torch.zeros((1, 2)), context), context)
    view = registry.resolve(route, context)
    with pytest.raises(PermissionError):
        registry.restrict(object(), (("E1",),))
    with pytest.raises(ValueError):
        registry.restrict(view, (("E1",), ()))
    with pytest.raises(ValueError):
        registry.restrict(view, (("E1", "E1"),))
    with pytest.raises(TypeError):
        registry.validate_mask(view, [[False, True]])
    with pytest.raises(TypeError):
        registry.validate_mask(view, torch.zeros((1, 2), dtype=torch.float32))
    with pytest.raises(TypeError):
        registry.validate_mask(view, torch.zeros((1, 2, 1), dtype=torch.bool))
    bad_mask = view.mask()
    bad_mask[0, 1] = False
    with pytest.raises(PermissionError):
        registry.validate_mask(view, bad_mask)


def test_dispatcher_constructor_rejects_unbound_components():
    """Dispatcher 不允许替换 scope registry 或注入不可调用 protected 回调。"""

    config, auth, coordinator, registry, _ = _components()
    with pytest.raises(TypeError):
        M0Dispatcher(object(), registry, lambda _state, _ids: torch.zeros((1, 1)))
    _, _, other_coordinator, _, _ = _components()
    with pytest.raises(ValueError):
        M0Dispatcher(
            other_coordinator, registry, lambda _state, _ids: torch.zeros((1, 1))
        )
    with pytest.raises(ValueError):
        M0Dispatcher(coordinator, registry, None)
