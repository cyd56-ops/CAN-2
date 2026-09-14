"""M0 session 生命周期与 prefill 编排。"""

from __future__ import annotations

import threading
import uuid
from enum import Enum
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from ..pretrained_gate.cache import CacheHandle, CacheRegistry
from ..pretrained_gate.host import PrefixState
from ..pretrained_gate.types import RouteKind
from .authentication import AuthExpert, ScopeCoordinator
from .dispatch import M0Dispatcher
from .host_bridge import M0TinyHostBridge
from .scope import ScopeRegistry
from .types import (
    ExpertSelection,
    M0DispatchResult,
    M0ExecutionConfig,
    M0StateError,
    RequestContext,
)


class SessionState(str, Enum):
    """描述 M0 session 生命周期。"""

    NEW = "new"
    PREFILLING = "prefilling"
    ACTIVE = "active"
    FAILED = "failed"
    FINISHED = "finished"


class M0Session:
    """串联 prefix、一次认证提交和固定 E1 dispatch。"""

    def __init__(
        self,
        bridge: M0TinyHostBridge,
        auth: AuthExpert,
        coordinator: ScopeCoordinator,
        registry: ScopeRegistry,
        config: M0ExecutionConfig,
        id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        """创建绑定到固定组件的 session。"""

        if (
            not isinstance(bridge, M0TinyHostBridge)
            or not isinstance(auth, AuthExpert)
            or not isinstance(coordinator, ScopeCoordinator)
            or not isinstance(registry, ScopeRegistry)
            or not isinstance(config, M0ExecutionConfig)
        ):
            raise TypeError("M0 session 参数非法")
        if (
            auth.config != config
            or coordinator.config != config
            or registry.config != config
        ):
            raise M0StateError("M0 组件 execution config 不一致")
        self.bridge, self.auth, self.coordinator, self.registry, self.config = (
            bridge,
            auth,
            coordinator,
            registry,
            config,
        )
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self.session_id = self._id_factory()
        self.state = SessionState.NEW
        self._lock = threading.RLock()
        self._context: Optional[RequestContext] = None
        self._evidence = None
        self._route = None
        self._view = None
        self._prefix: Optional[PrefixState] = None
        self._dispatcher: Optional[M0Dispatcher] = None
        self._input_ids: Optional[Tensor] = None
        self._attention_mask: Optional[Tensor] = None
        self._cache_registry: Optional[CacheRegistry] = None
        self._cache_handles: Optional[list[Optional[CacheHandle]]] = None
        self._pending_cache_handles: Optional[list[Optional[CacheHandle]]] = None
        self._decode_mode = False

    def prefill(
        self, input_ids: Tensor, attention_mask: Tensor, credential: Tensor
    ) -> M0DispatchResult:
        """执行首次 prefix、验证、授权提交和 protected dispatch。"""

        if not self._lock.acquire(blocking=False):
            raise M0StateError("concurrent_use")
        try:
            if self.state is not SessionState.NEW:
                raise M0StateError("inactive_session")
            if (
                not isinstance(input_ids, Tensor)
                or input_ids.ndim != 2
                or input_ids.dtype != torch.long
            ):
                raise TypeError("input_ids 必须是二维 LongTensor")
            if (
                input_ids.shape[0] < 1
                or input_ids.shape[0] > self.config.max_batch_size
                or input_ids.shape[1] < 1
                or input_ids.shape[1] > self.config.max_sequence_length
            ):
                raise ValueError("输入超过 M0 资源限制")
            if (
                not isinstance(attention_mask, Tensor)
                or attention_mask.shape != input_ids.shape
                or attention_mask.dtype != torch.bool
                or attention_mask.device != input_ids.device
            ):
                raise ValueError("attention_mask 必须与 input_ids 对齐")
            if not isinstance(credential, Tensor):
                raise TypeError("credential 必须是 Tensor")
            # 所有请求级结构和设备约束必须在 prefix 前完成，避免错误输入触发模型调用。
            host_embedding = getattr(
                self.bridge.host,
                "token_embedding",
                getattr(self.bridge.host, "embedding", None),
            )
            if not isinstance(host_embedding, torch.nn.Embedding):
                raise M0StateError("host_embedding_not_available")
            host_device = host_embedding.weight.device
            if input_ids.device != host_device:
                raise ValueError("input_ids 与 host device 不一致")
            self.auth.validate_input(credential, int(input_ids.shape[0]))
            # 在进入 prefix 前复制不可信输入，避免调用方原地改写本次请求。
            input_ids = input_ids.detach().clone()
            attention_mask = attention_mask.detach().clone()
            credential = credential.detach().clone()
            request_ids = tuple(
                f"{self.session_id}:{index}" for index in range(input_ids.shape[0])
            )
            context = RequestContext(
                request_ids,
                self.config.execution_config_id,
                self.session_id,
                self._id_factory(),
                0,
            )
            self._context = context
            self._input_ids = input_ids
            self._attention_mask = attention_mask
            self.state = SessionState.PREFILLING
            if self.config.cache_mode == "kv":
                if not hasattr(self.bridge.host, "blocks") or not hasattr(
                    self.bridge.host, "token_embedding"
                ):
                    raise M0StateError("kv_bridge_not_implemented")
                self._cache_registry = CacheRegistry(
                    self.config.execution_config_id,
                    self.config.model_sha256,
                    self.config.policy_id,
                    self.config.cut_layer,
                    len(self.bridge.host.blocks) - self.config.cut_layer,
                )
                self._cache_handles = [None] * len(request_ids)
            self._prefix = self.bridge.prefix_full(
                input_ids, attention_mask, request_ids
            )
            evidence = self.auth.verify(credential, context)
            self._evidence = evidence
            self._route = self.coordinator.commit(evidence, context)
            self._view = self.registry.resolve(self._route, context)
            self._dispatcher = M0Dispatcher(
                self.coordinator, self.registry, self._run_protected
            )
            selection = ExpertSelection(
                tuple("E1" if item else None for item in self._route.allowed_experts)
            )
            result = self._dispatcher.execute(
                self._prefix, self._route, self._view, selection, context
            )
            if self._pending_cache_handles is not None:
                self._cache_handles = self._pending_cache_handles
                self._pending_cache_handles = None
            self.state = SessionState.ACTIVE
            return result
        except BaseException:
            self._fail_and_clear()
            raise
        finally:
            self._lock.release()

    def decode_step(self, next_input_ids: Tensor) -> M0DispatchResult:
        """在既有 route 上推进无 KV 增量步，不重新验签或提交 route。"""

        if not self._lock.acquire(blocking=False):
            raise M0StateError("concurrent_use")
        try:
            if (
                self.state is not SessionState.ACTIVE
                or self._context is None
                or self._route is None
            ):
                raise M0StateError("inactive_session")
            if (
                not isinstance(next_input_ids, Tensor)
                or next_input_ids.dtype != torch.long
                or next_input_ids.ndim != 2
            ):
                raise TypeError("next_input_ids 必须是二维 LongTensor")
            if (
                self._input_ids is None
                or self._attention_mask is None
                or next_input_ids.shape != (self._input_ids.shape[0], 1)
            ):
                raise ValueError("decode batch 必须与 prefill 活动 batch 一致")
            # KV 预检必须使用追加 token 之前的请求长度与 mask，避免错误 cache 先参与计算。
            old_input_ids = self._input_ids
            old_attention_mask = self._attention_mask
            if self.config.cache_mode == "kv":
                if self._cache_registry is None or self._cache_handles is None:
                    raise M0StateError("cache_state_validation_failed")
                protected_rows = [
                    i
                    for i, item in enumerate(self._route.routes)
                    if item is RouteKind.PROTECTED
                ]
                handles = tuple(self._cache_handles[i] for i in protected_rows)
                if any(item is None for item in handles):
                    raise M0StateError("cache_state_validation_failed")
                ids = tuple(self._context.request_ids[i] for i in protected_rows)
                masks = tuple(
                    tuple(bool(value) for value in old_attention_mask[i].tolist())
                    for i in protected_rows
                )
                length = old_input_ids.shape[1]
                self._cache_registry.preflight_kv_batch(
                    handles, ids, masks, tuple(length for _ in protected_rows)
                )
            self._input_ids = torch.cat(
                (self._input_ids, next_input_ids.detach().clone()), dim=1
            )
            self._attention_mask = torch.cat(
                (
                    self._attention_mask,
                    torch.ones_like(next_input_ids, dtype=torch.bool),
                ),
                dim=1,
            )
            if self._input_ids.shape[1] > self.config.max_sequence_length:
                raise ValueError("decode 超过 M0 序列限制")
            context = RequestContext(
                self._context.request_ids,
                self._context.execution_config_id,
                self._context.session_id,
                self._context.attempt_id,
                self._context.step_id + 1,
            )
            state = self.bridge.prefix_full(
                self._input_ids, self._attention_mask, context.request_ids
            )
            view = self.registry.resolve(self._route, context)
            selection = ExpertSelection(
                tuple("E1" if item else None for item in self._route.allowed_experts)
            )
            self._decode_mode = self.config.cache_mode == "kv"
            result = self._dispatcher.execute(state, self._route, view, selection, context)  # type: ignore[union-attr]
            if self._pending_cache_handles is not None:
                old_handles = self._cache_handles or []
                for handle in old_handles:
                    if handle is not None:
                        self._cache_registry.finish(handle)  # type: ignore[union-attr]
                self._cache_handles = self._pending_cache_handles
                self._pending_cache_handles = None
            self._context = context
            return result
        except BaseException:
            self._fail_and_clear()
            raise
        finally:
            self._lock.release()

    def close(self) -> None:
        """原子地结束 session 并清理授权/cache 登记。"""

        with self._lock:
            if self.state is SessionState.FINISHED:
                return
            self.state = SessionState.FAILED
            self._revoke_authorization()
            self._route = self._view = self._prefix = self._context = (
                self._dispatcher
            ) = None
            self._evidence = None
            self._input_ids = self._attention_mask = None
            if self._cache_registry is not None:
                self._cache_registry.clear()
            self._cache_registry = None
            self._cache_handles = self._pending_cache_handles = None
            self.state = SessionState.FINISHED

    def _fail_and_clear(self) -> None:
        """在持锁状态下原子标记失败并清理内部句柄。"""

        self.state = SessionState.FAILED
        self._revoke_authorization()
        self._route = self._view = self._prefix = self._context = self._dispatcher = (
            None
        )
        self._evidence = None
        self._input_ids = self._attention_mask = None
        if self._cache_registry is not None:
            self._cache_registry.clear()
        self._cache_registry = None
        self._cache_handles = self._pending_cache_handles = None

    def _revoke_authorization(self) -> None:
        """在持锁状态下撤销 route/view/evidence 的可信登记。"""

        if self._route is not None:
            self.registry.revoke(self._route)
            self.coordinator.revoke(self._route)
        if self._evidence is not None:
            self.auth.revoke(self._evidence)

    def _run_protected(
        self, state: PrefixState, request_ids: Tuple[str, ...]
    ) -> Tensor:
        """仅由 Dispatcher 调用已绑定的 protected suffix。"""

        if self._cache_registry is None or not hasattr(self.bridge.host, "blocks"):
            return self.bridge.suffix_full(state, request_ids)
        if self._context is None:
            raise M0StateError("inactive_session")
        outputs = []
        new_handles: list[Optional[CacheHandle]] = [None] * len(
            self._context.request_ids
        )
        row_lookup = {
            value: index for index, value in enumerate(self._context.request_ids)
        }
        for row, request_id in enumerate(request_ids):
            original = row_lookup[request_id]
            one = PrefixState(
                state.hidden[row : row + 1],
                state.attention_mask[row : row + 1],
                state.position_ids[row : row + 1],
                (request_id,),
                state.host_state,
            )
            if self._decode_mode:
                if self._cache_handles is None or self._cache_handles[original] is None:
                    raise M0StateError("cache_state_validation_failed")
                old_keys, old_values = self._cache_registry.read_kv(
                    self._cache_handles[original]
                )
                output, keys, values = self.bridge.suffix_step(
                    one, (request_id,), old_keys, old_values
                )
            else:
                output, keys, values = self.bridge.suffix_with_cache(one, (request_id,))
            outputs.append(output)
            new_handles[original] = self._cache_registry.create_protected_kv(
                request_id,
                keys,
                values,
                tuple(bool(value) for value in one.attention_mask[0].tolist()),
                one.hidden.shape[1],
            )
        self._pending_cache_handles = new_handles
        return torch.cat(outputs, dim=0)
