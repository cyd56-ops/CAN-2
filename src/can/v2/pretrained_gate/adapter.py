"""与具体预训练宿主无关的 P1 protected/deny 调度器。"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import torch
from torch import Tensor, nn

from .authorization import FixedRelationVerifier, RouteCoordinator
from .host import HostPlugin, PrefixState
from .types import DispatchResult, RouteKind, TrustedRequestContext, _CommittedRoute


class BatchExecutionError(RuntimeError):
    """表示非流式 batch 在受保护执行阶段原子失败。"""

    def __init__(self, stage: str, cause: BaseException) -> None:
        """保存稳定失败阶段并隐藏底层异常文本。"""

        if not isinstance(stage, str) or not stage:
            raise ValueError("stage 必须是非空字符串")
        self.stage = stage
        self.cause_type = type(cause).__name__
        super().__init__(f"batch execution failed at {stage}")


class ProtectedDispatcher(nn.Module):
    """验证已提交 route，并仅调用 PROTECTED 子批。"""

    def __init__(self, coordinator: RouteCoordinator) -> None:
        """绑定唯一协调器。"""

        super().__init__()
        if not isinstance(coordinator, RouteCoordinator):
            raise TypeError("coordinator 必须是 RouteCoordinator")
        self.coordinator = coordinator

    def forward(
        self,
        hidden: Tensor,
        committed: _CommittedRoute,
        context: TrustedRequestContext,
        protected_fn: Callable[[Tensor, Tuple[str, ...]], Any],
    ) -> DispatchResult:
        """恒等选取合法 hidden 并执行一次受保护子批。

        参数:
            hidden: prefix 输出，第一维必须与请求 batch 对齐。
            committed: 唯一协调器提交的 route。
            context: 当前完整受信请求上下文。
            protected_fn: 真实 suffix/head 调用封装，接收子批和对应 request IDs。
        """

        if not isinstance(hidden, Tensor) or hidden.ndim < 1:
            raise TypeError("hidden 必须是至少一维 Tensor")
        if not hidden.dtype.is_floating_point:
            raise TypeError("hidden 必须使用浮点 dtype")
        if not isinstance(protected_fn, Callable):
            raise TypeError("protected_fn 必须可调用")
        routes = self.coordinator.validate_committed_route(committed, context)
        if hidden.shape[0] != len(routes):
            raise ValueError("hidden 与 route batch 大小不一致")

        protected_rows = [
            index for index, route in enumerate(routes) if route is RouteKind.PROTECTED
        ]
        denied_rows = [
            index for index, route in enumerate(routes) if route is RouteKind.DENY
        ]
        protected_indices = torch.tensor(
            protected_rows, dtype=torch.long, device=hidden.device
        )
        denied_indices = torch.tensor(
            denied_rows, dtype=torch.long, device=hidden.device
        )
        if not protected_rows:
            return DispatchResult(
                protected_output=None,
                protected_indices=protected_indices,
                denied_indices=denied_indices,
                request_ids=context.request_ids,
            )

        # index_select 保留 dtype/device/数值及到原 hidden 的梯度路径。
        protected_hidden = hidden.index_select(0, protected_indices)
        protected_ids = tuple(context.request_ids[index] for index in protected_rows)
        try:
            output = protected_fn(protected_hidden, protected_ids)
        except BaseException as error:
            raise BatchExecutionError("protected_execution", error) from error
        return DispatchResult(
            protected_output=output,
            protected_indices=protected_indices,
            denied_indices=denied_indices,
            request_ids=context.request_ids,
        )


class GatedHostAdapter(nn.Module):
    """按“结构预检 → prefix → Gate → dispatcher”执行 G0/P1 prefill。"""

    def __init__(
        self,
        host: HostPlugin,
        verifier: FixedRelationVerifier,
        coordinator: RouteCoordinator,
    ) -> None:
        """绑定宿主、verifier 和唯一协调器。"""

        super().__init__()
        if not isinstance(host, HostPlugin):
            raise TypeError("host 必须实现 HostPlugin")
        if not isinstance(verifier, FixedRelationVerifier):
            raise TypeError("verifier 必须是 FixedRelationVerifier")
        if not isinstance(coordinator, RouteCoordinator):
            raise TypeError("coordinator 必须是 RouteCoordinator")
        if coordinator.verifier is not verifier:
            raise PermissionError("coordinator 必须绑定同一 verifier")
        self.host = host
        self.verifier = verifier
        self.coordinator = coordinator
        self.dispatcher = ProtectedDispatcher(coordinator)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        credential: Tensor,
        context: TrustedRequestContext,
    ) -> DispatchResult:
        """执行一次非流式 P1 protected/deny prefill。

        结构错误在 prefix 前失败；逐行关系与数值拒绝在 prefix 后的图中间 Gate
        产生 DENY，且不调用该行 suffix/norm/head。
        """

        if not isinstance(context, TrustedRequestContext):
            raise TypeError("context 必须是 TrustedRequestContext")
        self.verifier.validate_credential_structure(credential)
        if credential.shape[0] != len(context.request_ids):
            raise ValueError("credential 与请求 batch 大小不一致")
        state = self.host.prefix(input_ids, attention_mask, context.request_ids)
        if state.request_ids != context.request_ids:
            raise PermissionError("宿主 prefix 改变了请求身份或顺序")
        evidence = self.verifier(credential)
        committed = self.coordinator.commit(evidence, context)

        def run_protected(hidden: Tensor, request_ids: Tuple[str, ...]) -> Any:
            """选择完整 prefix 状态并调用真实 protected suffix。"""

            original_rows = {
                value: index for index, value in enumerate(state.request_ids)
            }
            indices = torch.tensor(
                [original_rows[value] for value in request_ids],
                dtype=torch.long,
                device=hidden.device,
            )
            selected = self.host.select_prefix_state(state, indices, request_ids)
            if not torch.equal(selected.hidden, hidden):
                raise RuntimeError("宿主 select_prefix_state 改变了合法 hidden")
            return self.host.protected_suffix(selected, request_ids)

        return self.dispatcher(state.hidden, committed, context, run_protected)
