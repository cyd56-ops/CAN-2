"""M0 固定 E1 调度器。"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor

from ..pretrained_gate.adapter import BatchExecutionError
from ..pretrained_gate.host import PrefixState
from ..pretrained_gate.types import RouteKind
from .authentication import ScopeCoordinator
from .scope import ScopeRegistry
from .types import (
    AllowedExperts,
    CommittedRoute,
    ExpertSelection,
    M0AuthorizationError,
    M0DispatchResult,
    RequestContext,
)


class M0Dispatcher:
    """按可信 route/view 执行固定 protected E1，拒绝外部回调注入。"""

    def __init__(
        self,
        coordinator: ScopeCoordinator,
        registry: ScopeRegistry,
        protected_fn: Callable[[PrefixState, Tuple[str, ...]], Tensor],
    ) -> None:
        """绑定协调器、scope registry 和唯一受保护宿主回调。"""

        if not isinstance(coordinator, ScopeCoordinator) or not isinstance(
            registry, ScopeRegistry
        ):
            raise TypeError("coordinator/registry 类型不正确")
        if registry.coordinator is not coordinator or not callable(protected_fn):
            raise ValueError("registry 或 protected_fn 绑定不正确")
        self.coordinator = coordinator
        self.registry = registry
        self._protected_fn = protected_fn

    def execute(
        self,
        state: PrefixState,
        route: CommittedRoute,
        view: AllowedExperts,
        selection: ExpertSelection,
        context: RequestContext,
    ) -> M0DispatchResult:
        """完成整批预检后执行 E1，并保留原始 batch 索引。"""

        if not isinstance(state, PrefixState):
            raise TypeError("state 必须是 PrefixState")
        self.coordinator.validate(route, context)
        if (
            not isinstance(view, AllowedExperts)
            or view._registry_seal is not self.registry._seal
            or view.route is not route
            or view.context != context
            or not self.registry.is_registered(view)
        ):
            raise M0AuthorizationError("scope view 与 route 不匹配")
        if not isinstance(selection, ExpertSelection) or len(selection.values) != len(
            route.routes
        ):
            raise ValueError("selection batch 不匹配")
        if (
            state.hidden.shape[0] != len(route.routes)
            or state.request_ids != context.request_ids
        ):
            raise ValueError("prefix state 与 route 不匹配")
        protected, denied = [], []
        for index, (route_kind, allowed, chosen) in enumerate(
            zip(route.routes, view.expert_ids, selection.values)
        ):
            if chosen is not None and chosen not in allowed:
                raise M0AuthorizationError("selection 超出可信 allowed scope")
            if route_kind is RouteKind.PROTECTED:
                if chosen != "E1":
                    raise M0AuthorizationError("protected 行必须选择 E1")
                protected.append(index)
            else:
                if chosen is not None:
                    raise M0AuthorizationError("DENY 行必须选择 None")
                denied.append(index)
        protected_indices = torch.tensor(
            protected, dtype=torch.long, device=state.hidden.device
        )
        denied_indices = torch.tensor(
            denied, dtype=torch.long, device=state.hidden.device
        )
        output = None
        if protected:
            selected_state = _select_state(
                state,
                protected_indices,
                tuple(context.request_ids[i] for i in protected),
            )
            try:
                output = self._protected_fn(selected_state, selected_state.request_ids)
            except BaseException as error:
                raise BatchExecutionError("protected_execution", error) from error
            if not isinstance(output, Tensor) or not torch.isfinite(output).all():
                raise BatchExecutionError(
                    "protected_output", ValueError("non-finite or invalid output")
                )
        return M0DispatchResult(
            output, protected_indices, denied_indices, context.request_ids, route
        )


def _select_state(
    state: PrefixState, indices: Tensor, request_ids: Tuple[str, ...]
) -> PrefixState:
    """按原始索引选择 prefix state，保持 hidden 的梯度和数值。"""

    return PrefixState(
        state.hidden.index_select(0, indices),
        state.attention_mask.index_select(0, indices),
        state.position_ids.index_select(0, indices),
        request_ids,
        state.host_state,
    )
