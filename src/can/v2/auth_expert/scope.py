"""M0 scope registry：从可信 route 导出并只能收窄的 Expert 集合。"""

from __future__ import annotations

from typing import Tuple

import torch

from .authentication import ScopeCoordinator
from .types import (
    AllowedExperts,
    CommittedRoute,
    M0AuthorizationError,
    M0ExecutionConfig,
    RequestContext,
)


class ScopeRegistry:
    """维护 M0 固定 E0/E1 目录并生成只读 scope view。"""

    def __init__(
        self, coordinator: ScopeCoordinator, config: M0ExecutionConfig
    ) -> None:
        """绑定唯一协调器和不可变 execution config。"""

        if (
            not isinstance(coordinator, ScopeCoordinator)
            or coordinator.config != config
        ):
            raise TypeError("coordinator 必须与 config 匹配")
        self.coordinator = coordinator
        self.config = config
        self._seal = object()
        self._views = {}

    def resolve(self, route: CommittedRoute, context: RequestContext) -> AllowedExperts:
        """校验 route 后导出其允许的 Expert 集合。"""

        self.coordinator.validate(route, context)
        view = AllowedExperts(
            route.allowed_experts, route, context, context.step_id, self._seal
        )
        self._views[id(view)] = view
        return view

    def restrict(
        self,
        view: AllowedExperts,
        requested: Tuple[Tuple[str, ...], ...],
    ) -> AllowedExperts:
        """创建只包含原允许集合子集的新 view。"""

        if (
            not isinstance(view, AllowedExperts)
            or view._registry_seal is not self._seal
        ):
            raise M0AuthorizationError("scope view 来源不匹配")
        if not isinstance(requested, tuple) or len(requested) != len(view.expert_ids):
            raise ValueError("requested scope view batch 不匹配")
        result = []
        for original, selection in zip(view.expert_ids, requested):
            if not isinstance(selection, tuple) or len(set(selection)) != len(
                selection
            ):
                raise ValueError("requested Expert 集合必须是无重复 tuple")
            if any(item not in original for item in selection):
                raise M0AuthorizationError("scope restrict 不能扩大权限")
            result.append(selection)
        result_view = AllowedExperts(
            tuple(result), view.route, view.context, view.step_id, self._seal
        )
        self._views[id(result_view)] = result_view
        return result_view

    def is_registered(self, view: AllowedExperts) -> bool:
        """检查 view 是否仍由当前 registry 登记且未随 session 撤销。"""

        return isinstance(view, AllowedExperts) and self._views.get(id(view)) is view

    def revoke(self, route: CommittedRoute) -> None:
        """撤销 route 关联的全部 scope view。"""

        if not isinstance(route, CommittedRoute):
            raise TypeError("route 必须是 CommittedRoute")
        for view_id, view in tuple(self._views.items()):
            if view.route is route:
                del self._views[view_id]

    @staticmethod
    def validate_mask(view: AllowedExperts, mask: torch.Tensor) -> None:
        """验证 Router 提供的 mask 只是可信 view 的独立副本。"""

        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.ndim != 2
        ):
            raise TypeError("mask 必须是二维 BoolTensor")
        expected = view.mask()
        if not torch.equal(mask, expected):
            raise M0AuthorizationError("Router mask 与可信 scope 不一致")
