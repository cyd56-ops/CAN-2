"""M1a scope registry：只能收窄已提交的 Expert 集合。"""

from __future__ import annotations

from typing import Dict, Tuple

from .router import M1ARouteCoordinator
from .types import (
    M1AAllowedExperts,
    M1aAuthorizationError,
    M1ACommittedRoute,
    M1AContext,
)


class M1AScopeRegistry:
    """从可信 M1a route 生成并登记只能收窄的 scope view。"""

    def __init__(self, coordinator: M1ARouteCoordinator) -> None:
        """绑定唯一 route 协调器。"""

        if not isinstance(coordinator, M1ARouteCoordinator):
            raise TypeError("coordinator 必须是 M1ARouteCoordinator")
        self.coordinator = coordinator
        self._seal = object()
        self._views: Dict[int, M1AAllowedExperts] = {}

    def resolve(
        self, route: M1ACommittedRoute, context: M1AContext
    ) -> M1AAllowedExperts:
        """校验 route 后登记完整 allowed expert view。"""

        self.coordinator.validate(route, context.request_ids)
        view = M1AAllowedExperts(route.allowed_experts, route, context, self._seal)
        self._views[id(view)] = view
        return view

    def restrict(
        self, view: M1AAllowedExperts, requested: Tuple[Tuple[str, ...], ...]
    ) -> M1AAllowedExperts:
        """创建逐行仅为原集合子集的新 view，拒绝权限扩大。"""

        self.validate(view, view.route, view.context)
        if not isinstance(requested, tuple) or len(requested) != len(view.expert_ids):
            raise ValueError("requested scope batch 不匹配")
        result = []
        for original, narrowed in zip(view.expert_ids, requested):
            if not isinstance(narrowed, tuple) or len(set(narrowed)) != len(narrowed):
                raise ValueError("requested scope 必须为无重复 tuple")
            if any(item not in original for item in narrowed):
                raise M1aAuthorizationError("scope view 不能扩大权限")
            result.append(narrowed)
        restricted = M1AAllowedExperts(
            tuple(result), view.route, view.context, self._seal
        )
        self._views[id(restricted)] = restricted
        return restricted

    def validate(
        self, view: M1AAllowedExperts, route: M1ACommittedRoute, context: M1AContext
    ) -> None:
        """验证 view 来源、登记状态以及 route/context 绑定。"""

        if (
            not isinstance(view, M1AAllowedExperts)
            or self._views.get(id(view)) is not view
        ):
            raise M1aAuthorizationError("scope view 未由当前 registry 登记")
        if (
            view._registry_seal is not self._seal
            or view.route is not route
            or view.context != context
        ):
            raise M1aAuthorizationError("scope view 与 route/context 不匹配")
