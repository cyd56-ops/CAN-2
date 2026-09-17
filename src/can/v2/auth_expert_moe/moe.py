"""M1a Shared + Authenticated Routed tiny-MoE 执行器。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..pretrained_gate.types import RouteKind
from .experts import ProtectedExpert, SharedExpert
from .router import FixedConstrainedRouter, M1ARouteCoordinator
from .scope import M1AScopeRegistry
from .types import (
    CallLedger,
    M1AAllowedExperts,
    M1aAuthorizationError,
    M1ACommittedRoute,
    M1AConfig,
    M1AContext,
    M1AOutput,
)


class M1aMoE(nn.Module):
    """执行 shared 恒等路径与受信 routed E1 稀疏路径。"""

    def __init__(
        self,
        coordinator: M1ARouteCoordinator,
        config: M1AConfig = M1AConfig(),
        shared: Optional[SharedExpert] = None,
        protected: Optional[ProtectedExpert] = None,
        router: Optional[FixedConstrainedRouter] = None,
        registry: Optional[M1AScopeRegistry] = None,
    ) -> None:
        """构造冻结 E0/E1 tiny-MoE。"""

        super().__init__()
        if (
            not isinstance(coordinator, M1ARouteCoordinator)
            or coordinator.execution_config_id != config.execution_config_id
        ):
            raise M1aAuthorizationError("coordinator 与 M1a config 不匹配")
        self.config = config
        self.coordinator = coordinator
        self.router = router or FixedConstrainedRouter()
        self.registry = registry or M1AScopeRegistry(coordinator)
        self.shared = shared or SharedExpert(config.d_model)
        self.protected = protected or ProtectedExpert(config.d_model)

    def forward(
        self,
        hidden: Tensor,
        route: M1ACommittedRoute,
        context: M1AContext,
        ledger: Optional[CallLedger] = None,
        case_id: str = "default",
        view: Optional[M1AAllowedExperts] = None,
    ) -> M1AOutput:
        """按原始 batch 索引完成 shared + authorized routed 组合。"""

        if (
            not isinstance(hidden, Tensor)
            or hidden.ndim != 3
            or hidden.shape[2] != self.config.d_model
        ):
            raise ValueError("hidden 必须为 Tensor[B,T,d_model]")
        if hidden.shape[0] == 0 or hidden.shape[0] != len(context.request_ids):
            raise ValueError("hidden batch 与 context 不匹配")
        if hidden.dtype != self.config.dtype or not bool(
            torch.isfinite(hidden).all().item()
        ):
            raise ValueError("hidden dtype 或有限性不匹配")
        self.coordinator.validate(route, context.request_ids)
        if view is None:
            view = self.registry.resolve(route, context)
        self.registry.validate(view, route, context)
        if ledger is None:
            ledger = CallLedger(
                "m1a-local", self.config.execution_config_id, route.policy_id
            )
        all_indices = tuple(range(hidden.shape[0]))
        shared_output = self.shared(
            hidden, context.request_ids, ledger, all_indices, case_id
        )
        mask = torch.tensor(
            [["E0" in experts, "E1" in experts] for experts in view.expert_ids],
            dtype=torch.bool,
            device=hidden.device,
        )
        logits = torch.zeros(
            (hidden.shape[0], 2), dtype=hidden.dtype, device=hidden.device
        )
        selection = self.router.select(logits, mask)
        for kind, chosen, allowed in zip(
            route.routes, selection.values, view.expert_ids
        ):
            expected = (
                ("E1" if "E1" in allowed else None)
                if kind is RouteKind.PROTECTED
                else ("E0" if kind is RouteKind.PUBLIC else None)
            )
            if chosen != expected:
                raise M1aAuthorizationError("Router selection 与可信 route 不匹配")
        routed = [
            index for index, value in enumerate(selection.values) if value == "E1"
        ]
        denied = [
            index for index, value in enumerate(selection.values) if value is None
        ]
        if routed:
            indices = torch.tensor(routed, dtype=torch.long, device=hidden.device)
            routed_hidden = hidden.index_select(0, indices)
            routed_ids = tuple(context.request_ids[index] for index in routed)
            routed_output = self.protected(
                routed_hidden, routed_ids, ledger, tuple(routed), case_id
            )
            normalized = F.normalize(
                routed_output, p=2.0, dim=-1, eps=self.config.normalize_eps
            )
            output = shared_output.clone()
            output.index_add_(0, indices, self.config.alpha * normalized)
        else:
            output = shared_output
        return M1AOutput(
            output,
            shared_output,
            torch.tensor(routed, dtype=torch.long, device=hidden.device),
            torch.tensor(denied, dtype=torch.long, device=hidden.device),
            selection,
        )
