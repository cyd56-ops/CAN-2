"""M1a 固定约束 Router 和 policy route 协调器。"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from ..pretrained_gate.authorization import FixedRelationVerifier
from ..pretrained_gate.types import EvidenceReason, RouteKind
from .types import (
    M1aAuthorizationError,
    M1ABoundEvidence,
    M1ACommittedRoute,
    M1AContext,
    M1ASelection,
)

P1_POLICY = "p1-protected-or-deny-v1"
P2_POLICY = "p2-capability-routing-v1"


class M1AAuthExpert:
    """调用固定 verifier，并只产生与请求绑定的 evidence。"""

    def __init__(
        self, verifier: FixedRelationVerifier, execution_config_id: str
    ) -> None:
        """绑定 verifier 和不可替换的 execution config。"""

        if not isinstance(verifier, FixedRelationVerifier):
            raise TypeError("verifier 必须是 FixedRelationVerifier")
        if not isinstance(execution_config_id, str) or not execution_config_id:
            raise ValueError("execution_config_id 必须是非空字符串")
        self.verifier = verifier
        self.execution_config_id = execution_config_id
        self._seal = object()
        self._issued: Dict[int, M1ABoundEvidence] = {}

    def verify(self, credential: Tensor, context: M1AContext) -> M1ABoundEvidence:
        """生成 evidence，不提交任何能力 route。"""

        if (
            not isinstance(context, M1AContext)
            or context.execution_config_id != self.execution_config_id
        ):
            raise M1aAuthorizationError("evidence context 与 AuthExpert 不匹配")
        evidence = self.verifier(credential)
        if evidence.verified.shape[0] != len(context.request_ids):
            raise ValueError("credential 与 request_ids batch 不一致")
        bound = M1ABoundEvidence(evidence, context, self._seal)
        self._issued[id(bound)] = bound
        return bound

    def validate(self, bound: M1ABoundEvidence, context: M1AContext) -> None:
        """验证 evidence 来源、完整性和请求绑定。"""

        if (
            not isinstance(bound, M1ABoundEvidence)
            or self._issued.get(id(bound)) is not bound
        ):
            raise M1aAuthorizationError("evidence 未由当前 AuthExpert 签发")
        if bound._auth_seal is not self._seal or bound.context != context:
            raise M1aAuthorizationError("evidence 请求绑定不匹配")
        self.verifier.validate_evidence(bound.evidence)


class M1ARouteCoordinator:
    """将 verifier evidence 转成独立、绑定 policy 的 M1a route。"""

    def __init__(
        self,
        auth_expert: M1AAuthExpert,
        execution_config_id: str,
        policy: str = P1_POLICY,
    ) -> None:
        """绑定 AuthExpert、execution config 和 P1/P2 policy。"""

        if not isinstance(auth_expert, M1AAuthExpert):
            raise TypeError("auth_expert 必须是 M1AAuthExpert")
        if policy not in {P1_POLICY, P2_POLICY}:
            raise ValueError("未知 M1a policy")
        if auth_expert.execution_config_id != execution_config_id:
            raise M1aAuthorizationError("AuthExpert execution config 不匹配")
        self.auth_expert = auth_expert
        self.execution_config_id = execution_config_id
        self.policy = policy
        self._seal = object()
        self._routes = {}

    def commit(
        self, evidence: M1ABoundEvidence, context: M1AContext
    ) -> M1ACommittedRoute:
        """仅根据受信 evidence 提交一次不可变 P1/P2 route。"""

        self.auth_expert.validate(evidence, context)
        if context.execution_config_id != self.execution_config_id:
            raise M1aAuthorizationError("context execution config 不匹配")
        routes = []
        allowed = []
        for verified, reason in zip(
            evidence.evidence.verified.tolist(), evidence.evidence.reason_code.tolist()
        ):
            if verified:
                routes.append(RouteKind.PROTECTED)
                allowed.append(("E1",))
            elif self.policy == P2_POLICY and reason == int(
                EvidenceReason.RELATION_FAILED
            ):
                routes.append(RouteKind.PUBLIC)
                allowed.append(("E0",))
            else:
                routes.append(RouteKind.DENY)
                allowed.append(())
        route = M1ACommittedRoute(
            tuple(routes), tuple(allowed), context, self.policy, self._seal
        )
        self._routes[id(route)] = route
        return route

    def validate(self, route: M1ACommittedRoute, request_ids: Tuple[str, ...]) -> None:
        """验证 route 来源、policy、完整请求身份和允许集合。"""

        if (
            not isinstance(route, M1ACommittedRoute)
            or self._routes.get(id(route)) is not route
        ):
            raise M1aAuthorizationError("route 未由当前协调器登记")
        if route._seal is not self._seal or route.policy_id != self.policy:
            raise M1aAuthorizationError("route 来源或 policy 不匹配")
        if (
            route.context.request_ids != request_ids
            or route.context.execution_config_id != self.execution_config_id
        ):
            raise M1aAuthorizationError("route 请求或 execution config 不匹配")
        if len(route.routes) != len(request_ids) or len(route.allowed_experts) != len(
            request_ids
        ):
            raise ValueError("route batch 大小非法")
        for kind, experts in zip(route.routes, route.allowed_experts):
            expected = (
                ("E1",)
                if kind is RouteKind.PROTECTED
                else (("E0",) if kind is RouteKind.PUBLIC else ())
            )
            if experts != expected:
                raise M1aAuthorizationError("route allowed experts 被修改")


class FixedConstrainedRouter:
    """在可信 allowed mask 内执行固定 top-1 选择。"""

    def select(self, logits: Tensor, allowed_mask: Tensor) -> M1ASelection:
        """屏蔽越界 logits，并按 E1 优先规则输出选择。"""

        if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("logits 必须为 Tensor[B,2]")
        if (
            not isinstance(allowed_mask, Tensor)
            or allowed_mask.shape != logits.shape
            or allowed_mask.dtype != torch.bool
        ):
            raise ValueError("allowed_mask 必须为匹配的 BoolTensor")
        if logits.device != allowed_mask.device or not logits.dtype.is_floating_point:
            raise ValueError("logits/mask device 或 dtype 不匹配")
        values = []
        for score, mask in zip(logits, allowed_mask):
            if bool(mask[1]):
                values.append("E1")
            elif bool(mask[0]):
                values.append("E0")
            else:
                values.append(None)
        return M1ASelection(tuple(values))
