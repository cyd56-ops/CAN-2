"""M0 AuthExpert 与唯一 scope route 协调器。"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from ..pretrained_gate.authorization import FixedRelationVerifier
from ..pretrained_gate.types import RouteKind
from .types import (
    BoundEvidence,
    CommittedRoute,
    M0AuthorizationError,
    M0ExecutionConfig,
    RequestContext,
)


class AuthExpert:
    """包装 G0 固定 verifier，只生成绑定 evidence。"""

    def __init__(
        self, verifier: FixedRelationVerifier, config: M0ExecutionConfig
    ) -> None:
        """绑定不可替换的 verifier 和 M0 execution config。"""

        if not isinstance(verifier, FixedRelationVerifier):
            raise TypeError("verifier 必须是 FixedRelationVerifier")
        if not isinstance(config, M0ExecutionConfig):
            raise TypeError("config 必须是 M0ExecutionConfig")
        if verifier.profile_id != config.verifier_profile:
            raise M0AuthorizationError("verifier profile 与 M0 config 不匹配")
        self.verifier = verifier
        self.config = config
        self._seal = object()
        self._issued: Dict[int, RequestContext] = {}

    def validate_input(self, credential: Tensor, batch_size: int) -> None:
        """执行请求级 credential 结构检查，不产生业务调用。"""

        self.verifier.validate_credential_structure(credential)
        if credential.shape[0] != batch_size:
            raise ValueError("credential batch 与请求 batch 不一致")

    def verify(self, credential: Tensor, context: RequestContext) -> BoundEvidence:
        """委托 G0 verifier 生成一次绑定 evidence。"""

        if not isinstance(context, RequestContext):
            raise TypeError("context 必须是 RequestContext")
        self.validate_input(credential, len(context.request_ids))
        evidence = self.verifier(credential)
        bound = BoundEvidence(evidence, context, self._seal, object())
        self._issued[id(bound)] = context
        return bound

    def validate_evidence(
        self, evidence: BoundEvidence, context: RequestContext
    ) -> None:
        """拒绝复制、替换、跨请求或重复提交的 evidence。"""

        if not isinstance(evidence, BoundEvidence):
            raise TypeError("evidence 必须是 BoundEvidence")
        if id(evidence) not in self._issued or self._issued[id(evidence)] != context:
            raise M0AuthorizationError("evidence 未由当前 AuthExpert 为本请求签发")
        if evidence._auth_seal is not self._seal or evidence.context != context:
            raise M0AuthorizationError("evidence 来源或上下文不匹配")
        self.verifier.validate_evidence(evidence.evidence)

    def revoke(self, evidence: BoundEvidence) -> None:
        """撤销本次 session 签发的 evidence，使其不能再次提交。"""

        if not isinstance(evidence, BoundEvidence):
            raise TypeError("evidence 必须是 BoundEvidence")
        if evidence._auth_seal is not self._seal:
            raise M0AuthorizationError("evidence 来源不匹配")
        self._issued.pop(id(evidence), None)


class ScopeCoordinator:
    """将 AuthExpert evidence 提交为唯一 M0 protected/deny route。"""

    def __init__(self, auth_expert: AuthExpert, config: M0ExecutionConfig) -> None:
        """绑定 AuthExpert、固定 policy 和 scope 目录。"""

        if not isinstance(auth_expert, AuthExpert):
            raise TypeError("auth_expert 必须是 AuthExpert")
        if auth_expert.config != config:
            raise M0AuthorizationError("AuthExpert 与 config 不匹配")
        self.auth_expert = auth_expert
        self.config = config
        self._seal = object()
        self._committed: Dict[int, CommittedRoute] = {}
        self._submitted_evidence: set[int] = set()
        self._route_evidence: Dict[int, int] = {}

    def commit(
        self, evidence: BoundEvidence, context: RequestContext
    ) -> CommittedRoute:
        """仅接受本 session 的真实 evidence，并提交一次 route。"""

        self.auth_expert.validate_evidence(evidence, context)
        if context.execution_config_id != self.config.execution_config_id:
            raise M0AuthorizationError("context execution config 不匹配")
        if id(evidence) in self._submitted_evidence:
            raise M0AuthorizationError("evidence 已提交过 route")
        routes = tuple(
            RouteKind.PROTECTED if bool(value) else RouteKind.DENY
            for value in evidence.evidence.verified.tolist()
        )
        allowed = tuple(
            ("E1",) if route is RouteKind.PROTECTED else () for route in routes
        )
        route = CommittedRoute(
            routes, allowed, context, self.config.policy_id, self._seal, object()
        )
        self._submitted_evidence.add(id(evidence))
        self._committed[id(route)] = route
        self._route_evidence[id(route)] = id(evidence)
        return route

    def revoke(self, route: CommittedRoute) -> None:
        """撤销已登记 route，并释放其 evidence 提交记录。"""

        if not isinstance(route, CommittedRoute):
            raise TypeError("route 必须是 CommittedRoute")
        if route._coordinator_seal is not self._seal:
            raise M0AuthorizationError("route 来源协调器不匹配")
        self._committed.pop(id(route), None)
        evidence_id = self._route_evidence.pop(id(route), None)
        if evidence_id is not None:
            self._submitted_evidence.discard(evidence_id)

    def validate(self, route: CommittedRoute, context: RequestContext) -> None:
        """验证 route 必须是本协调器实际登记的对象。"""

        if not isinstance(route, CommittedRoute):
            raise TypeError("route 必须是 CommittedRoute")
        if route._coordinator_seal is not self._seal:
            raise M0AuthorizationError("route 来源协调器不匹配")
        registered = self._committed.get(id(route))
        if registered is not route:
            raise M0AuthorizationError("route 不是当前协调器登记对象")
        route_context = route.context
        same_request = (
            route_context.request_ids == context.request_ids
            and route_context.execution_config_id == context.execution_config_id
            and route_context.session_id == context.session_id
            and route_context.attempt_id == context.attempt_id
            and context.step_id >= route_context.step_id
        )
        if not same_request or route.policy_id != self.config.policy_id:
            raise M0AuthorizationError("route 上下文或 policy 不匹配")
        if len(route.routes) != len(context.request_ids):
            raise ValueError("route batch 大小不匹配")
        for route_kind, experts in zip(route.routes, route.allowed_experts):
            expected = ("E1",) if route_kind is RouteKind.PROTECTED else ()
            if experts != expected:
                raise M0AuthorizationError("route allowed experts 被修改")
