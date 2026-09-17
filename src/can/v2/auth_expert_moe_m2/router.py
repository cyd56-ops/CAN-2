"""M2 确定性 constrained top-1 Router 与授权协调器。"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from ..pretrained_gate.types import EvidenceReason, RouteKind
from .protocols import M2VerifierProtocol
from .types import *
P1_POLICY="p1-protected-or-deny-v1"; P2_POLICY="p2-capability-routing-v1"; ROUTED_ORDER=("E1","E2","E3")

class M2AuthExpert:
    """调用 A0 verifier 并绑定受信 scope assignment。"""
    def __init__(self,verifier:M2VerifierProtocol,execution_config_id:str)->None:
        """绑定 verifier 和配置。"""
        if not isinstance(verifier,M2VerifierProtocol): raise TypeError("verifier 类型错误")
        self.verifier,self.execution_config_id,self._seal=verifier,execution_config_id,object(); self._issued: Dict[int,M2BoundEvidence]={}; self._assignments: Dict[int,Tuple[M2ScopeAssignment,object]]={}
    def _register_assignment(self, assignment:M2ScopeAssignment, registry_seal:object)->None:
        """登记由绑定 registry 签发的 assignment，拒绝调用方伪造对象。"""
        if not isinstance(assignment,M2ScopeAssignment) or assignment._registry_seal is not registry_seal: raise M2AuthorizationError("M2A_SCOPE_GRANT_SOURCE_MISMATCH")
        self._assignments[id(assignment)]=(assignment,registry_seal)
    def verify(self,credential:Tensor,context:M2Context,assignment:M2ScopeAssignment)->M2BoundEvidence:
        """生成与 context/assignment 绑定的 evidence。"""
        registered=self._assignments.get(id(assignment))
        if registered is None or registered[0] is not assignment or assignment._registry_seal is not registered[1]: raise M2AuthorizationError("M2A_SCOPE_GRANT_SOURCE_MISMATCH")
        if context.execution_config_id!=self.execution_config_id or assignment.execution_config_id!=self.execution_config_id or assignment.request_ids!=context.request_ids: raise M2AuthorizationError("绑定不匹配")
        evidence=self.verifier(credential)
        if evidence.verified.shape[0]!=len(context.request_ids): raise ValueError("batch 不匹配")
        bound=M2BoundEvidence(evidence,context,assignment,self._seal); self._issued[id(bound)]=bound; return bound
    def validate(self,bound:M2BoundEvidence,context:M2Context)->None:
        """验证 evidence 来源、完整性和上下文。"""
        if self._issued.get(id(bound)) is not bound or bound._auth_seal is not self._seal or bound.context!=context: raise M2AuthorizationError("evidence 来源/绑定错误")
        self.verifier.validate_evidence(bound.evidence)

class M2RouteCoordinator:
    """依据 evidence 和 fixture assignment 提交 P1/P2 route。"""
    def __init__(self,auth:M2AuthExpert,execution_config_id:str,policy:str=P1_POLICY)->None:
        """绑定认证器、配置和 policy。"""
        if policy not in {P1_POLICY,P2_POLICY} or auth.execution_config_id!=execution_config_id: raise ValueError("M2 policy/config 非法")
        self.auth,self.execution_config_id,self.policy,self._seal=auth,execution_config_id,policy,object(); self._routes={}
    def commit(self,bound:M2BoundEvidence,context:M2Context)->M2CommittedRoute:
        """仅由受信 evidence/assignment 产生 route。"""
        self.auth.validate(bound,context); a=bound.assignment
        if a.policy_id!=self.policy or not a.case_id: raise M2AuthorizationError("assignment policy 不匹配")
        routes=[]; allowed=[]
        for verified,reason in zip(bound.evidence.verified.tolist(),bound.evidence.reason_code.tolist()):
            if verified and a.scope_id is not None: routes.append(RouteKind.PROTECTED); allowed.append(tuple(_scope_experts(a.scope_id)))
            elif self.policy==P2_POLICY and reason==int(EvidenceReason.RELATION_FAILED): routes.append(RouteKind.PUBLIC); allowed.append(("E0",))
            else: routes.append(RouteKind.DENY); allowed.append(())
        route=M2CommittedRoute(tuple(routes),tuple(allowed),context,self.policy,a.case_id,self._seal); self._routes[id(route)]=route; return route
    def validate(self,route:M2CommittedRoute,context:M2Context)->None:
        """验证 route 来源、policy、上下文和 allowed 集合。"""
        if self._routes.get(id(route)) is not route or route._seal is not self._seal or route.policy_id!=self.policy or route.context!=context: raise M2AuthorizationError("route 绑定错误")
        if len(route.routes)!=len(context.request_ids): raise ValueError("route batch 非法")
        for kind,allowed in zip(route.routes,route.allowed_experts):
            if kind is RouteKind.DENY and allowed: raise M2AuthorizationError("DENY 不得有 expert")
            if kind is RouteKind.PUBLIC and allowed!=("E0",): raise M2AuthorizationError("PUBLIC 集合非法")
            if kind is RouteKind.PROTECTED and (not allowed or any(x not in ROUTED_ORDER for x in allowed)): raise M2AuthorizationError("protected 集合非法")

def _scope_experts(scope:str)->Tuple[str,...]:
    """返回冻结 scope 的有序 routed 集合。"""
    values={"standard":("E1",),"advanced":("E1","E2"),"privileged":("E1","E2","E3")}
    if scope not in values: raise M2AuthorizationError("未知 scope")
    return values[scope]

class FixedConstrainedRouter:
    """在 [E1,E2,E3] allowed mask 中确定性选择 top-1。"""
    def select(self,logits:Tensor,allowed_mask:Tensor)->M2Selection:
        """屏蔽越界位置；零 logits 时选择最小 allowed 索引。"""
        if not isinstance(logits,Tensor) or logits.ndim!=2 or logits.shape[1]!=3 or not logits.dtype.is_floating_point: raise ValueError("logits 必须为 Tensor[B,3]")
        if not isinstance(allowed_mask,Tensor) or allowed_mask.shape!=logits.shape or allowed_mask.dtype is not torch.bool: raise ValueError("allowed_mask 非法")
        if logits.device != allowed_mask.device or not bool(torch.isfinite(logits).all().item()): raise ValueError("logits device 或有限性非法")
        values=[]
        for score,mask in zip(logits,allowed_mask):
            if not bool(mask.any()): values.append(None); continue
            masked=score.masked_fill(~mask,float("-inf")); values.append(ROUTED_ORDER[int(torch.argmax(masked).item())])
        return M2Selection(tuple(values))
