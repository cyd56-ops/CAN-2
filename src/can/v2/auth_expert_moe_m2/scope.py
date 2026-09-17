"""M2 scope lattice、assignment registry 与只能收窄 view。"""
from __future__ import annotations
from typing import Dict, Iterable, Mapping, Optional, Tuple
from .types import M2AllowedExperts, M2AuthorizationError, M2CommittedRoute, M2Context, M2ScopeAssignment
from .router import M2RouteCoordinator, _scope_experts

SCOPES=("standard","advanced","privileged")
PARENTS={"standard":(),"advanced":("standard",),"privileged":("advanced",)}

def validate_scope_lattice(scopes: Iterable[Mapping[str, object]], expert_ids: Iterable[str] = ("E0","E1","E2","E3")) -> Dict[str,Tuple[str,...]]:
    """严格验证 scope 拓扑、环和专家集合单调性。"""
    items=list(scopes); ids=[]; mapping={}
    for item in items:
        if not isinstance(item,Mapping) or set(item)!={"scope_id","parent_scope_ids","expert_ids"}: raise ValueError("scope schema 非法")
        sid=item["scope_id"]; parents=item["parent_scope_ids"]; experts=item["expert_ids"]
        if not isinstance(sid,str) or sid in mapping: raise ValueError("scope ID 重复/非法")
        if not isinstance(parents,list) or not isinstance(experts,list) or len(set(experts))!=len(experts): raise ValueError("scope 列表非法")
        ids.append(sid); mapping[sid]=(tuple(parents),tuple(experts))
    # 先独立检查 parent 图，确保环以稳定的 M2A_SCOPE_CYCLE 错误暴露。
    visiting=set(); visited=set()
    def visit(node:str)->None:
        """深度优先检查 scope parent 图无环。"""
        if node in visiting: raise M2AuthorizationError("M2A_SCOPE_CYCLE")
        if node in visited: return
        visiting.add(node)
        for parent in mapping[node][0]:
            if parent not in mapping: raise M2AuthorizationError("M2A_SCOPE_PARENT_MISSING")
            visit(parent)
        visiting.remove(node); visited.add(node)
    for node in ids: visit(node)
    if tuple(ids)!=SCOPES: raise ValueError("scope 必须按固定拓扑顺序声明")
    enabled=set(expert_ids)
    for sid in SCOPES:
        parents, experts=mapping[sid]
        if any(p not in mapping or ids.index(p)>=ids.index(sid) for p in parents): raise M2AuthorizationError("M2A_SCOPE_PARENT_MISSING")
        if any(e not in enabled or e=="E0" or e not in {"E1","E2","E3"} for e in experts): raise M2AuthorizationError("M2A_SCOPE_EXPERT_INVALID")
        if set(experts)!=set(_scope_experts(sid)): raise M2AuthorizationError("M2A_SCOPE_NON_MONOTONIC")
    return {sid: mapping[sid][1] for sid in SCOPES}

class M2ScopeRegistry:
    """由受信 fixture assignment 签发 scope view，禁止权限扩大。"""
    def __init__(self, coordinator:M2RouteCoordinator, assignments:Optional[Mapping[str,Optional[str]]]=None) -> None:
        """绑定协调器并加载受信 case→scope 注册表。"""
        if not isinstance(coordinator,M2RouteCoordinator): raise TypeError("coordinator 类型错误")
        self.coordinator=coordinator; self._seal=object(); self._views={}; self._assignments=dict(assignments or {})
    def assignment(self,case_id:str,context:M2Context)->M2ScopeAssignment:
        """为冻结 case 生成绑定上下文的内部 assignment。"""
        if case_id not in self._assignments: raise M2AuthorizationError("M2A_SCOPE_GRANT_SOURCE_MISMATCH")
        sid=self._assignments[case_id]
        if sid is not None and sid not in SCOPES: raise M2AuthorizationError("M2A_SCOPE_UNKNOWN")
        assignment=M2ScopeAssignment(case_id,sid,context.request_ids,self.coordinator.policy,self.coordinator.execution_config_id,self._seal)
        self.coordinator.auth._register_assignment(assignment,self._seal)
        return assignment
    def resolve(self,route:M2CommittedRoute,context:M2Context)->M2AllowedExperts:
        """将 route 转成完整逐行 allowed view。"""
        self.coordinator.validate(route,context); view=M2AllowedExperts(route.allowed_experts,route,context,self._seal); self._views[id(view)]=view; return view
    def restrict(self,view:M2AllowedExperts,requested:Tuple[Tuple[str,...],...])->M2AllowedExperts:
        """只允许从现有集合收窄。"""
        self.validate(view,view.route,view.context)
        if not isinstance(requested,tuple) or len(requested)!=len(view.expert_ids): raise ValueError("requested batch 非法")
        out=[]
        for original,narrow in zip(view.expert_ids,requested):
            if not isinstance(narrow,tuple) or len(set(narrow))!=len(narrow) or any(e not in original for e in narrow): raise M2AuthorizationError("scope 不能扩大")
            out.append(narrow)
        result=M2AllowedExperts(tuple(out),view.route,view.context,self._seal); self._views[id(result)]=result; return result
    def validate(self,view:M2AllowedExperts,route:M2CommittedRoute,context:M2Context)->None:
        """验证 view 登记、seal 与 route/context 绑定。"""
        if self._views.get(id(view)) is not view or view._registry_seal is not self._seal or view.route is not route or view.context!=context: raise M2AuthorizationError("scope view 绑定错误")
