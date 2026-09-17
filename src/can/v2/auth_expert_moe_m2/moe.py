"""M2 shared + constrained routed 稀疏执行器。"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ..pretrained_gate.types import RouteKind
from .experts import RoutedExpert, SharedExpert
from .router import FixedConstrainedRouter, M2RouteCoordinator, ROUTED_ORDER
from .scope import M2ScopeRegistry
from .types import M2AllowedExperts, M2AuthorizationError, M2CommittedRoute, M2Config, M2Context, M2CallLedger, M2Output

class M2MoE(nn.Module):
    """每行恒执行 E0，并按受信 mask 最多调用一个 routed expert。"""
    def __init__(self,coordinator:M2RouteCoordinator,config:M2Config=M2Config(),experts:Optional[Dict[str,nn.Module]]=None,router:Optional[FixedConstrainedRouter]=None,registry:Optional[M2ScopeRegistry]=None)->None:
        """构造冻结 E0/E1/E2/E3 tiny-MoE。"""
        super().__init__();
        if coordinator.execution_config_id!=config.execution_config_id: raise M2AuthorizationError("coordinator/config 不匹配")
        self.config,self.coordinator,self.router=config,coordinator,router or FixedConstrainedRouter(); self.registry=registry or M2ScopeRegistry(coordinator)
        self.shared=(experts.get("E0") if experts and experts.get("E0") is not None else SharedExpert(config.d_model))
        self.routed=nn.ModuleDict({e:(experts[e] if experts and e in experts else RoutedExpert(e,2000+i,config.d_model)) for i,e in enumerate(ROUTED_ORDER)})
    def forward(self,hidden:Tensor,route:M2CommittedRoute,context:M2Context,ledger:Optional[M2CallLedger]=None,case_id:str="default",view:Optional[M2AllowedExperts]=None)->M2Output:
        """执行 shared 恒等路径和受信 routed 稀疏组合。"""
        if not isinstance(hidden,Tensor) or hidden.ndim!=3 or hidden.shape[-1]!=self.config.d_model or hidden.dtype is not self.config.dtype or hidden.shape[0]==0 or hidden.shape[0]!=len(context.request_ids) or not bool(torch.isfinite(hidden).all()): raise ValueError("hidden 形状/dtype/有限性非法")
        self.coordinator.validate(route,context); view=view or self.registry.resolve(route,context); self.registry.validate(view,route,context)
        ledger=ledger or M2CallLedger("m2-local",self.config.execution_config_id,route.policy_id); all_idx=tuple(range(hidden.shape[0])); shared=self.shared(hidden,context.request_ids,ledger,all_idx,case_id)
        mask=torch.tensor([[e in allowed and kind is RouteKind.PROTECTED for e in ROUTED_ORDER] for kind,allowed in zip(route.routes,view.expert_ids)],dtype=torch.bool,device=hidden.device)
        selection=self.router.select(torch.zeros((hidden.shape[0],3),dtype=hidden.dtype,device=hidden.device),mask)
        routed_by:Dict[str,list[int]]={e:[] for e in ROUTED_ORDER}
        denied=[]
        for i,(kind,chosen) in enumerate(zip(route.routes,selection.expert_ids)):
            if chosen is not None and (kind is not RouteKind.PROTECTED or chosen not in view.expert_ids[i]): raise M2AuthorizationError("M2A_ROUTER_SCOPE_VIOLATION")
            expected=chosen if kind is RouteKind.PROTECTED else None
            if kind is RouteKind.PROTECTED and expected is None: denied.append(i)
            elif kind is RouteKind.DENY: denied.append(i)
            elif kind is RouteKind.PUBLIC: denied.append(i)
            if expected is not None: routed_by[expected].append(i)
        output=shared.clone(); routed_indices=[]
        for expert_id,indices in routed_by.items():
            if not indices: continue
            idx=torch.tensor(indices,dtype=torch.long,device=hidden.device); out=self.routed[expert_id](hidden.index_select(0,idx),tuple(context.request_ids[i] for i in indices),ledger,tuple(indices),case_id); output.index_add_(0,idx,self.config.alpha*F.normalize(out,p=2.0,dim=-1,eps=self.config.normalize_eps)); routed_indices.extend(indices)
        return M2Output(output,shared,selection,torch.tensor(sorted(routed_indices),dtype=torch.long,device=hidden.device),torch.tensor(sorted(set(denied)),dtype=torch.long,device=hidden.device))
