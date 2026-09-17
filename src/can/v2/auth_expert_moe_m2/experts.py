"""M2 冻结 shared/routed tiny experts。"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from .types import M2CallLedger

class _FrozenExpert(nn.Module):
    """带审计计数的冻结两层 FFN。"""
    def __init__(self, d_model: int, seed: int, expert_id: str, kind: str) -> None:
        """按 seed 初始化并冻结权重。"""
        super().__init__(); g=torch.Generator().manual_seed(seed); self.linear1=nn.Linear(d_model,2*d_model,bias=False); self.linear2=nn.Linear(2*d_model,d_model,bias=False)
        with torch.no_grad(): self.linear1.weight.copy_(torch.randn(self.linear1.weight.shape,generator=g)/d_model**0.5); self.linear2.weight.copy_(torch.randn(self.linear2.weight.shape,generator=g)/(2*d_model)**0.5)
        self.expert_id, self.kind, self.forward_calls = expert_id, kind, 0; self.eval()
        for p in self.parameters(): p.requires_grad_(False)
    def forward(self, hidden: Tensor, request_ids: Tuple[str,...], ledger: Optional[M2CallLedger]=None, batch_indices: Optional[Tuple[int,...]]=None, case_id: str="default") -> Tensor:
        """执行冻结 FFN 并记录原始 batch 索引。"""
        if not isinstance(hidden, Tensor) or hidden.ndim != 3 or hidden.shape[-1] != self.linear1.in_features or hidden.dtype != torch.float32: raise ValueError("hidden 形状/dtype 非法")
        if hidden.shape[0] != len(request_ids) or not torch.isfinite(hidden).all(): raise ValueError("hidden 或 request_ids 非法")
        idx=tuple(range(hidden.shape[0])) if batch_indices is None else batch_indices; self.forward_calls += 1
        if ledger is not None: ledger.record(case_id,self.expert_id,self.kind,idx)
        return self.linear2(torch.nn.functional.gelu(self.linear1(hidden)))
class SharedExpert(_FrozenExpert):
    """始终执行的通用 E0。"""
    def __init__(self,d_model:int=16,seed:int=2001)->None: super().__init__(d_model,seed,"E0","shared")
class RoutedExpert(_FrozenExpert):
    """授权后可路由的 E1/E2/E3。"""
    def __init__(self,expert_id:str,seed:int,d_model:int=16)->None:
        if expert_id not in {"E1","E2","E3"}: raise ValueError("未知 routed expert")
        super().__init__(d_model,seed,expert_id,"routed")

class ExpertE0(SharedExpert):
    """Shared E0 的语义别名，便于 manifest/runner 使用。"""

class ExpertE1(RoutedExpert):
    """Routed E1 的语义别名。"""
    def __init__(self,d_model:int=16,seed:int=2002)->None: super().__init__("E1",seed,d_model)

class ExpertE2(RoutedExpert):
    """Routed E2 的语义别名。"""
    def __init__(self,d_model:int=16,seed:int=2003)->None: super().__init__("E2",seed,d_model)

class ExpertE3(RoutedExpert):
    """Routed E3 的语义别名。"""
    def __init__(self,d_model:int=16,seed:int=2004)->None: super().__init__("E3",seed,d_model)
