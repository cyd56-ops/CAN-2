"""M2 多 routed experts 的可信类型定义。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from ..pretrained_gate.types import RouteKind, VerificationEvidence

class M2AuthorizationError(PermissionError):
    """表示 M2 对象来源、绑定或 scope 校验失败。"""

@dataclass(frozen=True)
class M2Config:
    """M2 固定 CPU tiny-MoE 执行配置。"""
    execution_config_id: str = "m2-multi-expert-v1"
    protocol_id: str = "m2-scope-lattice-v1"
    d_model: int = 16
    top_k: int = 1
    alpha: float = 1.0
    normalize_eps: float = 1e-12
    dtype: torch.dtype = torch.float32
    def __post_init__(self) -> None:
        """验证不可变配置。"""
        if not self.execution_config_id or not self.protocol_id: raise ValueError("配置 ID 不能为空")
        if isinstance(self.d_model, bool) or not isinstance(self.d_model, int) or self.d_model < 1: raise ValueError("d_model 非法")
        if self.top_k != 1 or self.alpha != 1.0 or self.normalize_eps != 1e-12: raise ValueError("M2 冻结参数不匹配")
        if self.dtype is not torch.float32: raise TypeError("M2 仅支持 float32")

@dataclass(frozen=True)
class M2Context:
    """保存服务端生成的请求身份。"""
    request_ids: Tuple[str, ...]
    execution_config_id: str
    def __post_init__(self) -> None:
        """验证请求 ID 唯一且非空。"""
        if not isinstance(self.request_ids, tuple) or not self.request_ids or any(not isinstance(x, str) or not x for x in self.request_ids): raise ValueError("request_ids 非法")
        if len(set(self.request_ids)) != len(self.request_ids): raise ValueError("request_ids 必须唯一")

@dataclass(frozen=True)
class M2ScopeAssignment:
    """受信 fixture registry 签发的 scope assignment。"""
    case_id: str
    scope_id: Optional[str]
    request_ids: Tuple[str, ...]
    policy_id: str
    execution_config_id: str
    _registry_seal: object = field(repr=False, compare=False)

@dataclass(frozen=True)
class M2BoundEvidence:
    """将 verifier evidence 与请求和受信 assignment 绑定。"""
    evidence: VerificationEvidence
    context: M2Context
    assignment: M2ScopeAssignment
    _auth_seal: object = field(repr=False, compare=False)

@dataclass(frozen=True)
class M2CommittedRoute:
    """协调器提交的不可变逐行 route。"""
    routes: Tuple[RouteKind, ...]
    allowed_experts: Tuple[Tuple[str, ...], ...]
    context: M2Context
    policy_id: str
    case_id: str
    _seal: object = field(repr=False, compare=False)

@dataclass(frozen=True)
class M2AllowedExperts:
    """scope registry 导出的只能收窄 view。"""
    expert_ids: Tuple[Tuple[str, ...], ...]
    route: M2CommittedRoute
    context: M2Context
    _registry_seal: object = field(repr=False, compare=False)

@dataclass(frozen=True)
class M2Selection:
    """保存逐行 routed 选择；None 表示 zero-call。"""
    expert_ids: Tuple[Optional[str], ...]

@dataclass(frozen=True)
class M2Output:
    """保存 MoE 输出和原始 batch 索引。"""
    output: Tensor
    shared_output: Tensor
    selection: M2Selection
    routed_indices: Tensor
    denied_indices: Tensor

@dataclass(frozen=True)
class M2CallEvent:
    """保存一次真实 Expert forward 的审计事件。"""
    sequence: int; case_id: str; stage: str; expert_id: str; kind: str; batch_indices: Tuple[int, ...]; count: int

class M2CallLedger:
    """记录真实 shared/routed 调用，不保存 hidden 或 credential。"""
    def __init__(self, run_id: str, execution_config_id: str, policy: str) -> None:
        """创建绑定运行标识的空台账。"""
        self.run_id, self.execution_config_id, self.policy = run_id, execution_config_id, policy; self._events: List[M2CallEvent] = []
    def record(self, case_id: str, expert_id: str, kind: str, indices: Tuple[int, ...]) -> None:
        """登记一次真实 forward。"""
        if not indices or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices): raise ValueError("调用索引非法")
        self._events.append(M2CallEvent(len(self._events), case_id, "forward", expert_id, kind, indices, 1))
    @property
    def events(self) -> Tuple[M2CallEvent, ...]:
        """返回不可变事件快照。"""
        return tuple(self._events)
    def to_dict(self) -> Dict[str, object]:
        """序列化为 JSON 兼容对象。"""
        return {"schema_version":1,"run_id":self.run_id,"execution_config_id":self.execution_config_id,"policy":self.policy,"events":[e.__dict__.copy() for e in self._events]}
