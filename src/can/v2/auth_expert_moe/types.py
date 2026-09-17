"""M1a tiny-MoE 的可信类型、路由和调用台账。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..pretrained_gate.types import RouteKind, VerificationEvidence


class M1aAuthorizationError(PermissionError):
    """表示 M1a route、scope 或 selection 来源不可信。"""


@dataclass(frozen=True)
class M1AConfig:
    """固定 M1a tiny-MoE 的执行配置。"""

    execution_config_id: str = "m1a-tiny-moe-v1"
    protocol_id: str = "m1a-contract-v1"
    d_model: int = 16
    top_k: int = 1
    alpha: float = 1.0
    normalize_eps: float = 1e-12
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        """验证 M1a 的固定数值和路由参数。"""

        if not self.execution_config_id or not self.protocol_id:
            raise ValueError("execution_config_id/protocol_id 不能为空")
        if (
            isinstance(self.d_model, bool)
            or not isinstance(self.d_model, int)
            or self.d_model < 1
        ):
            raise ValueError("d_model 必须为正整数")
        if self.top_k != 1:
            raise ValueError("M1a 只支持 top_k=1")
        if not torch.is_floating_point(torch.empty((), dtype=self.dtype)):
            raise TypeError("M1a dtype 必须为浮点 dtype")
        if self.alpha != 1.0 or self.normalize_eps != 1e-12:
            raise ValueError("M1a alpha/normalize_eps 必须使用冻结值")


@dataclass(frozen=True)
class M1AContext:
    """保存服务端生成的 M1a 请求身份。"""

    request_ids: Tuple[str, ...]
    execution_config_id: str

    def __post_init__(self) -> None:
        """验证 request ID 唯一且与 execution config 绑定。"""

        if not self.request_ids or any(
            not isinstance(v, str) or not v for v in self.request_ids
        ):
            raise ValueError("request_ids 必须是非空字符串 tuple")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request_ids 必须唯一")
        if (
            not isinstance(self.execution_config_id, str)
            or not self.execution_config_id
        ):
            raise ValueError("execution_config_id 必须是非空字符串")


@dataclass(frozen=True)
class M1ABoundEvidence:
    """将固定 verifier evidence 绑定到本次 M1a 请求。"""

    evidence: VerificationEvidence
    context: M1AContext
    _auth_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class M1ACommittedRoute:
    """由 M1a 协调器提交、调用方不能自行伪造的 route。"""

    routes: Tuple[RouteKind, ...]
    allowed_experts: Tuple[Tuple[str, ...], ...]
    context: M1AContext
    policy_id: str
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class M1AAllowedExperts:
    """保存由可信 scope registry 导出的只读允许集合。"""

    expert_ids: Tuple[Tuple[str, ...], ...]
    route: M1ACommittedRoute
    context: M1AContext
    _registry_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class M1ASelection:
    """保存每个 batch 行的 routed 选择；None 表示 routed zero-call。"""

    values: Tuple[Optional[str], ...]

    def __post_init__(self) -> None:
        """验证 M1a 选择只包含 E0、E1 或 routed zero-call。"""

        if not isinstance(self.values, tuple) or any(
            value not in {None, "E0", "E1"} for value in self.values
        ):
            raise ValueError("M1a selection 只能包含 E0、E1 或 None")


@dataclass(frozen=True)
class CallEvent:
    """保存一次 Expert 实际 forward 的稳定审计事件。"""

    sequence: int
    case_id: str
    stage: str
    expert_id: str
    kind: str
    batch_indices: Tuple[int, ...]
    count: int


class CallLedger:
    """记录真实 E0/E1 调用，不保存 hidden、credential 或 logits。"""

    def __init__(self, run_id: str, execution_config_id: str, policy: str) -> None:
        """创建空台账并绑定运行标识。"""

        self.run_id = run_id
        self.execution_config_id = execution_config_id
        self.policy = policy
        self._events: List[CallEvent] = []

    def record(
        self, case_id: str, expert_id: str, kind: str, indices: Tuple[int, ...]
    ) -> None:
        """追加一次真实 Expert 调用事件。"""

        if not indices:
            raise ValueError("调用事件 batch_indices 不能为空")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise ValueError("batch_indices 必须包含非负整数")
        # 一个事件代表一次模块 forward；batch_indices 只描述该次调用覆盖的原始行。
        self._events.append(
            CallEvent(
                len(self._events), case_id, "forward", expert_id, kind, indices, 1
            )
        )

    @property
    def events(self) -> Tuple[CallEvent, ...]:
        """返回不可变事件快照。"""

        return tuple(self._events)

    def to_dict(self) -> Dict[str, object]:
        """序列化为严格 JSON 兼容的调用台账对象。"""

        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "execution_config_id": self.execution_config_id,
            "policy": self.policy,
            "events": [event.__dict__.copy() for event in self._events],
        }


@dataclass(frozen=True)
class M1AOutput:
    """保存 MoE 输出、原始索引和本次 routed selection。"""

    output: Tensor
    shared_output: Tensor
    routed_indices: Tensor
    denied_indices: Tensor
    selection: M1ASelection
