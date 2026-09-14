"""冻结预训练宿主 Gate 的基础类型。

这些类型只描述可信进程内的证据、路由和调用结果，不是外部 wire schema，
也不提供白盒防伪能力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional, Tuple

import torch
from torch import Tensor


class EvidenceReason(IntEnum):
    """表示固定关系验证的稳定逐行结果码。"""

    SUCCESS = 0
    RELATION_FAILED = 1
    NON_FINITE = 2
    NUMERICAL_FAILURE = 3


class RouteKind(str, Enum):
    """表示协调器提交的业务执行路径。"""

    PROTECTED = "protected"
    PUBLIC = "public"
    DENY = "deny"


@dataclass(frozen=True)
class VerificationEvidence:
    """保存 verifier 产生的批量证据。

    参数:
        verified: 每行是否满足冻结关系，shape 为 ``[B]``。
        error_norm: 每行的 FP32 误差范数，shape 为 ``[B]``。
        reason_code: 稳定结果码，shape 为 ``[B]``。

    ``_verifier_seal`` 和 ``_integrity_tag`` 仅用于可信进程内来源与篡改检查。
    """

    verified: Tensor
    error_norm: Tensor
    reason_code: Tensor
    _verifier_seal: object = field(repr=False, compare=False)
    _integrity_tag: str = field(repr=False, compare=False)


@dataclass(frozen=True)
class TrustedRequestContext:
    """保存服务端建立的请求身份和执行配置。

    参数:
        request_ids: 服务端生成的非空、有序且唯一的请求身份。
        execution_config_id: 绑定模型、cut、后端和 verifier profile 的配置 ID。
    """

    request_ids: Tuple[str, ...]
    execution_config_id: str

    def __post_init__(self) -> None:
        """验证请求身份与配置 ID 的结构。"""

        if not isinstance(self.request_ids, tuple) or not self.request_ids:
            raise ValueError("request_ids 必须是非空 tuple")
        if any(not isinstance(value, str) or not value for value in self.request_ids):
            raise TypeError("每个 request ID 必须是非空字符串")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("同一 batch 的 request IDs 必须唯一")
        if (
            not isinstance(self.execution_config_id, str)
            or not self.execution_config_id
        ):
            raise ValueError("execution_config_id 必须是非空字符串")


@dataclass(frozen=True)
class _CommittedRoute:
    """保存协调器提交的进程内路由，禁止调用方自行构造。"""

    routes: Tuple[RouteKind, ...]
    policy_id: str
    request_ids: Tuple[str, ...]
    execution_config_id: str
    _coordinator_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class DispatchResult:
    """保存一次 P1 protected/deny 调度的稀疏结果。

    参数:
        protected_output: 仅包含 PROTECTED 行的宿主输出；全拒绝时为 ``None``。
        protected_indices: PROTECTED 行在原 batch 中的索引。
        denied_indices: DENY 行在原 batch 中的索引。
        request_ids: 原 batch 的完整有序身份。
    """

    protected_output: Optional[Any]
    protected_indices: Tensor
    denied_indices: Tensor
    request_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        """验证调度索引的基本 Tensor 契约。"""

        for name, value in (
            ("protected_indices", self.protected_indices),
            ("denied_indices", self.denied_indices),
        ):
            if not isinstance(value, Tensor) or value.dtype != torch.long:
                raise TypeError(f"{name} 必须是 LongTensor")
            if value.ndim != 1:
                raise ValueError(f"{name} 必须是一维 Tensor")


@dataclass(frozen=True)
class CallRecord:
    """记录一次真实模块调用涉及的请求身份。"""

    component: str
    stage: str
    request_ids: Tuple[str, ...]


@dataclass(frozen=True)
class FailureRecord:
    """保存脱敏的整批运行失败信息与实际调用历史。"""

    stage: str
    reason: str
    calls: Tuple[CallRecord, ...]
