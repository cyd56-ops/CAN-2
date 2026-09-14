"""M0 contract 的不可变可信对象和稳定错误类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import torch
from torch import Tensor

from ..pretrained_gate.types import RouteKind, VerificationEvidence


class M0AuthorizationError(PermissionError):
    """表示证据、route、scope 或 selection 来源不可信。"""


class M0StateError(RuntimeError):
    """表示 session 或授权对象不在允许生命周期。"""


class ExpertKind(str, Enum):
    """表示业务 Expert 的目录类别。"""

    PUBLIC = "public"
    PROTECTED = "protected"


@dataclass(frozen=True)
class M0ExpertSpec:
    """描述一个冻结的 M0 Expert 槽位。"""

    expert_id: str
    kind: ExpertKind
    enabled: bool

    def __post_init__(self) -> None:
        """验证 Expert 标识和布尔启用状态。"""

        if not isinstance(self.expert_id, str) or not self.expert_id:
            raise ValueError("expert_id 必须是非空字符串")
        if not isinstance(self.kind, ExpertKind):
            raise TypeError("kind 必须是 ExpertKind")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled 必须是 bool")


@dataclass(frozen=True)
class M0ScopeSpec:
    """描述一个 scope 到 Expert ID 的静态映射。"""

    scope_id: str
    expert_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        """验证 scope 非空、唯一且有序。"""

        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise ValueError("scope_id 必须是非空字符串")
        if not isinstance(self.expert_ids, tuple) or not self.expert_ids:
            raise ValueError("expert_ids 必须是非空 tuple")
        if any(not isinstance(value, str) or not value for value in self.expert_ids):
            raise TypeError("expert_ids 必须包含非空字符串")
        if len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("expert_ids 不得重复")


@dataclass(frozen=True)
class M0ExecutionConfig:
    """保存 M0 固定 policy、目录和宿主绑定。"""

    execution_config_id: str
    protocol_id: str
    policy_id: str
    verifier_profile: str
    model_type: str
    model_revision: str
    model_sha256: str
    relation_sha256: str
    num_hidden_layers: int
    cut_layer: int
    cache_mode: str
    max_batch_size: int
    max_sequence_length: int
    experts: Tuple[M0ExpertSpec, ...]
    scopes: Tuple[M0ScopeSpec, ...]

    def __post_init__(self) -> None:
        """严格验证 M0 当前只支持的 P1 配置。"""

        for name in (
            "execution_config_id",
            "protocol_id",
            "policy_id",
            "verifier_profile",
            "model_type",
            "model_revision",
            "model_sha256",
            "relation_sha256",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} 必须是非空字符串")
        if self.protocol_id != "m0-contract-v1":
            raise ValueError("protocol_id 不属于 M0 contract")
        if self.policy_id != "p1-protected-or-deny-v1":
            raise ValueError("M0 只支持 P1 protected-or-deny policy")
        if self.cache_mode not in {"none", "kv"}:
            raise ValueError("cache_mode 必须为 none 或 kv")
        ints = (
            self.num_hidden_layers,
            self.cut_layer,
            self.max_batch_size,
            self.max_sequence_length,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ints):
            raise TypeError("M0 资源限制必须是整数")
        if (
            self.num_hidden_layers < 2
            or not 1 <= self.cut_layer < self.num_hidden_layers
        ):
            raise ValueError("cut_layer 必须位于有效 block 边界")
        if self.max_batch_size < 1 or self.max_sequence_length < 1:
            raise ValueError("M0 资源限制必须为正数")
        if len(self.experts) != 2 or tuple(item.expert_id for item in self.experts) != (
            "E0",
            "E1",
        ):
            raise ValueError("M0 Expert 目录必须严格为 E0/E1")
        if self.experts[0] != M0ExpertSpec("E0", ExpertKind.PUBLIC, False):
            raise ValueError("M0 E0 必须是禁用 public Expert")
        if self.experts[1] != M0ExpertSpec("E1", ExpertKind.PROTECTED, True):
            raise ValueError("M0 E1 必须是启用 protected Expert")
        if len(self.scopes) != 1 or self.scopes[0] != M0ScopeSpec(
            "protected.default", ("E1",)
        ):
            raise ValueError("M0 scope 目录必须严格为 protected.default -> E1")


@dataclass(frozen=True)
class RequestContext:
    """保存服务端生成的 request 身份和单调执行步。"""

    request_ids: Tuple[str, ...]
    execution_config_id: str
    session_id: str
    attempt_id: str
    step_id: int = 0

    def __post_init__(self) -> None:
        """验证请求身份不可为空、重复或被调用方混淆。"""

        if not isinstance(self.request_ids, tuple) or not self.request_ids:
            raise ValueError("request_ids 必须是非空 tuple")
        if any(not isinstance(value, str) or not value for value in self.request_ids):
            raise TypeError("request_ids 必须包含非空字符串")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request_ids 必须唯一")
        for name in ("execution_config_id", "session_id", "attempt_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} 必须是非空字符串")
        if (
            isinstance(self.step_id, bool)
            or not isinstance(self.step_id, int)
            or self.step_id < 0
        ):
            raise ValueError("step_id 必须是非负整数")


@dataclass(frozen=True)
class BoundEvidence:
    """将 G0 evidence 绑定到 AuthExpert 和本次请求。"""

    evidence: VerificationEvidence
    context: RequestContext
    _auth_seal: object = field(repr=False, compare=False)
    _source_token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class CommittedRoute:
    """保存 ScopeCoordinator 唯一提交的 M0 route。"""

    routes: Tuple[RouteKind, ...]
    allowed_experts: Tuple[Tuple[str, ...], ...]
    context: RequestContext
    policy_id: str
    _coordinator_seal: object = field(repr=False, compare=False)
    _source_token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class AllowedExperts:
    """保存当前 route 在某一步的只读 scope view。"""

    expert_ids: Tuple[Tuple[str, ...], ...]
    route: CommittedRoute
    context: RequestContext
    step_id: int
    _registry_seal: object = field(repr=False, compare=False)

    def mask(self, expert_order: Tuple[str, ...] = ("E0", "E1")) -> Tensor:
        """返回新的 BoolTensor mask；调用方修改副本不会改变权限来源。"""

        result = torch.zeros(
            (len(self.expert_ids), len(expert_order)), dtype=torch.bool
        )
        positions = {value: index for index, value in enumerate(expert_order)}
        for row, experts in enumerate(self.expert_ids):
            for expert_id in experts:
                result[row, positions[expert_id]] = True
        return result


@dataclass(frozen=True)
class ExpertSelection:
    """保存 task Router 的逐行选择；M0 只允许 E1 或 None。"""

    values: Tuple[Optional[str], ...]

    def __post_init__(self) -> None:
        """验证 M0 selection 只能包含 E1 或 DENY 的 None。"""

        if not isinstance(self.values, tuple):
            raise TypeError("selection values 必须是 tuple")
        if any(value not in {None, "E1"} for value in self.values):
            raise ValueError("M0 selection 只能为 E1 或 None")


@dataclass(frozen=True)
class M0DispatchResult:
    """保存 M0 sparse dispatch 结果及原始 batch 索引。"""

    protected_output: Optional[Tensor]
    protected_indices: Tensor
    denied_indices: Tensor
    request_ids: Tuple[str, ...]
    route: CommittedRoute

    def __post_init__(self) -> None:
        """验证结果索引和请求身份的基本形状。"""

        for name, value in (
            ("protected_indices", self.protected_indices),
            ("denied_indices", self.denied_indices),
        ):
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.long
                or value.ndim != 1
            ):
                raise TypeError(f"{name} 必须是一维 LongTensor")
        if not isinstance(self.request_ids, tuple) or len(self.request_ids) != len(
            self.route.routes
        ):
            raise ValueError("request_ids 必须与 route 对齐")
        protected = set(self.protected_indices.tolist())
        denied = set(self.denied_indices.tolist())
        if protected & denied or protected | denied != set(
            range(len(self.request_ids))
        ):
            raise ValueError("protected/denied 索引必须互斥且完整覆盖 batch")
