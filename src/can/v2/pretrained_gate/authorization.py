"""固定关系 verifier 与唯一授权协调器。

本模块不复用历史软 Gate。所有模式下均执行 FP32 硬判定，verifier 只产生
evidence，只有 ``RouteCoordinator`` 能提交受信 route。
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .types import (
    EvidenceReason,
    RouteKind,
    TrustedRequestContext,
    VerificationEvidence,
    _CommittedRoute,
)

P1_POLICY_ID = "p1-protected-or-deny-v1"
TOY_REAL_FP32_PROFILE = "toy-real-fp32-v1"


def _require_nonempty_id(value: object, name: str) -> str:
    """验证并返回非空字符串标识符。"""

    if not isinstance(value, str):
        raise TypeError(f"{name} 必须是字符串")
    if not value:
        raise ValueError(f"{name} 不能为空")
    return value


def _evidence_integrity_tag(
    verified: Tensor,
    error_norm: Tensor,
    reason_code: Tensor,
    seal: object,
) -> str:
    """计算可信进程内 evidence 完整性标签。

    标签用于检测 evidence 产生后被原地改写；它不是密码学授权令牌，也不抵抗
    能修改 Python 运行时的白盒攻击者。
    """

    digest = hashlib.sha256()
    digest.update(str(id(seal)).encode("ascii"))
    for value in (verified, error_norm, reason_code):
        detached = value.detach().to(device="cpu").contiguous()
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(str(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


class FixedRelationVerifier(nn.Module):
    """执行 ``||Ac-b|| < threshold`` 的固定 FP32 批量判定。"""

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        error_threshold: float,
        profile_id: str = TOY_REAL_FP32_PROFILE,
    ) -> None:
        """冻结关系参数并建立 evidence 来源 seal。

        参数:
            A: FP32 公共矩阵，shape 为 ``[m, n]``。
            b: FP32 公共向量，shape 为 ``[m]``。
            error_threshold: 严格小于判据使用的有限正阈值。
            profile_id: 输入域和数值规则的稳定标识。
        """

        super().__init__()
        if not isinstance(A, np.ndarray) or A.dtype != np.float32 or A.ndim != 2:
            raise TypeError("A 必须是二维 np.float32 数组")
        if not isinstance(b, np.ndarray) or b.dtype != np.float32 or b.ndim != 1:
            raise TypeError("b 必须是一维 np.float32 数组")
        if A.shape[0] != b.shape[0] or A.shape[0] == 0 or A.shape[1] == 0:
            raise ValueError("A/b shape 不满足非空 [m,n]/[m]")
        if not np.isfinite(A).all() or not np.isfinite(b).all():
            raise ValueError("A/b 必须全部为有限值")
        if isinstance(error_threshold, bool) or not isinstance(error_threshold, Real):
            raise TypeError("error_threshold 必须是实数")
        threshold = float(error_threshold)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("error_threshold 必须是有限正数")

        self.profile_id = _require_nonempty_id(profile_id, "profile_id")
        self.error_threshold = threshold
        self.n = int(A.shape[1])
        self.m = int(A.shape[0])
        self.register_buffer("A", torch.from_numpy(A.copy()))
        self.register_buffer("b", torch.from_numpy(b.copy()))
        self._evidence_seal = object()

    def forward(self, credential: Tensor) -> VerificationEvidence:
        """验证严格规范化的 ``float32 Tensor[B,n]`` credential。

        结构错误整批抛出异常；shape 合法的非有限行和数值溢出行生成逐行拒绝证据。
        """

        self.validate_credential_structure(credential)
        finite_rows = torch.isfinite(credential).all(dim=1)
        safe = torch.where(
            finite_rows[:, None], credential, torch.zeros_like(credential)
        )

        # 禁用 autocast，认证关系始终使用已冻结的 FP32 算术。
        with torch.autocast(device_type=credential.device.type, enabled=False):
            residual = torch.matmul(safe, self.A.transpose(0, 1)) - self.b
            computed_norm = torch.linalg.vector_norm(residual, ord=2, dim=1)
        numerical_rows = finite_rows & torch.isfinite(computed_norm)
        error_norm = torch.where(
            numerical_rows,
            computed_norm,
            torch.full_like(computed_norm, float("inf")),
        )
        verified = numerical_rows & (error_norm < self.error_threshold)
        reason_code = torch.full(
            (credential.shape[0],),
            int(EvidenceReason.RELATION_FAILED),
            dtype=torch.long,
            device=credential.device,
        )
        reason_code = torch.where(
            verified,
            torch.full_like(reason_code, int(EvidenceReason.SUCCESS)),
            reason_code,
        )
        reason_code = torch.where(
            finite_rows,
            reason_code,
            torch.full_like(reason_code, int(EvidenceReason.NON_FINITE)),
        )
        reason_code = torch.where(
            finite_rows & ~numerical_rows,
            torch.full_like(reason_code, int(EvidenceReason.NUMERICAL_FAILURE)),
            reason_code,
        )
        tag = _evidence_integrity_tag(
            verified, error_norm, reason_code, self._evidence_seal
        )
        return VerificationEvidence(
            verified=verified,
            error_norm=error_norm,
            reason_code=reason_code,
            _verifier_seal=self._evidence_seal,
            _integrity_tag=tag,
        )

    def validate_evidence(self, evidence: VerificationEvidence) -> int:
        """验证 evidence 的来源、结构、语义一致性和完整性并返回 batch 大小。"""

        if not isinstance(evidence, VerificationEvidence):
            raise TypeError("evidence 必须是 VerificationEvidence")
        if evidence._verifier_seal is not self._evidence_seal:
            raise PermissionError("evidence 来源 verifier 不匹配")
        tensors = (evidence.verified, evidence.error_norm, evidence.reason_code)
        if any(not isinstance(value, Tensor) or value.ndim != 1 for value in tensors):
            raise ValueError("evidence 字段必须是一维 Tensor")
        if len({value.shape[0] for value in tensors}) != 1 or not tensors[0].numel():
            raise ValueError("evidence batch 必须非空且字段长度一致")
        if len({value.device for value in tensors}) != 1:
            raise ValueError("evidence 字段必须位于同一 device")
        if evidence.verified.dtype != torch.bool:
            raise TypeError("verified 必须是 BoolTensor")
        if evidence.error_norm.dtype != torch.float32:
            raise TypeError("error_norm 必须是 FloatTensor")
        if evidence.reason_code.dtype != torch.long:
            raise TypeError("reason_code 必须是 LongTensor")
        # 先检查产生来源和完整性，再解释字段语义，避免篡改被误报为业务拒绝。
        expected_tag = _evidence_integrity_tag(
            evidence.verified,
            evidence.error_norm,
            evidence.reason_code,
            self._evidence_seal,
        )
        if evidence._integrity_tag != expected_tag:
            raise PermissionError("evidence 内容在生成后被修改")
        valid_codes = {int(value) for value in EvidenceReason}
        if any(
            int(value) not in valid_codes for value in evidence.reason_code.tolist()
        ):
            raise ValueError("evidence 包含未知 reason code")
        success = evidence.reason_code == int(EvidenceReason.SUCCESS)
        if not torch.equal(evidence.verified, success):
            raise ValueError("verified 与 reason_code 语义不一致")
        if bool((success & ~torch.isfinite(evidence.error_norm)).any().item()):
            raise ValueError("成功 evidence 必须具有有限 error_norm")
        return int(evidence.verified.shape[0])

    def validate_credential_structure(self, credential: Tensor) -> None:
        """拒绝不满足固定 profile 的请求级 credential 结构。"""

        if not isinstance(credential, Tensor):
            raise TypeError("credential 必须是 Tensor")
        if credential.ndim != 2 or credential.shape[1] != self.n:
            raise ValueError(f"credential 必须具有 shape [B, {self.n}]")
        if credential.shape[0] == 0:
            raise ValueError("credential batch 不能为空")
        if credential.dtype != torch.float32:
            raise TypeError("credential 必须使用 torch.float32")
        if credential.device != self.A.device:
            raise ValueError("credential 与 verifier buffer 必须位于同一 device")
        if credential.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("固定认证配置要求关闭 CUDA matmul TF32")


class RouteCoordinator(nn.Module):
    """把受信 verifier evidence 提交为固定 P1 hard route。"""

    def __init__(
        self,
        verifier: FixedRelationVerifier,
        execution_config_id: str,
        policy_id: str = P1_POLICY_ID,
    ) -> None:
        """绑定 verifier、P1 policy 和 execution config。"""

        super().__init__()
        if not isinstance(verifier, FixedRelationVerifier):
            raise TypeError("verifier 必须是 FixedRelationVerifier")
        if policy_id != P1_POLICY_ID:
            raise ValueError("G0/P1 只允许固定 protected-or-deny policy")
        self.verifier = verifier
        self.policy_id = policy_id
        self.execution_config_id = _require_nonempty_id(
            execution_config_id, "execution_config_id"
        )
        self._coordinator_seal = object()

    def commit(
        self,
        evidence: VerificationEvidence,
        context: TrustedRequestContext,
    ) -> _CommittedRoute:
        """校验证据和受信上下文并提交不可变 hard route。"""

        batch_size = self.verifier.validate_evidence(evidence)
        if not isinstance(context, TrustedRequestContext):
            raise TypeError("context 必须是 TrustedRequestContext")
        if context.execution_config_id != self.execution_config_id:
            raise PermissionError("请求 execution config 与协调器绑定不一致")
        if len(context.request_ids) != batch_size:
            raise ValueError("request IDs 与 evidence batch 大小不一致")
        routes = tuple(
            RouteKind.PROTECTED if bool(value) else RouteKind.DENY
            for value in evidence.verified.tolist()
        )
        return _CommittedRoute(
            routes=routes,
            policy_id=self.policy_id,
            request_ids=context.request_ids,
            execution_config_id=self.execution_config_id,
            _coordinator_seal=self._coordinator_seal,
        )

    def validate_committed_route(
        self,
        committed: _CommittedRoute,
        context: TrustedRequestContext,
    ) -> Tuple[RouteKind, ...]:
        """验证 route 来源与完整上下文绑定并返回路由 tuple。"""

        if not isinstance(committed, _CommittedRoute):
            raise TypeError("committed 必须来自 RouteCoordinator")
        if committed._coordinator_seal is not self._coordinator_seal:
            raise PermissionError("route 来源协调器不匹配")
        if not isinstance(context, TrustedRequestContext):
            raise TypeError("context 必须是 TrustedRequestContext")
        if committed.policy_id != self.policy_id:
            raise PermissionError("route policy 不匹配")
        if committed.execution_config_id != self.execution_config_id:
            raise PermissionError("route execution config 不匹配")
        if context.execution_config_id != committed.execution_config_id:
            raise PermissionError("请求 execution config 不匹配")
        if committed.request_ids != context.request_ids:
            raise PermissionError("route 与完整有序 request IDs 不匹配")
        if not committed.routes or len(committed.routes) != len(context.request_ids):
            raise ValueError("route batch 大小非法")
        if any(not isinstance(route, RouteKind) for route in committed.routes):
            raise ValueError("route 包含未知枚举值")
        if any(route is RouteKind.PUBLIC for route in committed.routes):
            raise PermissionError("P1 policy 不允许 PUBLIC route")
        return committed.routes
