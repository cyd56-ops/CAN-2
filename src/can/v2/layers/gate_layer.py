"""Neural Gate Layer 的 PyTorch 实现。

该模块将 toy LWE 验证编译为批量 Tensor 计算，并在网络中间把验证结果
应用到浅层特征。当前实现用于科研原型，不提供生产级密码学安全保证，也
不处理 credential replay。
"""

from dataclasses import dataclass
from enum import IntEnum
from numbers import Real
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..crypto.lwe import LWEParams


class ReasonCode(IntEnum):
    """表示 LWE 验证或输入规范化的稳定结果码。"""

    SUCCESS = 0
    LWE_VERIFICATION_FAILED = 1
    INVALID_SHAPE = 2
    NON_FINITE = 3
    WRONG_DTYPE = 4
    DIMENSION_MISMATCH = 5


@dataclass(frozen=True)
class VerificationEvidence:
    """保存批量 LWE 验证证据，不直接授予深层访问能力。

    参数:
        verified: 每个样本是否通过 LWE 验证，shape 为 ``[B]``。
        error_norm: 每个样本的 L2 误差范数，shape 为 ``[B]``。
        reason_code: 每个样本的稳定结果码，shape 为 ``[B]``。
    """

    verified: Tensor
    error_norm: Tensor
    reason_code: Tensor


@dataclass(frozen=True)
class AuthorizationDecision:
    """保存协调器提交的批量授权决定和门控信号。

    参数:
        allow: 是否允许样本进入受保护路径，shape 为 ``[B]``。
        gate_signal: 训练时为软值、推理时为硬值，shape 为 ``[B]``。
        evidence: 产生本决定的验证证据。
    """

    allow: Tensor
    gate_signal: Tensor
    evidence: VerificationEvidence


def _validate_positive_finite(value: object, name: str) -> float:
    """验证并返回有限正浮点参数。

    参数:
        value: 待验证的数值。
        name: 用于异常信息的参数名称。

    返回:
        转换后的有限正浮点数。
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} 必须是实数")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return result


def _validate_public_array(
    value: object,
    name: str,
    expected_shape: Tuple[int, ...],
) -> np.ndarray:
    """验证 LWE 公共数组的类型、形状和有限性。

    参数:
        value: 待验证的 NumPy 数组。
        name: 数组名称。
        expected_shape: 设计要求的精确形状。

    返回:
        验证通过的原始 NumPy 数组。
    """

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} 必须是 np.ndarray")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} 必须使用浮点 dtype")
    if value.shape != expected_shape:
        raise ValueError(f"{name} shape 必须是 {expected_shape}，得到 {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} 必须全部为有限值")
    return value


class LWEVerifier(nn.Module):
    """执行无状态、批量化的 LWE 确定性验证。"""

    def __init__(self, A: np.ndarray, b: np.ndarray, params: LWEParams) -> None:
        """初始化验证器并冻结 LWE 公共参数。

        参数:
            A: LWE 公钥矩阵，shape 为 ``[m, n]``。
            b: LWE 公钥向量，shape 为 ``[m]``。
            params: LWE 参数，包含维度和误差阈值。
        """

        super().__init__()
        if not isinstance(params, LWEParams):
            raise TypeError("params 必须是 LWEParams")

        A_array = _validate_public_array(A, "A", (params.m, params.n))
        b_array = _validate_public_array(b, "b", (params.m,))
        self.error_threshold = _validate_positive_finite(
            params.error_threshold, "error_threshold"
        )
        self.params = params
        self.n = params.n
        self.m = params.m

        # clone 防止调用方在构造后通过原 NumPy 数组修改已注册 buffer。
        self.register_buffer("A", torch.as_tensor(A_array, dtype=torch.float32).clone())
        self.register_buffer("b", torch.as_tensor(b_array, dtype=torch.float32).clone())

    def forward(self, credential: Tensor) -> VerificationEvidence:
        """验证已规范化的批量 credential。

        参数:
            credential: float32 Tensor，shape 为 ``[B, n]``，device 与 A 相同。

        返回:
            每个样本独立生成的 ``VerificationEvidence``。
        """

        if not isinstance(credential, Tensor):
            raise TypeError("credential 必须是 Tensor")
        if credential.ndim != 2 or credential.shape[1] != self.n:
            raise ValueError(f"credential 必须具有 shape [B, {self.n}]")
        if credential.dtype != torch.float32:
            raise TypeError("LWEVerifier 仅接受已规范化的 float32 credential")
        if credential.device != self.A.device:
            raise ValueError("credential 与 LWE buffer 必须位于同一 device")

        batch_size = credential.shape[0]
        finite_rows = torch.isfinite(credential).all(dim=1)

        # 非有限行先替换为零，避免 NaN/Inf 污染矩阵乘法和其他样本。
        safe_credential = torch.where(
            finite_rows[:, None], credential, torch.zeros_like(credential)
        )
        residual = torch.matmul(safe_credential, self.A.transpose(0, 1)) - self.b
        computed_norm = torch.linalg.vector_norm(residual, ord=2, dim=1)
        error_norm = torch.where(
            finite_rows,
            computed_norm,
            torch.full(
                (batch_size,),
                float("inf"),
                device=credential.device,
                dtype=credential.dtype,
            ),
        )

        verified = finite_rows & (error_norm < self.error_threshold)
        reason_code = torch.full(
            (batch_size,),
            int(ReasonCode.LWE_VERIFICATION_FAILED),
            device=credential.device,
            dtype=torch.long,
        )
        reason_code = torch.where(
            verified,
            torch.full_like(reason_code, int(ReasonCode.SUCCESS)),
            reason_code,
        )
        reason_code = torch.where(
            finite_rows,
            reason_code,
            torch.full_like(reason_code, int(ReasonCode.NON_FINITE)),
        )
        return VerificationEvidence(verified, error_norm, reason_code)


class AuthorizationCoordinator(nn.Module):
    """根据验证证据生成唯一的批量授权决定。"""

    def __init__(self, params: LWEParams, temperature: float = 5.0) -> None:
        """初始化协调器。

        参数:
            params: LWE 参数，提供误差阈值。
            temperature: 训练模式软门控的有限正温度。
        """

        super().__init__()
        if not isinstance(params, LWEParams):
            raise TypeError("params 必须是 LWEParams")
        self.error_threshold = _validate_positive_finite(
            params.error_threshold, "error_threshold"
        )
        self.temperature = _validate_positive_finite(temperature, "temperature")

    def forward(self, evidence: VerificationEvidence) -> AuthorizationDecision:
        """把结构化证据提交为训练软门控或推理硬门控决定。

        参数:
            evidence: LWEVerifier 产生的批量验证证据。

        返回:
            与证据 batch 对齐的 ``AuthorizationDecision``。
        """

        self._validate_evidence(evidence)
        success = evidence.reason_code == int(ReasonCode.SUCCESS)
        parsed = success | (
            evidence.reason_code == int(ReasonCode.LWE_VERIFICATION_FAILED)
        )
        allow = evidence.verified & success

        if self.training:
            # 对非法输入使用安全替代值，再通过 mask 强制门控为零。
            safe_error_norm = torch.where(
                parsed,
                evidence.error_norm,
                torch.full_like(evidence.error_norm, self.error_threshold),
            )
            soft_gate = torch.sigmoid(
                (self.error_threshold - safe_error_norm) / self.temperature
            )
            gate_signal = torch.where(parsed, soft_gate, torch.zeros_like(soft_gate))
        else:
            gate_signal = allow.to(dtype=evidence.error_norm.dtype)

        return AuthorizationDecision(allow, gate_signal, evidence)

    @staticmethod
    def _validate_evidence(evidence: VerificationEvidence) -> None:
        """验证 evidence 的 Tensor 类型、形状和 device 一致性。

        参数:
            evidence: 待提交的批量验证证据。
        """

        if not isinstance(evidence, VerificationEvidence):
            raise TypeError("evidence 必须是 VerificationEvidence")
        tensors = (evidence.verified, evidence.error_norm, evidence.reason_code)
        if any(not isinstance(value, Tensor) for value in tensors):
            raise TypeError("evidence 的所有字段必须是 Tensor")
        if any(value.ndim != 1 for value in tensors):
            raise ValueError("evidence 的所有字段必须是一维 batch Tensor")
        if len({value.shape[0] for value in tensors}) != 1:
            raise ValueError("evidence 字段的 batch 大小必须一致")
        if len({value.device for value in tensors}) != 1:
            raise ValueError("evidence 字段必须位于同一 device")
        if evidence.verified.dtype != torch.bool:
            raise TypeError("verified 必须是 BoolTensor")
        if not evidence.error_norm.dtype.is_floating_point:
            raise TypeError("error_norm 必须是浮点 Tensor")
        if evidence.reason_code.dtype != torch.long:
            raise TypeError("reason_code 必须是 LongTensor")


class FeatureGate(nn.Module):
    """将协调器产生的门控信号应用到浅层特征。"""

    def forward(
        self, shallow_features: Tensor, decision: AuthorizationDecision
    ) -> Tensor:
        """执行保持 shape、dtype 和 device 的逐样本特征门控。

        参数:
            shallow_features: 浅层特征图，shape 为 ``[B, C, H, W]``。
            decision: 与特征 batch 对齐的授权决定。

        返回:
            门控后的浅层特征，shape 与输入相同。
        """

        if not isinstance(decision, AuthorizationDecision):
            raise TypeError("decision 必须是 AuthorizationDecision")
        if decision.gate_signal.ndim != 1:
            raise ValueError("gate_signal 必须是一维 batch Tensor")
        if decision.gate_signal.shape[0] != shallow_features.shape[0]:
            raise ValueError("shallow_features 与 gate_signal 的 batch 不一致")

        gate = decision.gate_signal.to(
            device=shallow_features.device, dtype=shallow_features.dtype
        )[:, None, None, None]
        return shallow_features * gate


class GateLayer(nn.Module):
    """组合 LWE 验证、授权协调和中间特征门控。

    Gate Layer 无可训练参数，但训练模式的软门控允许梯度从 gated_features
    回传到浅层网络。推理模式使用确定性硬门控。
    """

    _ALLOWED_CREDENTIAL_DTYPES = {
        torch.float16,
        torch.float32,
        torch.float64,
    }
    _ALLOWED_NUMPY_DTYPES = {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        params: LWEParams,
        temperature: float = 5.0,
    ) -> None:
        """初始化 Gate Layer。

        参数:
            A: LWE 公钥矩阵，shape 为 ``[m, n]``。
            b: LWE 公钥向量，shape 为 ``[m]``。
            params: LWE 参数。
            temperature: 训练模式软门控温度。
        """

        super().__init__()
        self.verifier = LWEVerifier(A, b, params)
        self.coordinator = AuthorizationCoordinator(params, temperature)
        self.feature_gate = FeatureGate()

    def forward(
        self,
        shallow_features: Tensor,
        credential: Union[Tensor, np.ndarray],
    ) -> Tuple[Tensor, AuthorizationDecision]:
        """验证 credential 并将决定应用到计算图中间特征。

        参数:
            shallow_features: 浅层特征，shape 为 ``[B, C, H, W]``。
            credential: 单个 ``[n]`` 或批量 ``[B, n]`` 浮点 credential。

        返回:
            ``(gated_features, decision)``；两者均与输入 batch 对齐。
        """

        batch_size = self._validate_features(shallow_features)
        normalized, error = self._normalize_credential(credential, batch_size)
        if error is not None:
            return self._reject_request(shallow_features, error)

        # error 为 None 时，规范化函数保证 normalized 是 Tensor。
        assert normalized is not None
        evidence = self.verifier(normalized)
        decision = self.coordinator(evidence)
        gated_features = self.feature_gate(shallow_features, decision)
        return gated_features, decision

    def _validate_features(self, shallow_features: Tensor) -> int:
        """验证浅层特征并返回 batch 大小。

        参数:
            shallow_features: 待验证的浅层特征。

        返回:
            特征的 batch 大小。
        """

        if not isinstance(shallow_features, Tensor):
            raise TypeError("shallow_features 必须是 Tensor")
        if shallow_features.ndim != 4:
            raise ValueError("shallow_features 必须是 4D Tensor[B, C, H, W]")
        if not shallow_features.dtype.is_floating_point:
            raise TypeError("shallow_features 必须使用浮点 dtype")
        if shallow_features.device != self.verifier.A.device:
            raise ValueError("shallow_features 与 Gate Layer 必须位于同一 device")
        if not torch.isfinite(shallow_features).all():
            raise ValueError("shallow_features 包含 NaN 或 Inf")
        return shallow_features.shape[0]

    def _normalize_credential(
        self,
        credential: Union[Tensor, np.ndarray],
        batch_size: int,
    ) -> Tuple[Optional[Tensor], Optional[ReasonCode]]:
        """按稳定顺序验证并规范化 credential 请求。

        参数:
            credential: Tensor 或 NumPy credential。
            batch_size: shallow_features 提供的目标 batch 大小。

        返回:
            ``(normalized, None)`` 或 ``(None, reason_code)``。
        """

        if isinstance(credential, np.ndarray):
            if credential.ndim not in (1, 2):
                return None, ReasonCode.INVALID_SHAPE
            if credential.dtype not in self._ALLOWED_NUMPY_DTYPES:
                return None, ReasonCode.WRONG_DTYPE
            try:
                tensor = torch.from_numpy(credential)
            except (TypeError, ValueError):
                return None, ReasonCode.WRONG_DTYPE
        elif isinstance(credential, Tensor):
            tensor = credential
        else:
            return None, ReasonCode.WRONG_DTYPE

        if tensor.ndim not in (1, 2):
            return None, ReasonCode.INVALID_SHAPE
        if tensor.dtype not in self._ALLOWED_CREDENTIAL_DTYPES:
            return None, ReasonCode.WRONG_DTYPE
        if tensor.shape[-1] != self.verifier.n:
            return None, ReasonCode.DIMENSION_MISMATCH

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        source_batch = tensor.shape[0]
        if source_batch == batch_size:
            pass
        elif source_batch == 1:
            tensor = tensor.expand(batch_size, -1)
        else:
            return None, ReasonCode.DIMENSION_MISMATCH

        normalized = tensor.to(device=self.verifier.A.device, dtype=torch.float32)
        return normalized, None

    def _reject_request(
        self, shallow_features: Tensor, reason: ReasonCode
    ) -> Tuple[Tensor, AuthorizationDecision]:
        """为 credential 请求级错误构造批量拒绝结果。

        参数:
            shallow_features: 已验证的浅层特征。
            reason: 稳定的请求拒绝原因。

        返回:
            全零特征和逐样本拒绝决定。
        """

        batch_size = shallow_features.shape[0]
        device = shallow_features.device
        verified = torch.zeros(batch_size, dtype=torch.bool, device=device)
        error_norm = torch.full(
            (batch_size,), float("inf"), dtype=torch.float32, device=device
        )
        reason_code = torch.full(
            (batch_size,), int(reason), dtype=torch.long, device=device
        )
        evidence = VerificationEvidence(verified, error_norm, reason_code)
        decision = AuthorizationDecision(
            allow=verified.clone(),
            gate_signal=torch.zeros(batch_size, dtype=torch.float32, device=device),
            evidence=evidence,
        )
        return torch.zeros_like(shallow_features), decision


__all__ = [
    "AuthorizationDecision",
    "GateLayer",
    "ReasonCode",
    "VerificationEvidence",
]
