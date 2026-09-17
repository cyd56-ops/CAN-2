"""G1-a reference 的不可变类型和稳定错误。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class G1AError(ValueError):
    """表示 G1-a canonical 输入或关系校验失败。"""

    def __init__(self, code: str, message: str) -> None:
        """使用稳定错误码构造异常。"""
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class G1AConfig:
    """保存 G1-a 模整数协议的候选配置。"""

    q: int = 3329
    n: int = 256
    m: int = 512
    tau: int = 2
    norm: str = "linf"
    protocol_id: str = "g1a-modint-v1"
    execution_config_id: str = "g1a-reference-cpu-v1"
    master_seed: int = 20260915
    prng_id: str = "numpy-pcg64"
    prng_version: str = "numpy"

    def __post_init__(self) -> None:
        """验证模数、维度、阈值和 seed 的 canonical 范围。"""
        integer_fields = ("q", "n", "m", "tau", "master_seed")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise G1AError("G1A_CONFIG_TYPE", f"{name} 必须是整数")
        if self.q < 3 or self.q % 2 == 0:
            raise G1AError("G1A_CONFIG_Q", "q 必须是大于 2 的奇数")
        if self.n < 1 or self.m < 1:
            raise G1AError("G1A_CONFIG_DIM", "n 和 m 必须为正数")
        if self.tau < 0 or self.tau > self.q // 2:
            raise G1AError("G1A_CONFIG_TAU", "tau 必须位于 centered 域内")
        if self.norm != "linf":
            raise G1AError("G1A_CONFIG_NORM", "当前 reference 只支持 linf")
        if self.master_seed < 0:
            raise G1AError("G1A_CONFIG_SEED", "master_seed 不能为负数")
        if not self.protocol_id or not self.execution_config_id:
            raise G1AError("G1A_CONFIG_ID", "协议和执行配置 ID 不能为空")


@dataclass(frozen=True)
class G1AParameters:
    """保存公开矩阵 A、公开向量 b 及其协议配置。"""

    A: np.ndarray
    b: np.ndarray
    config: G1AConfig

    def __post_init__(self) -> None:
        """复制并冻结公开参数，防止调用方事后修改摘要内容。"""
        if not isinstance(self.A, np.ndarray) or not isinstance(self.b, np.ndarray):
            raise G1AError("G1A_PARAMETER_TYPE", "A 和 b 必须是 NumPy 数组")
        A = np.array(self.A, dtype="<i8", copy=True, order="C")
        b = np.array(self.b, dtype="<i8", copy=True, order="C")
        if A.shape != (self.config.m, self.config.n):
            raise G1AError("G1A_PARAMETER_SHAPE", "A 的形状与 m/n 不匹配")
        if b.shape != (self.config.m,):
            raise G1AError("G1A_PARAMETER_SHAPE", "b 的形状与 m 不匹配")
        if np.any(A < 0) or np.any(A >= self.config.q) or np.any(b < 0) or np.any(b >= self.config.q):
            raise G1AError("G1A_PARAMETER_DOMAIN", "A 和 b 必须位于 [0,q)")
        A.setflags(write=False)
        b.setflags(write=False)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)


@dataclass(frozen=True)
class G1aEvidence:
    """保存不含完整残差的逐行 verifier evidence。"""

    protocol_id: str
    parameter_digest: str
    accepted: Tuple[bool, ...]
    max_abs_residual: Tuple[int, ...]
    residual_digest: str
    reason_code: Tuple[str, ...]
    batch_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        """验证 evidence 各字段长度、索引和稳定 reason code。"""
        lengths = {len(self.accepted), len(self.max_abs_residual), len(self.reason_code), len(self.batch_indices)}
        if len(lengths) != 1:
            raise G1AError("G1A_EVIDENCE_LENGTH", "evidence 逐行字段长度必须一致")
        if any(type(value) is not bool for value in self.accepted):
            raise G1AError("G1A_EVIDENCE_TYPE", "accepted 必须只包含 bool")
        if any(type(value) is not int or value < 0 for value in self.max_abs_residual):
            raise G1AError("G1A_EVIDENCE_TYPE", "max_abs_residual 必须是非负 Python int")
        if tuple(self.batch_indices) != tuple(sorted(self.batch_indices)) or len(set(self.batch_indices)) != len(self.batch_indices):
            raise G1AError("G1A_EVIDENCE_INDEX", "batch_indices 必须严格递增")
        allowed = {"RELATION_WITHIN_BOUND", "RELATION_OUT_OF_BOUND"}
        if any(value not in allowed for value in self.reason_code):
            raise G1AError("G1A_EVIDENCE_REASON", "reason_code 不在登记集合中")

    def to_dict(self) -> dict:
        """序列化为不含秘密中间值的 JSON 对象。"""
        return {
            "protocol_id": self.protocol_id,
            "parameter_digest": self.parameter_digest,
            "accepted": list(self.accepted),
            "max_abs_residual": list(self.max_abs_residual),
            "residual_digest": self.residual_digest,
            "reason_code": list(self.reason_code),
            "batch_indices": list(self.batch_indices),
        }


@dataclass(frozen=True)
class G1AReferenceResult:
    """保存 evidence 以及仅供内部复核使用的完整整数结果。"""

    evidence: G1aEvidence
    residuals: Tuple[Tuple[int, ...], ...]
    reduced: Tuple[Tuple[int, ...], ...]
    raw_values: Optional[Tuple[Tuple[int, ...], ...]] = None
