"""CAN v2 神经网络层模块

包含可嵌入计算图中间的认证层实现。

当前实现:
- gate_layer.py: GateLayer，融合浅层特征与 LWE credential 验证结果
"""

from .gate_layer import (
    AuthorizationDecision,
    GateLayer,
    ReasonCode,
    VerificationEvidence,
)

__all__ = [
    "AuthorizationDecision",
    "GateLayer",
    "ReasonCode",
    "VerificationEvidence",
]
