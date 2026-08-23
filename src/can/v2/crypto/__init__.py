"""CAN v2 密码原语模块

提供可嵌入神经网络的密码学验证组件。

当前实现:
- LWE (Learning With Errors): 基于格的密码验证方案
"""

from .lwe import (
    LWEParams,
    generate_keypair,
    verify,
    compute_error_norm,
    V_ref,
    generate_random_credential,
    generate_perturbed_credential,
)

__all__ = [
    'LWEParams',
    'generate_keypair',
    'verify',
    'compute_error_norm',
    'V_ref',
    'generate_random_credential',
    'generate_perturbed_credential',
]
