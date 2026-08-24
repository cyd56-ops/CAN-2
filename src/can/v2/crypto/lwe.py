"""Learning With Errors (LWE) 密码原语

基于 LWE 问题的简化认证方案，用于 Gate Layer 的密码验证。
这是一个 toy profile，使用浮点数近似模运算，便于嵌入神经网络。

参考：
- Regev, "On Lattices, Learning with Errors, Random Linear Codes, and Cryptography"
- Shamir et al., "How to Securely Implement Cryptography in Deep Neural Networks"
- NIST PQC: Kyber (LWE-based KEM), Dilithium (Module-LWE based signature)

Toy 简化说明：
这个实现是用于研究原型的简化版本，目标是演示"将 LWE 验证嵌入神经网络"的可行性。
真实的 LWE 方案需要严格的模运算、rejection sampling 等，本实现用浮点数近似。

LWE 问题：
    给定 (A, b = A*s + e mod q)，其中：
    - A: m×n 随机矩阵（公钥）
    - s: n 维小范数向量（秘密）
    - e: m 维小噪声向量（从高斯分布采样）
    - q: 模数
    找到 s 在计算上困难（Decision-LWE 假设）

认证方案：
    - 密钥生成：生成 (A, s, b = A*s + e)
    - 公钥：(A, b)
    - 秘密（credential）：s
    - 验证：检查 ||A*credential - b|| < threshold
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np


@dataclass
class LWEParams:
    """LWE 参数

    参数:
        n: 秘密维度（credential 向量长度）
        m: 公钥维度（验证方程数量），通常 m > n
        q: 模数（理论上的模运算基数，toy 实现中用于缩放）
        sigma: 噪声标准差（高斯分布参数）
        secret_bound: 秘密向量的范数上界
        error_threshold: 验证时允许的误差范数上界

    典型参数选择（Toy Profile）：
        - n=128, m=256: 中等安全强度
        - sigma=1.0: 小噪声（保证验证成功率）
        - secret_bound=2.0: 秘密向量小范数约束
    """

    n: int = 128  # 秘密维度
    m: int = 256  # 公钥维度
    q: float = 8380417.0  # 模数（约 2^23，toy 中用于缩放）
    sigma: float = 1.0  # 噪声标准差
    secret_bound: float = 2.0  # 秘密范数上界
    error_threshold: float = None  # 验证阈值（自动计算）

    def __post_init__(self):
        """自动计算验证阈值"""
        if self.error_threshold is None:
            # 阈值设为：期望噪声范数 + 3倍标准差
            # E[||e||] ≈ sigma * sqrt(m)
            # 留出足够余量容纳噪声波动
            #
            # 对于 valid credential: residual = -e, ||residual|| ≈ sigma * sqrt(m)
            # 对于 random credential: residual = A*(random - s) - e
            #                          ||residual|| ≈ ||random - s|| * sqrt(m)
            #                                        (因为 A 的每个元素约为 1/sqrt(n))
            #
            # 当 random 和 s 差异大时，||random - s|| >> sigma，验证会失败
            self.error_threshold = self.sigma * np.sqrt(self.m) * 3.0


def generate_keypair(
    params: LWEParams,
    rng: Optional[Union[np.random.Generator, np.random.RandomState]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 LWE 密钥对

    参数:
        params: LWE 参数
        rng: 可选显式随机源；未提供时兼容使用 NumPy 全局随机源。

    返回:
        (A, secret, b):
            - A: 公钥矩阵，shape [m, n]
            - secret: 秘密向量（credential），shape [n]
            - b: 公钥向量，b = A*s + e，shape [m]

    Toy 实现说明：
        - A 从标准正态分布采样（而非均匀分布 mod q）
        - 模运算用浮点数近似（不做严格的 mod q）
        - 这样便于嵌入神经网络（全是可微分的浮点运算）
    """
    # 生成公钥矩阵 A（标准正态分布，不归一化）
    # Toy 简化：用 N(0,1) 而非 Uniform(0, q)
    # 不归一化是为了让 valid 和 invalid credentials 产生显著不同的误差
    random_source = rng if rng is not None else np.random
    if not hasattr(random_source, "standard_normal") and not hasattr(
        random_source, "randn"
    ):
        raise TypeError("rng 必须提供 standard_normal 或 randn")
    normal = getattr(random_source, "standard_normal", None)
    if normal is None:
        normal = getattr(random_source, "randn")
    A = normal((params.m, params.n)).astype(np.float32)

    # 生成秘密向量 s（小范数）
    # 从 N(0, 0.5) 采样后截断到 [-secret_bound/2, secret_bound/2]
    secret = (normal(params.n) * 0.5).astype(np.float32)
    secret = np.clip(secret, -params.secret_bound / 2, params.secret_bound / 2)

    # 生成噪声向量 e
    noise = (normal(params.m) * params.sigma).astype(np.float32)

    # 计算公钥向量 b = A*s + e
    # Toy 简化：不做 mod q 运算
    b = (np.dot(A, secret) + noise).astype(np.float32)

    return A, secret, b


def verify(
    credential: np.ndarray, A: np.ndarray, b: np.ndarray, params: LWEParams
) -> bool:
    """验证 LWE credential

    参数:
        credential: 待验证的凭证向量，shape [n]
        A: 公钥矩阵，shape [m, n]
        b: 公钥向量，shape [m]
        params: LWE 参数

    返回:
        True 如果 credential 有效，False 否则

    验证逻辑：
        1. 计算 residual = A * credential - b
        2. 检查 ||residual|| < threshold

        如果 credential = secret，则 residual = A*s - (A*s + e) = -e
        所以 ||residual|| = ||e|| ≈ sigma * sqrt(m)，应该小于阈值
    """
    try:
        # 检查输入形状
        if credential.shape != (params.n,):
            return False
        if A.shape != (params.m, params.n):
            return False
        if b.shape != (params.m,):
            return False

        # 步骤 1: 计算 A * credential
        Ac = np.dot(A, credential)

        # 步骤 2: 计算残差
        residual = Ac - b

        # 步骤 3: 计算残差的 L2 范数
        error_norm = np.linalg.norm(residual)

        # 步骤 4: 判断是否小于阈值
        is_valid = error_norm < params.error_threshold

        return bool(is_valid)

    except (ValueError, TypeError, AttributeError):
        # 任何异常都返回 False（fail-closed）
        return False


def V_ref(
    credential_dict: Dict, A: np.ndarray, b: np.ndarray, params: LWEParams
) -> int:
    """参考验证器：将 credential 验证映射为 {0, 1}

    这是用于差分测试的参考实现，Gate Layer 的神经验证器必须与此一致。

    参数:
        credential_dict: 凭证字典，包含:
            - 'vector': np.ndarray，shape [n]，凭证向量
        A: 公钥矩阵
        b: 公钥向量
        params: LWE 参数

    返回:
        0 或 1：1 表示验证通过，0 表示验证失败
    """
    try:
        # 提取 credential 向量
        credential = credential_dict.get("vector")

        if credential is None:
            return 0

        # 确保是 numpy 数组
        if not isinstance(credential, np.ndarray):
            credential = np.array(credential, dtype=np.float32)

        # 调用 verify 函数
        is_valid = verify(credential, A, b, params)

        return 1 if is_valid else 0

    except Exception:
        # 任何异常都返回 0（fail-closed）
        return 0


def compute_error_norm(credential: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    """计算验证误差范数（用于调试和分析）

    参数:
        credential: 凭证向量，shape [n]
        A: 公钥矩阵，shape [m, n]
        b: 公钥向量，shape [m]

    返回:
        error_norm: ||A*credential - b||
    """
    residual = np.dot(A, credential) - b
    return float(np.linalg.norm(residual))


# ============================================================================
# 辅助函数：生成 invalid credentials（用于训练和测试）
# ============================================================================


def generate_random_credential(params: LWEParams, scale: float = 1.0) -> np.ndarray:
    """生成随机的 invalid credential

    参数:
        params: LWE 参数
        scale: 缩放因子（控制随机程度）

    返回:
        random_cred: shape [n]，随机向量
    """
    return np.random.randn(params.n).astype(np.float32) * scale


def generate_perturbed_credential(secret: np.ndarray, noise_level: float) -> np.ndarray:
    """生成轻微扰动的 credential（用于测试鲁棒性）

    参数:
        secret: 原始秘密向量，shape [n]
        noise_level: 扰动强度

    返回:
        perturbed: shape [n]，扰动后的向量
    """
    noise = np.random.randn(*secret.shape).astype(np.float32) * noise_level
    return secret + noise
