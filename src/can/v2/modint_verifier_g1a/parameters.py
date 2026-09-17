"""G1-a 公开参数的确定性生成与 canonical 摘要。"""

from __future__ import annotations

import hashlib
import json
from typing import Tuple

import numpy as np

from .types import G1AConfig, G1AParameters


def derive_seed(master_seed: int, label: str) -> int:
    """从 master seed 和固定标签派生可复现的 64 位 PRNG seed。"""
    payload = f"g1a-seed-v1|{int(master_seed)}|{label}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def public_parameter_bytes(parameters: G1AParameters) -> bytes:
    """用排序 JSON 编码 A、b 和协议参数，生成稳定摘要输入。"""
    config = parameters.config
    payload = {
        "A": parameters.A.tolist(),
        "b": parameters.b.tolist(),
        "m": config.m,
        "n": config.n,
        "norm": config.norm,
        "q": config.q,
        "tau": config.tau,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def parameter_digest(parameters: G1AParameters) -> str:
    """计算公开参数和接受阈值的 SHA-256。"""
    return hashlib.sha256(public_parameter_bytes(parameters)).hexdigest()


def generate_parameters(config: G1AConfig) -> Tuple[G1AParameters, np.ndarray, np.ndarray]:
    """使用固定 seed 生成 A、secret s 和小误差 e，并返回公开 b。"""
    # A、secret 和误差使用不同派生标签，避免随机流相互耦合。
    rng_A = np.random.Generator(np.random.PCG64(derive_seed(config.master_seed, "A")))
    rng_s = np.random.Generator(np.random.PCG64(derive_seed(config.master_seed, "secret")))
    rng_e = np.random.Generator(np.random.PCG64(derive_seed(config.master_seed, "error")))
    A = rng_A.integers(0, config.q, size=(config.m, config.n), dtype=np.int64).astype("<i8", copy=False)
    secret = rng_s.integers(0, config.q, size=(config.n,), dtype=np.int64).astype("<i8", copy=False)
    error = rng_e.integers(-config.tau, config.tau + 1, size=(config.m,), dtype=np.int64).astype("<i8", copy=False)
    b = np.asarray([(sum(int(A[i, j]) * int(secret[j]) for j in range(config.n)) + int(error[i])) % config.q for i in range(config.m)], dtype="<i8")
    parameters = G1AParameters(A=A, b=b, config=config)
    return parameters, secret, error
