"""G1-b 固定 torch.int64 kernel 和解析位宽检查。"""

from __future__ import annotations

import hashlib
import struct
from math import ceil, log2
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..modint_verifier_g1a.parameters import parameter_digest
from ..modint_verifier_g1a.types import G1AError, G1AParameters, G1AReferenceResult, G1aEvidence
from .types import G1BError, G1BResult, IntegerBounds


INT64_MAX = 2**63 - 1


def derive_integer_bounds(parameters: G1AParameters) -> IntegerBounds:
    """从 canonical 域解析计算乘积、累加器和所需有符号位宽。"""
    config = parameters.config
    max_input = config.q - 1
    max_product = max_input * max_input
    max_accumulator = config.n * max_product + max_input
    required_bits = ceil(log2(max_accumulator + 1)) + 1
    bounds = IntegerBounds(config.q, config.n, config.m, max_input, max_product, max_accumulator, required_bits)
    if max_accumulator > INT64_MAX:
        raise G1BError("G1B_ARITH_WIDTH_UNSAFE", "canonical 域超出 int64 安全范围")
    return bounds


def _require_int64(value: Tensor, name: str) -> None:
    """检查 tensor 是 CPU contiguous int64，拒绝隐式浮点转换。"""
    if not isinstance(value, Tensor) or value.dtype is not torch.int64:
        raise G1BError("G1B_INPUT_DTYPE", f"{name} 必须是 torch.int64")
    if value.device.type != "cpu":
        raise G1BError("G1B_INPUT_DEVICE", f"{name} 当前必须位于 CPU")
    if not value.is_contiguous():
        raise G1BError("G1B_INPUT_LAYOUT", f"{name} 必须是 contiguous")


def canonical_mod_q(value: Tensor, q: int) -> Tensor:
    """执行跨后端明确的 [0,q) canonical 模约简。"""
    _require_int64(value, "value")
    if type(q) is not int or q <= 0:
        raise G1BError("G1B_MOD_Q", "q 必须是正 Python int")
    # 对正 q 显式执行两次 remainder，避免把负余数语义留给调用方。
    return torch.remainder(torch.remainder(value, q) + q, q)


def centered_lift_tensor(value: Tensor, q: int) -> Tensor:
    """将 [0,q) tensor 映射到奇数 q 的 centered 区间。"""
    _require_int64(value, "value")
    if type(q) is not int or q < 3 or q % 2 == 0:
        raise G1BError("G1B_LIFT_Q", "centered lift 要求奇数 q")
    if bool(torch.any(value < 0).item()) or bool(torch.any(value >= q).item()):
        raise G1BError("G1B_LIFT_DOMAIN", "lift 输入必须位于 [0,q)")
    return torch.where(value <= q // 2, value, value - q)


def _tensor_credentials(credentials: object, n: int) -> Tensor:
    """把 credential 转换为严格 CPU int64 batch，不接受浮点输入。"""
    if isinstance(credentials, Tensor):
        tensor = credentials
    elif isinstance(credentials, np.ndarray):
        if credentials.dtype.kind not in "iu" or credentials.dtype.itemsize != 8 or credentials.dtype.byteorder not in ("<", "="):
            raise G1BError("G1B_INPUT_DTYPE", "NumPy credential 必须是 little-endian int64")
        tensor = torch.from_numpy(np.asarray(credentials, dtype="<i8", order="C"))
    else:
        # 先检查 Python 序列的叶子类型，避免 torch.tensor 把 1.0/True 静默转成 int64。
        if not isinstance(credentials, (list, tuple)) or not credentials or any(
            not isinstance(row, (list, tuple))
            or len(row) != n
            or any(type(item) is not int for item in row)
            for row in credentials
        ):
            raise G1BError("G1B_INPUT_TYPE", "credential 序列必须只包含 Python int")
        try:
            tensor = torch.tensor(credentials, dtype=torch.int64)
        except (TypeError, ValueError) as exc:
            raise G1BError("G1B_INPUT_TYPE", "credential 无法转换为 int64") from exc
    _require_int64(tensor, "credentials")
    if tensor.ndim != 2 or tensor.shape[1] != n or tensor.shape[0] == 0:
        raise G1BError("G1B_INPUT_SHAPE", "credential 必须是非空 [B,n] batch")
    return tensor


def _residual_digest(residuals: Tuple[Tuple[int, ...], ...]) -> str:
    """使用 little-endian int64 对完整 residual 计算 SHA-256。"""
    payload = bytearray()
    for row in residuals:
        for value in row:
            try:
                payload.extend(struct.pack("<q", value))
            except struct.error as exc:
                raise G1BError("G1B_RESIDUAL_WIDTH", "residual 超出 int64 digest 范围") from exc
    return hashlib.sha256(bytes(payload)).hexdigest()


def verify_tensor_batch(
    credentials: object,
    parameters: G1AParameters,
    batch_indices: Optional[Sequence[int]] = None,
) -> G1BResult:
    """使用固定 CPU int64 kernel 生成与 G1-a 相同的离散 evidence。"""
    bounds = derive_integer_bounds(parameters)
    tensor = _tensor_credentials(credentials, parameters.config.n)
    if bool(torch.any(tensor < 0).item()) or bool(torch.any(tensor >= parameters.config.q).item()):
        raise G1BError("G1B_INPUT_DOMAIN", "credential 分量必须位于 [0,q)")
    if batch_indices is None:
        indices = tuple(range(tensor.shape[0]))
    else:
        indices = tuple(batch_indices)
        if len(indices) != tensor.shape[0] or indices != tuple(sorted(indices)) or len(set(indices)) != len(indices) or any(type(index) is not int or index < 0 for index in indices):
            raise G1BError("G1B_INPUT_INDEX", "batch_indices 必须严格递增并与 batch 对齐")
    A = torch.from_numpy(np.array(parameters.A, dtype="<i8", order="C", copy=True))
    b = torch.from_numpy(np.array(parameters.b, dtype="<i8", order="C", copy=True))
    # 上界已经覆盖完整 canonical 域，此处的 int64 matmul 不允许发生环绕。
    raw = torch.matmul(tensor, A.transpose(0, 1)) - b
    reduced = canonical_mod_q(raw, parameters.config.q)
    residual = centered_lift_tensor(reduced, parameters.config.q)
    max_abs = torch.amax(torch.abs(residual), dim=1)
    accepted_tensor = max_abs <= parameters.config.tau
    raw_rows = tuple(tuple(int(value) for value in row) for row in raw.tolist())
    residual_rows = tuple(tuple(int(value) for value in row) for row in residual.tolist())
    accepted = tuple(bool(value) for value in accepted_tensor.tolist())
    maximum = tuple(int(value) for value in max_abs.tolist())
    reasons = tuple("RELATION_WITHIN_BOUND" if value else "RELATION_OUT_OF_BOUND" for value in accepted)
    evidence = G1aEvidence(
        protocol_id=parameters.config.protocol_id,
        parameter_digest=parameter_digest(parameters),
        accepted=accepted,
        max_abs_residual=maximum,
        residual_digest=_residual_digest(residual_rows),
        reason_code=reasons,
        batch_indices=indices,
    )
    return G1BResult(evidence=evidence, residuals=residual_rows, raw_values=raw_rows, bounds=bounds)
