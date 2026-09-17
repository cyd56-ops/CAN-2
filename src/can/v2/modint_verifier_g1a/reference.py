"""G1-a 纯 Python int reference 运算。"""

from __future__ import annotations

import hashlib
import struct
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .parameters import parameter_digest
from .types import G1AError, G1AParameters, G1AReferenceResult, G1aEvidence


def canonical_mod(value: int, q: int) -> int:
    """返回 [0,q) 内的 canonical 余数，不依赖语言的负余数约定。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise G1AError("G1A_MOD_TYPE", "模约简输入必须是 Python int")
    if isinstance(q, bool) or not isinstance(q, int) or q <= 0:
        raise G1AError("G1A_MOD_Q", "模数必须是正整数")
    return ((value % q) + q) % q


def centered_lift(value: int, q: int) -> int:
    """把 [0,q) 代表元映射到奇数 q 的 centered 区间。"""
    if isinstance(q, bool) or not isinstance(q, int) or q < 3 or q % 2 == 0:
        raise G1AError("G1A_LIFT_Q", "centered lift 要求奇数 q")
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < q:
        raise G1AError("G1A_LIFT_DOMAIN", "lift 输入必须位于 [0,q)")
    return value if value <= q // 2 else value - q


def _parse_batch(credentials: object, n: int, q: int) -> Tuple[Tuple[int, ...], ...]:
    """严格解析二维 credential batch，拒绝浮点、布尔和非 canonical 数值。"""
    if isinstance(credentials, np.ndarray):
        if credentials.ndim != 2 or credentials.shape[1] != n or credentials.shape[0] == 0:
            raise G1AError("G1A_INPUT_SHAPE", "credential 必须是非空 [B,n] 数组")
        if credentials.dtype.kind not in "iu" or credentials.dtype.itemsize != 8 or credentials.dtype.byteorder not in ("<", "="):
            raise G1AError("G1A_INPUT_DTYPE", "credential 必须是 little-endian int64")
        if not credentials.flags.c_contiguous:
            raise G1AError("G1A_INPUT_LAYOUT", "credential 必须是 C-contiguous")
        rows = credentials.tolist()
    elif isinstance(credentials, (list, tuple)):
        if not credentials:
            raise G1AError("G1A_INPUT_EMPTY", "credential batch 不能为空")
        rows = credentials
    else:
        raise G1AError("G1A_INPUT_TYPE", "credential 必须是二维序列或 NumPy 数组")
    parsed = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise G1AError("G1A_INPUT_SHAPE", "每个 credential 行长度必须为 n")
        values = []
        for value in row:
            if type(value) is not int or not 0 <= value < q:
                raise G1AError("G1A_INPUT_DOMAIN", "credential 分量必须是 [0,q) 内的 Python int")
            values.append(value)
        parsed.append(tuple(values))
    return tuple(parsed)


def _residual_digest(residuals: Tuple[Tuple[int, ...], ...]) -> str:
    """对完整 residual 使用显式 little-endian int64 编码计算摘要。"""
    payload = bytearray()
    for row in residuals:
        for value in row:
            try:
                payload.extend(struct.pack("<q", value))
            except struct.error as exc:
                raise G1AError("G1A_RESIDUAL_WIDTH", "residual 超出 int64 digest 编码范围") from exc
    return hashlib.sha256(bytes(payload)).hexdigest()


def verify_batch(
    credentials: object,
    parameters: G1AParameters,
    batch_indices: Optional[Sequence[int]] = None,
) -> G1AReferenceResult:
    """逐行计算模整数关系并生成 reference evidence。"""
    rows = _parse_batch(credentials, parameters.config.n, parameters.config.q)
    if batch_indices is None:
        indices = tuple(range(len(rows)))
    else:
        indices = tuple(batch_indices)
        if len(indices) != len(rows) or any(type(i) is not int or i < 0 for i in indices) or indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise G1AError("G1A_INPUT_INDEX", "batch_indices 必须与 batch 对齐且严格递增")
    raw_values = []
    reduced = []
    residuals = []
    accepted = []
    max_abs = []
    reasons = []
    for row in rows:
        raw_row = []
        reduced_row = []
        residual_row = []
        for i in range(parameters.config.m):
            z = sum(int(parameters.A[i, j]) * row[j] for j in range(parameters.config.n)) - int(parameters.b[i])
            u = canonical_mod(z, parameters.config.q)
            r = centered_lift(u, parameters.config.q)
            raw_row.append(z)
            reduced_row.append(u)
            residual_row.append(r)
        maximum = max(abs(value) for value in residual_row)
        is_accepted = maximum <= parameters.config.tau
        raw_values.append(tuple(raw_row))
        reduced.append(tuple(reduced_row))
        residuals.append(tuple(residual_row))
        accepted.append(is_accepted)
        max_abs.append(maximum)
        reasons.append("RELATION_WITHIN_BOUND" if is_accepted else "RELATION_OUT_OF_BOUND")
    residual_tuple = tuple(residuals)
    evidence = G1aEvidence(
        protocol_id=parameters.config.protocol_id,
        parameter_digest=parameter_digest(parameters),
        accepted=tuple(accepted),
        max_abs_residual=tuple(max_abs),
        residual_digest=_residual_digest(residual_tuple),
        reason_code=tuple(reasons),
        batch_indices=indices,
    )
    return G1AReferenceResult(evidence=evidence, residuals=residual_tuple, reduced=tuple(reduced), raw_values=tuple(raw_values))
