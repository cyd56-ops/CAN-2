"""G1-b CPU backend 的不可变类型和稳定错误。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from ..modint_verifier_g1a.types import G1AError, G1AReferenceResult, G1aEvidence


class G1BError(ValueError):
    """表示 G1-b backend 输入、位宽或计算契约失败。"""

    def __init__(self, code: str, message: str) -> None:
        """使用稳定错误码构造异常。"""
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class IntegerBounds:
    """保存 canonical 域的解析整数上界。"""

    q: int
    n: int
    m: int
    max_input_value: int
    max_product_abs: int
    max_accumulator_abs: int
    required_signed_bits: int
    formula_version: str = "g1b-bounds-v1"

    def __post_init__(self) -> None:
        """验证上界字段为非负 Python int。"""
        values = (self.q, self.n, self.m, self.max_input_value, self.max_product_abs, self.max_accumulator_abs, self.required_signed_bits)
        if any(type(value) is not int or value < 0 for value in values):
            raise G1BError("G1B_BOUNDS_SCHEMA", "integer_bounds 字段必须是非负 Python int")
        if not self.formula_version:
            raise G1BError("G1B_BOUNDS_SCHEMA", "formula_version 不能为空")

    def to_dict(self) -> dict:
        """转换为 JSON 兼容摘要。"""
        return {
            "q": self.q,
            "n": self.n,
            "m": self.m,
            "max_input_value": self.max_input_value,
            "max_product_abs": self.max_product_abs,
            "max_accumulator_abs": self.max_accumulator_abs,
            "required_signed_bits": self.required_signed_bits,
            "formula_version": self.formula_version,
        }


@dataclass(frozen=True)
class G1BResult:
    """保存 G1-b evidence、残差和解析上界。"""

    evidence: G1aEvidence
    residuals: Tuple[Tuple[int, ...], ...]
    raw_values: Tuple[Tuple[int, ...], ...]
    bounds: IntegerBounds
    backend_id: str = "g1b-torch-int64-cpu-v1"

    def __post_init__(self) -> None:
        """验证 backend 结果使用固定 CPU backend 标识。"""
        if self.backend_id != "g1b-torch-int64-cpu-v1":
            raise G1BError("G1B_BACKEND_ID", "当前只允许 CPU int64 backend")
