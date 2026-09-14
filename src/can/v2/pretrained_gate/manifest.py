"""G0/P0/P1 严格 provenance manifest 的基础 schema 校验。"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from pathlib import Path
from typing import Any

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "protocol_id",
    "execution_config_id",
    "model",
    "environment",
    "inputs",
    "tolerance",
}
_NESTED_FIELDS = {
    "model": {"revision"},
    "environment": {"python"},
    "inputs": {"sha256"},
    "tolerance": {"dtype", "atol", "rtol"},
}


def file_sha256(path: Path) -> str:
    """计算文件 SHA-256。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("目标文件不存在")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """读取 G0 schema v1 manifest，严格拒绝重复、未知及非规范字段。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("manifest 不存在")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """拒绝同一 JSON object 中的重复字段。"""

        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"manifest 字段重复: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        """拒绝 JSON 标准之外的 NaN 与 Infinity 常量。"""

        raise ValueError(f"manifest 包含非有限常量: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError("manifest 根必须是 object")
    unknown = set(value) - _TOP_LEVEL_FIELDS
    if unknown or not _TOP_LEVEL_FIELDS.issubset(value):
        raise ValueError("manifest 缺少字段或包含未知字段")
    if isinstance(value["schema_version"], bool) or value["schema_version"] != 1:
        raise ValueError("schema_version 必须严格等于 1")
    for key in ("protocol_id", "execution_config_id"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"{key} 必须是非空字符串")
    for key in ("model", "environment", "inputs", "tolerance"):
        if not isinstance(value[key], dict):
            raise TypeError(f"{key} 必须是 object")
        if set(value[key]) != _NESTED_FIELDS[key]:
            raise ValueError(f"{key} 缺少字段或包含未知字段")

    _require_nonempty_string(value["model"]["revision"], "model.revision")
    _require_nonempty_string(value["environment"]["python"], "environment.python")
    _require_sha256(value["inputs"]["sha256"], "inputs.sha256")
    tolerance = value["tolerance"]
    if tolerance["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("tolerance.dtype 只允许 float32 或 bfloat16")
    for key in ("atol", "rtol"):
        number = tolerance[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise TypeError(f"tolerance.{key} 必须是实数")
        if not math.isfinite(float(number)) or float(number) < 0.0:
            raise ValueError(f"tolerance.{key} 必须是有限非负数")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    """验证 manifest 中的非空字符串字段。"""

    if not isinstance(value, str):
        raise TypeError(f"{name} 必须是字符串")
    if not value:
        raise ValueError(f"{name} 不能为空")
    return value


def _require_sha256(value: Any, name: str) -> str:
    """验证并规范化 64 位十六进制 SHA-256。"""

    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 64 位十六进制 SHA-256")
    return value.lower()


def verify_manifest_sha256(path: Path, expected_sha256: str) -> None:
    """校验 manifest 本身摘要，摘要必须从可信来源提供。"""

    expected = _require_sha256(expected_sha256, "expected_sha256")
    if not hmac.compare_digest(file_sha256(path), expected):
        raise ValueError("manifest SHA-256 不匹配")
