"""G1-b CPU backend manifest 的严格 schema。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .types import G1BError, IntegerBounds


_FIELDS = {
    "schema_version",
    "execution_config_id",
    "protocol_id",
    "backend_id",
    "dtype",
    "device",
    "source_parameter_sha256",
    "integer_bounds",
    "vectors_sha256",
    "provenance",
}


def _sha(value: object, name: str, lengths=(64,)) -> None:
    """验证摘要字段使用小写十六进制。"""
    if not isinstance(value, str) or len(value) not in lengths or value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise G1BError("G1B_MANIFEST_SCHEMA", f"{name} 摘要格式非法")


def build_g1b_manifest(
    source_parameter_sha256: str,
    bounds: IntegerBounds,
    vectors_sha256: str,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """生成固定 CPU int64 backend manifest。"""
    _sha(source_parameter_sha256, "source_parameter_sha256")
    _sha(vectors_sha256, "vectors_sha256")
    if not isinstance(provenance, dict) or set(provenance) != {"git_commit", "dirty", "python", "torch"}:
        raise G1BError("G1B_MANIFEST_PROVENANCE", "provenance schema 不匹配")
    _sha(provenance["git_commit"], "git_commit", (40, 64))
    if type(provenance["dirty"]) is not bool:
        raise G1BError("G1B_MANIFEST_PROVENANCE", "dirty 必须为 bool")
    return {
        "schema_version": 1,
        "execution_config_id": "g1b-modint-neural-v1",
        "protocol_id": "g1b-modint-v1",
        "backend_id": "g1b-torch-int64-cpu-v1",
        "dtype": "torch.int64",
        "device": "cpu",
        "source_parameter_sha256": source_parameter_sha256,
        "integer_bounds": bounds.to_dict(),
        "vectors_sha256": vectors_sha256,
        "provenance": provenance,
    }


def validate_g1b_manifest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """严格校验 G1-b manifest 的字段、backend 和上界 schema。"""
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise G1BError("G1B_MANIFEST_SCHEMA", "manifest 字段集合不匹配")
    if payload["schema_version"] != 1 or payload["execution_config_id"] != "g1b-modint-neural-v1" or payload["protocol_id"] != "g1b-modint-v1":
        raise G1BError("G1B_MANIFEST_SCHEMA", "协议或 schema 版本不匹配")
    if payload["backend_id"] != "g1b-torch-int64-cpu-v1" or payload["dtype"] != "torch.int64" or payload["device"] != "cpu":
        raise G1BError("G1B_MANIFEST_BACKEND", "当前 manifest 必须描述 CPU int64 backend")
    _sha(payload["source_parameter_sha256"], "source_parameter_sha256")
    _sha(payload["vectors_sha256"], "vectors_sha256")
    bounds = payload["integer_bounds"]
    required = {"q", "n", "m", "max_input_value", "max_product_abs", "max_accumulator_abs", "required_signed_bits", "formula_version"}
    if not isinstance(bounds, dict) or set(bounds) != required:
        raise G1BError("G1B_MANIFEST_BOUNDS", "integer_bounds 字段不完整")
    IntegerBounds(**bounds)
    provenance = payload["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"git_commit", "dirty", "python", "torch"}:
        raise G1BError("G1B_MANIFEST_PROVENANCE", "provenance schema 不匹配")
    _sha(provenance["git_commit"], "git_commit", (40, 64))
    if type(provenance["dirty"]) is not bool:
        raise G1BError("G1B_MANIFEST_PROVENANCE", "dirty 必须为 bool")
    return payload


def load_g1b_manifest(path: Path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """按原始 bytes 校验摘要后读取并验证 G1-b manifest。"""
    raw = Path(path).read_bytes()
    if expected_sha256 is not None:
        _sha(expected_sha256, "expected_sha256")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise G1BError("G1B_MANIFEST_DIGEST_MISMATCH", "manifest bytes 摘要不匹配")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G1BError("G1B_MANIFEST_JSON", "manifest 不是合法 JSON") from exc
    return validate_g1b_manifest(payload)
