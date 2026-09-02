"""Phase 5 正式实验冻结记录的加载与一致性校验。"""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional


def load_freeze_record(path: Path) -> Mapping[str, Any]:
    """读取并验证 freeze record，拒绝缺失核心字段或非法 JSON。"""
    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("freeze record 不存在")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("freeze record 不是有效 JSON") from exc
    if not isinstance(record, dict):
        raise ValueError("freeze record 顶层必须是对象")
    required = ("freeze_version", "generator_version", "batch_size", "cache_mode")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"freeze record 缺少字段: {', '.join(missing)}")
    if (
        not isinstance(record["batch_size"], int)
        or isinstance(record["batch_size"], bool)
        or record["batch_size"] < 6
        or record["batch_size"] % 3
    ):
        raise ValueError("freeze record.batch_size 必须是 >=6 的 3 倍数")
    if record["cache_mode"] not in {"none", "kv"}:
        raise ValueError("freeze record.cache_mode 必须为 none 或 kv")
    if (
        not isinstance(record["generator_version"], str)
        or not record["generator_version"]
    ):
        raise ValueError("freeze record.generator_version 非法")
    return record


def freeze_record_sha256(path: Path) -> str:
    """计算 freeze record 原始字节 SHA-256。"""
    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("freeze record 不存在")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_runtime_against_freeze(
    record: Mapping[str, Any],
    *,
    batch_size: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    cache_mode: Optional[str] = None,
    generator_version: Optional[str] = None,
) -> None:
    """校验运行时覆盖项与冻结值一致，任何偏差立即失败。"""
    if not isinstance(record, Mapping):
        raise TypeError("record 必须是 Mapping")
    checks = {
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "cache_mode": cache_mode,
        "generator_version": generator_version,
    }
    for key, value in checks.items():
        if value is not None and key in record and record[key] != value:
            raise ValueError(
                f"运行时 {key}={value} 与 freeze record 的 {record[key]} 不一致"
            )


__all__ = [
    "freeze_record_sha256",
    "load_freeze_record",
    "validate_runtime_against_freeze",
]
