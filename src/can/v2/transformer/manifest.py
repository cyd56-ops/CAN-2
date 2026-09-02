"""Phase 5 T1 的 checkpoint/data manifest 完整性工具。"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def sha256_file(path: Path) -> str:
    """按二进制内容计算文件 SHA-256。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("manifest 目标文件不存在")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_manifest(
    checkpoint_root: Path,
    checkpoints: Mapping[str, Mapping[str, Any]],
    version: int = 1,
) -> Dict[str, Any]:
    """构造独立 checkpoint manifest，不读取 checkpoint 内部摘要作为信任源。"""

    if not isinstance(checkpoint_root, Path) or not checkpoint_root.is_dir():
        raise NotADirectoryError("checkpoint_root 必须是目录")
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("checkpoints 不能为空映射")
    entries: Dict[str, Dict[str, Any]] = {}
    for key, metadata in checkpoints.items():
        if (
            not isinstance(key, str)
            or not key
            or Path(key).is_absolute()
            or "\\" in key
        ):
            raise ValueError("checkpoint key 必须是相对 POSIX 路径")
        path = checkpoint_root.joinpath(*key.split("/"))
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在: {key}")
        if not isinstance(metadata, Mapping):
            raise TypeError("checkpoint metadata 必须是 Mapping")
        entries[key] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            **dict(metadata),
        }
    return {"manifest_version": version, "checkpoints": entries}


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    """规范化写入 manifest 并返回 manifest 自身 SHA-256。"""

    if not isinstance(path, Path) or not isinstance(manifest, Mapping):
        raise TypeError("path 和 manifest 类型非法")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return sha256_file(path)


def verify_manifest_entry(
    checkpoint: Path, manifest: Mapping[str, Any], key: str
) -> None:
    """校验 checkpoint 与独立 manifest 的摘要和大小。"""

    if not isinstance(key, str) or not isinstance(manifest, Mapping):
        raise TypeError("manifest 参数类型非法")
    entries = manifest.get("checkpoints")
    if not isinstance(entries, Mapping) or key not in entries:
        raise KeyError("checkpoint key 不在 manifest 中")
    entry = entries[key]
    if not isinstance(entry, Mapping):
        raise ValueError("manifest checkpoint 条目非法")
    if entry.get("sha256") != sha256_file(checkpoint):
        raise ValueError("checkpoint SHA-256 不匹配")
    if entry.get("size_bytes") != checkpoint.stat().st_size:
        raise ValueError("checkpoint 文件大小不匹配")


def verify_checkpoint_integrity(
    checkpoint: Path,
    expected_sha256: Optional[str] = None,
    manifest: Optional[Mapping[str, Any]] = None,
    manifest_key: Optional[str] = None,
) -> str:
    """按显式摘要或独立 manifest 校验 checkpoint，返回实际摘要。"""

    if expected_sha256 is not None and manifest is not None:
        raise ValueError("expected_sha256 与 manifest 不能同时提供")
    actual = sha256_file(checkpoint)
    if expected_sha256 is not None:
        if actual != expected_sha256:
            raise ValueError("checkpoint SHA-256 不匹配")
    elif manifest is not None:
        if not manifest_key:
            raise ValueError("manifest 模式必须提供 manifest_key")
        verify_manifest_entry(checkpoint, manifest, manifest_key)
    return actual


__all__ = [
    "build_checkpoint_manifest",
    "sha256_file",
    "verify_manifest_entry",
    "verify_checkpoint_integrity",
    "write_manifest",
]
