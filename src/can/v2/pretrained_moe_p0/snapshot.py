"""冻结真实宿主 snapshot 的盘点、只读复核和受控下载工具。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .p0a import canonical_json_bytes, sha256_file
from .real_types import SnapshotFile, SnapshotManifest
from .types import CandidateSpec, P0Error

_P0_V1_MAX_SNAPSHOT_BYTES = 80 * 1024**3


def _safe_path(path: Path, root: Path) -> str:
    """将路径转换为安全 POSIX 相对路径并拒绝符号链接。"""

    if path.is_symlink():
        raise P0Error("snapshot_symlink_rejected", f"snapshot 含符号链接: {path}")
    relative = path.relative_to(root).as_posix()
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise P0Error("snapshot_path_escape", f"snapshot 路径越界: {relative}")
    return relative


def inventory_snapshot(root: Path) -> Tuple[SnapshotFile, ...]:
    """盘点 snapshot 的普通文件并计算 SHA-256。"""

    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise P0Error("snapshot_missing", "snapshot 根目录不存在或为符号链接")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = _safe_path(path, root)
        # 生成的 manifest 是证据旁车文件，不属于模型 snapshot inventory。
        if relative == "snapshot_manifest.json":
            continue
        files.append(
            SnapshotFile(
                relative,
                path.stat().st_size,
                sha256_file(path),
            )
        )
    if not files:
        raise P0Error("snapshot_empty", "snapshot 不得为空")
    return tuple(files)


def verify_read_only(root: Path, files: Iterable[SnapshotFile]) -> bool:
    """验证 snapshot 文件和目录没有写权限。"""

    paths = [root, *(root / item.path for item in files)]
    for path in paths:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            return False
    return True


def freeze_snapshot(root: Path) -> None:
    """移除 snapshot 内文件和目录的全部写权限，根目录最后处理。"""

    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise P0Error("snapshot_missing", "snapshot 根目录不存在或为符号链接")
    paths = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in [*paths, root]:
        if path.is_symlink():
            raise P0Error("snapshot_symlink_rejected", f"snapshot 含符号链接: {path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _existing_snapshot_bytes(parent: Path) -> int:
    """统计同一 P0-v1 snapshot 根下已有 manifest 的冻结字节数。"""

    total = 0
    if not parent.exists():
        return total
    for path in parent.rglob("snapshot_manifest.json"):
        manifest = load_snapshot_manifest(path)
        value = manifest["total_bytes"]
        total += int(value)
    return total


def _declared_snapshot_size(candidate: CandidateSpec) -> int:
    """从 Hub 固定 revision 获取完整声明大小，并复核 resolved commit。"""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise P0Error("optional_dependency_missing", "需要 huggingface_hub") from exc
    try:
        info = HfApi().model_info(
            candidate.repository_id,
            revision=candidate.resolved_commit_sha,
            files_metadata=True,
        )
    except Exception as exc:  # pragma: no cover - 服务器联网路径
        raise P0Error(
            "snapshot_metadata_unavailable", "无法读取 snapshot 元数据"
        ) from exc
    if getattr(info, "sha", None) != candidate.resolved_commit_sha:
        raise P0Error("revision_not_immutable", "Hub 返回 revision 与 registry 不一致")
    siblings = tuple(getattr(info, "siblings", ()) or ())
    sizes = tuple(getattr(item, "size", None) for item in siblings)
    if not sizes or any(type(value) is not int or value < 0 for value in sizes):
        raise P0Error("snapshot_metadata_incomplete", "Hub 文件大小元数据不完整")
    return sum(sizes)


def validate_download_budget(
    candidate: CandidateSpec, root: Path, declared_size_bytes: int
) -> None:
    """在下载前校验候选、P0-v1 总预算和目标磁盘可用空间。"""

    if type(declared_size_bytes) is not int or declared_size_bytes <= 0:
        raise P0Error("snapshot_metadata_incomplete", "snapshot 声明大小非法")
    if declared_size_bytes > candidate.max_snapshot_bytes:
        raise P0Error("snapshot_size_invalid", "snapshot 声明大小超过候选预算")
    root.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_snapshot_bytes(root.parent)
    if existing + declared_size_bytes > _P0_V1_MAX_SNAPSHOT_BYTES:
        raise P0Error("snapshot_total_budget_exceeded", "P0-v1 snapshot 超过 80 GiB")
    if shutil.disk_usage(root.parent).free < declared_size_bytes:
        raise P0Error("snapshot_disk_space_insufficient", "下载目标磁盘空间不足")


def manifest_payload(
    candidate: CandidateSpec,
    root: Path,
    files: Tuple[SnapshotFile, ...],
    read_only_verified: bool = False,
    offline_verified: bool = False,
) -> Dict[str, Any]:
    """构造不含秘密的 snapshot manifest payload。"""

    return {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "repository_id": candidate.repository_id,
        "resolved_commit_sha": candidate.resolved_commit_sha,
        "root": str(root.resolve()),
        "files": [
            {"path": item.path, "size_bytes": item.size_bytes, "sha256": item.sha256}
            for item in files
        ],
        "total_bytes": sum(item.size_bytes for item in files),
        "read_only_verified": bool(read_only_verified),
        "offline_verified": bool(offline_verified),
    }


def write_snapshot_manifest(
    candidate: CandidateSpec,
    root: Path,
    output: Path,
    *,
    offline_verified: bool = False,
    freeze: bool = False,
) -> SnapshotManifest:
    """盘点 snapshot 并以不可覆盖方式写入 manifest。"""

    files = inventory_snapshot(root)
    read_only = freeze or verify_read_only(root, files)
    payload = manifest_payload(candidate, root, files, read_only, offline_verified)
    raw = canonical_json_bytes(payload)
    if output.exists():
        raise P0Error("artifact_exists", "snapshot manifest 已存在")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if freeze:
        freeze_snapshot(root)
        if not verify_read_only(root, files):
            raise P0Error("snapshot_not_read_only", "snapshot 只读冻结失败")
    return SnapshotManifest(
        candidate.candidate_id,
        candidate.repository_id,
        candidate.resolved_commit_sha,
        str(root.resolve()),
        files,
        int(payload["total_bytes"]),
        digest,
        read_only,
        offline_verified,
    )


def load_snapshot_manifest(path: Path) -> Mapping[str, Any]:
    """读取 snapshot manifest 并拒绝重复字段、非有限数值和未知 schema。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(path)

    def duplicate(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise P0Error("artifact_duplicate_field", f"重复字段: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except P0Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise P0Error(
            "artifact_json_invalid", "snapshot manifest 不是规范 JSON"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise P0Error("artifact_schema_mismatch", "snapshot manifest 版本非法")
    required = {
        "schema_version",
        "candidate_id",
        "repository_id",
        "resolved_commit_sha",
        "root",
        "files",
        "total_bytes",
        "read_only_verified",
        "offline_verified",
    }
    if set(payload) != required or not isinstance(payload["files"], list):
        raise P0Error("artifact_schema_mismatch", "snapshot manifest 字段不完整")
    if type(payload["total_bytes"]) is not int or payload["total_bytes"] < 1:
        raise P0Error("artifact_schema_mismatch", "snapshot total_bytes 非法")
    if (
        type(payload["read_only_verified"]) is not bool
        or type(payload["offline_verified"]) is not bool
    ):
        raise P0Error("artifact_schema_mismatch", "snapshot 状态字段非法")
    return payload


def verify_snapshot_against_manifest(
    candidate: CandidateSpec, root: Path, manifest: Mapping[str, Any]
) -> SnapshotManifest:
    """重新盘点并验证 snapshot 与冻结 manifest、registry 完全一致。"""

    if manifest.get("candidate_id") != candidate.candidate_id:
        raise P0Error("snapshot_binding_mismatch", "snapshot candidate 不匹配")
    if manifest.get("repository_id") != candidate.repository_id:
        raise P0Error("snapshot_binding_mismatch", "snapshot repository 不匹配")
    if manifest.get("resolved_commit_sha") != candidate.resolved_commit_sha:
        raise P0Error("revision_not_immutable", "snapshot revision 不匹配")
    files = inventory_snapshot(root)
    expected = tuple(
        SnapshotFile(item["path"], item["size_bytes"], item["sha256"])
        for item in manifest["files"]
        if isinstance(item, dict)
        and set(item) == {"path", "size_bytes", "sha256"}
        and type(item["size_bytes"]) is int
    )
    if len(expected) != len(manifest["files"]) or files != expected:
        raise P0Error("snapshot_digest_mismatch", "snapshot inventory 漂移")
    read_only = verify_read_only(root, files)
    if not read_only:
        raise P0Error("snapshot_not_read_only", "snapshot 仍可写")
    total = sum(item.size_bytes for item in files)
    if total != manifest["total_bytes"] or total > candidate.max_snapshot_bytes:
        raise P0Error("snapshot_size_invalid", "snapshot 超出冻结预算")
    raw = canonical_json_bytes(dict(manifest))
    return SnapshotManifest(
        candidate.candidate_id,
        candidate.repository_id,
        candidate.resolved_commit_sha,
        str(root.resolve()),
        files,
        total,
        hashlib.sha256(raw).hexdigest(),
        True,
        bool(manifest["offline_verified"]),
    )


def download_snapshot(  # pragma: no cover - 服务器 Hugging Face 下载路径
    candidate: CandidateSpec, root: Path
) -> SnapshotManifest:
    """按 registry 固定 revision 下载完整 snapshot；依赖仅在服务器导入。"""

    if candidate.p0a_status != "passed":
        raise P0Error("p0a_rejected", "只有正式 P0-A passed 候选允许下载")
    if root.exists():
        raise P0Error("artifact_exists", "snapshot 目录已存在，拒绝覆盖")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise P0Error("optional_dependency_missing", "需要 huggingface_hub") from exc
    declared_size = _declared_snapshot_size(candidate)
    validate_download_budget(candidate, root, declared_size)
    try:
        snapshot_download(
            repo_id=candidate.repository_id,
            revision=candidate.resolved_commit_sha,
            local_dir=str(root),
            local_dir_use_symlinks=False,
            allow_patterns=None,
            resume_download=False,
        )
    except Exception as exc:  # pragma: no cover - 服务器依赖路径
        raise P0Error("snapshot_download_failed", "snapshot 下载失败") from exc
    files = inventory_snapshot(root)
    total = sum(item.size_bytes for item in files)
    if total > candidate.max_snapshot_bytes:
        raise P0Error("snapshot_size_invalid", "下载后 snapshot 超出冻结预算")
    return write_snapshot_manifest(
        candidate,
        root,
        root / "snapshot_manifest.json",
        freeze=True,
    )
