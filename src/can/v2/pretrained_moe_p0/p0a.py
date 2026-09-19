"""P0-A 元数据盘点、人工审阅校验与正式 registry 生成。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .artifacts import ArtifactWriter
from .registry import validate_registry
from .types import P0Error, require_exact_keys

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
_SCAN_PATTERNS = (
    "subprocess",
    "requests",
    "urllib",
    "socket",
    "open(",
    "os.environ",
    "hf_hub_download",
    "snapshot_download",
    "torch.hub",
    "credential",
)


@dataclass(frozen=True)
class P0ACandidatePreset:
    """保存一个 P0-A 候选的冻结身份和加载 profile。"""

    candidate_id: str
    repository_id: str
    attempt_order: int
    metadata_filename: str
    metadata_directory: str
    profile_id: str
    dtype: str
    quantization_config: Mapping[str, Any]
    allow_remote_code: bool
    expected_architecture_family: str
    expected_moe_variant: str
    max_snapshot_bytes: int = 40 * 1024**3


P0A_CANDIDATES: Tuple[P0ACandidatePreset, ...] = (
    P0ACandidatePreset(
        "C1",
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        1,
        "c1-qwen-metadata.json",
        "c1-files",
        "c1-bnb-nf4-bf16-v1",
        "bf16",
        {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        False,
        "qwen2_moe",
        "native_shared_and_routed",
    ),
    P0ACandidatePreset(
        "C2",
        "deepseek-ai/deepseek-moe-16b-chat",
        2,
        "c2-deepseek-metadata.json",
        "c2-files",
        "c2-bnb-nf4-bf16-v1",
        "bf16",
        {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        True,
        "deepseek",
        "native_shared_and_routed",
    ),
    P0ACandidatePreset(
        "C3",
        "ibm-granite/granite-3.1-1b-a400m-instruct",
        3,
        "c3-granite-metadata.json",
        "c3-files",
        "c3-bf16-v1",
        "bf16",
        {},
        False,
        "granitemoe",
        "unknown_requires_verification",
    ),
)


def canonical_json_bytes(payload: object) -> bytes:
    """将结构化对象编码为稳定的 ASCII JSON bytes。"""

    try:
        return json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise P0Error("artifact_json_invalid", "对象无法规范编码") from exc


def sha256_file(path: Path) -> str:
    """流式计算普通文件的 SHA-256。"""

    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise P0Error("artifact_validation_failed", f"不是普通文件: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """构造 JSON 对象并拒绝重复字段。"""

    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise P0Error("artifact_duplicate_field", f"重复字段 {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> object:
    """读取 UTF-8 JSON，并拒绝重复字段和非有限常量。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(path)
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except P0Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise P0Error("artifact_json_invalid", f"非法 JSON: {path}") from exc


def _safe_relative_path(value: object, context: str) -> str:
    """校验 artifact 中的 POSIX 相对路径，禁止逃逸根目录。"""

    if not isinstance(value, str) or not value or "\\" in value:
        raise P0Error("artifact_schema_mismatch", f"{context} 路径非法")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise P0Error("artifact_schema_mismatch", f"{context} 路径越界")
    return value


def _inventory(directory: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    """盘点候选小型元数据目录，并标记意外出现的权重文件。"""

    if not directory.is_dir() or directory.is_symlink():
        raise P0Error("metadata_unresolved", f"元数据目录不存在: {directory}")
    files: List[Dict[str, object]] = []
    forbidden: List[str] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise P0Error("artifact_validation_failed", f"元数据目录含符号链接: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if relative.lower().endswith(_WEIGHT_SUFFIXES):
            forbidden.append(relative)
    if not files:
        raise P0Error("metadata_unresolved", f"元数据目录为空: {directory}")
    return files, forbidden


def _validate_hub_metadata(
    payload: object, preset: P0ACandidatePreset
) -> Dict[str, Any]:
    """校验此前从官方 Hub 采集的元数据摘要。"""

    allowed = {
        "candidate_id",
        "repository_id",
        "requested_revision",
        "resolved_commit_sha",
        "pipeline_tag",
        "library_name",
        "tags",
        "siblings",
        "expected_moe_variant",
    }
    required = {
        "candidate_id",
        "repository_id",
        "requested_revision",
        "resolved_commit_sha",
        "siblings",
    }
    if (
        not isinstance(payload, dict)
        or not required.issubset(payload)
        or not set(payload).issubset(allowed)
    ):
        raise P0Error("artifact_schema_mismatch", "Hub metadata 字段集合不匹配")
    if (
        payload["candidate_id"] != preset.candidate_id
        or payload["repository_id"] != preset.repository_id
    ):
        raise P0Error("artifact_validation_failed", "候选身份与预登记不一致")
    requested = payload["requested_revision"]
    resolved = payload["resolved_commit_sha"]
    if not isinstance(requested, str) or not requested:
        raise P0Error("revision_not_immutable", "requested revision 非法")
    if not isinstance(resolved, str) or _COMMIT.fullmatch(resolved) is None:
        raise P0Error("revision_not_immutable", "resolved commit 必须为 40 位小写 SHA")
    siblings = payload["siblings"]
    if not isinstance(siblings, list) or not siblings:
        raise P0Error("metadata_unresolved", "Hub 文件清单为空")
    for item in siblings:
        if not isinstance(item, dict) or "rfilename" not in item or "size" not in item:
            raise P0Error("artifact_schema_mismatch", "Hub sibling 字段不完整")
        _safe_relative_path(item["rfilename"], "Hub sibling")
        size = item["size"]
        if size is not None and (type(size) is not int or size < 0):
            raise P0Error("artifact_schema_mismatch", "Hub sibling size 非法")
    return payload


def _architecture_fields(config: Mapping[str, Any]) -> Dict[str, Any]:
    """提取与 MoE/shared/router 结构相关的配置字段。"""

    terms = (
        "expert",
        "moe",
        "router",
        "route",
        "topk",
        "top_k",
        "shared",
        "routed",
        "intermediate",
    )
    return {
        key: config[key]
        for key in sorted(config)
        if isinstance(key, str) and any(term in key.lower() for term in terms)
    }


def _config_hints(config: Mapping[str, Any]) -> Tuple[bool, bool]:
    """仅从配置字段生成 shared/routed 的非授权性提示。"""

    shared_values = (
        config.get("num_shared_experts"),
        config.get("n_shared_experts"),
        config.get("shared_expert_intermediate_size"),
        config.get("shared_intermediate_size"),
    )
    routed_values = (
        config.get("num_experts"),
        config.get("num_local_experts"),
        config.get("n_routed_experts"),
    )
    shared = any(type(value) is int and value > 0 for value in shared_values)
    routed = any(type(value) is int and value > 0 for value in routed_values)
    return shared, routed


def _scan_python_files(
    directory: Path, inventory: Sequence[Mapping[str, object]]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """列出 Python 文件并执行保守的危险调用文本扫描。"""

    python_files: List[Dict[str, object]] = []
    findings: List[Dict[str, object]] = []
    for item in inventory:
        relative = str(item["path"])
        if not relative.endswith(".py"):
            continue
        python_files.append(dict(item))
        path = directory / PurePosixPath(relative)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise P0Error(
                "remote_code_rejected", f"Python 文件不是 UTF-8: {relative}"
            ) from exc
        for line_number, line in enumerate(lines, 1):
            lowered = line.lower().replace(" ", "")
            for pattern in _SCAN_PATTERNS:
                if pattern.lower().replace(" ", "") in lowered:
                    findings.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "pattern": pattern,
                        }
                    )
    return python_files, findings


def _environment_record() -> Dict[str, object]:
    """记录 P0-A 生成环境，不读取或保存秘密环境变量。"""

    packages: Dict[str, Optional[str]] = {}
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface-hub",
        "bitsandbytes",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def _review_reference(path: str, digest: str) -> Dict[str, str]:
    """构造人工审阅证据的路径/摘要引用。"""

    return {"path": path, "sha256": digest}


def _pretty_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    """以便于人工编辑的格式创建新 JSON，拒绝覆盖。"""

    if path.exists():
        raise P0Error("artifact_exists", f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
    except OSError as exc:
        raise P0Error("artifact_write_failed", f"无法创建 {path}") from exc


def _validate_prepared_manifest(
    input_root: Path,
    manifest: Mapping[str, Any],
    preset: P0ACandidatePreset,
) -> None:
    """根据原始 Hub JSON、config 和文件目录复算 prepared manifest。"""

    expected_keys = {
        "schema_version",
        "candidate_id",
        "repository_id",
        "attempt_order",
        "requested_revision",
        "resolved_commit_sha",
        "profile",
        "expected_architecture_family",
        "expected_moe_variant",
        "max_snapshot_bytes",
        "hub_metadata",
        "local_metadata",
        "architecture",
        "remote_code_scan",
        "environment",
    }
    require_exact_keys(manifest, expected_keys, "metadata manifest")
    fixed = (
        manifest["schema_version"] == 1
        and manifest["candidate_id"] == preset.candidate_id
        and manifest["repository_id"] == preset.repository_id
        and manifest["attempt_order"] == preset.attempt_order
        and manifest["expected_architecture_family"]
        == preset.expected_architecture_family
        and manifest["expected_moe_variant"] == preset.expected_moe_variant
        and manifest["max_snapshot_bytes"] == preset.max_snapshot_bytes
        and manifest["profile"]
        == {
            "profile_id": preset.profile_id,
            "dtype": preset.dtype,
            "quantization_config": dict(preset.quantization_config),
            "allow_remote_code": preset.allow_remote_code,
        }
    )
    if not fixed:
        raise P0Error("artifact_validation_failed", "metadata manifest 固定配置漂移")

    metadata_path = input_root / preset.metadata_filename
    metadata = _validate_hub_metadata(load_strict_json(metadata_path), preset)
    if (
        manifest["requested_revision"] != metadata["requested_revision"]
        or manifest["resolved_commit_sha"] != metadata["resolved_commit_sha"]
    ):
        raise P0Error("artifact_validation_failed", "metadata manifest revision 漂移")

    hub = require_exact_keys(
        manifest["hub_metadata"],
        {
            "source_path",
            "sha256",
            "file_count",
            "snapshot_size_complete",
            "snapshot_size_bytes",
        },
        "hub metadata binding",
    )
    sizes = [item.get("size") for item in metadata["siblings"]]
    complete = all(type(value) is int for value in sizes)
    total = sum(int(value) for value in sizes) if complete else None
    if hub != {
        "source_path": preset.metadata_filename,
        "sha256": sha256_file(metadata_path),
        "file_count": len(metadata["siblings"]),
        "snapshot_size_complete": complete,
        "snapshot_size_bytes": total,
    }:
        raise P0Error("snapshot_digest_mismatch", "Hub metadata binding 漂移")

    directory = input_root / preset.metadata_directory
    inventory, forbidden = _inventory(directory)
    local = require_exact_keys(
        manifest["local_metadata"],
        {
            "source_directory",
            "inventory",
            "inventory_sha256",
            "forbidden_weight_files",
        },
        "local metadata binding",
    )
    inventory_sha = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    if local != {
        "source_directory": preset.metadata_directory,
        "inventory": inventory,
        "inventory_sha256": inventory_sha,
        "forbidden_weight_files": forbidden,
    }:
        raise P0Error("snapshot_digest_mismatch", "本地 metadata inventory 漂移")

    config_path = directory / "config.json"
    config = load_strict_json(config_path)
    if not isinstance(config, dict):
        raise P0Error("metadata_unresolved", "config 不是 JSON 对象")
    shared_hint, routed_hint = _config_hints(config)
    architecture = require_exact_keys(
        manifest["architecture"],
        {
            "config_sha256",
            "model_type",
            "architectures",
            "fields",
            "config_shared_hint",
            "config_routed_hint",
        },
        "architecture binding",
    )
    if architecture != {
        "config_sha256": sha256_file(config_path),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "fields": _architecture_fields(config),
        "config_shared_hint": shared_hint,
        "config_routed_hint": routed_hint,
    }:
        raise P0Error("snapshot_digest_mismatch", "architecture evidence 漂移")

    python_files, findings = _scan_python_files(directory, inventory)
    remote_scan = require_exact_keys(
        manifest["remote_code_scan"], {"python_files", "findings"}, "remote scan"
    )
    if remote_scan != {"python_files": python_files, "findings": findings}:
        raise P0Error("snapshot_digest_mismatch", "remote-code scan 漂移")


def prepare_p0a_reviews(input_root: Path, output_root: Path) -> Mapping[str, object]:
    """盘点三个固定候选，并生成可人工填写的 P0-A 审阅模板。"""

    if not isinstance(input_root, Path) or not input_root.is_dir():
        raise P0Error("metadata_unresolved", "P0-A input root 不存在")
    if not isinstance(output_root, Path) or output_root.exists():
        raise P0Error("artifact_exists", "P0-A review output 已存在")

    prepared: List[Dict[str, object]] = []
    environment = _environment_record()
    # 先完成所有输入校验，避免输入错误留下半套模板。
    records: List[
        Tuple[
            P0ACandidatePreset,
            Dict[str, Any],
            List[Dict[str, object]],
            List[str],
            Dict[str, Any],
        ]
    ] = []
    for preset in P0A_CANDIDATES:
        metadata_path = input_root / preset.metadata_filename
        metadata = _validate_hub_metadata(load_strict_json(metadata_path), preset)
        directory = input_root / preset.metadata_directory
        inventory, forbidden = _inventory(directory)
        config_path = directory / "config.json"
        config = load_strict_json(config_path)
        if not isinstance(config, dict):
            raise P0Error(
                "metadata_unresolved", f"{preset.candidate_id} config 不是对象"
            )
        records.append((preset, metadata, inventory, forbidden, config))

    output_root.mkdir(parents=True)
    for preset, metadata, inventory, forbidden, config in records:
        candidate_dir = output_root / preset.candidate_id
        candidate_dir.mkdir()
        shared_hint, routed_hint = _config_hints(config)
        python_files, scan_findings = _scan_python_files(
            input_root / preset.metadata_directory, inventory
        )
        sibling_sizes = [item.get("size") for item in metadata["siblings"]]
        snapshot_size_complete = all(type(value) is int for value in sibling_sizes)
        snapshot_size = (
            sum(int(value) for value in sibling_sizes)
            if snapshot_size_complete
            else None
        )
        inventory_sha = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
        manifest = {
            "schema_version": 1,
            "candidate_id": preset.candidate_id,
            "repository_id": preset.repository_id,
            "attempt_order": preset.attempt_order,
            "requested_revision": metadata["requested_revision"],
            "resolved_commit_sha": metadata["resolved_commit_sha"],
            "profile": {
                "profile_id": preset.profile_id,
                "dtype": preset.dtype,
                "quantization_config": dict(preset.quantization_config),
                "allow_remote_code": preset.allow_remote_code,
            },
            "expected_architecture_family": preset.expected_architecture_family,
            "expected_moe_variant": preset.expected_moe_variant,
            "max_snapshot_bytes": preset.max_snapshot_bytes,
            "hub_metadata": {
                "source_path": preset.metadata_filename,
                "sha256": sha256_file(input_root / preset.metadata_filename),
                "file_count": len(metadata["siblings"]),
                "snapshot_size_complete": snapshot_size_complete,
                "snapshot_size_bytes": snapshot_size,
            },
            "local_metadata": {
                "source_directory": preset.metadata_directory,
                "inventory": inventory,
                "inventory_sha256": inventory_sha,
                "forbidden_weight_files": forbidden,
            },
            "architecture": {
                "config_sha256": sha256_file(
                    input_root / preset.metadata_directory / "config.json"
                ),
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "fields": _architecture_fields(config),
                "config_shared_hint": shared_hint,
                "config_routed_hint": routed_hint,
            },
            "remote_code_scan": {
                "python_files": python_files,
                "findings": scan_findings,
            },
            "environment": environment,
        }
        manifest_path = candidate_dir / "metadata_manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))

        config_reference = _review_reference(
            f"{preset.metadata_directory}/config.json",
            str(manifest["architecture"]["config_sha256"]),
        )
        source_evidence = [config_reference] + [
            _review_reference(
                f"{preset.metadata_directory}/{item['path']}", str(item["sha256"])
            )
            for item in python_files
        ]
        # 内置 Transformers 源码审阅常以独立文本保存，也纳入摘要绑定。
        prefix = preset.candidate_id.lower()
        for review_path in sorted(input_root.glob(f"{prefix}-*-source-review.txt")):
            source_evidence.append(
                _review_reference(
                    review_path.relative_to(input_root).as_posix(),
                    sha256_file(review_path),
                )
            )
        license_files = [
            _review_reference(
                f"{preset.metadata_directory}/{item['path']}", str(item["sha256"])
            )
            for item in inventory
            if Path(str(item["path"])).name.lower().startswith(("license", "readme"))
        ]
        source_review = {
            "schema_version": 1,
            "candidate_id": preset.candidate_id,
            "resolved_commit_sha": metadata["resolved_commit_sha"],
            "native_shared_expert": shared_hint,
            "native_routed_experts": routed_hint,
            "shared_always_executes": False,
            "mask_before_dispatch_static_feasibility": False,
            "expert_call_observability_static_feasibility": False,
            "evidence_files": source_evidence,
            "reviewer": "",
            "reviewed_at_utc": "",
            "status": "pending",
            "rationale": "",
        }
        license_review = {
            "schema_version": 1,
            "candidate_id": preset.candidate_id,
            "resolved_commit_sha": metadata["resolved_commit_sha"],
            "declared_license": "",
            "license_files": license_files,
            "usage_restrictions": [],
            "reviewer": "",
            "reviewed_at_utc": "",
            "status": "pending",
            "rationale": "",
        }
        remote_status = "pending" if preset.allow_remote_code else "not_required"
        review_findings = [
            {
                **finding,
                "review_disposition": "pending",
                "rationale": "",
            }
            for finding in scan_findings
        ]
        remote_review = {
            "schema_version": 1,
            "candidate_id": preset.candidate_id,
            "resolved_commit_sha": metadata["resolved_commit_sha"],
            "remote_code_required": preset.allow_remote_code,
            "allow_remote_code": preset.allow_remote_code,
            "reviewed_python_files": [
                _review_reference(
                    f"{preset.metadata_directory}/{item['path']}", str(item["sha256"])
                )
                for item in python_files
            ],
            "dangerous_call_findings": review_findings,
            "reviewer": (
                "" if remote_status == "pending" else "automatic-metadata-check"
            ),
            "reviewed_at_utc": (
                ""
                if remote_status == "pending"
                else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ),
            "status": remote_status,
            "rationale": (
                ""
                if remote_status == "pending"
                else "当前 profile 不启用 remote code。"
            ),
        }
        _pretty_write_new(candidate_dir / "source_review.json", source_review)
        _pretty_write_new(candidate_dir / "license_review.json", license_review)
        _pretty_write_new(candidate_dir / "remote_code_review.json", remote_review)
        prepared.append(
            {
                "candidate_id": preset.candidate_id,
                "resolved_commit_sha": metadata["resolved_commit_sha"],
                "metadata_manifest_sha256": sha256_file(manifest_path),
                "review_directory": preset.candidate_id,
            }
        )

    summary = {
        "schema_version": 1,
        "status": "review_required",
        "candidates": prepared,
    }
    (output_root / "prepare_summary.json").write_bytes(canonical_json_bytes(summary))
    return summary


def _validate_reference_list(
    value: object, input_root: Path, context: str
) -> Tuple[Mapping[str, str], ...]:
    """校验审阅证据引用，并复核对应文件摘要。"""

    if not isinstance(value, list):
        raise P0Error("artifact_schema_mismatch", f"{context} 必须是数组")
    result: List[Mapping[str, str]] = []
    for item in value:
        record = require_exact_keys(item, {"path", "sha256"}, context)
        relative = _safe_relative_path(record["path"], context)
        digest = record["sha256"]
        if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
            raise P0Error("artifact_schema_mismatch", f"{context} 摘要非法")
        target = input_root / PurePosixPath(relative)
        try:
            target.resolve().relative_to(input_root.resolve())
        except ValueError as exc:
            raise P0Error("artifact_schema_mismatch", f"{context} 路径越界") from exc
        if sha256_file(target) != digest:
            raise P0Error("snapshot_digest_mismatch", f"{context} 摘要漂移: {relative}")
        result.append({"path": relative, "sha256": digest})
    return tuple(result)


def _validate_review_identity(
    payload: Mapping[str, Any], manifest: Mapping[str, Any], context: str
) -> None:
    """验证人工审阅记录与当前候选和 revision 严格绑定。"""

    if (
        payload["schema_version"] != 1
        or payload["candidate_id"] != manifest["candidate_id"]
        or payload["resolved_commit_sha"] != manifest["resolved_commit_sha"]
    ):
        raise P0Error("artifact_validation_failed", f"{context} 身份不匹配")


def _validate_human_fields(payload: Mapping[str, Any], context: str) -> None:
    """校验已完成人工审阅的审阅人、UTC 时间和理由。"""

    if not isinstance(payload["reviewer"], str) or not payload["reviewer"].strip():
        raise P0Error("p0a_review_incomplete", f"{context} 缺少 reviewer")
    timestamp = payload["reviewed_at_utc"]
    if not isinstance(timestamp, str) or _UTC.fullmatch(timestamp) is None:
        raise P0Error("p0a_review_incomplete", f"{context} UTC 时间非法")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise P0Error("p0a_review_incomplete", f"{context} 缺少 rationale")


def _load_reviews(
    input_root: Path,
    prepared_dir: Path,
    preset: P0ACandidatePreset,
    *,
    require_human_review: bool = True,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """读取并校验一个候选的 manifest 和三类审阅记录。

    参数 require_human_review 为 False 时只校验机器可复核的结构证据，
    允许 prepare 生成的 pending 模板直接进入 provisional 预筛；这不构成
    正式 P0-A 通过，正式模式仍要求完整人工字段。
    """

    candidate_dir = prepared_dir / preset.candidate_id
    manifest = load_strict_json(candidate_dir / "metadata_manifest.json")
    if not isinstance(manifest, dict):
        raise P0Error("artifact_schema_mismatch", "metadata manifest 不是对象")
    _validate_prepared_manifest(input_root, manifest, preset)

    source_keys = {
        "schema_version",
        "candidate_id",
        "resolved_commit_sha",
        "native_shared_expert",
        "native_routed_experts",
        "shared_always_executes",
        "mask_before_dispatch_static_feasibility",
        "expert_call_observability_static_feasibility",
        "evidence_files",
        "reviewer",
        "reviewed_at_utc",
        "status",
        "rationale",
    }
    license_keys = {
        "schema_version",
        "candidate_id",
        "resolved_commit_sha",
        "declared_license",
        "license_files",
        "usage_restrictions",
        "reviewer",
        "reviewed_at_utc",
        "status",
        "rationale",
    }
    remote_keys = {
        "schema_version",
        "candidate_id",
        "resolved_commit_sha",
        "remote_code_required",
        "allow_remote_code",
        "reviewed_python_files",
        "dangerous_call_findings",
        "reviewer",
        "reviewed_at_utc",
        "status",
        "rationale",
    }
    source = require_exact_keys(
        load_strict_json(candidate_dir / "source_review.json"),
        source_keys,
        "source review",
    )
    license_review = require_exact_keys(
        load_strict_json(candidate_dir / "license_review.json"),
        license_keys,
        "license review",
    )
    remote = require_exact_keys(
        load_strict_json(candidate_dir / "remote_code_review.json"),
        remote_keys,
        "remote review",
    )
    for review, name in (
        (source, "source review"),
        (license_review, "license review"),
        (remote, "remote review"),
    ):
        _validate_review_identity(review, manifest, name)
        if require_human_review:
            _validate_human_fields(review, name)

    allowed_source_status = {"approved", "rejected"}
    if not require_human_review:
        allowed_source_status.add("pending")
    if source["status"] not in allowed_source_status:
        raise P0Error("p0a_review_incomplete", "source review 状态非法")
    for field in (
        "native_shared_expert",
        "native_routed_experts",
        "shared_always_executes",
        "mask_before_dispatch_static_feasibility",
        "expert_call_observability_static_feasibility",
    ):
        if type(source[field]) is not bool:
            raise P0Error("artifact_schema_mismatch", f"source review {field} 非布尔")
    source_references = _validate_reference_list(
        source["evidence_files"], input_root, "source evidence"
    )
    if not source["evidence_files"]:
        raise P0Error("p0a_review_incomplete", "source review 缺少证据文件")
    implementation_evidence = any(
        reference["path"].endswith(".py")
        or reference["path"].endswith("-source-review.txt")
        for reference in source_references
    )
    if source["status"] == "approved" and not implementation_evidence:
        raise P0Error("p0a_review_incomplete", "批准的源码审阅缺少实现证据")

    allowed_license_status = {"approved", "rejected"}
    if not require_human_review:
        allowed_license_status.add("pending")
    if license_review["status"] not in allowed_license_status:
        raise P0Error("p0a_review_incomplete", "license review 状态非法")
    if not isinstance(license_review["declared_license"], str):
        raise P0Error("artifact_schema_mismatch", "declared_license 非字符串")
    if not isinstance(license_review["usage_restrictions"], list) or any(
        not isinstance(value, str) for value in license_review["usage_restrictions"]
    ):
        raise P0Error("artifact_schema_mismatch", "usage_restrictions 非字符串数组")
    _validate_reference_list(
        license_review["license_files"], input_root, "license evidence"
    )
    if license_review["status"] == "approved" and (
        not license_review["declared_license"].strip()
        or not license_review["license_files"]
    ):
        raise P0Error("license_unresolved", "批准的许可证审阅缺少声明或证据")

    if (
        type(remote["remote_code_required"]) is not bool
        or type(remote["allow_remote_code"]) is not bool
        or remote["remote_code_required"] != preset.allow_remote_code
        or remote["allow_remote_code"] != preset.allow_remote_code
    ):
        raise P0Error("remote_code_rejected", "remote-code profile 与预登记不一致")
    allowed_remote_status = (
        {"approved", "rejected"} if preset.allow_remote_code else {"not_required"}
    )
    if not require_human_review and preset.allow_remote_code:
        allowed_remote_status.add("pending")
    if remote["status"] not in allowed_remote_status:
        raise P0Error("p0a_review_incomplete", "remote-code review 状态非法或未完成")
    remote_references = _validate_reference_list(
        remote["reviewed_python_files"], input_root, "remote-code evidence"
    )
    if preset.allow_remote_code and not remote["reviewed_python_files"]:
        raise P0Error("remote_code_rejected", "remote-code 审阅缺少 Python 文件")
    if not isinstance(remote["dangerous_call_findings"], list):
        raise P0Error("artifact_schema_mismatch", "dangerous_call_findings 非数组")
    expected_python = {
        (
            f"{preset.metadata_directory}/{item['path']}",
            item["sha256"],
        )
        for item in manifest["remote_code_scan"]["python_files"]
    }
    actual_python = {
        (reference["path"], reference["sha256"]) for reference in remote_references
    }
    if actual_python != expected_python:
        raise P0Error("remote_code_rejected", "remote-code Python 文件清单不完整")
    finding_keys = {"path", "line", "pattern", "review_disposition", "rationale"}
    for finding in remote["dangerous_call_findings"]:
        record = require_exact_keys(finding, finding_keys, "dangerous call finding")
        _safe_relative_path(record["path"], "dangerous call finding")
        if type(record["line"]) is not int or record["line"] < 1:
            raise P0Error("artifact_schema_mismatch", "finding line 非法")
        allowed_dispositions = {"accepted", "rejected"}
        if not require_human_review:
            allowed_dispositions.add("pending")
        if record["review_disposition"] not in allowed_dispositions:
            raise P0Error("p0a_review_incomplete", "危险调用命中尚未人工处置")
        if require_human_review and (
            not isinstance(record["rationale"], str) or not record["rationale"].strip()
        ):
            raise P0Error("p0a_review_incomplete", "危险调用命中缺少解释")
    expected_findings = {
        (item["path"], item["line"], item["pattern"])
        for item in manifest["remote_code_scan"]["findings"]
    }
    actual_findings = {
        (item["path"], item["line"], item["pattern"])
        for item in remote["dangerous_call_findings"]
    }
    if actual_findings != expected_findings:
        raise P0Error("remote_code_rejected", "remote-code 扫描命中清单不完整")
    if (
        any(
            finding["review_disposition"] == "rejected"
            for finding in remote["dangerous_call_findings"]
        )
        and remote["status"] != "rejected"
    ):
        raise P0Error("remote_code_rejected", "存在拒绝命中但 remote review 未拒绝")
    return manifest, source, license_review, remote


def _decision_for(
    manifest_path: Path,
    source_path: Path,
    license_path: Path,
    remote_path: Path,
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    license_review: Mapping[str, Any],
    remote: Mapping[str, Any],
    *,
    approval_mode: str = "human_review",
) -> Dict[str, Any]:
    """根据机器检查和审阅状态生成 fail-closed P0-A 决策。"""

    hub = manifest["hub_metadata"]
    local = manifest["local_metadata"]
    static_checks = {
        "revision_immutable": bool(_COMMIT.fullmatch(manifest["resolved_commit_sha"])),
        "metadata_complete": bool(hub["snapshot_size_complete"]),
        "snapshot_within_limit": (
            type(hub["snapshot_size_bytes"]) is int
            and hub["snapshot_size_bytes"] <= manifest["max_snapshot_bytes"]
        ),
        "no_weight_files_downloaded": not local["forbidden_weight_files"],
    }
    if approval_mode == "machine_only":
        static_checks["config_shared_hint"] = source["native_shared_expert"]
        static_checks["config_routed_hint"] = source["native_routed_experts"]
        static_checks["remote_scan_clear"] = not bool(
            manifest["remote_code_scan"]["findings"]
        )
    if approval_mode == "human_review":
        checks = {
            **static_checks,
            "license_approved": license_review["status"] == "approved",
            "remote_code_approved": remote["status"] in {"approved", "not_required"},
            "source_review_approved": source["status"] == "approved",
            "native_shared_expert": source["native_shared_expert"],
            "native_routed_experts": source["native_routed_experts"],
            "shared_always_executes": source["shared_always_executes"],
            "mask_before_dispatch_static_feasibility": source[
                "mask_before_dispatch_static_feasibility"
            ],
            "expert_call_observability_static_feasibility": source[
                "expert_call_observability_static_feasibility"
            ],
        }
    elif approval_mode == "machine_only":
        checks = static_checks
    else:
        raise P0Error("artifact_schema_mismatch", "approval_mode 非法")
    failure_map = {
        "revision_immutable": "revision_not_immutable",
        "metadata_complete": "metadata_unresolved",
        "config_shared_hint": "native_shared_expert_missing",
        "config_routed_hint": "native_routed_expert_missing",
        "remote_scan_clear": "remote_code_scan_findings",
        "license_approved": "license_unresolved",
        "remote_code_approved": "remote_code_rejected",
        "source_review_approved": "source_review_rejected",
        "native_shared_expert": "native_shared_expert_missing",
        "native_routed_experts": "native_routed_expert_missing",
        "shared_always_executes": "shared_execution_unverified",
        "mask_before_dispatch_static_feasibility": "host_interface_not_controllable",
        "expert_call_observability_static_feasibility": "expert_call_unobservable",
        "snapshot_within_limit": "resource_limit_exceeded",
        "no_weight_files_downloaded": "unexpected_weight_file",
    }
    failure_codes = [failure_map[name] for name, passed in checks.items() if not passed]
    unverified_checks: List[str] = []
    if approval_mode == "machine_only":
        unverified_checks = [
            "license_approved",
            "remote_code_approved",
            "source_review_approved",
            "shared_always_executes",
            "mask_before_dispatch_static_feasibility",
            "expert_call_observability_static_feasibility",
        ]
        failure_codes.append("human_review_required")
        status = "rejected" if len(failure_codes) > 1 else "provisional"
    else:
        status = "passed" if not failure_codes else "rejected"
    return {
        "schema_version": 1,
        "approval_mode": approval_mode,
        "human_review_required": approval_mode == "machine_only",
        "formal_acceptance": approval_mode == "human_review",
        "candidate_id": manifest["candidate_id"],
        "repository_id": manifest["repository_id"],
        "resolved_commit_sha": manifest["resolved_commit_sha"],
        "checks": checks,
        "unverified_checks": unverified_checks,
        "status": status,
        "failure_codes": failure_codes,
        "evidence_sha256": {
            "metadata_manifest": sha256_file(manifest_path),
            "source_review": sha256_file(source_path),
            "license_review": sha256_file(license_path),
            "remote_code_review": sha256_file(remote_path),
        },
    }


def _write_new_canonical_json(path: Path, payload: Mapping[str, Any]) -> str:
    """原子创建规范 JSON 文件，并返回最终 SHA-256。"""

    if path.exists():
        raise P0Error("artifact_exists", f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload)
    temp = path.with_name(f".{path.name}.tmp")
    if temp.exists():
        raise P0Error("artifact_exists", f"临时文件已存在: {temp}")
    try:
        temp.write_bytes(raw)
        os.replace(temp, path)
    except OSError as exc:
        if temp.exists():
            temp.unlink()
        raise P0Error("artifact_write_failed", f"无法写入 {path}") from exc
    return hashlib.sha256(raw).hexdigest()


def finalize_p0a_registry(
    input_root: Path,
    prepared_root: Path,
    output_root: Path,
    registry_path: Path,
    *,
    require_human_review: bool = True,
) -> Mapping[str, object]:
    """生成 P0-A 决策、registry 和摘要 sidecar。

    Python API 默认保持严格人工模式以兼容既有调用者；服务器 CLI 默认
    使用 machine-only 预筛，并将结果标记为 provisional。只有显式人工模式
    才能产生 formal acceptance，provisional 永远不能进入 P0-B/C/D。
    """

    if type(require_human_review) is not bool:
        raise P0Error("artifact_schema_mismatch", "require_human_review 必须为 bool")

    if (
        output_root.exists()
        or registry_path.exists()
        or registry_path.with_suffix(registry_path.suffix + ".sha256").exists()
    ):
        raise P0Error("artifact_exists", "决策输出或正式 registry 已存在")

    collected: List[Tuple[P0ACandidatePreset, Mapping[str, Any], Dict[str, Any]]] = []
    for preset in P0A_CANDIDATES:
        candidate_dir = prepared_root / preset.candidate_id
        manifest, source, license_review, remote = _load_reviews(
            input_root,
            prepared_root,
            preset,
            require_human_review=require_human_review,
        )
        decision = _decision_for(
            candidate_dir / "metadata_manifest.json",
            candidate_dir / "source_review.json",
            candidate_dir / "license_review.json",
            candidate_dir / "remote_code_review.json",
            manifest,
            source,
            license_review,
            remote,
            approval_mode=("human_review" if require_human_review else "machine_only"),
        )
        collected.append((preset, manifest, decision))

    writer = ArtifactWriter(output_root)
    candidates: List[Dict[str, object]] = []
    decision_summaries: List[Dict[str, object]] = []
    for preset, manifest, decision in collected:
        decision_name = f"{preset.candidate_id.lower()}_p0a_decision.json"
        decision_sha = writer.write_json(decision_name, decision)
        candidates.append(
            {
                "candidate_id": preset.candidate_id,
                "repository_id": preset.repository_id,
                "attempt_order": preset.attempt_order,
                "requested_revision": manifest["requested_revision"],
                "resolved_commit_sha": manifest["resolved_commit_sha"],
                "profile": manifest["profile"],
                "expected_architecture_family": preset.expected_architecture_family,
                "expected_moe_variant": preset.expected_moe_variant,
                "max_snapshot_bytes": preset.max_snapshot_bytes,
                "license_review_status": (
                    (
                        "approved"
                        if decision["checks"]["license_approved"]
                        else "rejected"
                    )
                    if require_human_review
                    else (
                        "machine_detected"
                        if any(
                            Path(str(item["path"]))
                            .name.lower()
                            .startswith(("license", "readme"))
                            for item in manifest["local_metadata"]["inventory"]
                        )
                        else "unresolved"
                    )
                ),
                "metadata_source_sha256": decision["evidence_sha256"][
                    "metadata_manifest"
                ],
                "p0a_status": decision["status"],
                "p0a_decision_sha256": decision_sha,
                "p0a_failure_codes": decision["failure_codes"],
            }
        )
        decision_summaries.append(
            {
                "candidate_id": preset.candidate_id,
                "status": decision["status"],
                "approval_mode": decision["approval_mode"],
                "human_review_required": decision["human_review_required"],
                "formal_acceptance": decision["formal_acceptance"],
                "failure_codes": decision["failure_codes"],
                "decision_file": decision_name,
                "decision_sha256": decision_sha,
            }
        )

    registry = {
        "schema_version": 1,
        "registry_id": "p0-moe-host-v1",
        "candidates": candidates,
    }
    # 使用正式 loader 同源校验生成结果，再写入磁盘。
    validate_registry(registry)
    registry_sha = _write_new_canonical_json(registry_path, registry)
    sidecar = registry_path.with_suffix(registry_path.suffix + ".sha256")
    try:
        sidecar.write_text(f"{registry_sha}  {registry_path.name}\n", encoding="ascii")
    except OSError as exc:
        raise P0Error(
            "artifact_write_failed", "无法写入 registry 摘要 sidecar"
        ) from exc
    summary = {
        "schema_version": 1,
        "status": "complete",
        "approval_mode": "human_review" if require_human_review else "machine_only",
        "formal_acceptance": require_human_review,
        "human_review_required": not require_human_review,
        "registry_path": registry_path.as_posix(),
        "registry_sha256": registry_sha,
        "candidates": decision_summaries,
    }
    writer.write_json("summary.json", summary)
    return summary
