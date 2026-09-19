"""P0-MoE 候选 registry 的严格解析和固定顺序校验。"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import CandidateProfile, CandidateSpec, P0Error, require_exact_keys

_HASH = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TOP_KEYS = {"schema_version", "registry_id", "candidates"}
_CANDIDATE_KEYS = {
    "candidate_id",
    "repository_id",
    "attempt_order",
    "requested_revision",
    "resolved_commit_sha",
    "profile",
    "expected_architecture_family",
    "expected_moe_variant",
    "max_snapshot_bytes",
    "license_review_status",
    "metadata_source_sha256",
    "p0a_status",
    "p0a_decision_sha256",
    "p0a_failure_codes",
}
_PROFILE_KEYS = {"profile_id", "dtype", "quantization_config", "allow_remote_code"}


def _duplicate_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """解析 JSON 对象并拒绝重复字段。"""
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise P0Error("artifact_duplicate_field", f"重复字段 {key}")
        result[key] = value
    return result


def _sha(value: object, name: str) -> str:
    """校验小写 SHA-256 字符串。"""
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise P0Error("artifact_schema_mismatch", f"{name} 不是 SHA-256")
    return value


def validate_registry(payload: object) -> Tuple[CandidateSpec, ...]:
    """校验 registry 并按 attempt_order 返回不可变候选。"""
    root = require_exact_keys(payload, _TOP_KEYS, "registry")
    if root["schema_version"] != 1 or root["registry_id"] != "p0-moe-host-v1":
        raise P0Error("artifact_schema_mismatch", "registry 版本不匹配")
    candidates = root["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise P0Error("artifact_schema_mismatch", "candidates 必须为非空数组")
    parsed: List[CandidateSpec] = []
    seen_ids: Set[str] = set()
    seen_orders: Set[int] = set()
    for item in candidates:
        candidate = require_exact_keys(item, _CANDIDATE_KEYS, "candidate")
        profile_data = require_exact_keys(
            candidate["profile"], _PROFILE_KEYS, "profile"
        )
        candidate_id = candidate["candidate_id"]
        order = candidate["attempt_order"]
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen_ids
            or type(order) is not int
            or order < 1
            or order in seen_orders
        ):
            raise P0Error("registry_order_invalid", "candidate id/order 重复或非法")
        if (
            not isinstance(candidate["repository_id"], str)
            or not candidate["repository_id"]
        ):
            raise P0Error("artifact_schema_mismatch", "repository_id 非法")
        if (
            not isinstance(candidate["requested_revision"], str)
            or not candidate["requested_revision"]
        ):
            raise P0Error("artifact_schema_mismatch", "requested_revision 非法")
        resolved = candidate["resolved_commit_sha"]
        if not isinstance(resolved, str) or _COMMIT.fullmatch(resolved) is None:
            raise P0Error(
                "revision_not_immutable", "resolved_commit_sha 必须为 40 位 commit"
            )
        if (
            type(candidate["max_snapshot_bytes"]) is not int
            or candidate["max_snapshot_bytes"] <= 0
        ):
            raise P0Error("artifact_schema_mismatch", "max_snapshot_bytes 非法")
        profile_id = profile_data["profile_id"]
        if not isinstance(profile_id, str) or not profile_id:
            raise P0Error("artifact_schema_mismatch", "profile_id 非法")
        if not isinstance(profile_data["dtype"], str) or profile_data["dtype"] not in {
            "bf16",
            "float16",
            "float32",
        }:
            raise P0Error("artifact_schema_mismatch", "dtype 不支持")
        if type(profile_data["allow_remote_code"]) is not bool:
            raise P0Error("artifact_schema_mismatch", "allow_remote_code 必须为 bool")
        _sha(candidate["metadata_source_sha256"], "metadata_source_sha256")
        _sha(candidate["p0a_decision_sha256"], "p0a_decision_sha256")
        if candidate["license_review_status"] not in {
            "approved",
            "rejected",
            "machine_detected",
            "unresolved",
        }:
            raise P0Error("license_unresolved", "license_review_status 非法")
        status = candidate["p0a_status"]
        failure_codes = candidate["p0a_failure_codes"]
        if status not in {"passed", "rejected", "provisional"}:
            raise P0Error("p0a_decision_invalid", "p0a_status 非法")
        if (
            not isinstance(failure_codes, list)
            or any(not isinstance(code, str) or not code for code in failure_codes)
            or len(set(failure_codes)) != len(failure_codes)
        ):
            raise P0Error("p0a_decision_invalid", "P0-A failure codes 非法")
        if status == "passed":
            if failure_codes or candidate["license_review_status"] != "approved":
                raise P0Error("p0a_decision_invalid", "通过候选仍含失败或许可证未批准")
        elif status == "provisional":
            if "human_review_required" not in failure_codes:
                raise P0Error(
                    "p0a_decision_invalid",
                    "provisional 候选必须明确要求人工审阅",
                )
        elif not failure_codes:
            raise P0Error("p0a_decision_invalid", "拒绝候选必须登记失败码")
        seen_ids.add(candidate_id)
        seen_orders.add(order)
        parsed.append(
            CandidateSpec(
                candidate_id=candidate_id,
                repository_id=candidate["repository_id"],
                attempt_order=order,
                requested_revision=candidate["requested_revision"],
                resolved_commit_sha=resolved,
                profile=CandidateProfile(
                    profile_id=profile_id,
                    dtype=profile_data["dtype"],
                    quantization_config=profile_data["quantization_config"],
                    allow_remote_code=profile_data["allow_remote_code"],
                ),
                expected_architecture_family=candidate["expected_architecture_family"],
                expected_moe_variant=candidate["expected_moe_variant"],
                max_snapshot_bytes=candidate["max_snapshot_bytes"],
                license_review_status=candidate["license_review_status"],
                metadata_source_sha256=candidate["metadata_source_sha256"],
                p0a_status=status,
                p0a_decision_sha256=candidate["p0a_decision_sha256"],
                p0a_failure_codes=tuple(failure_codes),
            )
        )
    orders = sorted(seen_orders)
    if orders != list(range(1, len(orders) + 1)):
        raise P0Error("registry_order_invalid", "attempt_order 必须连续从 1 开始")
    return tuple(sorted(parsed, key=lambda item: item.attempt_order))


def load_registry(
    path: Path, expected_sha256: Optional[str] = None
) -> Tuple[CandidateSpec, ...]:
    """读取 registry 原始 bytes，校验摘要和严格 schema。"""
    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("registry 不存在")
    raw = path.read_bytes()
    if expected_sha256 is not None:
        _sha(expected_sha256, "expected_sha256")
        if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256):
            raise P0Error("snapshot_digest_mismatch", "registry 摘要不匹配")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except P0Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise P0Error("artifact_json_invalid", "registry 不是规范 JSON") from exc
    return validate_registry(payload)
