"""P0-MoE 24 条公开能力 fixture 的严格校验和评分规范。"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import FixtureCase, P0Error, require_exact_keys

_ROOT_KEYS = {"schema_version", "fixture_id", "cases"}
_BASE_CASE_KEYS = {
    "case_id",
    "group",
    "system_text",
    "user_text",
    "expected_text",
    "metric",
    "max_new_tokens",
    "content_sha256",
}
_GROUPS = {"format_copy", "single_hop", "two_hop"}


def _content_digest(item: Dict[str, Any]) -> str:
    """计算去除摘要字段后的规范 fixture 摘要。"""
    body = {key: item[key] for key in sorted(item) if key != "content_sha256"}
    raw = json.dumps(
        body, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_fixture(payload: object) -> Tuple[FixtureCase, ...]:
    """校验 24 条 fixture、组计数、公开事实来源和内容摘要。"""
    root = require_exact_keys(payload, _ROOT_KEYS, "fixture")
    if root["schema_version"] != 1 or root["fixture_id"] != "p0-moe-public-fixture-v1":
        raise P0Error("artifact_schema_mismatch", "fixture 版本不匹配")
    cases = root["cases"]
    if not isinstance(cases, list) or len(cases) != 24:
        raise P0Error("fixture_group_invalid", "fixture 必须恰好包含 24 条 case")
    parsed: List[FixtureCase] = []
    seen: Set[str] = set()
    counts = {group: 0 for group in _GROUPS}
    for item in cases:
        if not isinstance(item, dict):
            raise P0Error("fixture_schema_mismatch", "fixture case 必须为对象")
        base_keys = set(_BASE_CASE_KEYS)
        group_hint = item.get("group")
        if group_hint in {"single_hop", "two_hop"}:
            case = require_exact_keys(
                item, base_keys | {"fact_source", "rationale"}, "fixture case"
            )
        elif group_hint == "format_copy":
            case = require_exact_keys(item, base_keys, "fixture case")
        else:
            raise P0Error("fixture_group_invalid", "未知 fixture group")
        case_id = case["case_id"]
        group = case["group"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise P0Error("fixture_duplicate_case", "case_id 重复或为空")
        if group not in _GROUPS:
            raise P0Error("fixture_group_invalid", "未知 fixture group")
        text_fields = ("system_text", "user_text", "expected_text")
        if any(
            not isinstance(case[field], str) or not case[field] for field in text_fields
        ):
            raise P0Error("fixture_content_invalid", "文本字段不能为空")
        if (
            type(case["max_new_tokens"]) is not int
            or not 1 <= case["max_new_tokens"] <= 24
        ):
            raise P0Error("fixture_content_invalid", "max_new_tokens 超出冻结范围")
        metric = case["metric"]
        expected_metric = "strict_em" if group == "format_copy" else "normalized_em"
        if metric != expected_metric:
            raise P0Error("fixture_metric_invalid", "metric 与 group 不匹配")
        if group in {"single_hop", "two_hop"}:
            if (
                case["fact_source"] != "stated_in_prompt"
                or not isinstance(case["rationale"], str)
                or not case["rationale"]
            ):
                raise P0Error(
                    "fixture_fact_source_invalid",
                    "公开事实必须来自 prompt 且有 rationale",
                )
        digest = case["content_sha256"]
        if not isinstance(digest, str) or digest != _content_digest(case):
            raise P0Error("fixture_digest_mismatch", "case content_sha256 不匹配")
        seen.add(case_id)
        counts[group] += 1
        parsed.append(FixtureCase(**case))
    if counts != {group: 8 for group in _GROUPS}:
        raise P0Error("fixture_group_invalid", "每组必须恰好 8 条 case")
    return tuple(parsed)


def load_fixture(
    path: Path, expected_sha256: Optional[str] = None
) -> Tuple[FixtureCase, ...]:
    """读取 fixture 原始 bytes，校验摘要和严格 schema。"""
    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("fixture 不存在")
    raw = path.read_bytes()
    if (
        expected_sha256 is not None
        and hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise P0Error("snapshot_digest_mismatch", "fixture 摘要不匹配")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P0Error("artifact_json_invalid", "fixture 不是合法 JSON") from exc
    return validate_fixture(payload)


def strict_em(expected: str, actual: str) -> bool:
    """执行 format_copy 的 NFC 和首尾 ASCII 空白规范化比较。"""
    normalize = (
        lambda value: unicodedata.normalize("NFC", value)
        .replace("\r\n", "\n")
        .strip(" \t\r\n")
    )
    return normalize(expected) == normalize(actual)


def normalized_em(expected: str, actual: str) -> bool:
    """执行 single/two-hop 的冻结 normalized EM 比较。"""

    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFC", value).lower()
        value = "".join(" " if char in ",.!?;:" else char for char in value)
        tokens = [token for token in value.split() if token not in {"a", "an", "the"}]
        return " ".join(tokens)

    return normalize(expected) == normalize(actual)
