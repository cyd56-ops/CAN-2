"""M1a manifest 的严格 JSON/SHA-256 校验。"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .router import P1_POLICY, P2_POLICY

_HASH = re.compile(r"^[0-9a-f]{64}$")


def load_m1a_manifest(path: Path, expected_sha256: str) -> Dict[str, Any]:
    """按完整 UTF-8 bytes 摘要并严格读取 M1a manifest。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("manifest 不存在")
    if not isinstance(expected_sha256, str) or _HASH.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 必须是 64 位小写十六进制")
    raw = path.read_bytes()
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256):
        raise ValueError("M1a manifest SHA-256 不匹配")

    def duplicate(pairs: List[Any]) -> Dict[str, Any]:
        """拒绝 JSON 重复字段。"""

        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"manifest 字段重复: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        """拒绝 NaN/Infinity。"""

        raise ValueError(f"manifest 非有限常量: {value}")

    payload = json.loads(
        raw.decode("utf-8"), object_pairs_hook=duplicate, parse_constant=constant
    )
    required = {
        "schema_version",
        "execution_config_id",
        "protocol_id",
        "policy",
        "expert_specs",
        "alpha",
        "top_k",
        "router",
        "model",
        "fixture",
        "provenance",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("M1a manifest 根字段不匹配")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise ValueError("schema_version 必须为 1")
    if payload["policy"] not in {P1_POLICY, P2_POLICY}:
        raise ValueError("M1a policy 不受支持")
    if (
        payload["execution_config_id"] != "m1a-tiny-moe-v1"
        or payload["protocol_id"] != "m1a-contract-v1"
    ):
        raise ValueError("M1a execution/protocol ID 不匹配")
    if payload["alpha"] != 1.0 or payload["top_k"] != 1:
        raise ValueError("M1a alpha/top_k 不匹配")
    if not isinstance(payload["router"], dict) or set(payload["router"]) != {
        "type",
        "normalize_eps",
    }:
        raise ValueError("router 字段不匹配")
    if (
        payload["router"]["type"] != "fixed"
        or payload["router"]["normalize_eps"] != 1e-12
    ):
        raise ValueError("router 配置不匹配")
    if not isinstance(payload["model"], dict) or set(payload["model"]) != {
        "d_model",
        "dtype",
        "device",
    }:
        raise ValueError("model 字段不匹配")
    if payload["model"] != {"d_model": 16, "dtype": "float32", "device": "cpu"}:
        raise ValueError("model 配置不匹配")
    if not isinstance(payload["provenance"], dict) or set(payload["provenance"]) != {
        "git_commit"
    }:
        raise ValueError("provenance 字段不匹配")
    if (
        not isinstance(payload["provenance"]["git_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", payload["provenance"]["git_commit"]) is None
    ):
        raise ValueError("provenance.git_commit 格式错误")
    if (
        not isinstance(payload["expert_specs"], list)
        or len(payload["expert_specs"]) != 2
    ):
        raise ValueError("expert_specs 必须包含 E0/E1")
    expert_keys = {
        "expert_id",
        "kind",
        "capability_level",
        "scope_ids",
        "architecture_revision",
        "weights_sha256",
        "train_data_scope",
        "max_context_length",
        "enabled",
    }
    for item in payload["expert_specs"]:
        if not isinstance(item, dict) or set(item) != expert_keys:
            raise ValueError("ExpertSpec 字段不匹配")
        if _HASH.fullmatch(item["weights_sha256"]) is None:
            raise ValueError("weights_sha256 格式错误")
        if not isinstance(item["expert_id"], str) or item["expert_id"] not in {
            "E0",
            "E1",
        }:
            raise ValueError("expert_id 必须为 E0/E1")
        if not isinstance(item["enabled"], bool) or not isinstance(
            item["scope_ids"], list
        ):
            raise ValueError("ExpertSpec 类型错误")
        if (
            isinstance(item["max_context_length"], bool)
            or not isinstance(item["max_context_length"], int)
            or item["max_context_length"] < 1
        ):
            raise ValueError("max_context_length 必须为正整数")
    if not isinstance(payload["fixture"], dict) or set(payload["fixture"]) != {
        "inputs_sha256",
        "reference_outputs_sha256",
    }:
        raise ValueError("fixture 字段不匹配")
    for value in payload["fixture"].values():
        if not isinstance(value, str) or _HASH.fullmatch(value) is None:
            raise ValueError("fixture 摘要格式错误")
    return payload
