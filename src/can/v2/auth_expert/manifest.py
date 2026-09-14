"""M0 contract manifest 的严格读取和摘要校验。"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

from .types import ExpertKind, M0ExecutionConfig, M0ExpertSpec, M0ScopeSpec

_HASH = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    """读取文件并返回 SHA-256 小写摘要。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("manifest 不存在")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_m0_manifest(path: Path, expected_sha256: str) -> M0ExecutionConfig:
    """校验同一份 manifest bytes 的摘要并解析严格 M0 配置。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("manifest 不存在")
    if not isinstance(expected_sha256, str) or _HASH.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 必须是 64 位小写十六进制")
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("manifest 超过 1 MiB")
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256):
        raise ValueError("manifest SHA-256 不匹配")

    def duplicate(pairs: List[Any]) -> Dict[str, Any]:
        """拒绝 object 内重复字段。"""

        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"manifest 字段重复: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        """拒绝 JSON 非标准 NaN/Infinity。"""

        raise ValueError(f"manifest 非有限常量: {value}")

    payload = json.loads(
        raw.decode("utf-8"), object_pairs_hook=duplicate, parse_constant=constant
    )
    if not isinstance(payload, dict):
        raise ValueError("manifest 根必须是 object")
    expected = {
        "schema_version",
        "protocol_id",
        "execution_config_id",
        "policy_id",
        "verifier",
        "host",
        "experts",
        "scopes",
        "limits",
        "fixture",
        "provenance",
    }
    if set(payload) != expected:
        raise ValueError("manifest 根字段不匹配")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise ValueError("schema_version 必须严格为整数 1")
    for name in ("protocol_id", "execution_config_id", "policy_id"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ValueError(f"{name} 必须是非空字符串")
    verifier = payload["verifier"]
    host = payload["host"]
    limits = payload["limits"]
    if set(verifier) != {
        "backend",
        "profile_id",
        "n",
        "m",
        "threshold",
        "dtype",
        "relation_sha256",
    }:
        raise ValueError("verifier 字段不匹配")
    if verifier["backend"] != "a0-fixed-relation" or verifier["dtype"] != "float32":
        raise ValueError("M0 verifier backend/dtype 不匹配")
    if set(host) != {
        "model_type",
        "model_revision",
        "model_sha256",
        "num_hidden_layers",
        "cut_layer",
        "dtype",
        "attention_backend",
        "cache_mode",
    }:
        raise ValueError("host 字段不匹配")
    if set(limits) != {"max_batch_size", "max_sequence_length"}:
        raise ValueError("limits 字段不匹配")
    for name in ("relation_sha256",):
        if (
            not isinstance(verifier[name], str)
            or _HASH.fullmatch(verifier[name]) is None
        ):
            raise ValueError(f"{name} 摘要格式错误")
    if not isinstance(verifier["profile_id"], str) or not verifier["profile_id"]:
        raise ValueError("verifier.profile_id 必须是非空字符串")
    if (
        isinstance(verifier["n"], bool)
        or not isinstance(verifier["n"], int)
        or verifier["n"] < 1
    ):
        raise ValueError("verifier.n 必须为正整数")
    if (
        isinstance(verifier["m"], bool)
        or not isinstance(verifier["m"], int)
        or verifier["m"] < verifier["n"]
    ):
        raise ValueError("verifier.m 必须不小于 n")
    if (
        isinstance(verifier["threshold"], bool)
        or not isinstance(verifier["threshold"], (int, float))
        or not math.isfinite(float(verifier["threshold"]))
        or verifier["threshold"] <= 0
    ):
        raise ValueError("verifier.threshold 必须为有限正数")
    for name in (
        "model_type",
        "model_revision",
        "model_sha256",
        "dtype",
        "attention_backend",
        "cache_mode",
    ):
        if not isinstance(host[name], str) or not host[name]:
            raise ValueError(f"host.{name} 必须是非空字符串")
    if _HASH.fullmatch(host["model_sha256"]) is None:
        raise ValueError("host.model_sha256 摘要格式错误")
    if (
        host["model_type"] not in {"tiny_decoder", "tiny_kv_decoder"}
        or host["dtype"] != "float32"
        or host["attention_backend"] != "eager"
        or host["cache_mode"] not in {"none", "kv"}
    ):
        raise ValueError("host dtype/cache_mode 不匹配")
    ints = (
        host["num_hidden_layers"],
        host["cut_layer"],
        limits["max_batch_size"],
        limits["max_sequence_length"],
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in ints
    ):
        raise ValueError("host/limits 整数域非法")
    experts = payload["experts"]
    if not isinstance(experts, list) or len(experts) != 2:
        raise ValueError("experts 必须包含 E0/E1")
    parsed_items = []
    for item in experts:
        if not isinstance(item, dict) or set(item) != {"expert_id", "kind", "enabled"}:
            raise ValueError("ExpertSpec 字段不匹配")
        parsed_items.append(
            M0ExpertSpec(item["expert_id"], ExpertKind(item["kind"]), item["enabled"])
        )
    parsed_experts = tuple(parsed_items)
    scopes = payload["scopes"]
    if not isinstance(scopes, list) or len(scopes) != 1:
        raise ValueError("scopes 必须包含 protected.default")
    if not isinstance(scopes[0], dict) or set(scopes[0]) != {"scope_id", "expert_ids"}:
        raise ValueError("ScopeSpec 字段不匹配")
    parsed_scopes = (
        M0ScopeSpec(scopes[0]["scope_id"], tuple(scopes[0]["expert_ids"])),
    )
    fixture = payload["fixture"]
    provenance = payload["provenance"]
    if not isinstance(fixture, dict) or set(fixture) != {
        "seed",
        "inputs_sha256",
        "credential_fixture_sha256",
    }:
        raise ValueError("fixture 字段不匹配")
    if not isinstance(provenance, dict) or set(provenance) != {
        "git_commit",
        "dirty",
        "source_tree_sha256",
        "python_version",
        "torch_version",
        "numpy_version",
        "device",
    }:
        raise ValueError("provenance 字段不匹配")
    if isinstance(fixture["seed"], bool) or not isinstance(fixture["seed"], int):
        raise TypeError("fixture.seed 必须是整数")
    for name in ("inputs_sha256", "credential_fixture_sha256", "source_tree_sha256"):
        value = fixture.get(name, provenance.get(name))
        if not isinstance(value, str) or _HASH.fullmatch(value) is None:
            raise ValueError(f"{name} 摘要格式错误")
    if (
        not isinstance(provenance["git_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", provenance["git_commit"]) is None
    ):
        raise ValueError("provenance.git_commit 格式错误")
    if not isinstance(provenance["dirty"], bool) or provenance["device"] != "cpu":
        raise ValueError("provenance dirty/device 不匹配")
    for name in ("python_version", "torch_version", "numpy_version"):
        if not isinstance(provenance[name], str) or not provenance[name]:
            raise ValueError(f"provenance.{name} 必须是非空字符串")
    return M0ExecutionConfig(
        execution_config_id=payload["execution_config_id"],
        protocol_id=payload["protocol_id"],
        policy_id=payload["policy_id"],
        verifier_profile=verifier["profile_id"],
        model_type=host["model_type"],
        model_revision=host["model_revision"],
        model_sha256=host["model_sha256"],
        relation_sha256=verifier["relation_sha256"],
        num_hidden_layers=host["num_hidden_layers"],
        cut_layer=host["cut_layer"],
        cache_mode=host["cache_mode"],
        max_batch_size=limits["max_batch_size"],
        max_sequence_length=limits["max_sequence_length"],
        experts=parsed_experts,
        scopes=parsed_scopes,
    )
