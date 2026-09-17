"""G1-a manifest 的严格 schema 与摘要校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .types import G1AConfig, G1AError, G1AParameters
from .parameters import parameter_digest


_REQUIRED = {
    "schema_version",
    "execution_config_id",
    "protocol_id",
    "q",
    "n",
    "m",
    "tau",
    "norm",
    "integer_encoding",
    "parameter_sha256",
    "vectors_sha256",
    "reference_impl_sha256",
    "seeds",
    "provenance",
}


def _digest(value: str, field: str) -> None:
    """验证 digest 是小写 64 位十六进制字符串。"""
    if not isinstance(value, str) or len(value) != 64 or value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise G1AError("G1A_MANIFEST_FIELD", f"{field} 必须是小写 SHA-256")


def _git_digest(value: str) -> None:
    """验证 Git commit 为常见 40 位或扩展 64 位小写十六进制摘要。"""
    if not isinstance(value, str) or len(value) not in (40, 64) or value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise G1AError("G1A_MANIFEST_PROVENANCE", "git_commit 必须是 40/64 位小写十六进制")


def validate_g1a_manifest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """严格校验 manifest 字段、类型和候选配置。"""
    if not isinstance(payload, dict) or set(payload) != _REQUIRED:
        raise G1AError("G1A_MANIFEST_SCHEMA", "manifest 字段集合不匹配")
    if payload["schema_version"] != 1:
        raise G1AError("G1A_MANIFEST_VERSION", "schema_version 必须为 1")
    for field in ("execution_config_id", "protocol_id", "norm", "integer_encoding"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise G1AError("G1A_MANIFEST_FIELD", f"{field} 必须是非空字符串")
    seeds = payload["seeds"]
    if not isinstance(seeds, dict):
        raise G1AError("G1A_MANIFEST_SEEDS", "seeds 必须是对象")
    config = G1AConfig(q=payload["q"], n=payload["n"], m=payload["m"], tau=payload["tau"], norm=payload["norm"], protocol_id=payload["protocol_id"], execution_config_id=payload["execution_config_id"], master_seed=seeds.get("master_seed", 0))
    if payload["integer_encoding"] != "little-endian-int64-v1":
        raise G1AError("G1A_MANIFEST_ENCODING", "integer_encoding 未登记")
    for field in ("parameter_sha256", "vectors_sha256", "reference_impl_sha256"):
        _digest(payload[field], field)
    if not isinstance(seeds, dict) or set(seeds) != {"master_seed", "seed_A", "seed_secret", "seed_error", "prng_id", "prng_version", "seed_derivation"}:
        raise G1AError("G1A_MANIFEST_SEEDS", "seeds schema 不匹配")
    for field in ("master_seed", "seed_A", "seed_secret", "seed_error"):
        if type(seeds[field]) is not int or seeds[field] < 0:
            raise G1AError("G1A_MANIFEST_SEEDS", f"{field} 必须是非负整数")
    for field in ("prng_id", "prng_version", "seed_derivation"):
        if not isinstance(seeds[field], str) or not seeds[field]:
            raise G1AError("G1A_MANIFEST_SEEDS", f"{field} 必须是非空字符串")
    from .parameters import derive_seed

    expected_seeds = {
        "seed_A": derive_seed(seeds["master_seed"], "A"),
        "seed_secret": derive_seed(seeds["master_seed"], "secret"),
        "seed_error": derive_seed(seeds["master_seed"], "error"),
    }
    if any(seeds[name] != value for name, value in expected_seeds.items()):
        raise G1AError("G1A_MANIFEST_SEEDS", "派生 seed 与 master_seed 不一致")
    if not isinstance(payload["provenance"], dict) or set(payload["provenance"]) != {"git_commit", "dirty", "python", "numpy"}:
        raise G1AError("G1A_MANIFEST_PROVENANCE", "provenance schema 不匹配")
    if type(payload["provenance"]["dirty"]) is not bool:
        raise G1AError("G1A_MANIFEST_PROVENANCE", "dirty 必须是 bool")
    _git_digest(payload["provenance"]["git_commit"])
    return payload


def build_manifest(config: G1AConfig, parameters: G1AParameters, vectors_sha256: str, reference_impl_sha256: str, provenance: Dict[str, Any]) -> Dict[str, Any]:
    """根据配置和公开参数生成待写入的严格 manifest 对象。"""
    for field, value in (("vectors_sha256", vectors_sha256), ("reference_impl_sha256", reference_impl_sha256)):
        _digest(value, field)
    return {
        "schema_version": 1,
        "execution_config_id": config.execution_config_id,
        "protocol_id": config.protocol_id,
        "q": config.q,
        "n": config.n,
        "m": config.m,
        "tau": config.tau,
        "norm": config.norm,
        "integer_encoding": "little-endian-int64-v1",
        "parameter_sha256": parameter_digest(parameters),
        "vectors_sha256": vectors_sha256,
        "reference_impl_sha256": reference_impl_sha256,
        "seeds": {
            "master_seed": config.master_seed,
            "seed_A": _seed(config.master_seed, "A"),
            "seed_secret": _seed(config.master_seed, "secret"),
            "seed_error": _seed(config.master_seed, "error"),
            "prng_id": config.prng_id,
            "prng_version": config.prng_version,
            "seed_derivation": "sha256(g1a-seed-v1|master_seed|label)[:8] little-endian",
        },
        "provenance": provenance,
    }


def _seed(master_seed: int, label: str) -> int:
    """延迟导入 seed 派生函数，避免模块初始化循环。"""
    from .parameters import derive_seed

    return derive_seed(master_seed, label)


def load_g1a_manifest(path: Path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """按原始 UTF-8 bytes 校验摘要后解析并校验 manifest。"""
    path = Path(path)
    raw = path.read_bytes()
    if expected_sha256 is not None:
        _digest(expected_sha256, "expected_sha256")
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise G1AError("G1A_MANIFEST_DIGEST_MISMATCH", "manifest bytes 摘要不匹配")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G1AError("G1A_MANIFEST_JSON", "manifest 不是合法 UTF-8 JSON") from exc
    return validate_g1a_manifest(payload)
