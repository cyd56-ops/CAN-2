"""I1 manifest 的构造、严格校验与摘要绑定。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..auth_expert_moe_m2.router import P1_POLICY, P2_POLICY
from ..modint_verifier_g1a.parameters import parameter_digest
from ..modint_verifier_g1a.types import G1AParameters
from ..modint_verifier_g1b.kernel import derive_integer_bounds
from ..modint_verifier_g1b.types import IntegerBounds
from .types import (
    FP32_EXACT_INTEGER_LIMIT,
    I1_EXECUTION_CONFIG_ID,
    I1_PROTOCOL_ID,
    I1Error,
)

_FIELDS = {
    "schema_version",
    "execution_config_id",
    "protocol_id",
    "policy",
    "m2_manifest_sha256",
    "m2_assignment_sha256",
    "g1a_parameter_sha256",
    "g1b_backend_manifest_sha256",
    "g1b_protocol_id",
    "g1b_backend_id",
    "q",
    "n",
    "m",
    "tau",
    "norm",
    "integer_bounds",
    "max_abs_residual_upper_bound",
    "fp32_exact_integer_limit",
    "fp32_lossless_verified",
    "scope_schema_sha256",
    "router",
    "model",
    "provenance",
}


def _require_sha256(value: object, name: str, lengths: Tuple[int, ...] = (64,)) -> str:
    """验证小写十六进制摘要。"""
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise I1Error("I1_MANIFEST_DIGEST", f"{name} 摘要格式非法")
    return value


def build_i1_manifest(
    *,
    policy: str,
    parameters: G1AParameters,
    m2_manifest_sha256: str,
    m2_assignment_sha256: str,
    g1b_backend_manifest_sha256: str,
    scope_schema_sha256: str,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """根据已绑定依赖构造 I1 manifest。"""
    config = parameters.config
    bounds = derive_integer_bounds(parameters)
    upper_bound = config.q // 2
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "execution_config_id": I1_EXECUTION_CONFIG_ID,
        "protocol_id": I1_PROTOCOL_ID,
        "policy": policy,
        "m2_manifest_sha256": m2_manifest_sha256,
        "m2_assignment_sha256": m2_assignment_sha256,
        "g1a_parameter_sha256": parameter_digest(parameters),
        "g1b_backend_manifest_sha256": g1b_backend_manifest_sha256,
        "g1b_protocol_id": "g1b-modint-v1",
        "g1b_backend_id": "g1b-torch-int64-cpu-v1",
        "q": config.q,
        "n": config.n,
        "m": config.m,
        "tau": config.tau,
        "norm": config.norm,
        "integer_bounds": bounds.to_dict(),
        "max_abs_residual_upper_bound": upper_bound,
        "fp32_exact_integer_limit": FP32_EXACT_INTEGER_LIMIT,
        "fp32_lossless_verified": upper_bound <= FP32_EXACT_INTEGER_LIMIT,
        "scope_schema_sha256": scope_schema_sha256,
        "router": {
            "type": "fixed-constrained-top1",
            "top_k": 1,
            "routed_expert_order": ["E1", "E2", "E3"],
        },
        "model": {"d_model": 16, "dtype": "float32", "device": "cpu"},
        "provenance": dict(provenance),
    }
    return validate_i1_manifest(payload)


def validate_i1_manifest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """严格验证 I1 manifest 的完整字段和可复算范围。"""
    if not isinstance(payload, Mapping) or set(payload) != _FIELDS:
        raise I1Error("I1_MANIFEST_SCHEMA", "manifest 字段集合不匹配")
    data = dict(payload)
    if (
        data["schema_version"] != 1
        or data["execution_config_id"] != I1_EXECUTION_CONFIG_ID
        or data["protocol_id"] != I1_PROTOCOL_ID
        or data["policy"] not in {P1_POLICY, P2_POLICY}
    ):
        raise I1Error("I1_MANIFEST_SCHEMA", "协议、配置或 policy 不匹配")
    for name in (
        "m2_manifest_sha256",
        "m2_assignment_sha256",
        "g1a_parameter_sha256",
        "g1b_backend_manifest_sha256",
        "scope_schema_sha256",
    ):
        _require_sha256(data[name], name)
    if (
        data["g1b_protocol_id"] != "g1b-modint-v1"
        or data["g1b_backend_id"] != "g1b-torch-int64-cpu-v1"
    ):
        raise I1Error("I1_MANIFEST_BACKEND", "G1-b protocol/backend 不匹配")
    for name in ("q", "n", "m", "tau"):
        if type(data[name]) is not int:
            raise I1Error("I1_MANIFEST_NUMERIC", f"{name} 必须是整数")
    if (
        data["q"] < 3
        or data["q"] % 2 == 0
        or data["n"] < 1
        or data["m"] < 1
        or data["tau"] < 0
        or data["tau"] > data["q"] // 2
    ):
        raise I1Error("I1_MANIFEST_NUMERIC", "模数、维度或阈值非法")
    if data["norm"] != "linf":
        raise I1Error("I1_MANIFEST_NUMERIC", "当前 I1 只允许 linf")
    bounds = data["integer_bounds"]
    if (
        not isinstance(bounds, Mapping)
        or bounds.get("q") != data["q"]
        or bounds.get("n") != data["n"]
        or bounds.get("m") != data["m"]
    ):
        raise I1Error("I1_MANIFEST_BOUNDS", "integer_bounds 与 q/n/m 不匹配")
    try:
        IntegerBounds(**dict(bounds))
    except (TypeError, ValueError) as exc:
        raise I1Error("I1_MANIFEST_BOUNDS", "integer_bounds schema 非法") from exc
    upper_bound = data["q"] // 2
    if data["max_abs_residual_upper_bound"] != upper_bound:
        raise I1Error("I1_MANIFEST_FP32", "max residual 上界自报不一致")
    if data["fp32_exact_integer_limit"] != FP32_EXACT_INTEGER_LIMIT:
        raise I1Error("I1_MANIFEST_FP32", "FP32 连续整数上限不匹配")
    lossless = upper_bound <= FP32_EXACT_INTEGER_LIMIT
    if data["fp32_lossless_verified"] is not lossless or not lossless:
        raise I1Error("I1_MANIFEST_FP32", "当前 evidence schema 不能无损保存整数诊断")
    if data["router"] != {
        "type": "fixed-constrained-top1",
        "top_k": 1,
        "routed_expert_order": ["E1", "E2", "E3"],
    }:
        raise I1Error("I1_MANIFEST_ROUTER", "Router 配置不匹配")
    if data["model"] != {"d_model": 16, "dtype": "float32", "device": "cpu"}:
        raise I1Error("I1_MANIFEST_MODEL", "模型配置不匹配")
    provenance = data["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "git_commit",
        "dirty",
        "python",
        "torch",
        "numpy",
    }:
        raise I1Error("I1_MANIFEST_PROVENANCE", "provenance schema 不匹配")
    _require_sha256(provenance["git_commit"], "git_commit", (40, 64))
    if type(provenance["dirty"]) is not bool:
        raise I1Error("I1_MANIFEST_PROVENANCE", "dirty 必须是 bool")
    if any(
        not isinstance(provenance[name], str) or not provenance[name]
        for name in ("python", "torch", "numpy")
    ):
        raise I1Error("I1_MANIFEST_PROVENANCE", "环境版本必须是非空字符串")
    return data


def load_i1_manifest(
    path: Path, expected_sha256: Optional[str] = None
) -> Dict[str, Any]:
    """拒绝重复字段并按原始 bytes 摘要加载 I1 manifest。"""
    raw = Path(path).read_bytes()
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, "expected_sha256")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise I1Error("I1_MANIFEST_DIGEST_MISMATCH", "manifest bytes 摘要不匹配")

    def reject_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """将 JSON pairs 转成 dict 并拒绝重复 key。"""
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise I1Error("I1_MANIFEST_JSON", "manifest 包含重复字段")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                I1Error("I1_MANIFEST_JSON", "manifest 包含非有限常量")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise I1Error("I1_MANIFEST_JSON", "manifest 不是合法 UTF-8 JSON") from exc
    return validate_i1_manifest(payload)
