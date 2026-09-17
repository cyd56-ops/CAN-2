"""I1 配对 fixture、调用台账和 summary 的严格校验。"""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..auth_expert_moe_m2.router import P1_POLICY, P2_POLICY
from .types import I1_EXECUTION_CONFIG_ID, I1_PROTOCOL_ID, I1Error

_HASH = re.compile(r"^[0-9a-f]{64}$")
BOUNDARY_ROLES = {
    "nominal_valid",
    "nominal_failure",
    "exact_accept_boundary",
    "first_reject_boundary",
    "format_dtype",
    "format_rank",
    "format_shape",
    "domain_below_zero",
    "domain_at_q",
}


def _credential_descriptor(value: object, name: str) -> None:
    """验证不含 raw credential 的摘要描述符。"""
    required = {"encoding", "sha256", "dtype", "shape"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise I1Error("I1_FIXTURE_CREDENTIAL", f"{name} 描述符字段不匹配")
    if (
        value["encoding"] != "sha256-only"
        or not isinstance(value["sha256"], str)
        or _HASH.fullmatch(value["sha256"]) is None
    ):
        raise I1Error("I1_FIXTURE_CREDENTIAL", f"{name} 摘要非法")
    if (
        not isinstance(value["dtype"], str)
        or not isinstance(value["shape"], list)
        or any(type(item) is not int or item < 0 for item in value["shape"])
    ):
        raise I1Error("I1_FIXTURE_CREDENTIAL", f"{name} dtype/shape 非法")


def validate_i1_fixture(payload: Mapping[str, Any]) -> None:
    """验证配对 fixture schema、case 标注和边界覆盖。"""
    required = {
        "schema_version",
        "execution_config_id",
        "protocol_id",
        "policy",
        "cases",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload["schema_version"] != 1
    ):
        raise I1Error("I1_FIXTURE_SCHEMA", "fixture 根字段不匹配")
    if (
        payload["execution_config_id"] != I1_EXECUTION_CONFIG_ID
        or payload["protocol_id"] != I1_PROTOCOL_ID
        or payload["policy"] not in {P1_POLICY, P2_POLICY}
    ):
        raise I1Error("I1_FIXTURE_SCHEMA", "fixture config/protocol/policy 不匹配")
    cases = payload["cases"]
    case_fields = {
        "case_id",
        "expected_relation_class",
        "boundary_role",
        "a0_credential",
        "a0_expected_reason",
        "g1b_credential",
        "g1b_expected_reason",
        "rationale",
        "expected_scope",
        "request_ids",
        "expected_routes",
    }
    if not isinstance(cases, list) or not cases:
        raise I1Error("I1_FIXTURE_SCHEMA", "cases 必须是非空数组")
    identifiers = []
    roles = set()
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != case_fields:
            raise I1Error("I1_FIXTURE_CASE", "case 字段不匹配")
        if not isinstance(case["case_id"], str) or not case["case_id"]:
            raise I1Error("I1_FIXTURE_CASE", "case_id 非法")
        identifiers.append(case["case_id"])
        if case["expected_relation_class"] not in {"valid", "failure", "format_error"}:
            raise I1Error("I1_FIXTURE_CASE", "relation class 非法")
        if case["boundary_role"] not in BOUNDARY_ROLES:
            raise I1Error("I1_FIXTURE_BOUNDARY", "boundary_role 未登记")
        roles.add(case["boundary_role"])
        _credential_descriptor(case["a0_credential"], "a0_credential")
        _credential_descriptor(case["g1b_credential"], "g1b_credential")
        for name in ("a0_expected_reason", "g1b_expected_reason", "rationale"):
            if not isinstance(case[name], str) or not case[name]:
                raise I1Error("I1_FIXTURE_CASE", f"{name} 必须是非空字符串")
        if case["expected_scope"] not in {None, "standard", "advanced", "privileged"}:
            raise I1Error("I1_FIXTURE_CASE", "expected_scope 非法")
        if (
            not isinstance(case["request_ids"], list)
            or not case["request_ids"]
            or any(
                not isinstance(value, str) or not value for value in case["request_ids"]
            )
            or len(set(case["request_ids"])) != len(case["request_ids"])
        ):
            raise I1Error("I1_FIXTURE_CASE", "request_ids 非法")
        if (
            not isinstance(case["expected_routes"], list)
            or len(case["expected_routes"]) != len(case["request_ids"])
            or any(
                value not in {"protected", "public", "deny", "error"}
                for value in case["expected_routes"]
            )
        ):
            raise I1Error("I1_FIXTURE_CASE", "expected_routes 非法")
    if len(set(identifiers)) != len(identifiers):
        raise I1Error("I1_FIXTURE_CASE", "case_id 必须唯一")
    if roles != BOUNDARY_ROLES:
        raise I1Error("I1_FIXTURE_BOUNDARY", "fixture 未覆盖完整 boundary_role 集合")


def validate_i1_call_ledger(payload: Mapping[str, Any]) -> None:
    """验证 I1 台账的事件顺序、绑定摘要和零调用原因。"""
    required = {"schema_version", "run_id", "execution_config_id", "policy", "events"}
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload["schema_version"] != 1
    ):
        raise I1Error("I1_LEDGER_SCHEMA", "ledger 根字段不匹配")
    if payload["execution_config_id"] != I1_EXECUTION_CONFIG_ID or payload[
        "policy"
    ] not in {P1_POLICY, P2_POLICY}:
        raise I1Error("I1_LEDGER_SCHEMA", "ledger config/policy 不匹配")
    events = payload["events"]
    event_fields = {
        "sequence",
        "implementation",
        "case_id",
        "expert_id",
        "kind",
        "batch_indices",
        "count",
        "verifier_backend_id",
        "verifier_evidence_digest",
        "route_digest",
        "zero_call_reason",
    }
    if not isinstance(events, list):
        raise I1Error("I1_LEDGER_SCHEMA", "events 必须是数组")
    for index, event in enumerate(events):
        if (
            not isinstance(event, Mapping)
            or set(event) != event_fields
            or event["sequence"] != index
        ):
            raise I1Error("I1_LEDGER_EVENT", "ledger event 字段或顺序非法")
        if event["implementation"] not in {"a0-m2", "g1b-i1"} or event["count"] not in {
            0,
            1,
        }:
            raise I1Error("I1_LEDGER_EVENT", "implementation/count 非法")
        if not isinstance(event["batch_indices"], list) or any(
            type(value) is not int or value < 0 for value in event["batch_indices"]
        ):
            raise I1Error("I1_LEDGER_EVENT", "batch_indices 非法")
        for name in ("verifier_evidence_digest", "route_digest"):
            if not isinstance(event[name], str) or _HASH.fullmatch(event[name]) is None:
                raise I1Error("I1_LEDGER_EVENT", f"{name} 非法")
        if event["count"] == 0 and not event["zero_call_reason"]:
            raise I1Error("I1_LEDGER_EVENT", "zero-call 必须登记原因")
        if event["count"] == 1 and (
            not event["batch_indices"] or event["zero_call_reason"] is not None
        ):
            raise I1Error("I1_LEDGER_EVENT", "真实调用必须有索引且无 zero-call 原因")


def validate_i1_summary(payload: Mapping[str, Any]) -> None:
    """验证 I1 summary 的状态、差分计数和 provenance。"""
    required = {
        "schema_version",
        "run_id",
        "execution_config_id",
        "protocol_id",
        "policy",
        "status",
        "exit_code",
        "case_count",
        "mathematical_difference_count",
        "adapter_difference_count",
        "route_difference_count",
        "unauthorized_routed_calls",
        "negative_case_count",
        "determinism",
        "coverage",
        "artifacts",
        "provenance",
        "failure",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise I1Error("I1_SUMMARY_SCHEMA", "summary 字段集合不匹配")
    if (
        payload["schema_version"] != 1
        or payload["execution_config_id"] != I1_EXECUTION_CONFIG_ID
        or payload["protocol_id"] != I1_PROTOCOL_ID
    ):
        raise I1Error("I1_SUMMARY_SCHEMA", "summary protocol/config 不匹配")
    if (
        payload["policy"] not in {P1_POLICY, P2_POLICY}
        or payload["status"] not in {"complete", "failed"}
        or type(payload["exit_code"]) is not int
    ):
        raise I1Error("I1_SUMMARY_SCHEMA", "summary policy/status/exit_code 非法")
    for name in (
        "case_count",
        "mathematical_difference_count",
        "adapter_difference_count",
        "route_difference_count",
        "unauthorized_routed_calls",
        "negative_case_count",
    ):
        if type(payload[name]) is not int or payload[name] < 0:
            raise I1Error("I1_SUMMARY_SCHEMA", f"{name} 必须是非负整数")
    for name in ("determinism", "coverage", "artifacts", "provenance"):
        if not isinstance(payload[name], Mapping):
            raise I1Error("I1_SUMMARY_SCHEMA", f"{name} 必须是对象")
    if payload["failure"] is not None and not isinstance(payload["failure"], Mapping):
        raise I1Error("I1_SUMMARY_SCHEMA", "failure 必须是 null 或对象")
