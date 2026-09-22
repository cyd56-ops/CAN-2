"""P0-C 真实 MoE 结构、mask、zero-call 和 mixed-batch 检查。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .real_types import (
    ArchitectureMap,
    ProbeRequest,
    ProbeResult,
    validate_reassembly_indices,
)
from .types import P0Error


def allowed_mask_for_mixed(
    batch: int, sequence: int, experts: int, top_k: int
) -> Tuple[Tuple[Tuple[bool, ...], ...], ...]:
    """构造一行 shared-only、一行 partial 的确定性 mask。"""

    if min(batch, sequence, experts, top_k) <= 0 or top_k > experts:
        raise P0Error("mask_shape_invalid", "mixed mask 维度或 top-k 非法")
    result: List[Tuple[Tuple[bool, ...], ...]] = []
    partial = tuple(index < top_k + 1 for index in range(experts))
    for batch_index in range(batch):
        rows = []
        for _ in range(sequence):
            rows.append(
                tuple(False for _ in range(experts)) if batch_index == 0 else partial
            )
        result.append(tuple(rows))
    return tuple(result)


def make_mixed_probe_request(
    *, batch: int = 2, sequence: int = 2, experts: int = 5, top_k: int = 4
) -> ProbeRequest:
    """创建固定的 mixed batch 结构探查请求。"""

    if batch != 2:
        raise P0Error("probe_fixture_invalid", "正式 mixed fixture 固定 batch=2")
    return ProbeRequest(
        hidden_shape=(batch, sequence, 8),
        allowed_mask=allowed_mask_for_mixed(batch, sequence, experts, top_k),
        native_top_k=top_k,
        padding_mask=tuple(tuple(True for _ in range(sequence)) for _ in range(batch)),
        request_ids=("p0-request-0", "p0-request-1"),
    )


def _flatten_allowed(request: ProbeRequest) -> Tuple[Tuple[bool, ...], ...]:
    """按 global_row 展平 allowed mask。"""
    return tuple(row for batch_rows in request.allowed_mask for row in batch_rows)


def validate_probe_result(
    request: ProbeRequest, result: ProbeResult, experts: int
) -> None:
    """校验 selected IDs、zero-call、padding 和 mixed 索引重组契约。"""

    batch, sequence, _ = request.hidden_shape
    if len(request.allowed_mask) != batch or any(
        len(rows) != sequence for rows in request.allowed_mask
    ):
        raise P0Error("mask_shape_invalid", "allowed_mask 与 hidden shape 不匹配")
    flattened = _flatten_allowed(request)
    if len(result.selected_ids) != len(flattened):
        raise P0Error("probe_result_invalid", "selected rows 数量不匹配")
    for row_index, selected in enumerate(result.selected_ids):
        allowed = flattened[row_index]
        if not any(allowed) and selected:
            raise P0Error("routed_call_not_zero", "shared-only 行仍选择 routed expert")
        if any(allowed):
            if len(selected) != request.native_top_k:
                raise P0Error("topk_changed", "partial mask 改变了 native top-k")
            if any(
                type(expert) is not int
                or expert < 0
                or expert >= experts
                or not allowed[expert]
                for expert in selected
            ):
                raise P0Error(
                    "forbidden_expert_selected", "selected expert 超出 allowed mask"
                )
    valid_indices = tuple(
        index
        for index, _ in enumerate(flattened)
        if request.padding_mask[index // sequence][index % sequence]
    )
    validate_reassembly_indices(
        result.original_indices, result.reassembled_indices, len(valid_indices)
    )
    if result.original_indices != tuple(sorted(valid_indices)):
        raise P0Error("batch_index_not_preserved", "original_indices 未覆盖有效输入行")
    for call in result.expert_calls:
        if call.global_row < 0 or call.global_row >= batch * sequence:
            raise P0Error("expert_call_index_invalid", "expert call global_row 越界")
        if call.branch == "routed":
            if not any(flattened[call.global_row]):
                raise P0Error(
                    "routed_call_not_zero", "shared-only 行实际调用 routed expert"
                )
            try:
                expert_index = int(call.expert_id.removeprefix("E"))
            except (AttributeError, ValueError) as exc:
                raise P0Error(
                    "expert_call_index_invalid", "routed expert ID 不是规范整数"
                ) from exc
            if expert_index < 0 or expert_index >= experts:
                raise P0Error(
                    "expert_call_index_invalid", "routed expert ID 超出登记范围"
                )
            if not flattened[call.global_row][expert_index]:
                raise P0Error(
                    "forbidden_expert_called", "forbidden routed expert 实际被调用"
                )
    if not result.kv_bound:
        raise P0Error("kv_semantics_unsupported", "probe 未绑定 request/cache/position")


def inspect_real_structure(
    adapter: Any, architecture: ArchitectureMap
) -> Dict[str, Any]:
    """执行 P0-C 结构门；任一真实执行证据缺失均 fail closed。"""

    gates = {
        "stable_moe_boundary": bool(
            architecture.moe_layers
            and architecture.router_paths
            and architecture.routed_paths
        ),
        "mask_before_dispatch": architecture.supports_row_mask,
        "expert_zero_call_observable": architecture.supports_execution_counter,
        "native_shared_branch": bool(
            architecture.shared_paths and architecture.routed_paths
        ),
        "mixed_batch_indices": False,
        "kv_identity_binding": False,
        "state_dict_stable": False,
    }
    failure_codes: List[str] = []
    if architecture.native_top_k <= 0:
        gates["stable_moe_boundary"] = False
        failure_codes.append("native_top_k_unresolved")
    before = adapter.state_digest()
    try:
        all_allowed = adapter.observe_all_allowed()
        if (
            all_allowed.get("output_exact") is not True
            or int(all_allowed.get("routed_actual_calls", 0)) <= 0
            or int(all_allowed.get("shared_actual_calls", 0)) <= 0
        ):
            raise P0Error(
                "host_interface_not_controllable",
                "all-allowed 原始路径或只读执行观察不成立",
            )
        expert_count = architecture.routed_expert_count
        if expert_count < architecture.native_top_k:
            raise P0Error("host_architecture_unresolved", "routed expert 数量无法解析")
        request = make_mixed_probe_request(
            top_k=architecture.native_top_k,
            experts=expert_count,
        )
        with adapter.install_probe():
            result = adapter.run_moe_probe(request)
        validate_probe_result(request, result, len(request.allowed_mask[0][0]))
        gates["mixed_batch_indices"] = True
        gates["kv_identity_binding"] = result.kv_bound and adapter.verify_kv_semantics()
    except P0Error as exc:
        failure_codes.append(exc.code)
    finally:
        adapter.uninstall_probe()
    after = adapter.state_digest()
    gates["state_dict_stable"] = before == after
    if not gates["state_dict_stable"]:
        failure_codes.append("host_interface_not_controllable")
    if not gates["stable_moe_boundary"]:
        failure_codes.append("host_architecture_unresolved")
    if not gates["expert_zero_call_observable"]:
        failure_codes.append("expert_call_unobservable")
    if not gates["mask_before_dispatch"]:
        failure_codes.append("host_interface_not_controllable")
    if not gates["native_shared_branch"]:
        failure_codes.append("native_shared_expert_missing")
    if not gates["kv_identity_binding"]:
        failure_codes.append("kv_semantics_unsupported")
    if not gates["mixed_batch_indices"]:
        failure_codes.append("batch_index_not_preserved")
    return {
        "architecture": architecture.__dict__,
        "gates": gates,
        "failure_codes": tuple(dict.fromkeys(failure_codes)),
        "passed": not failure_codes and all(gates.values()),
        "all_allowed": locals().get("all_allowed", {}),
        "probe_result": locals().get("result"),
    }
