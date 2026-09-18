"""P0-MoE 只读宿主适配协议与 fake-host 结构门探查。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from .types import HostCapabilities, HostInspection, P0Error


class HostAdapter(Protocol):
    """宿主只读探查与受控 instrumentation 的最小协议。"""

    def capabilities(self) -> HostCapabilities:
        """返回不修改宿主的结构能力摘要。"""

    def hook_points(self) -> Tuple[str, ...]:
        """返回可记录但不执行授权逻辑的 hook 点。"""


@dataclass(frozen=True)
class FakeHostAdapter:
    """用于本地 fail-closed 测试的不可变 fake host。"""

    profile: str = "native"

    def capabilities(self) -> HostCapabilities:
        """按预设 profile 返回结构能力。"""
        profiles = {
            "native": HostCapabilities(
                "fake-moe", True, True, True, True, True, 1, True, True, True
            ),
            "dense": HostCapabilities(
                "dense",
                False,
                False,
                False,
                False,
                False,
                1,
                True,
                True,
                True,
                native_shared_routed=False,
            ),
            "no_shared": HostCapabilities(
                "fake-moe", False, True, True, True, True, 1, True, True, True
            ),
            "post_dispatch_mask": HostCapabilities(
                "fake-moe", True, True, False, True, True, 1, True, True, True
            ),
            "no_row_mask": HostCapabilities(
                "fake-moe", True, True, True, False, True, 1, True, True, True
            ),
            "no_counter": HostCapabilities(
                "fake-moe", True, True, True, True, False, 1, True, True, True
            ),
            "topk2": HostCapabilities(
                "fake-moe", True, True, True, True, True, 2, True, True, True
            ),
            "batch_reorder": HostCapabilities(
                "fake-moe", True, True, True, True, True, 1, False, True, True
            ),
            "kv_unbound": HostCapabilities(
                "fake-moe", True, True, True, True, True, 1, True, False, True
            ),
            "state_mutates": HostCapabilities(
                "fake-moe", True, True, True, True, True, 1, True, True, False
            ),
        }
        try:
            return profiles[self.profile]
        except KeyError as exc:
            raise P0Error(
                "host_interface_not_controllable", f"未知 fake profile: {self.profile}"
            ) from exc

    def hook_points(self) -> Tuple[str, ...]:
        """返回 fake host 的只读、预期 P1 插入边界。"""
        if self.profile == "dense":
            return tuple()
        return (
            "model.layers[0].mlp.gate.pre_dispatch",
            "model.layers[0].mlp.expert.forward",
        )


def inspect_host(adapter: HostAdapter) -> HostInspection:
    """执行七项结构门探查，并在失败时返回稳定 reason code。"""
    capabilities = adapter.capabilities()
    gates = {
        "stable_moe_boundary": capabilities.has_routed_experts
        and capabilities.native_shared_routed,
        "mask_before_dispatch": capabilities.mask_before_dispatch,
        "expert_zero_call_observable": capabilities.expert_call_counter,
        "native_shared_branch": capabilities.has_shared_expert
        and capabilities.has_routed_experts,
        "mixed_batch_indices": capabilities.per_row_mask
        and capabilities.preserves_batch_indices,
        "kv_identity_binding": capabilities.kv_identity_binding,
        "state_dict_stable": capabilities.state_dict_stable,
    }
    if capabilities.top_k != 1:
        gates["stable_moe_boundary"] = False
    failure_codes = (
        ("stable_moe_boundary", "host_interface_not_controllable"),
        ("mask_before_dispatch", "host_interface_not_controllable"),
        ("expert_zero_call_observable", "host_interface_not_controllable"),
        ("native_shared_branch", "host_interface_not_controllable"),
        ("mixed_batch_indices", "host_interface_not_controllable"),
        ("kv_identity_binding", "kv_semantics_unsupported"),
        ("state_dict_stable", "host_interface_not_controllable"),
    )
    rejected = next((code for gate, code in failure_codes if not gates[gate]), None)
    return HostInspection(capabilities, tuple(adapter.hook_points()), gates, rejected)
