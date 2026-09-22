"""Qwen1.5-MoE C1 的保守真实宿主 adapter。"""

from __future__ import annotations

from dataclasses import replace

from .real_adapter import TransformersHostAdapter
from .real_types import ArchitectureMap, ProbeRequest, ProbeResult
from .types import P0Error


class QwenHostAdapter(TransformersHostAdapter):
    """审计 Qwen packed experts，并拒绝仅凭 router selection 证明执行。"""

    def inspect_architecture(self) -> ArchitectureMap:
        """返回 Qwen 架构；packed 容器默认不具备逐 expert 执行计数。"""
        architecture = super().inspect_architecture()
        return replace(
            architecture,
            supports_row_mask=True,
            supports_execution_counter=False,
        )

    def run_moe_probe(self, request: ProbeRequest) -> ProbeResult:
        """拒绝把 packed 容器调用或 selected IDs 误作实际 matmul 证据。"""
        if not self._probe_installed:
            raise P0Error("probe_not_installed", "必须在 probe context 内执行")
        raise P0Error(
            "expert_call_unobservable",
            "C1 packed expert 原生 backend 无逐 expert matmul hook；禁止切换 backend",
        )
