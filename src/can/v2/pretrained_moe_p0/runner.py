"""P0-MoE 本地 runner：按固定顺序执行 fake-host 结构门。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, Mapping, Optional

from .host_adapter import HostAdapter, inspect_host
from .types import CandidateSpec, HostInspection, P0Error


class P0Runner:
    """执行不下载模型的候选结构预检状态机。"""

    def __init__(self, candidates: Iterable[CandidateSpec]) -> None:
        """按 attempt_order 固定候选列表。"""
        self._candidates = tuple(candidates)
        if not self._candidates:
            raise P0Error("registry_order_invalid", "候选列表不能为空")
        orders = [candidate.attempt_order for candidate in self._candidates]
        if orders != sorted(orders) or orders != list(range(1, len(orders) + 1)):
            raise P0Error("registry_order_invalid", "候选顺序必须连续")

    def run_structure_only(
        self, adapters: Mapping[str, HostAdapter]
    ) -> Dict[str, object]:
        """按固定顺序探查 fake host，返回首个通过者或 no_suitable_host。"""
        attempts = []
        selected: Optional[str] = None
        for candidate in self._candidates:
            if candidate.p0a_status != "passed":
                attempts.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": "not_run",
                        "reason": "p0a_rejected",
                        "failure_codes": list(candidate.p0a_failure_codes),
                    }
                )
                continue
            adapter = adapters.get(candidate.candidate_id)
            if adapter is None:
                attempts.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": "not_run",
                        "reason": "metadata_unresolved",
                    }
                )
                continue
            inspection = inspect_host(adapter)
            attempts.append(self._attempt_record(candidate, inspection))
            if inspection.passed:
                selected = candidate.candidate_id
                break
        return {
            "schema_version": 1,
            "status": "selected" if selected is not None else "no_suitable_host",
            "selected_candidate": selected,
            "attempts": attempts,
        }

    @staticmethod
    def _attempt_record(
        candidate: CandidateSpec, inspection: HostInspection
    ) -> Dict[str, object]:
        """把探查结果转换为不含权重的摘要记录。"""
        return {
            "candidate_id": candidate.candidate_id,
            "attempt_order": candidate.attempt_order,
            "status": "passed" if inspection.passed else "failed",
            "failure_reason": inspection.rejected_reason,
            "gates": dict(inspection.gate_results),
            "hook_points": list(inspection.hook_points),
            "capabilities": asdict(inspection.capabilities),
        }
