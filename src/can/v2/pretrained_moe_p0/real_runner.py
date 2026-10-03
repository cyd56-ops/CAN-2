"""P0-A-runtime/B/C/D 真实宿主 runner 状态机。"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .capability_eval import evaluate_fixture
from .deepseek_adapter import DeepSeekHostAdapter
from .p0a import load_strict_json
from .qwen_adapter import QwenHostAdapter
from .real_adapter import TransformersHostAdapter
from .real_artifacts import write_preflight_artifact, write_run_artifacts
from .real_loader import load_transformers_host
from .real_types import PreflightResult, RealRunResult, ResourceSample
from .registry import load_registry
from .resource_probe import (
    candidate_load_smoke,
    infrastructure_preflight,
    sample_resource,
    validate_resource_gates,
)
from .snapshot import load_snapshot_manifest, verify_snapshot_against_manifest
from .structure_probe import inspect_real_structure
from .types import CandidateSpec, P0Error


class RealP0Runner:
    """按 C1→C2 固定顺序执行真实宿主 P0；不接受外部授权结果。"""

    def __init__(self, candidates: Tuple[CandidateSpec, ...]) -> None:
        """绑定已通过正式 P0-A-static loader 的有序候选。"""
        if not candidates:
            raise P0Error("registry_order_invalid", "真实 runner 候选不能为空")
        self.candidates = candidates

    @classmethod
    def from_registry(
        cls, registry: Path, expected_sha256: Optional[str] = None
    ) -> "RealP0Runner":
        """从严格 registry 构造真实 runner。"""
        return cls(load_registry(registry, expected_sha256))

    @staticmethod
    def _candidate(
        candidates: Tuple[CandidateSpec, ...], candidate_id: str
    ) -> CandidateSpec:
        """按 ID 取候选，并拒绝非正式 passed 状态。"""
        for candidate in candidates:
            if candidate.candidate_id == candidate_id:
                if candidate.p0a_status != "passed":
                    raise P0Error("p0a_rejected", "候选未通过正式 P0-A")
                return candidate
        raise P0Error("candidate_unknown", "候选不在冻结 registry")

    def run_infrastructure_preflight(
        self, output_dir: Path, snapshot_root: Optional[Path] = None
    ) -> PreflightResult:
        """执行下载前 non-formal 基础设施检查并写入独立目录。"""
        result = infrastructure_preflight(snapshot_root, require_snapshot=False)
        write_preflight_artifact(output_dir, result)
        return result

    def run_candidate_preflight(
        self, candidate_id: str, snapshot_root: Path, output_dir: Path
    ) -> PreflightResult:
        """加载 C1/C2 并执行 non-formal backend/load smoke。"""
        candidate = self._candidate(self.candidates, candidate_id)
        manifest = load_snapshot_manifest(snapshot_root / "snapshot_manifest.json")
        verify_snapshot_against_manifest(candidate, snapshot_root, manifest)
        loaded = load_transformers_host(candidate, snapshot_root)
        adapter: TransformersHostAdapter
        if candidate.candidate_id == "C1":
            adapter = QwenHostAdapter(loaded)
        elif candidate.candidate_id == "C2":
            adapter = DeepSeekHostAdapter(loaded)
        else:
            adapter = TransformersHostAdapter(loaded)
        result = candidate_load_smoke(loaded, adapter.inspect_architecture())
        write_preflight_artifact(output_dir, result)
        return result

    def run_formal_candidate(
        self,
        candidate_id: str,
        snapshot_root: Path,
        fixture_cases: Tuple[Any, ...],
        output_dir: Path,
        *,
        snapshot_manifest_path: Optional[Path] = None,
        prior_candidate_summary: Optional[Path] = None,
        cross_process_p0b: Optional[Tuple[Mapping[str, Any], ...]] = None,
        controller_elapsed_seconds: float = 0.0,
    ) -> RealRunResult:
        """执行一个候选的正式 P0-A-runtime/B/C/D runner。"""
        candidate = self._candidate(self.candidates, candidate_id)
        started = time.monotonic()
        failures = []
        generations = []
        resources = []
        router_ledger = []
        expert_calls = []
        architecture = None
        p0a_runtime = "failed"
        p0b = "not_run"
        p0c = "not_run"
        p0d = "not_run"
        p0b_metrics: Dict[str, Any] = {}
        p0c_gates: Dict[str, bool] = {}
        p0c_metrics: Dict[str, Any] = {}
        determinism: Dict[str, Any] = {}
        resource_metrics: Dict[str, Any] = {}
        stage_timings: Dict[str, float] = {}
        try:
            if controller_elapsed_seconds > 4 * 60 * 60:
                raise P0Error("p0_total_timeout", "P0-v1 总运行超过 4 小时")
            if controller_elapsed_seconds > 90 * 60:
                raise P0Error("candidate_timeout", "候选 controller 已超过 90 分钟")
            if candidate.attempt_order > 1:
                self._validate_prior_failure(candidate, prior_candidate_summary)
            infra = infrastructure_preflight(snapshot_root)
            if infra.status != "passed":
                raise P0Error(
                    "offline_isolation_unavailable",
                    "正式 run 缺少可验证基础设施隔离",
                )
            if snapshot_manifest_path is None:
                snapshot_manifest_path = snapshot_root / "snapshot_manifest.json"
            manifest = load_snapshot_manifest(snapshot_manifest_path)
            snapshot = verify_snapshot_against_manifest(
                candidate, snapshot_root, manifest
            )
            if not snapshot.read_only_verified:
                raise P0Error("snapshot_not_read_only", "snapshot 未通过只读检查")
            p0a_runtime = "passed"
            loaded = load_transformers_host(candidate, snapshot_root)
            stage_timings["load_seconds"] = float(loaded.load_seconds)
            adapter: TransformersHostAdapter
            if candidate.candidate_id == "C1":
                adapter = QwenHostAdapter(loaded)
            elif candidate.candidate_id == "C2":
                adapter = DeepSeekHostAdapter(loaded)
            else:
                adapter = TransformersHostAdapter(loaded)
            architecture = adapter.inspect_architecture()
            resources.append(sample_resource("load", started, loaded.model))
            capability_started = time.monotonic()
            p0b = "failed"
            capability = evaluate_fixture(adapter, fixture_cases)
            capability_seconds = time.monotonic() - capability_started
            stage_timings["p0b_seconds"] = capability_seconds
            generations.extend(capability["record_objects"])
            p0b_metrics = {
                "protocol_id": capability.get("protocol_id", "p0b-canonical-cache-v1"),
                "canonical_use_cache": capability.get("canonical_use_cache", True),
                "groups": capability.get("groups", {}),
                "thresholds": capability.get("thresholds", {}),
                "groups_by_mode": capability.get("groups_by_mode", {}),
                "error_count": len(capability.get("errors", ())),
                "canonical_error_count": capability.get("canonical_error_count", 0),
                "diagnostic_error_count": capability.get("diagnostic_error_count", 0),
                "error_code_counts": self._p0b_error_code_counts(capability),
                "error_examples": self._p0b_error_examples(capability),
            }
            determinism = {
                "in_process_run_count": capability.get("in_process_run_count", 0),
                "in_process_difference_count": capability.get(
                    "in_process_difference_count", 1
                ),
                "cache_difference_count": capability.get("cache_difference_count", 1),
                "diagnostic_in_process_difference_count": capability.get(
                    "diagnostic_in_process_difference_count", 1
                ),
                "diagnostic_deterministic": capability.get(
                    "diagnostic_deterministic", False
                ),
                "new_process_run_count": 0,
                "new_process_difference_count": None,
            }
            if capability["status"] != "passed":
                raise P0Error("capability_below_threshold", "P0-B 能力门失败")
            if capability_seconds > 60 * 60:
                raise P0Error("suite_timeout", "P0-B suite 超过 60 分钟")
            self._validate_cross_process_p0b(capability, cross_process_p0b)
            determinism["new_process_run_count"] = 3
            determinism["new_process_difference_count"] = 0
            p0b = "passed"
            p0c = "failed"
            structure_started = time.monotonic()
            structure = inspect_real_structure(adapter, architecture)
            stage_timings["p0c_seconds"] = time.monotonic() - structure_started
            p0c_gates = dict(structure.get("gates", {}))
            all_allowed = dict(structure.get("all_allowed", {}))
            router_ledger.append(
                {
                    "phase": "all_allowed",
                    "selected_ids": [
                        list(row) for row in all_allowed.get("selected_ids", ())
                    ],
                    "routed_actual_calls": int(
                        all_allowed.get("routed_actual_calls", 0)
                    ),
                    "shared_actual_calls": int(
                        all_allowed.get("shared_actual_calls", 0)
                    ),
                }
            )
            all_allowed_invocations = tuple(
                all_allowed.get("grouped_mm_invocations", ())
            )
            if all_allowed_invocations:
                router_ledger.append(
                    {
                        "phase": "all_allowed_grouped_mm_execution",
                        "execution_observer_kind": all_allowed.get(
                            "execution_observer_kind"
                        ),
                        "invocations": [
                            asdict(item) for item in all_allowed_invocations
                        ],
                    }
                )
            probe_result = structure.get("probe_result")
            if probe_result is not None:
                router_ledger.extend(
                    {
                        "global_row": index,
                        "selected_ids": list(selected),
                        "phase": "mixed",
                    }
                    for index, selected in enumerate(probe_result.selected_ids)
                )
                expert_calls.extend(probe_result.expert_calls)
                if probe_result.grouped_mm_invocations:
                    router_ledger.append(
                        {
                            "phase": "grouped_mm_execution",
                            "execution_observer_kind": probe_result.execution_observer_kind,
                            "invocations": [
                                asdict(item)
                                for item in probe_result.grouped_mm_invocations
                            ],
                        }
                    )
            if not structure["passed"]:
                failures.extend(structure["failure_codes"])
                code = (
                    structure["failure_codes"][0]
                    if structure["failure_codes"]
                    else "host_architecture_unresolved"
                )
                raise P0Error(code, "P0-C 结构门失败")
            mixed_shared = sum(item.branch == "shared" for item in expert_calls)
            mixed_routed = sum(item.branch == "routed" for item in expert_calls)
            mixed_invocations = (
                tuple(probe_result.grouped_mm_invocations)
                if probe_result is not None
                else tuple()
            )
            p0c_metrics = {
                "shared_actual_calls": int(all_allowed.get("shared_actual_calls", 0))
                + mixed_shared,
                "routed_actual_calls": int(all_allowed.get("routed_actual_calls", 0))
                + mixed_routed,
                "mixed_shared_actual_calls": mixed_shared,
                "mixed_routed_actual_calls": mixed_routed,
                "execution_observer_kind": (
                    probe_result.execution_observer_kind
                    if probe_result is not None
                    else None
                ),
                "grouped_mm_invocation_count": (
                    len(probe_result.grouped_mm_invocations)
                    if probe_result is not None
                    else 0
                ),
                "grouped_mm_counts": (
                    list(mixed_invocations[0].counts) if mixed_invocations else []
                ),
                "all_allowed_grouped_mm_invocation_count": len(all_allowed_invocations),
                "all_allowed_grouped_mm_counts": (
                    list(all_allowed_invocations[0].counts)
                    if all_allowed_invocations
                    else []
                ),
                "mixed_grouped_mm_original_rows_by_expert": (
                    [
                        list(rows)
                        for rows in mixed_invocations[0].original_rows_by_expert
                    ]
                    if mixed_invocations
                    else []
                ),
                "mixed_grouped_mm_zero_call_count": (
                    sum(count == 0 for count in mixed_invocations[0].counts)
                    if mixed_invocations
                    else 0
                ),
            }
            p0c = "passed"
            resources.append(sample_resource("structure", started, loaded.model))
            p0d = "failed"
            p0d_sample = sample_resource("complete", started, loaded.model)
            resources.append(p0d_sample)
            validate_resource_gates(
                loaded.load_seconds,
                controller_elapsed_seconds + time.monotonic() - started,
                tuple(resources),
                max_reserved_bytes=candidate.profile.max_reserved_bytes,
                min_free_bytes=candidate.profile.min_free_bytes,
            )
            reserved = [
                item.cuda_reserved_bytes
                for item in resources
                if item.cuda_reserved_bytes is not None
            ]
            free = [
                item.gpu_free_bytes
                for item in resources
                if item.gpu_free_bytes is not None
            ]
            host_rss = [
                item.host_rss_bytes
                for item in resources
                if item.host_rss_bytes is not None
            ]
            resource_metrics = {
                "max_cuda_reserved_bytes": max(reserved),
                "min_gpu_free_bytes": min(free),
                "max_host_rss_bytes": max(host_rss) if host_rss else None,
                "cpu_offload_detected": any(
                    item.cpu_offload_detected for item in resources
                ),
                "disk_offload_detected": any(
                    item.disk_offload_detected for item in resources
                ),
            }
            p0d = "passed"
            stage_timings["total_seconds"] = (
                controller_elapsed_seconds + time.monotonic() - started
            )
            result = RealRunResult(
                candidate_id=candidate_id,
                profile_id=candidate.profile.profile_id,
                status="passed",
                p0a_runtime=p0a_runtime,
                p0b=p0b,
                p0c=p0c,
                p0d=p0d,
                failure_codes=tuple(),
                generations=tuple(generations),
                architecture=architecture,
                resource_samples=tuple(resources),
                router_ledger=tuple(router_ledger),
                expert_calls=tuple(expert_calls),
                p0b_metrics=p0b_metrics,
                p0c_gates=p0c_gates,
                p0c_metrics=p0c_metrics,
                determinism=determinism,
                resource_metrics=resource_metrics,
                stage_timings=stage_timings,
                exit_code=0,
            )
        except P0Error as exc:
            failures.append(exc.code)
            stage_timings["total_seconds"] = (
                controller_elapsed_seconds + time.monotonic() - started
            )
            result = RealRunResult(
                candidate_id=candidate_id,
                profile_id=candidate.profile.profile_id,
                status="failed",
                p0a_runtime=p0a_runtime,
                p0b=p0b,
                p0c=p0c,
                p0d=p0d,
                failure_codes=tuple(dict.fromkeys(failures)),
                generations=tuple(generations),
                architecture=architecture,
                resource_samples=tuple(resources),
                router_ledger=tuple(router_ledger),
                expert_calls=tuple(expert_calls),
                p0b_metrics=p0b_metrics,
                p0c_gates=p0c_gates,
                p0c_metrics=p0c_metrics,
                determinism=determinism,
                resource_metrics=resource_metrics,
                stage_timings=stage_timings,
                exit_code=2,
            )
        write_run_artifacts(
            output_dir,
            candidate,
            result,
            {
                "schema_version": 1,
                "candidate_id": candidate.candidate_id,
                "profile_id": candidate.profile.profile_id,
                "registry_candidate_digest": candidate.p0a_decision_sha256,
                "fixture_case_count": len(fixture_cases),
                "non_formal": False,
            },
            result.architecture,
            locals().get("manifest"),
        )
        return result

    def run_p0b_worker(
        self,
        candidate_id: str,
        snapshot_root: Path,
        fixture_cases: Tuple[Any, ...],
        *,
        prior_candidate_summary: Optional[Path] = None,
    ) -> Mapping[str, Any]:
        """在全新进程中只执行正式 P0-B，并返回可精确比较的签名。"""
        candidate = self._candidate(self.candidates, candidate_id)
        if candidate.attempt_order > 1:
            self._validate_prior_failure(candidate, prior_candidate_summary)
        infra = infrastructure_preflight(snapshot_root)
        if infra.status != "passed":
            raise P0Error(
                "offline_isolation_unavailable", "P0-B worker 缺少基础设施隔离"
            )
        manifest = load_snapshot_manifest(snapshot_root / "snapshot_manifest.json")
        verify_snapshot_against_manifest(candidate, snapshot_root, manifest)
        loaded = load_transformers_host(candidate, snapshot_root)
        if candidate.candidate_id == "C1":
            adapter: TransformersHostAdapter = QwenHostAdapter(loaded)
        elif candidate.candidate_id == "C2":
            adapter = DeepSeekHostAdapter(loaded)
        else:
            adapter = TransformersHostAdapter(loaded)
        capability = evaluate_fixture(adapter, fixture_cases)
        return self._p0b_signature(capability)

    @staticmethod
    def _p0b_error_code_counts(capability: Mapping[str, Any]) -> Mapping[str, int]:
        """统计 P0-B 稳定错误码，不持久化异常文本或敏感输入。"""

        errors = capability.get("errors", ())
        if not isinstance(errors, (tuple, list)):
            return {}
        counts = Counter(
            item.get("code")
            for item in errors
            if isinstance(item, Mapping) and isinstance(item.get("code"), str)
        )
        return {code: counts[code] for code in sorted(counts)}

    @staticmethod
    def _p0b_error_examples(
        capability: Mapping[str, Any], limit: int = 8
    ) -> Tuple[Mapping[str, Any], ...]:
        """保留有限错误定位样本和安全摘要，避免写入 prompt/token/异常文本。"""

        errors = capability.get("errors", ())
        if not isinstance(errors, (tuple, list)):
            return ()
        examples = []
        for item in errors:
            if not isinstance(item, Mapping):
                continue
            case_id = item.get("case_id")
            use_cache = item.get("use_cache")
            repeat = item.get("repeat")
            code = item.get("code")
            if not all(
                isinstance(value, str) for value in (case_id, use_cache, repeat, code)
            ):
                continue
            examples.append(
                {
                    "case_id": case_id,
                    "use_cache": use_cache,
                    "repeat": repeat,
                    "code": code,
                }
            )
            details = item.get("details")
            if isinstance(details, Mapping):
                safe_details = {
                    key: value
                    for key, value in details.items()
                    if key in {"value", "output", "prompt_length", "output_length"}
                    and isinstance(value, (str, int, float, list, dict, tuple))
                }
                if safe_details:
                    examples[-1]["details"] = safe_details
            if len(examples) >= limit:
                break
        return tuple(examples)

    @staticmethod
    def _p0b_signature(capability: Mapping[str, Any]) -> Mapping[str, Any]:
        """提取 canonical cache 模式的不含 prompt token/停止签名。"""
        records = capability.get("record_objects")
        if not isinstance(records, tuple):
            raise P0Error("cross_process_evidence_invalid", "P0-B records 缺失")
        canonical_use_cache = bool(capability.get("canonical_use_cache", True))
        canonical_records = [
            item for item in records if item.use_cache == canonical_use_cache
        ]
        if not canonical_records:
            raise P0Error(
                "cross_process_evidence_invalid", "canonical P0-B records 缺失"
            )
        return {
            "status": capability.get("status"),
            "protocol_id": capability.get("protocol_id", "p0b-canonical-cache-v1"),
            "groups": capability.get("groups"),
            "signatures": [
                {
                    "case_id": item.case_id,
                    "use_cache": item.use_cache,
                    "continuation_tokens": list(item.continuation_tokens),
                    "stop_reason": item.stop_reason,
                    "matched": item.matched,
                }
                for item in canonical_records
            ],
        }

    @classmethod
    def _validate_cross_process_p0b(
        cls,
        capability: Mapping[str, Any],
        workers: Optional[Tuple[Mapping[str, Any], ...]],
    ) -> None:
        """要求三个全新 worker 与当前进程的输出逐 token 完全一致。"""
        if workers is None or len(workers) != 3:
            raise P0Error(
                "cross_process_evidence_missing", "必须提供三个全新 worker 结果"
            )
        expected = cls._p0b_signature(capability)
        for worker in workers:
            if dict(worker) != expected:
                raise P0Error(
                    "cross_process_determinism_failed", "P0-B 新进程输出不一致"
                )

    def _validate_prior_failure(
        self, candidate: CandidateSpec, summary_path: Optional[Path]
    ) -> None:
        """验证 C2 只能在前一候选形成正式失败摘要后运行。"""
        if summary_path is None or not summary_path.is_file():
            raise P0Error("prior_candidate_failure_missing", "缺少前序候选失败摘要")
        sidecar = summary_path.with_name(summary_path.name + ".sha256")
        if not sidecar.is_file():
            raise P0Error("prior_candidate_failure_missing", "缺少前序摘要 sidecar")
        expected = sidecar.read_text(encoding="ascii").strip()
        actual = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        if len(expected) != 64 or not hmac.compare_digest(expected, actual):
            raise P0Error("snapshot_digest_mismatch", "前序摘要 sidecar 不匹配")
        payload = load_strict_json(summary_path)
        prior = self.candidates[candidate.attempt_order - 2]
        if (
            not isinstance(payload, dict)
            or payload.get("candidate_id") != prior.candidate_id
            or payload.get("status") != "failed"
            or not isinstance(payload.get("failure_codes"), list)
            or not payload["failure_codes"]
        ):
            raise P0Error(
                "prior_candidate_failure_invalid", "前序候选没有可验证正式失败"
            )
