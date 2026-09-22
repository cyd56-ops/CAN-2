"""真实宿主 P0 预检和正式 run 的不可覆盖 artifact 写入。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

from .artifacts import ArtifactWriter
from .real_types import ArchitectureMap, PreflightResult, RealRunResult
from .types import CandidateSpec, P0Error


def _jsonable(value: Any) -> Any:
    """将 dataclass、tuple 和嵌套结构转为规范 JSON 类型。"""
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def write_preflight_artifact(
    output_dir: Path, result: PreflightResult
) -> Mapping[str, str]:
    """写入带 `non_formal=true` 的 preflight artifact。"""

    writer = ArtifactWriter(output_dir)
    summary = {
        "schema_version": 1,
        "non_formal": True,
        "mode": result.mode,
        "status": result.status,
        "checks": _jsonable(dict(result.checks)),
        "failure_codes": list(result.failure_codes),
        "details": _jsonable(dict(result.details)),
    }
    hashes = {
        "preflight_summary_sha256": writer.write_json(
            "preflight_summary.json", summary
        ),
        "environment_sha256": writer.write_json(
            "environment.json",
            {"python": __import__("sys").version, "non_formal": True},
        ),
    }
    writer.write_json(
        "preflight_manifest.json",
        {"schema_version": 1, "non_formal": True, "artifact_sha256": hashes},
    )
    return hashes


def write_run_artifacts(
    output_dir: Path,
    candidate: CandidateSpec,
    result: RealRunResult,
    manifest: Mapping[str, Any],
    architecture: Optional[ArchitectureMap] = None,
    snapshot_manifest: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, str]:
    """原子写入正式 run 的 manifest、generation、architecture、resource 和 summary。"""

    writer = ArtifactWriter(output_dir)
    hashes = {}
    hashes["manifest_sha256"] = writer.write_json(
        "manifest.json", _jsonable(dict(manifest))
    )
    hashes["snapshot_manifest_sha256"] = writer.write_json(
        "snapshot_manifest.json", _jsonable(dict(snapshot_manifest or {}))
    )
    hashes["generations_sha256"] = writer.write_jsonl(
        "generations.jsonl", [_jsonable(item) for item in result.generations]
    )
    hashes["resource_samples_sha256"] = writer.write_jsonl(
        "resource_samples.jsonl", [_jsonable(item) for item in result.resource_samples]
    )
    hashes["architecture_sha256"] = writer.write_json(
        "architecture.json", _jsonable(architecture or result.architecture or {})
    )
    hashes["router_ledger_sha256"] = writer.write_jsonl(
        "router_ledger.jsonl", [_jsonable(item) for item in result.router_ledger]
    )
    hashes["expert_calls_sha256"] = writer.write_jsonl(
        "expert_calls.jsonl", [_jsonable(item) for item in result.expert_calls]
    )
    summary = {
        "schema_version": 1,
        "status": result.status,
        "candidate_id": candidate.candidate_id,
        "profile_id": candidate.profile.profile_id,
        "p0a_runtime": result.p0a_runtime,
        "p0b": result.p0b,
        "p0c": result.p0c,
        "p0d": result.p0d,
        "failure_codes": list(result.failure_codes),
        "exit_code": result.exit_code,
        "generation_count": len(result.generations),
        "p0b_metrics": _jsonable(dict(result.p0b_metrics)),
        "determinism": _jsonable(dict(result.determinism)),
        "p0c_gates": _jsonable(dict(result.p0c_gates)),
        "p0c_metrics": _jsonable(dict(result.p0c_metrics)),
        "resource_metrics": _jsonable(dict(result.resource_metrics)),
        "stage_timings": _jsonable(dict(result.stage_timings)),
        "artifact_sha256": hashes,
    }
    hashes["summary_sha256"] = writer.write_json("summary.json", summary)
    writer.write_text("summary.json.sha256", hashes["summary_sha256"] + "\n")
    return hashes
