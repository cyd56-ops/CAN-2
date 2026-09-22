"""执行 P0-MoE 真实宿主的 preflight 或正式 runner。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Tuple

from can.v2.pretrained_moe_p0.fixture import load_fixture
from can.v2.pretrained_moe_p0.p0a import canonical_json_bytes, load_strict_json
from can.v2.pretrained_moe_p0.real_runner import RealP0Runner
from can.v2.pretrained_moe_p0.resource_probe import infrastructure_preflight


def _worker_failure(worker_index: int, code: str) -> Mapping[str, Any]:
    """构造不包含 stderr、环境变量或凭据的稳定 worker 失败占位。"""

    return {"worker_error": code, "worker_index": worker_index}


def _collect_p0b_workers(
    args: argparse.Namespace, controller_started: float
) -> Tuple[Mapping[str, Any], ...]:
    """在候选 90 分钟和 P0-v1 四小时预算内启动三个新进程。"""

    worker_results = []
    with tempfile.TemporaryDirectory(prefix="can-p0b-workers-") as temp_directory:
        for worker_index in range(3):
            elapsed = time.monotonic() - controller_started
            remaining = min(90 * 60 - elapsed, 4 * 60 * 60 - elapsed)
            if remaining <= 0:
                worker_results.extend(
                    _worker_failure(index, "controller_timeout")
                    for index in range(worker_index, 3)
                )
                break
            worker_output = Path(temp_directory) / f"worker-{worker_index}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--registry",
                str(args.registry),
                "--fixture",
                str(args.fixture),
                "--candidate-id",
                args.candidate_id,
                "--snapshot-root",
                str(args.snapshot_root),
                "--output-root",
                str(args.output_root),
                "--p0b-worker-output",
                str(worker_output),
            ]
            if args.prior_candidate_summary is not None:
                command.extend(
                    ["--prior-candidate-summary", str(args.prior_candidate_summary)]
                )
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    timeout=remaining,
                    env=dict(os.environ),
                    capture_output=True,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                worker_results.append(_worker_failure(worker_index, "worker_timeout"))
                continue
            if completed.returncode != 0 or not worker_output.is_file():
                # stderr 可能含路径、token 或远端实现细节，正式 artifact 不持久化。
                worker_results.append(_worker_failure(worker_index, "worker_failed"))
            else:
                payload = load_strict_json(worker_output)
                if not isinstance(payload, dict):
                    worker_results.append(
                        _worker_failure(worker_index, "worker_output_invalid")
                    )
                else:
                    worker_results.append(payload)
    return tuple(worker_results)


def main() -> int:
    """执行明确指定的 infrastructure/candidate preflight 或正式 candidate run。"""
    parser = argparse.ArgumentParser(description="Run P0-MoE real-host checks")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight", choices=("infrastructure", "candidate"))
    parser.add_argument("--prior-candidate-summary", type=Path)
    parser.add_argument("--p0b-worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    runner = RealP0Runner.from_registry(args.registry)
    cases = tuple(load_fixture(args.fixture))
    if args.p0b_worker_output is not None:
        if args.p0b_worker_output.exists():
            raise FileExistsError(args.p0b_worker_output)
        result = runner.run_p0b_worker(
            args.candidate_id,
            args.snapshot_root,
            cases,
            prior_candidate_summary=args.prior_candidate_summary,
        )
        args.p0b_worker_output.write_bytes(canonical_json_bytes(result))
        return 0
    if args.preflight == "infrastructure":
        result = runner.run_infrastructure_preflight(
            args.output_root, args.snapshot_root
        )
        print(json.dumps(result.__dict__, ensure_ascii=True, indent=2, default=str))
        return 0 if result.status == "passed" else 2
    if args.preflight == "candidate":
        result = runner.run_candidate_preflight(
            args.candidate_id, args.snapshot_root, args.output_root
        )
        print(json.dumps(result.__dict__, ensure_ascii=True, indent=2, default=str))
        return 0 if result.status == "passed" else 2
    controller_started = time.monotonic()
    worker_results = _collect_p0b_workers(args, controller_started)
    result = runner.run_formal_candidate(
        args.candidate_id,
        args.snapshot_root,
        cases,
        args.output_root,
        prior_candidate_summary=args.prior_candidate_summary,
        cross_process_p0b=worker_results,
        controller_elapsed_seconds=time.monotonic() - controller_started,
    )
    print(json.dumps(result.__dict__, ensure_ascii=True, indent=2, default=str))
    return 0 if result.status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
