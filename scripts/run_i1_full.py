"""运行 I1 G1-b→M2 双 policy CPU 集成。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from can.v2.modint_verifier_m2_i1.runner import run_i1_artifacts


def main() -> int:
    """解析固定依赖 run 并生成不可覆盖的 I1 交付物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g1b-dir",
        type=Path,
        default=ROOT / "results" / "g1b-modint-v1" / "run-20260916-02",
    )
    parser.add_argument("--m2-run-id", default="run-20260915-04")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--statement-coverage", type=float, required=True)
    parser.add_argument("--branch-coverage", type=float, required=True)
    parser.add_argument("--tau", type=int, default=2)
    args = parser.parse_args()
    result = run_i1_artifacts(
        repository_root=ROOT,
        g1b_dir=args.g1b_dir,
        m2_run_id=args.m2_run_id,
        run_id=args.run_id,
        coverage_report=args.coverage_report,
        statement_coverage=args.statement_coverage,
        branch_coverage=args.branch_coverage,
        tau=args.tau,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
