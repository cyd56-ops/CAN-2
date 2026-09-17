"""生成 G1-b CPU int64 contract 交付物。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """解析输出目录并运行一次不可覆盖的 CPU H/G 对照。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="生成 G1-b CPU verifier 交付物")
    parser.add_argument("--output", type=Path, default=Path("results/g1b-modint-v1/run-local"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from can.v2.modint_verifier_g1b.artifacts import generate_g1b_artifacts

    summary = generate_g1b_artifacts(args.output)
    print(f"status={summary['status']} output={args.output}")
    print(f"h_g_difference_count={summary['h_g_difference_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
