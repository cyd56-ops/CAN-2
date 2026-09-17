"""生成 G1-a CPU reference 交付物；默认使用独立的小型候选 fixture。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """解析输出目录并生成一次不可覆盖的 reference 运行。"""
    # Windows 控制台可能使用本地代码页，runner 输出统一使用 UTF-8。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="生成 G1-a 模整数 reference 交付物")
    parser.add_argument("--output", type=Path, default=Path("results/g1a-modint-v1/run-local"))
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from can.v2.modint_verifier_g1a.artifacts import generate_g1a_artifacts

    summary = generate_g1a_artifacts(args.output)
    print(f"status={summary['status']} output={args.output}")
    print(f"parameter_sha256={summary['parameter_sha256']}")
    print(f"vectors_sha256={summary['vectors_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
