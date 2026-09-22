"""按正式 P0-A registry 下载冻结真实宿主 snapshot。"""

from __future__ import annotations

import argparse
from pathlib import Path

from can.v2.pretrained_moe_p0.registry import load_registry
from can.v2.pretrained_moe_p0.snapshot import download_snapshot


def main() -> int:
    """解析 registry/candidate 并执行一次不可覆盖下载。"""
    parser = argparse.ArgumentParser(description="Download a frozen P0-MoE snapshot")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    candidates = load_registry(args.registry)
    candidate = next(
        item for item in candidates if item.candidate_id == args.candidate_id
    )
    manifest = download_snapshot(candidate, args.snapshot_root)
    print(
        {
            "candidate_id": manifest.candidate_id,
            "total_bytes": manifest.total_bytes,
            "manifest_sha256": manifest.manifest_sha256,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
