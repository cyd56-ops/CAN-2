"""生成 Phase 5 checkpoint manifest 的最小 CLI。"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.can.v2.transformer.manifest import build_checkpoint_manifest, write_manifest


def main() -> int:
    """解析参数、生成独立 manifest 并打印其摘要。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--epoch", type=int, default=None)
    args = parser.parse_args()
    metadata = {
        key: {
            **({"seed": args.seed} if args.seed is not None else {}),
            **({"stage": args.stage} if args.stage is not None else {}),
            **({"epoch": args.epoch} if args.epoch is not None else {}),
        }
        for key in args.checkpoint
    }
    manifest = build_checkpoint_manifest(args.root, metadata)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["generator"] = "generate_phase5_manifest.py"
    digest = write_manifest(args.output, manifest)
    print(
        json.dumps({"manifest": str(args.output), "sha256": digest}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
