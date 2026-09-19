"""根据已完成的 P0-A 人工审阅自动生成决策与正式 registry。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from can.v2.pretrained_moe_p0.p0a import finalize_p0a_registry


def main() -> int:
    """校验审阅记录并生成不可覆盖的 P0-A 正式产物。"""

    parser = argparse.ArgumentParser(description="Finalize P0-A candidate registry")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    args = parser.parse_args()
    summary = finalize_p0a_registry(
        args.input_root,
        args.prepared_root,
        args.output_root,
        args.registry_output,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
