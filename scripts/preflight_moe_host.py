"""运行 P0-MoE 本地 fake-host 结构预检；不下载或加载真实模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from can.v2.pretrained_moe_p0 import FakeHostAdapter, P0Runner, load_registry


def main() -> int:
    """解析 registry 并执行显式 fake profile 的本地结构预检。"""
    parser = argparse.ArgumentParser(description="P0-MoE local fake-host preflight")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-id", action="append", default=[])
    args = parser.parse_args()
    candidates = load_registry(args.registry)
    selected_ids = set(args.candidate_id)
    adapters = {
        candidate.candidate_id: FakeHostAdapter("native")
        for candidate in candidates
        if not selected_ids or candidate.candidate_id in selected_ids
    }
    result = P0Runner(candidates).run_structure_only(adapters)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
