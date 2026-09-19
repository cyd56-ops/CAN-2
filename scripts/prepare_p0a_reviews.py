"""从服务器已有的小型元数据生成 P0-A 盘点结果和人工审阅模板。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from can.v2.pretrained_moe_p0.p0a import prepare_p0a_reviews


def main() -> int:
    """解析目录参数，生成不可覆盖的 P0-A 审阅工作区。"""

    parser = argparse.ArgumentParser(description="Prepare P0-A metadata reviews")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="包含 c1/c2/c3 metadata JSON 和 *-files 目录的路径",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="新建的人工审阅工作区；已存在时拒绝覆盖",
    )
    args = parser.parse_args()
    summary = prepare_p0a_reviews(args.input_root, args.output_root)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
