"""生成 P0-A registry；默认执行无需人工填写的机器预筛。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from can.v2.pretrained_moe_p0.p0a import finalize_p0a_registry


def main() -> int:
    """解析参数并生成不可覆盖的 P0-A 产物。

    默认 machine-only 只生成 provisional 结果；--require-human-review
    才会要求 reviewer、时间、理由和逐条 finding 处置，并允许正式 passed。
    """

    parser = argparse.ArgumentParser(description="Finalize P0-A candidate registry")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument(
        "--require-human-review",
        action="store_true",
        help="启用正式人工审阅校验；默认只生成 machine-only provisional 结果",
    )
    args = parser.parse_args()
    summary = finalize_p0a_registry(
        args.input_root,
        args.prepared_root,
        args.output_root,
        args.registry_output,
        require_human_review=args.require_human_review,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
