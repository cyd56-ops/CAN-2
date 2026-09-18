"""生成 P0-MoE 的 24 条公开能力 fixture。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List


def _digest(item: Dict[str, object]) -> str:
    """计算去除 content_sha256 字段后的 canonical 摘要。"""
    body = {key: item[key] for key in sorted(item) if key != "content_sha256"}
    raw = json.dumps(
        body, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _make_case(index: int, group: str) -> Dict[str, object]:
    """构造一条不依赖预训练知识的公开 case。"""
    if group == "format_copy":
        user = f"Copy the identifier exactly. Identifier: PUBLIC-COPY-{index:02d}."
        expected = f"PUBLIC-COPY-{index:02d}"
        item: Dict[str, object] = {
            "case_id": f"format-copy-{index:02d}",
            "group": group,
            "system_text": "Answer with only the requested text. Do not explain.",
            "user_text": user,
            "expected_text": expected,
            "metric": "strict_em",
            "max_new_tokens": 24,
        }
    elif group == "single_hop":
        user = f"The public fact is: marker-{index:02d} maps to answer-{index:02d}. What is the answer for marker-{index:02d}?"
        item = {
            "case_id": f"single-hop-{index:02d}",
            "group": group,
            "system_text": "Answer with only the requested text. Do not explain.",
            "user_text": user,
            "expected_text": f"answer-{index:02d}",
            "metric": "normalized_em",
            "max_new_tokens": 24,
            "fact_source": "stated_in_prompt",
            "rationale": "The marker-to-answer mapping is stated verbatim in user_text.",
        }
    else:
        user = f"Fact one: node-{index:02d} links to hub-{index:02d}. Fact two: hub-{index:02d} links to result-{index:02d}. Which result does node-{index:02d} reach?"
        item = {
            "case_id": f"two-hop-{index:02d}",
            "group": group,
            "system_text": "Answer with only the requested text. Do not explain.",
            "user_text": user,
            "expected_text": f"result-{index:02d}",
            "metric": "normalized_em",
            "max_new_tokens": 24,
            "fact_source": "stated_in_prompt",
            "rationale": "Both links needed for the two-hop answer are stated verbatim in user_text.",
        }
    item["content_sha256"] = _digest(item)
    return item


def main() -> int:
    """写入不可覆盖的正式 fixture 文件。"""
    output = Path("experiments/p0_moe_host_v1/fixture_v1.json")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 fixture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cases: List[Dict[str, object]] = []
    for group in ("format_copy", "single_hop", "two_hop"):
        cases.extend(_make_case(index, group) for index in range(8))
    payload = {
        "schema_version": 1,
        "fixture_id": "p0-moe-public-fixture-v1",
        "cases": cases,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(hashlib.sha256(output.read_bytes()).hexdigest(), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
