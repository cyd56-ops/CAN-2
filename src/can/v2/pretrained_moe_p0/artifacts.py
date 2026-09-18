"""P0-MoE artifact 的原子写入与摘要工具。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, List, Mapping

from .types import P0Error


class ArtifactWriter:
    """在不可覆盖目录中原子写入 JSON/JSONL 交付物。"""

    def __init__(self, output_dir: Path) -> None:
        """创建空输出目录；已有目录默认拒绝。"""
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir 必须为 Path")
        if output_dir.exists():
            raise P0Error("artifact_exists", "输出目录已存在，拒绝覆盖")
        output_dir.mkdir(parents=True)
        self.output_dir = output_dir

    def write_json(self, name: str, payload: Mapping[str, Any]) -> str:
        """以规范 JSON 原子写入文件并返回 SHA-256。"""
        if not name.endswith(".json") or Path(name).name != name:
            raise P0Error("artifact_schema_mismatch", "artifact 文件名非法")
        raw = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return self._write(name, raw)

    def write_jsonl(self, name: str, rows: List[Mapping[str, Any]]) -> str:
        """以规范 JSONL 原子写入文件并返回 SHA-256。"""
        if not name.endswith(".jsonl") or Path(name).name != name:
            raise P0Error("artifact_schema_mismatch", "JSONL 文件名非法")
        raw = b"".join(
            json.dumps(
                row, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + b"\n"
            for row in rows
        )
        return self._write(name, raw)

    def _write(self, name: str, raw: bytes) -> str:
        """先写临时文件，再使用原子替换提交 artifact。"""
        target = self.output_dir / name
        temp = self.output_dir / f".{name}.tmp"
        if target.exists() or temp.exists():
            raise P0Error("artifact_exists", "artifact 文件已存在")
        try:
            temp.write_bytes(raw)
            os.replace(temp, target)
        except OSError as exc:
            if temp.exists():
                temp.unlink()
            raise P0Error("artifact_write_failed", "artifact 原子写入失败") from exc
        return hashlib.sha256(raw).hexdigest()
