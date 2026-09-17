"""M1a fixture、调用台账和 summary 的严格解析工具。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from .types import CallLedger

_HASH = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """按 key、shape 和 little-endian C-contiguous bytes 计算 Expert 权重摘要。"""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict 必须为非空 mapping")
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if (
            not isinstance(key, str)
            or not isinstance(value, torch.Tensor)
            or value.is_sparse
        ):
            raise TypeError("state_dict 只允许 dense Tensor")
        array = value.detach().to(device="cpu").contiguous().numpy()
        little = np.asarray(array, dtype=array.dtype.newbyteorder("<"), order="C")
        digest.update(key.encode("utf-8"))
        digest.update(str(little.dtype).encode("ascii"))
        digest.update(str(tuple(little.shape)).encode("ascii"))
        digest.update(little.tobytes(order="C"))
    return digest.hexdigest()


def _digest(path: Path) -> str:
    """计算文件原始 bytes 的 SHA-256。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_m1a_fixture(path: Path) -> Dict[str, Any]:
    """加载 fixture.json 并核对关联 little-endian NPY 摘要。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("fixture 不存在")
    raw = path.read_bytes()

    def duplicate(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        """拒绝重复 JSON 字段。"""
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"fixture 字段重复: {key}")
            result[key] = value
        return result

    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=duplicate,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    required = {"schema_version", "seeds", "input", "cases", "reference", "expected"}
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"] != 1
    ):
        raise ValueError("fixture 根字段不匹配")
    if not isinstance(payload["seeds"], dict) or not {
        "e0_init",
        "e1_init",
        "router_init",
        "input_init",
    }.issubset(payload["seeds"]):
        raise ValueError("fixture seeds 字段不完整")
    for seed in payload["seeds"].values():
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("fixture seed 必须为非负整数")
    for section, filename in (
        ("input", "inputs.npy"),
        ("reference", "reference_outputs.npy"),
    ):
        item = payload[section]
        if (
            not isinstance(item, dict)
            or item.get("path") != filename
            or item.get("format", "npy-v1") != "npy-v1"
        ):
            raise ValueError(f"fixture {section} 文件描述不匹配")
        if item.get("dtype") != "<f4" or not isinstance(item.get("shape"), list):
            raise ValueError(f"fixture {section} dtype/shape 描述不匹配")
        if section == "reference" and (
            item.get("atol") != 1e-6 or item.get("rtol") != 1e-5
        ):
            raise ValueError("fixture reference 容差不匹配")
        digest = item.get("sha256")
        if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
            raise ValueError(f"fixture {section} 摘要格式错误")
        target = path.parent / filename
        if not target.is_file() or _digest(target) != digest:
            raise ValueError(f"fixture {section} 摘要不匹配")
        array = np.load(target, allow_pickle=False)
        if array.dtype.byteorder not in {"<", "="} or not array.flags.c_contiguous:
            raise ValueError(f"fixture {section} 必须是 little-endian C-contiguous")
        if array.dtype != np.dtype("<f4") or list(array.shape) != item["shape"]:
            raise ValueError(f"fixture {section} dtype/shape 与文件不一致")
    if not isinstance(payload["cases"], list) or not isinstance(
        payload["expected"], list
    ):
        raise ValueError("fixture cases/expected 必须为数组")
    return payload


def validate_call_ledger(payload: Mapping[str, Any]) -> None:
    """严格验证调用台账根字段、事件顺序和计数语义。"""

    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "run_id",
        "execution_config_id",
        "policy",
        "events",
    }:
        raise ValueError("call ledger 根字段不匹配")
    if payload["schema_version"] != 1 or not all(
        isinstance(payload[key], str) and payload[key]
        for key in ("run_id", "execution_config_id", "policy")
    ):
        raise ValueError("call ledger 标识字段非法")
    events = payload["events"]
    if not isinstance(events, list):
        raise ValueError("call ledger events 必须为数组")
    keys = {
        "sequence",
        "case_id",
        "stage",
        "expert_id",
        "kind",
        "batch_indices",
        "count",
    }
    for index, event in enumerate(events):
        if (
            not isinstance(event, Mapping)
            or set(event) != keys
            or event["sequence"] != index
            or event["count"] != 1
        ):
            raise ValueError("call ledger event 顺序或字段非法")
        if (
            not isinstance(event["batch_indices"], (list, tuple))
            or not event["batch_indices"]
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in event["batch_indices"]
            )
        ):
            raise ValueError("call ledger batch_indices 非法")


def validate_m1a_summary(payload: Mapping[str, Any]) -> None:
    """验证 M1a summary 的根字段、状态、退出码和失败对象。"""

    required = {
        "schema_version",
        "run_id",
        "execution_config_id",
        "protocol_id",
        "policy",
        "status",
        "exit_code",
        "started_at",
        "finished_at",
        "cases",
        "metrics",
        "determinism",
        "coverage",
        "artifacts",
        "provenance",
        "failure",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload["schema_version"] != 1
    ):
        raise ValueError("summary 根字段不匹配")
    if (
        payload["status"] not in {"complete", "failed", "not_run"}
        or isinstance(payload["exit_code"], bool)
        or not isinstance(payload["exit_code"], int)
    ):
        raise ValueError("summary status/exit_code 非法")
    if payload["failure"] is not None and (
        not isinstance(payload["failure"], Mapping)
        or set(payload["failure"]) != {"code", "stage", "case_id", "retryable"}
        or payload["failure"]["retryable"] is not False
    ):
        raise ValueError("summary failure schema 非法")
