"""Phase 5.5/T2 的严格配置、原子状态与一次性 test ledger。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .model import TransformerConfig
from .t2_data import T2_CAP_SUITE, T2_GENERATOR_VERSION, T2_MEM_SUITE
from .t2_metrics import T2_NORMALIZATION_VERSION
from .tokenizer import ByteTokenizer

T2_RUNTIME_SCHEMA_VERSION = 1
T2_TOKENIZER_VERSION = "byte-tokenizer-v1"
_SHA256_LENGTH = 64
_MODES = {"dev-pilot", "frozen-validation"}


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    """将 JSON 对象转换为 dict，并拒绝重复字段。"""

    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复字段: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    """拒绝 JSON 的 NaN、Infinity 和 -Infinity 扩展。"""

    raise ValueError(f"JSON 包含非有限数值: {value}")


def load_strict_json(path: Path) -> Dict[str, Any]:
    """读取 UTF-8 JSON，拒绝重复字段、非对象根节点和非有限值。"""

    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path")
    if not path.is_file():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("JSON 必须使用 UTF-8 编码") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON 根节点必须是 object")
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """以规范 JSON 原子替换目标文件，并返回文件 SHA-256。"""

    if not isinstance(path, Path) or not isinstance(payload, Mapping):
        raise TypeError("path 必须是 Path，payload 必须是 Mapping")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = (
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("JSON payload 必须可序列化且不含非有限数值") from exc
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """流式计算普通文件的 SHA-256。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("摘要目标文件不存在")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, name: str) -> int:
    """验证并返回非 bool 正整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是非 bool 整数")
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


def _positive_float(value: Any, name: str) -> float:
    """验证并返回有限正浮点数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是有限正数")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    """拒绝结构化配置中的未知字段和缺失字段。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是 object")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ValueError(
            f"{name} 字段不匹配；missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


@dataclass(frozen=True)
class T2RuntimeConfig:
    """保存 T2 T-pretrain 运行所需的完整、可比较配置。"""

    mode: str
    suite_id: str
    prompt_group: str
    seed: int
    models: str
    split_counts: Mapping[str, int]
    batch_size: int
    token_budget: int
    validation_interval_tokens: int
    learning_rate: float
    max_new_tokens: int
    cache_mode: str
    model_config: TransformerConfig
    lwe_n: int = 32
    lwe_m: int = 64
    lwe_sigma: float = 1.0

    def __post_init__(self) -> None:
        """验证运行模式、四元组边界、数据身份和模型参数。"""

        if self.mode not in _MODES:
            raise ValueError("mode 必须为 dev-pilot 或 frozen-validation")
        if self.suite_id not in {T2_CAP_SUITE, T2_MEM_SUITE}:
            raise ValueError("suite_id 必须为 t2_nl_cap 或 t2_nl_mem")
        if self.prompt_group not in {"C0", "C1", "C2"}:
            raise ValueError("prompt_group 必须为 C0、C1 或 C2")
        _positive_int(self.seed, "seed")
        if self.models not in {"both", "plain", "can"}:
            raise ValueError("models 必须为 both、plain 或 can")
        _exact_keys(
            self.split_counts,
            {"train", "dev", "validation", "test"},
            "split_counts",
        )
        for split, count in self.split_counts.items():
            _positive_int(count, f"split_counts.{split}")
        _positive_int(self.batch_size, "batch_size")
        if self.batch_size < 4 or self.batch_size % 4:
            raise ValueError("T2 batch_size 必须至少为 4 且能被 4 整除")
        _positive_int(self.token_budget, "token_budget")
        _positive_int(self.validation_interval_tokens, "validation_interval_tokens")
        _positive_float(self.learning_rate, "learning_rate")
        _positive_int(self.max_new_tokens, "max_new_tokens")
        if self.cache_mode not in {"none", "kv"}:
            raise ValueError("cache_mode 必须为 none 或 kv")
        if not isinstance(self.model_config, TransformerConfig):
            raise TypeError("model_config 必须是 TransformerConfig")
        _positive_int(self.lwe_n, "lwe_n")
        _positive_int(self.lwe_m, "lwe_m")
        _positive_float(self.lwe_sigma, "lwe_sigma")
        if self.lwe_m < self.lwe_n:
            raise ValueError("lwe_m 必须不小于 lwe_n")

    def to_dict(self) -> Dict[str, Any]:
        """转换为可写入 resolved_config.json 的规范字典。"""

        value = asdict(self)
        value["schema_version"] = T2_RUNTIME_SCHEMA_VERSION
        value["generator_version"] = T2_GENERATOR_VERSION
        value["normalization_version"] = T2_NORMALIZATION_VERSION
        value["tokenizer_version"] = T2_TOKENIZER_VERSION
        return value


def runtime_config_from_mapping(payload: Mapping[str, Any]) -> T2RuntimeConfig:
    """从严格字段映射构造 T2RuntimeConfig。"""

    expected = {
        "schema_version",
        "mode",
        "suite_id",
        "prompt_group",
        "seed",
        "models",
        "split_counts",
        "batch_size",
        "token_budget",
        "validation_interval_tokens",
        "learning_rate",
        "max_new_tokens",
        "cache_mode",
        "model_config",
        "lwe_n",
        "lwe_m",
        "lwe_sigma",
        "generator_version",
        "normalization_version",
        "tokenizer_version",
    }
    _exact_keys(payload, expected, "runtime config")
    if payload["schema_version"] != T2_RUNTIME_SCHEMA_VERSION:
        raise ValueError("runtime schema_version 不受支持")
    identities = {
        "generator_version": T2_GENERATOR_VERSION,
        "normalization_version": T2_NORMALIZATION_VERSION,
        "tokenizer_version": T2_TOKENIZER_VERSION,
    }
    if any(payload[name] != value for name, value in identities.items()):
        raise ValueError("runtime 的 generator/normalizer/tokenizer identity 不匹配")
    model_payload = payload["model_config"]
    model_fields = set(TransformerConfig.__dataclass_fields__)
    _exact_keys(model_payload, model_fields, "model_config")
    return T2RuntimeConfig(
        mode=payload["mode"],
        suite_id=payload["suite_id"],
        prompt_group=payload["prompt_group"],
        seed=payload["seed"],
        models=payload["models"],
        split_counts=dict(payload["split_counts"]),
        batch_size=payload["batch_size"],
        token_budget=payload["token_budget"],
        validation_interval_tokens=payload["validation_interval_tokens"],
        learning_rate=payload["learning_rate"],
        max_new_tokens=payload["max_new_tokens"],
        cache_mode=payload["cache_mode"],
        model_config=TransformerConfig(**dict(model_payload)),
        lwe_n=payload["lwe_n"],
        lwe_m=payload["lwe_m"],
        lwe_sigma=payload["lwe_sigma"],
    )


def load_runtime_config(path: Path) -> T2RuntimeConfig:
    """从严格 JSON 文件读取 T2 runtime 配置。"""

    return runtime_config_from_mapping(load_strict_json(path))


def load_frozen_runtime(path: Path, expected_sha256: str) -> T2RuntimeConfig:
    """校验可信外部摘要并加载 frozen-validation 配置。"""

    if not isinstance(expected_sha256, str) or len(expected_sha256) != _SHA256_LENGTH:
        raise ValueError("expected freeze SHA-256 必须是 64 位小写十六进制")
    if any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError("expected freeze SHA-256 必须是小写十六进制")
    if file_sha256(path) != expected_sha256:
        raise ValueError("freeze record SHA-256 不匹配")
    config = load_runtime_config(path)
    if config.mode != "frozen-validation":
        raise ValueError("freeze record mode 必须为 frozen-validation")
    if config.models != "both":
        raise ValueError("正式 frozen-validation 必须运行 Plain/CAN 成对模型")
    return config


def initialize_run_state(path: Path, identity: Mapping[str, Any]) -> Dict[str, Any]:
    """创建新的 T2 pair run state，拒绝覆盖已有状态。"""

    if path.exists():
        raise FileExistsError("run_state.json 已存在；恢复必须显式使用 --resume")
    state = {
        "schema_version": 1,
        "status": "running",
        "lifecycle_stage": "T-pretrain",
        "identity": dict(identity),
        "models": {},
        "completed_models": [],
        "stages": {
            "T-pretrain": "running",
            "A": "not_run",
            "B": "not_run",
            "C": "not_run",
        },
    }
    atomic_write_json(path, state)
    return state


def load_run_state(path: Path, expected_identity: Mapping[str, Any]) -> Dict[str, Any]:
    """读取 run state 并拒绝跨实验身份恢复。"""

    state = load_strict_json(path)
    required = {
        "schema_version",
        "status",
        "lifecycle_stage",
        "identity",
        "models",
        "completed_models",
        "stages",
    }
    _exact_keys(state, required, "run state")
    if state["schema_version"] != 1 or state["lifecycle_stage"] != "T-pretrain":
        raise ValueError("run state schema/lifecycle 不匹配")
    if state["identity"] != dict(expected_identity):
        raise ValueError("run state identity 与本次运行不一致")
    return state


def begin_test_access(run_directory: Path, identity: Mapping[str, Any]) -> Path:
    """在生成 test 数据前永久占用一次性访问权并写入 started ledger。"""

    if not isinstance(run_directory, Path):
        raise TypeError("run_directory 必须是 Path")
    run_directory.mkdir(parents=True, exist_ok=True)
    claim = run_directory / "test_access.claim"
    ledger = run_directory / "test_access_ledger.json"
    try:
        with claim.open("x", encoding="ascii") as handle:
            handle.write("claimed\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError("test split 已被 started/failed/completed 运行占用") from exc
    if ledger.exists():
        raise RuntimeError("test ledger 已存在，拒绝再次读取 test")
    atomic_write_json(
        ledger,
        {
            "schema_version": 1,
            "status": "started",
            "identity": dict(identity),
        },
    )
    return ledger


def finish_test_access(
    ledger_path: Path,
    identity: Mapping[str, Any],
    status: str,
    error: Optional[str] = None,
) -> str:
    """将已 started 的 test ledger 更新为 completed 或 failed。"""

    if status not in {"completed", "failed"}:
        raise ValueError("test ledger 终态必须为 completed 或 failed")
    ledger = load_strict_json(ledger_path)
    if ledger.get("status") != "started" or ledger.get("identity") != dict(identity):
        raise RuntimeError("只能更新身份一致的 started test ledger")
    result = {
        "schema_version": 1,
        "status": status,
        "identity": dict(identity),
        "error": error if status == "failed" else None,
    }
    return atomic_write_json(ledger_path, result)


__all__ = [
    "T2RuntimeConfig",
    "T2_RUNTIME_SCHEMA_VERSION",
    "T2_TOKENIZER_VERSION",
    "atomic_write_json",
    "begin_test_access",
    "file_sha256",
    "finish_test_access",
    "initialize_run_state",
    "load_frozen_runtime",
    "load_run_state",
    "load_runtime_config",
    "load_strict_json",
    "runtime_config_from_mapping",
]
