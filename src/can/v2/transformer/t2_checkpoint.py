"""Phase 5.5/T2 checkpoint、RNG 恢复和 keypair 身份工具。"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from ..crypto.lwe import LWEParams, generate_keypair
from .model import GatedDecoderTransformer, TransformerConfig
from .plain_model import PlainDecoderTransformer

T2CheckpointModel = Union[GatedDecoderTransformer, PlainDecoderTransformer]
T2_CHECKPOINT_SCHEMA_VERSION = 1


def array_sha256(value: np.ndarray) -> str:
    """计算含 dtype 与 shape 绑定的 NumPy 数组 SHA-256。"""

    if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
        raise ValueError("摘要输入必须是有限 NumPy 数组")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def model_tensor_sha256(model: torch.nn.Module, *, trainable_only: bool = False) -> str:
    """按参数名和 tensor 内容计算稳定模型摘要。"""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model 必须是 torch.nn.Module")
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if trainable_only and name not in trainable:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def capture_rng_state() -> Dict[str, Any]:
    """捕获 Python、NumPy、Torch 和可用 CUDA RNG 状态。"""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """恢复 checkpoint 中的完整进程 RNG 状态。"""

    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torch",
        "cuda",
    }:
        raise ValueError("checkpoint RNG state 不完整")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint 包含 CUDA RNG，但当前 CUDA 不可用")
        torch.cuda.set_rng_state_all(state["cuda"])


def derive_lwe_keypair(
    seed: int, params: LWEParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用独立派生 seed 确定性生成 T2 toy LWE keypair。"""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed 必须是整数")
    if not isinstance(params, LWEParams):
        raise TypeError("params 必须是 LWEParams")
    return generate_keypair(params, np.random.default_rng(seed + 10_000))


def save_t2_checkpoint(
    path: Path,
    model: T2CheckpointModel,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
    progress: Mapping[str, int],
    credential_rng_state: Optional[Mapping[str, Any]] = None,
) -> None:
    """原子保存 T2 checkpoint；CAN 仅保存公共 A/b，不保存 secret。"""

    if not isinstance(path, Path) or not isinstance(
        model, (GatedDecoderTransformer, PlainDecoderTransformer)
    ):
        raise TypeError("path/model 类型非法")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer 必须是 torch.optim.Optimizer")
    if not isinstance(metadata, Mapping) or not isinstance(progress, Mapping):
        raise TypeError("metadata/progress 必须是 Mapping")
    if "secret" in metadata or any("secret" in str(key).casefold() for key in metadata):
        raise ValueError("checkpoint metadata 禁止包含 secret")
    payload: Dict[str, Any] = {
        "schema_version": T2_CHECKPOINT_SCHEMA_VERSION,
        "lifecycle_stage": "T-pretrain",
        "model_kind": "can" if isinstance(model, GatedDecoderTransformer) else "plain",
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metadata": dict(metadata),
        "progress": dict(progress),
        "rng_state": capture_rng_state(),
        "credential_rng_state": (
            dict(credential_rng_state) if credential_rng_state is not None else None
        ),
    }
    if isinstance(model, GatedDecoderTransformer):
        verifier = model.gate_layer.verifier
        matrix = verifier.A.detach().cpu().numpy().copy()
        vector = verifier.b.detach().cpu().numpy().copy()
        params = verifier.params
        payload["lwe_public"] = {
            "params": asdict(params),
            "A": matrix,
            "b": vector,
            "A_sha256": array_sha256(matrix),
            "b_sha256": array_sha256(vector),
        }
    else:
        payload["lwe_public"] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_t2_checkpoint(
    path: Path,
    model: T2CheckpointModel,
    optimizer: Optional[torch.optim.Optimizer],
    expected_metadata: Mapping[str, Any],
    *,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """校验身份后加载 T2 checkpoint，并可恢复 optimizer/RNG。"""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError("T2 checkpoint 不存在")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema_version",
        "lifecycle_stage",
        "model_kind",
        "model_config",
        "model_state",
        "optimizer_state",
        "metadata",
        "progress",
        "rng_state",
        "credential_rng_state",
        "lwe_public",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("T2 checkpoint schema 字段不匹配")
    expected_kind = "can" if isinstance(model, GatedDecoderTransformer) else "plain"
    if payload["schema_version"] != 1 or payload["lifecycle_stage"] != "T-pretrain":
        raise ValueError("T2 checkpoint schema/lifecycle 不匹配")
    if payload["model_kind"] != expected_kind or payload["model_config"] != asdict(
        model.config
    ):
        raise ValueError("T2 checkpoint 模型身份不匹配")
    if payload["metadata"] != dict(expected_metadata):
        raise ValueError("T2 checkpoint metadata 与本次运行不一致")
    if expected_kind == "can":
        public = payload["lwe_public"]
        if not isinstance(public, Mapping):
            raise ValueError("CAN checkpoint 缺少 LWE 公共参数")
        matrix = public.get("A")
        vector = public.get("b")
        if not isinstance(matrix, np.ndarray) or not isinstance(vector, np.ndarray):
            raise ValueError("CAN checkpoint 的 A/b 类型非法")
        if public.get("A_sha256") != array_sha256(matrix) or public.get(
            "b_sha256"
        ) != array_sha256(vector):
            raise ValueError("CAN checkpoint 的 A/b 摘要不匹配")
        verifier = model.gate_layer.verifier
        if not np.array_equal(
            matrix, verifier.A.detach().cpu().numpy()
        ) or not np.array_equal(vector, verifier.b.detach().cpu().numpy()):
            raise ValueError("CAN checkpoint 的 keypair 与派生 keypair 不匹配")
    elif payload["lwe_public"] is not None:
        raise ValueError("Plain checkpoint 不得包含 LWE 公共参数")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload


__all__ = [
    "T2_CHECKPOINT_SCHEMA_VERSION",
    "array_sha256",
    "capture_rng_state",
    "derive_lwe_keypair",
    "load_t2_checkpoint",
    "model_tensor_sha256",
    "restore_rng_state",
    "save_t2_checkpoint",
]
