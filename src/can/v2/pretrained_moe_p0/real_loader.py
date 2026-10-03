"""Transformers 真实宿主的离线加载器和结构盘点。

所有重依赖都在函数内部延迟导入；这保证本地 schema/stand-in 测试不需要
GPU、联网或安装 Transformers。
"""

from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .real_types import ArchitectureMap
from .types import CandidateSpec, P0Error


@dataclass
class LoadedRealHost:
    """保存已加载的 tokenizer/model 和加载资源事实。"""

    candidate: CandidateSpec
    tokenizer: Any
    model: Any
    architecture: ArchitectureMap
    load_seconds: float
    parameter_devices: Tuple[str, ...]
    parameter_dtypes: Tuple[str, ...]
    quantization_class: str
    packed_expert_records: Tuple[Dict[str, Any], ...]
    cpu_offload_detected: bool
    disk_offload_detected: bool
    cuda_peak_allocated: Optional[int]
    cuda_peak_reserved: Optional[int]


def _torch() -> Any:
    """延迟导入 torch，缺失时返回稳定错误。"""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 服务器环境
        raise P0Error("optional_dependency_missing", "需要 PyTorch") from exc
    return torch


def _transformers() -> Tuple[Any, Any, Any]:  # pragma: no cover - 服务器依赖边界
    """延迟导入 Transformers 的 Auto 类。"""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - 服务器环境
        raise P0Error("optional_dependency_missing", "需要 transformers") from exc
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        BitsAndBytesConfig = None
    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _reset_cuda_peak_memory(torch: Any, device_index: int) -> None:
    """显式选择 CUDA 设备后再重置峰值统计，兼容部分运行时初始化顺序。"""

    # 某些 PyTorch/CUDA 组合在未显式建立当前设备上下文时，直接 reset
    # 会报 Invalid device argument；先选择设备可使 runner 与独立诊断一致。
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)


def _quantization_config(candidate: CandidateSpec, bits_config: Any) -> Any:
    """根据 registry 的冻结字段创建量化配置，不接受调用方覆盖。"""

    config = dict(candidate.profile.quantization_config or {})
    if not config:
        return None
    if bits_config is None:
        raise P0Error(
            "optional_dependency_missing", "NF4 profile 需要 BitsAndBytesConfig"
        )
    torch = _torch()
    dtype_name = config.get("bnb_4bit_compute_dtype")
    if dtype_name != "bfloat16" or config.get("bnb_4bit_quant_type") != "nf4":
        raise P0Error(
            "quantization_profile_unverifiable", "registry 不是冻结 NF4 profile"
        )
    return bits_config(
        load_in_4bit=config.get("load_in_4bit") is True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _module_path_records(model: Any) -> Tuple[Tuple[str, Any], ...]:
    """返回模型模块路径，兼容普通 torch.nn.Module。"""

    named = getattr(model, "named_modules", None)
    if not callable(named):
        raise P0Error("host_architecture_unresolved", "模型没有 named_modules")
    return tuple(named())


def _infer_architecture(candidate: CandidateSpec, model: Any) -> ArchitectureMap:
    """从实际模块类和属性盘点 MoE/router/shared/routed 路径。"""

    modules = _module_path_records(model)
    moe_layers = []
    routers = []
    shared = []
    routed = []
    backend = "unknown"
    native_top_k = 0
    routed_expert_count = 0
    supports_counter = False
    supports_row_mask = False
    for path, module in modules:
        name = module.__class__.__name__.lower()
        lower_path = path.lower()
        if "moe" in name or "moe" in lower_path or "sparsemoe" in name:
            if path:
                moe_layers.append(path)
        if "router" in name or "gate" in name and "mlp" not in lower_path:
            if path:
                routers.append(path)
            for attr in ("top_k", "num_experts_per_tok", "num_selected_experts"):
                value = getattr(module, attr, None)
                if type(value) is int and value > 0:
                    native_top_k = value
                    break
        is_shared = "shared" in lower_path or "sharedexpert" in name
        if is_shared:
            shared.append(path)
        if ("expert" in lower_path or "experts" in lower_path) and not is_shared:
            routed.append(path)
        if hasattr(module, "expert_idx") or hasattr(module, "expert_id"):
            supports_counter = True
        if callable(getattr(module, "set_allowed_mask", None)):
            supports_row_mask = True
        implementation = getattr(module, "use_experts_implementation", None)
        if isinstance(implementation, str):
            backend = implementation
    config = getattr(model, "config", None)
    config_implementation = getattr(config, "_experts_implementation", None)
    if isinstance(config_implementation, str):
        backend = config_implementation
    for attr in (
        "num_experts_per_tok",
        "num_selected_experts",
        "num_experts_per_token",
    ):
        value = getattr(config, attr, None)
        if type(value) is int and value > 0:
            native_top_k = value
            break
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        value = getattr(config, attr, None)
        if type(value) is int and value > 0:
            routed_expert_count = value
            break
    return ArchitectureMap(
        architecture_family=candidate.expected_architecture_family,
        model_class=model.__class__.__name__,
        moe_layers=tuple(dict.fromkeys(moe_layers)),
        router_paths=tuple(dict.fromkeys(routers)),
        shared_paths=tuple(dict.fromkeys(shared)),
        routed_paths=tuple(dict.fromkeys(routed)),
        native_top_k=native_top_k,
        backend_id=backend,
        supports_row_mask=supports_row_mask,
        supports_execution_counter=supports_counter,
        routed_expert_count=routed_expert_count,
    )


def _packed_expert_records(model: Any) -> Tuple[Dict[str, Any], ...]:
    """盘点 packed expert 参数的类型、dtype、设备、存储和量化 metadata。"""

    records = []
    for path, module in _module_path_records(model):
        lower = path.lower()
        if "expert" not in lower and "experts" not in lower:
            continue
        named_parameters = getattr(module, "named_parameters", None)
        if not callable(named_parameters):
            continue
        for name, parameter in named_parameters(recurse=False):
            full_name = f"{path}.{name}" if path else name
            data = getattr(parameter, "data", parameter)
            numel = int(getattr(data, "numel", lambda: 0)())
            element_size = int(getattr(data, "element_size", lambda: 0)())
            quant_state = getattr(parameter, "quant_state", None)
            quant_type = getattr(quant_state, "quant_type", None)
            if quant_type is None:
                quant_type = getattr(parameter, "quant_type", None)
            records.append(
                {
                    "name": full_name,
                    "python_type": f"{parameter.__class__.__module__}.{parameter.__class__.__name__}",
                    "dtype": str(getattr(data, "dtype", "unknown")),
                    "device": str(getattr(data, "device", "unknown")),
                    "shape": tuple(int(item) for item in getattr(data, "shape", ())),
                    "numel": numel,
                    "storage_bytes": numel * element_size,
                    "quant_type": None if quant_type is None else str(quant_type),
                    "is_params4bit": parameter.__class__.__name__ == "Params4bit",
                }
            )
    return tuple(records)


def _device_facts(model: Any) -> Tuple[Tuple[str, ...], Tuple[str, ...], bool, bool]:
    """枚举参数/缓冲区设备并检测 CPU/disk offload。"""

    devices = []
    dtypes = []
    offload_cpu = False
    offload_disk = False
    for parameter in list(getattr(model, "parameters", lambda: ())()) + list(
        getattr(model, "buffers", lambda: ())()
    ):
        device = str(getattr(parameter, "device", "unknown"))
        devices.append(device)
        dtypes.append(str(getattr(parameter, "dtype", "unknown")))
        if device.startswith("cpu"):
            offload_cpu = True
        if device.startswith("meta") or "disk" in device:
            offload_disk = True
    return (
        tuple(sorted(set(devices))),
        tuple(sorted(set(dtypes))),
        offload_cpu,
        offload_disk,
    )


def load_transformers_host(
    candidate: CandidateSpec, snapshot_root: Path, device_index: int = 0
) -> LoadedRealHost:  # pragma: no cover - 服务器 CUDA/Transformers 集成路径
    """从冻结本地 snapshot 加载真实 Transformers 宿主并执行资源硬检查。"""

    if candidate.p0a_status != "passed":
        raise P0Error("p0a_rejected", "候选未通过正式 P0-A")
    if not isinstance(snapshot_root, Path) or not snapshot_root.is_dir():
        raise P0Error("snapshot_missing", "snapshot 根目录不存在")
    torch = _torch()
    if not torch.cuda.is_available():
        raise P0Error("cuda_unavailable", "真实宿主 runner 需要 CUDA")
    if type(device_index) is not int or device_index < 0:
        raise P0Error("device_invalid", "CUDA device index 非法")
    if not torch.cuda.device_count() > device_index:
        raise P0Error("device_invalid", "请求的 CUDA device 不存在")
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _transformers()
    if (
        candidate.profile.allow_remote_code
        and candidate.expected_architecture_family == "deepseek"
    ):
        trust_remote_code = True
    else:
        trust_remote_code = candidate.profile.allow_remote_code
    quantization = _quantization_config(candidate, BitsAndBytesConfig)
    dtype = torch.bfloat16 if candidate.profile.dtype == "bf16" else torch.float16
    kwargs: Dict[str, Any] = {
        "revision": candidate.resolved_commit_sha,
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "device_map": {"": device_index},
        "low_cpu_mem_usage": True,
    }
    if quantization is not None:
        kwargs["quantization_config"] = quantization
    started = time.monotonic()
    if torch.cuda.is_available():
        _reset_cuda_peak_memory(torch, device_index)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_root),
            **{
                "revision": candidate.resolved_commit_sha,
                "local_files_only": True,
                "trust_remote_code": trust_remote_code,
            },
        )
        model = AutoModelForCausalLM.from_pretrained(str(snapshot_root), **kwargs)
    except Exception as exc:  # pragma: no cover - 服务器依赖路径
        raise P0Error("host_load_failed", "真实宿主加载失败") from exc
    elapsed = time.monotonic() - started
    devices, dtypes, cpu_offload, disk_offload = _device_facts(model)
    if cpu_offload or disk_offload:
        raise P0Error("offload_detected", "模型发生 CPU/disk offload")
    expected_device = f"cuda:{device_index}"
    if any(device != expected_device for device in devices):
        raise P0Error("device_placement_invalid", "模型未完整位于目标 GPU")
    architecture = _infer_architecture(candidate, model)
    records = _packed_expert_records(model)
    quantized = bool(records) and all(
        item["is_params4bit"] and item["quant_type"] == "nf4" for item in records
    )
    if candidate.profile.quantization_config and not quantized:
        raise P0Error("packed_expert_not_nf4", "packed expert 未证明为 NF4")
    return LoadedRealHost(
        candidate,
        tokenizer,
        model,
        architecture,
        elapsed,
        devices,
        dtypes,
        "BitsAndBytesConfig:nf4" if quantization is not None else "none",
        records,
        cpu_offload,
        disk_offload,
        int(torch.cuda.max_memory_allocated(device_index)),
        int(torch.cuda.max_memory_reserved(device_index)),
    )
