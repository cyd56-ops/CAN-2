"""P0-A infrastructure、C1 load smoke 和 P0-D 资源采样。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .real_loader import LoadedRealHost
from .real_types import PreflightResult, ResourceSample
from .types import P0Error


def infrastructure_preflight(
    snapshot_root: Optional[Path] = None, *, require_snapshot: bool = True
) -> PreflightResult:
    """检查隔离、冻结依赖和 CUDA；下载前允许 snapshot 尚未创建。"""

    checks = {
        "network_isolation": os.environ.get("CAN_NETWORK_ISOLATION_VERIFIED") == "1",
        "offline_environment": all(
            os.environ.get(name) == "1"
            for name in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
            )
        ),
        "readonly_mount": os.environ.get("CAN_READONLY_SNAPSHOT_VERIFIED") == "1",
        "python": True,
    }
    details: Dict[str, Any] = {"network_isolation_source": "environment_marker"}
    if snapshot_root is not None:
        details["snapshot_root"] = str(snapshot_root.resolve())
        if require_snapshot:
            checks["snapshot_root_exists"] = snapshot_root.exists()
    try:
        import torch

        checks["cuda_visible"] = bool(torch.cuda.is_available())
        details["cuda_device_count"] = int(torch.cuda.device_count())
    except ImportError:
        checks["cuda_visible"] = False
        details["cuda_device_count"] = 0
    failures = tuple(name for name, passed in checks.items() if not passed)
    return PreflightResult(
        "infrastructure",
        True,
        "passed" if not failures else "failed",
        checks,
        failures,
        details,
    )


def candidate_load_smoke(
    loaded: LoadedRealHost, architecture: Optional[Any] = None
) -> PreflightResult:
    """按冻结 profile 检查量化、设备和逐 expert 可观测性。"""

    records = loaded.packed_expert_records
    architecture = loaded.architecture if architecture is None else architecture
    profile = getattr(loaded, "candidate", None)
    profile = getattr(profile, "profile", None)
    quantization_config = getattr(profile, "quantization_config", None)
    # 没有 candidate 的旧 CPU stand-in 默认按 NF4 处理，保持旧测试和旧摘要语义。
    expects_nf4 = quantization_config is None or bool(quantization_config)
    packed_experts_nf4 = bool(records) and all(
        item.get("is_params4bit") and item.get("quant_type") == "nf4"
        for item in records
    )
    packed_experts_bf16 = bool(records) and all(
        item.get("dtype") in {"torch.bfloat16", "bfloat16"}
        and not item.get("is_params4bit")
        and item.get("quant_type") is None
        and str(item.get("device", "")).startswith("cuda:")
        for item in records
    )
    checks = {
        "packed_experts_present": bool(records),
        "packed_experts_nf4": packed_experts_nf4,
        "packed_experts_bf16": packed_experts_bf16,
        "single_device": len(loaded.parameter_devices) == 1
        and loaded.parameter_devices[0].startswith("cuda:"),
        "no_cpu_offload": not loaded.cpu_offload_detected,
        "no_disk_offload": not loaded.disk_offload_detected,
        "expert_execution_observable": architecture.supports_execution_counter,
        "native_top_k_resolved": architecture.native_top_k > 0,
    }
    failure_codes = []
    if expects_nf4 and not packed_experts_nf4:
        failure_codes.append("packed_expert_not_nf4")
    if not expects_nf4 and not packed_experts_bf16:
        failure_codes.append("packed_expert_not_bf16")
    if (
        not checks["single_device"]
        or not checks["no_cpu_offload"]
        or not checks["no_disk_offload"]
    ):
        failure_codes.append("offload_detected")
    if not checks["expert_execution_observable"]:
        failure_codes.append("expert_call_unobservable")
    if not checks["native_top_k_resolved"]:
        failure_codes.append("native_top_k_unresolved")
    return PreflightResult(
        "candidate",
        True,
        "passed" if not failure_codes else "failed",
        checks,
        tuple(dict.fromkeys(failure_codes)),
        {
            "load_seconds": loaded.load_seconds,
            "parameter_devices": loaded.parameter_devices,
            "parameter_dtypes": loaded.parameter_dtypes,
            "quantization_class": loaded.quantization_class,
            "packed_expert_records": loaded.packed_expert_records,
            "cuda_peak_allocated": loaded.cuda_peak_allocated,
            "cuda_peak_reserved": loaded.cuda_peak_reserved,
        },
    )


def sample_resource(stage: str, started: float, model: Any = None) -> ResourceSample:
    """采样 CPU/GPU 资源，不修改模型状态。"""

    rss: Optional[int] = None
    try:
        import psutil

        rss = int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    allocated = reserved = free = total = None
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = int(torch.cuda.memory_allocated(device))
            reserved = int(torch.cuda.memory_reserved(device))
            free, total = (int(value) for value in torch.cuda.mem_get_info(device))
    except (ImportError, RuntimeError):
        pass
    device_map = getattr(model, "hf_device_map", {})
    placements = (
        tuple(str(value).lower() for value in device_map.values())
        if isinstance(device_map, dict)
        else ()
    )
    return ResourceSample(
        stage,
        time.monotonic() - started,
        rss,
        allocated,
        reserved,
        free,
        total,
        any(value == "cpu" for value in placements),
        any(value == "disk" for value in placements),
    )


def validate_resource_gates(
    load_seconds: float,
    total_seconds: float,
    samples: tuple[ResourceSample, ...],
    *,
    max_reserved_bytes: int = int(14.5 * 1024**3),
    min_free_bytes: int = 1024**3,
) -> None:
    """执行 profile 预登记的时长、显存和 offload 硬门。"""
    if load_seconds > 20 * 60:
        raise P0Error("load_timeout", "模型加载超过 20 分钟")
    if total_seconds > 90 * 60:
        raise P0Error("candidate_timeout", "候选总运行超过 90 分钟")
    if not samples:
        raise P0Error("resource_evidence_missing", "缺少资源采样")
    if any(item.cpu_offload_detected or item.disk_offload_detected for item in samples):
        raise P0Error("offload_detected", "资源采样检测到 CPU/disk offload")
    reserved = [
        item.cuda_reserved_bytes
        for item in samples
        if item.cuda_reserved_bytes is not None
    ]
    free = [item.gpu_free_bytes for item in samples if item.gpu_free_bytes is not None]
    if not reserved or not free:
        raise P0Error("resource_evidence_missing", "缺少 CUDA 显存采样")
    if max(reserved) > max_reserved_bytes:
        raise P0Error("gpu_memory_limit_exceeded", "reserved 显存超过 profile 门槛")
    if min(free) < min_free_bytes:
        raise P0Error("gpu_memory_headroom_insufficient", "设备余量低于 profile 门槛")
