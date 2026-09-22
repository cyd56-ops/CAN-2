"""真实预训练 MoE P0 runner 的严格数据类型。

本模块只保存可序列化的请求、探查和资源结果，不在导入时依赖
Transformers、bitsandbytes 或 CUDA。真实宿主的可选依赖由 loader 延迟导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SnapshotFile:
    """记录 snapshot 中一个普通文件的相对路径、大小和摘要。"""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SnapshotManifest:
    """绑定候选、revision 和本地 snapshot inventory。"""

    candidate_id: str
    repository_id: str
    resolved_commit_sha: str
    root: str
    files: Tuple[SnapshotFile, ...]
    total_bytes: int
    manifest_sha256: str
    read_only_verified: bool = False
    offline_verified: bool = False


@dataclass(frozen=True)
class GenerationRequest:
    """描述一条固定解码请求。"""

    case_id: str
    system_text: str
    user_text: str
    expected_text: str
    metric: str
    max_new_tokens: int
    use_cache: bool


@dataclass(frozen=True)
class GenerationRecord:
    """保存一条生成结果及其确定性诊断信息。"""

    case_id: str
    generated_text: str
    canonical_text: str
    expected_text: str
    matched: bool
    prompt_tokens: Tuple[int, ...]
    continuation_tokens: Tuple[int, ...]
    stop_reason: str
    use_cache: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class ProbeRequest:
    """描述结构探查输入和逐行 allowed mask。"""

    hidden_shape: Tuple[int, int, int]
    allowed_mask: Tuple[Tuple[Tuple[bool, ...], ...], ...]
    native_top_k: int
    padding_mask: Tuple[Tuple[bool, ...], ...]
    request_ids: Tuple[str, ...]
    cache_position: int = 0
    processed_length: int = 0


@dataclass(frozen=True)
class ExpertCall:
    """记录一次真实 routed/shared expert 调用。"""

    expert_id: str
    batch_index: int
    token_index: int
    global_row: int
    branch: str


@dataclass(frozen=True)
class ProbeResult:
    """保存一次受控 forward 的结构和索引结果。"""

    output_shape: Tuple[int, ...]
    selected_ids: Tuple[Tuple[int, ...], ...]
    expert_calls: Tuple[ExpertCall, ...]
    original_indices: Tuple[int, ...]
    reassembled_indices: Tuple[int, ...]
    kv_bound: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class ArchitectureMap:
    """记录实际模型对象中识别出的 MoE 边界。"""

    architecture_family: str
    model_class: str
    moe_layers: Tuple[str, ...]
    router_paths: Tuple[str, ...]
    shared_paths: Tuple[str, ...]
    routed_paths: Tuple[str, ...]
    native_top_k: int
    backend_id: str
    supports_row_mask: bool
    supports_execution_counter: bool
    routed_expert_count: int = 0


@dataclass(frozen=True)
class ResourceSample:
    """记录一次资源采样。"""

    stage: str
    elapsed_seconds: float
    host_rss_bytes: Optional[int]
    cuda_allocated_bytes: Optional[int]
    cuda_reserved_bytes: Optional[int]
    gpu_free_bytes: Optional[int]
    gpu_total_bytes: Optional[int]
    cpu_offload_detected: bool
    disk_offload_detected: bool


@dataclass(frozen=True)
class PreflightResult:
    """保存非正式 preflight 的可审计结果。"""

    mode: str
    non_formal: bool
    status: str
    checks: Mapping[str, bool]
    failure_codes: Tuple[str, ...] = field(default_factory=tuple)
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RealRunResult:
    """保存一个正式 candidate run 的四门摘要。"""

    candidate_id: str
    profile_id: str
    status: str
    p0a_runtime: str
    p0b: str
    p0c: str
    p0d: str
    failure_codes: Tuple[str, ...]
    generations: Tuple[GenerationRecord, ...] = field(default_factory=tuple)
    architecture: Optional[ArchitectureMap] = None
    resource_samples: Tuple[ResourceSample, ...] = field(default_factory=tuple)
    router_ledger: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    expert_calls: Tuple[ExpertCall, ...] = field(default_factory=tuple)
    p0b_metrics: Mapping[str, Any] = field(default_factory=dict)
    p0c_gates: Mapping[str, bool] = field(default_factory=dict)
    p0c_metrics: Mapping[str, Any] = field(default_factory=dict)
    determinism: Mapping[str, Any] = field(default_factory=dict)
    resource_metrics: Mapping[str, Any] = field(default_factory=dict)
    stage_timings: Mapping[str, float] = field(default_factory=dict)
    exit_code: int = 2


def tensor_index_contract(batch: int, sequence: int) -> Tuple[int, ...]:
    """返回 `[B,S,H]` 展平时的严格 batch/token 全局索引。"""

    if (
        type(batch) is not int
        or type(sequence) is not int
        or batch <= 0
        or sequence <= 0
    ):
        raise ValueError("batch 和 sequence 必须为正整数")
    return tuple(
        batch_index * sequence + token_index
        for batch_index in range(batch)
        for token_index in range(sequence)
    )


def validate_reassembly_indices(
    original_indices: Sequence[int],
    reassembled_indices: Sequence[int],
    valid_count: int,
) -> None:
    """校验 mixed batch 拆分/重组的一一映射和顺序。"""

    if type(valid_count) is not int or valid_count < 0:
        raise ValueError("valid_count 非法")
    original = tuple(original_indices)
    reassembled = tuple(reassembled_indices)
    if len(original) != valid_count or len(reassembled) != valid_count:
        raise ValueError("有效行数量不匹配")
    if any(type(index) is not int or index < 0 for index in original + reassembled):
        raise ValueError("索引必须为非负整数")
    if len(set(original)) != len(original) or len(set(reassembled)) != len(reassembled):
        raise ValueError("索引不能重复")
    if original != tuple(sorted(original)) or reassembled != original:
        raise ValueError("索引必须单调且重组顺序保持")
