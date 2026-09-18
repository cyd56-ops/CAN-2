"""P0-MoE 的不可变类型、错误码和结构能力结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set, Tuple


class P0Error(ValueError):
    """表示 P0 输入、供应链或宿主能力违反冻结契约。"""

    def __init__(self, code: str, message: str) -> None:
        """使用稳定 reason code 构造错误。"""
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class CandidateProfile:
    """描述一个不可变的模型加载 profile。"""

    profile_id: str
    dtype: str
    quantization_config: Optional[Mapping[str, Any]]
    allow_remote_code: bool


@dataclass(frozen=True)
class CandidateSpec:
    """描述候选模型及其固定尝试顺序。"""

    candidate_id: str
    repository_id: str
    attempt_order: int
    requested_revision: str
    resolved_commit_sha: str
    profile: CandidateProfile
    expected_architecture_family: str
    expected_moe_variant: str
    max_snapshot_bytes: int
    license_review_status: str
    metadata_source_sha256: str


@dataclass(frozen=True)
class FixtureCase:
    """描述一条公开、非敏感的能力 fixture。"""

    case_id: str
    group: str
    system_text: str
    user_text: str
    expected_text: str
    metric: str
    max_new_tokens: int
    content_sha256: str
    fact_source: Optional[str] = None
    rationale: Optional[str] = None


@dataclass(frozen=True)
class HostCapabilities:
    """记录 fake 或真实适配器暴露的只读结构能力。"""

    architecture_family: str
    has_shared_expert: bool
    has_routed_experts: bool
    mask_before_dispatch: bool
    per_row_mask: bool
    expert_call_counter: bool
    top_k: int
    preserves_batch_indices: bool
    kv_identity_binding: bool
    state_dict_stable: bool
    native_shared_routed: bool = True
    tied_weights: bool = False


@dataclass(frozen=True)
class HostInspection:
    """保存只读结构探查结果和七道 P0-C 门状态。"""

    capabilities: HostCapabilities
    hook_points: Tuple[str, ...]
    gate_results: Mapping[str, bool]
    rejected_reason: Optional[str]

    @property
    def passed(self) -> bool:
        """返回所有结构门是否通过。"""
        return self.rejected_reason is None and all(self.gate_results.values())


def require_exact_keys(
    payload: object, expected: Set[str], context: str
) -> Dict[str, Any]:
    """校验映射且拒绝未知或缺失字段。"""
    if not isinstance(payload, dict) or set(payload) != expected:
        raise P0Error("artifact_schema_mismatch", f"{context} 字段集合不匹配")
    return payload
