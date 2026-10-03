"""P0-MoE 真实预训练宿主的离线预检契约。

本包只负责严格解析、结构探查和 fake-host 预检，不加载或修改真实模型。
"""

from .artifacts import ArtifactWriter
from .deepseek_adapter import DeepSeekHostAdapter
from .fixture import load_fixture, validate_fixture
from .grouped_mm_observer import GroupedMMInvocation, GroupedMMObserver
from .host_adapter import FakeHostAdapter, HostAdapter, inspect_host
from .p0a import finalize_p0a_registry, prepare_p0a_reviews
from .qwen_adapter import QwenHostAdapter
from .real_adapter import RealHostAdapter, TransformersHostAdapter
from .real_runner import RealP0Runner
from .real_types import (
    ArchitectureMap,
    GenerationRecord,
    GenerationRequest,
    PreflightResult,
    ProbeRequest,
    ProbeResult,
    RealRunResult,
    ResourceSample,
    SnapshotManifest,
)
from .registry import load_registry, validate_registry
from .runner import P0Runner
from .types import P0Error

__all__ = [
    "ArtifactWriter",
    "FakeHostAdapter",
    "HostAdapter",
    "P0Error",
    "P0Runner",
    "inspect_host",
    "finalize_p0a_registry",
    "load_fixture",
    "load_registry",
    "prepare_p0a_reviews",
    "validate_fixture",
    "validate_registry",
    "ArchitectureMap",
    "DeepSeekHostAdapter",
    "GenerationRecord",
    "GenerationRequest",
    "GroupedMMInvocation",
    "GroupedMMObserver",
    "PreflightResult",
    "ProbeRequest",
    "ProbeResult",
    "QwenHostAdapter",
    "RealHostAdapter",
    "RealP0Runner",
    "RealRunResult",
    "ResourceSample",
    "SnapshotManifest",
    "TransformersHostAdapter",
]
