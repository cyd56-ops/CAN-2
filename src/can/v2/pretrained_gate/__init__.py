"""固定预训练宿主 Gate 的 G0 CPU 契约。"""

from .adapter import BatchExecutionError, GatedHostAdapter, ProtectedDispatcher
from .authorization import (
    P1_POLICY_ID,
    TOY_REAL_FP32_PROFILE,
    FixedRelationVerifier,
    RouteCoordinator,
)
from .cache import CacheHandle, CacheRegistry, CacheStateError
from .host import HostPlugin, HostSpec, PrefixState, TinyDecoderHost, TinyKVDecoderHost
from .instrumentation import CallLedger, InstrumentedModuleCall
from .manifest import file_sha256, load_manifest, verify_manifest_sha256
from .types import (
    CallRecord,
    DispatchResult,
    EvidenceReason,
    FailureRecord,
    RouteKind,
    TrustedRequestContext,
    VerificationEvidence,
)

__all__ = [
    "BatchExecutionError",
    "CallLedger",
    "CallRecord",
    "CacheHandle",
    "CacheRegistry",
    "CacheStateError",
    "DispatchResult",
    "EvidenceReason",
    "FailureRecord",
    "FixedRelationVerifier",
    "GatedHostAdapter",
    "HostPlugin",
    "HostSpec",
    "InstrumentedModuleCall",
    "P1_POLICY_ID",
    "PrefixState",
    "TinyDecoderHost",
    "TinyKVDecoderHost",
    "ProtectedDispatcher",
    "RouteCoordinator",
    "RouteKind",
    "TOY_REAL_FP32_PROFILE",
    "TrustedRequestContext",
    "VerificationEvidence",
    "file_sha256",
    "load_manifest",
    "verify_manifest_sha256",
]
