"""I1 G1-b 到 M2 的集成接口。"""

from .adapter import G1BVerifierAdapter
from .artifacts import (
    BOUNDARY_ROLES,
    validate_i1_call_ledger,
    validate_i1_fixture,
    validate_i1_summary,
)
from .manifest import build_i1_manifest, load_i1_manifest, validate_i1_manifest
from .runner import run_i1_artifacts
from .types import (
    FP32_EXACT_INTEGER_LIMIT,
    I1_EXECUTION_CONFIG_ID,
    I1_PROTOCOL_ID,
    I1Error,
)

__all__ = [
    "FP32_EXACT_INTEGER_LIMIT",
    "BOUNDARY_ROLES",
    "G1BVerifierAdapter",
    "I1_EXECUTION_CONFIG_ID",
    "I1_PROTOCOL_ID",
    "I1Error",
    "build_i1_manifest",
    "load_i1_manifest",
    "run_i1_artifacts",
    "validate_i1_call_ledger",
    "validate_i1_fixture",
    "validate_i1_manifest",
    "validate_i1_summary",
]
