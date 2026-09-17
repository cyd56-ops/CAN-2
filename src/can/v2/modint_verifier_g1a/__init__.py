"""G1-a 模整数 verifier 的 CPU reference 实现。"""

from .manifest import build_manifest, load_g1a_manifest, validate_g1a_manifest
from .artifacts import generate_g1a_artifacts
from .parameters import (
    derive_seed,
    generate_parameters,
    parameter_digest,
    public_parameter_bytes,
)
from .reference import canonical_mod, centered_lift, verify_batch
from .types import (
    G1AConfig,
    G1AError,
    G1AParameters,
    G1AReferenceResult,
    G1aEvidence,
)

__all__ = [
    "G1AConfig",
    "G1AError",
    "G1AParameters",
    "G1AReferenceResult",
    "G1aEvidence",
    "build_manifest",
    "generate_g1a_artifacts",
    "canonical_mod",
    "centered_lift",
    "derive_seed",
    "generate_parameters",
    "load_g1a_manifest",
    "parameter_digest",
    "public_parameter_bytes",
    "validate_g1a_manifest",
    "verify_batch",
]
