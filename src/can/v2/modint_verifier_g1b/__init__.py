"""G1-b 固定模整数 neural verifier 的 CPU backend。"""

from .kernel import (
    canonical_mod_q,
    centered_lift_tensor,
    derive_integer_bounds,
    verify_tensor_batch,
)
from .manifest import build_g1b_manifest, load_g1b_manifest, validate_g1b_manifest
from .artifacts import generate_g1b_artifacts
from .types import G1BError, G1BResult, IntegerBounds
from .verifier import ModIntNeuralVerifier

__all__ = [
    "G1BError",
    "G1BResult",
    "IntegerBounds",
    "ModIntNeuralVerifier",
    "canonical_mod_q",
    "centered_lift_tensor",
    "derive_integer_bounds",
    "verify_tensor_batch",
    "build_g1b_manifest",
    "load_g1b_manifest",
    "validate_g1b_manifest",
    "generate_g1b_artifacts",
]
