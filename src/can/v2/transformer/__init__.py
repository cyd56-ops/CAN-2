"""Phase 5 小型 Transformer 能力分级原型的公开接口。"""

from .baselines import BaselineConfig, validate_baseline_compatibility
from .data import (
    EntityTripletBatchSampler,
    KnowledgeExample,
    SyntheticKnowledgeDataset,
    build_memorization_validation,
    collate_causal_lm_batch,
    generate_synthetic_corpus,
)
from .evaluator import (
    CapabilityMetrics,
    Phase5Evaluator,
    RefusalMetrics,
    TeacherComparison,
    evaluate_pretrain_validation,
)
from .experiments import (
    ProbeConfig,
    ProbeResult,
    RecoveryConfig,
    RecoveryResult,
    compute_recovery_rate,
    run_probe,
)
from .freeze import (
    freeze_record_sha256,
    load_freeze_record,
    validate_runtime_against_freeze,
)
from .manifest import (
    build_checkpoint_manifest,
    sha256_file,
    verify_checkpoint_integrity,
    verify_manifest_entry,
    write_manifest,
)
from .model import (
    GatedDecoderTransformer,
    GenerationOutput,
    TransformerConfig,
    TransformerInferenceOutput,
    TransformerTrainingOutput,
)
from .normalization import (
    NORMALIZATION_VERSION,
    RECOVERY_EPSILON,
    classify_refusal,
    normalize_answer,
)
from .reference import (
    DirectReferenceEquivalence,
    MixedRoutingValidation,
    ReferenceTrace,
    validate_direct_reference,
    validate_generation_reference,
)
from .tokenizer import ByteTokenizer
from .training import (
    Phase5Trainer,
    PretrainMetrics,
    causal_distillation_loss,
    configure_stage,
    count_non_padding_input_tokens,
    freeze_teacher,
    masked_causal_lm_loss,
    pretrain_go_no_go,
    validate_mixed_routing,
)

__all__ = [
    "ByteTokenizer",
    "CapabilityMetrics",
    "EntityTripletBatchSampler",
    "GatedDecoderTransformer",
    "GenerationOutput",
    "KnowledgeExample",
    "NORMALIZATION_VERSION",
    "Phase5Trainer",
    "Phase5Evaluator",
    "ProbeConfig",
    "ProbeResult",
    "RecoveryConfig",
    "RecoveryResult",
    "RefusalMetrics",
    "TeacherComparison",
    "evaluate_pretrain_validation",
    "PretrainMetrics",
    "SyntheticKnowledgeDataset",
    "TransformerConfig",
    "TransformerInferenceOutput",
    "TransformerTrainingOutput",
    "causal_distillation_loss",
    "count_non_padding_input_tokens",
    "build_memorization_validation",
    "collate_causal_lm_batch",
    "configure_stage",
    "freeze_teacher",
    "generate_synthetic_corpus",
    "masked_causal_lm_loss",
    "RECOVERY_EPSILON",
    "classify_refusal",
    "compute_recovery_rate",
    "run_probe",
    "freeze_record_sha256",
    "load_freeze_record",
    "validate_runtime_against_freeze",
    "normalize_answer",
    "BaselineConfig",
    "build_checkpoint_manifest",
    "validate_baseline_compatibility",
    "sha256_file",
    "verify_manifest_entry",
    "verify_checkpoint_integrity",
    "write_manifest",
    "MixedRoutingValidation",
    "ReferenceTrace",
    "validate_generation_reference",
    "validate_direct_reference",
    "DirectReferenceEquivalence",
    "pretrain_go_no_go",
    "validate_mixed_routing",
]
