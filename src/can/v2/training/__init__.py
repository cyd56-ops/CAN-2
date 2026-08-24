"""CAN v2 Phase 2 训练组件。"""

from .data import (
    CIFAR2_MAPPING,
    DATA_MAPPING_VERSION,
    CIFAR10WithCoarse,
    CredentialBatch,
    CredentialGenerator,
    fine_to_coarse,
    get_cifar_transforms,
    make_worker_init_fn,
    split_indices,
)
from .loss import LossOutput, compute_training_loss
from .metrics import EvaluationMetricAccumulator, training_accuracy
from .trainer import GatedResNetTrainer, checkpoint_sha256

__all__ = [
    "CIFAR10WithCoarse",
    "CIFAR2_MAPPING",
    "DATA_MAPPING_VERSION",
    "CredentialBatch",
    "CredentialGenerator",
    "LossOutput",
    "compute_training_loss",
    "EvaluationMetricAccumulator",
    "fine_to_coarse",
    "GatedResNetTrainer",
    "checkpoint_sha256",
    "get_cifar_transforms",
    "make_worker_init_fn",
    "split_indices",
    "training_accuracy",
]
