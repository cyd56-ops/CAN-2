"""Phase 5 T1 的 probe/recovery 配置与结果框架。"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .normalization import RECOVERY_EPSILON


@dataclass(frozen=True)
class ProbeConfig:
    """冻结 TM-REP probe 的表示层、样本和 seed 预算。"""

    target_layer: int
    probe_type: str = "logistic"
    samples_per_entity: int = 1
    train_entities: int = 8
    test_entities: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        """验证 probe 配置为有限正预算。"""
        if self.target_layer < 0:
            raise ValueError("target_layer 必须为非负整数")
        if self.probe_type not in {"logistic", "linear"}:
            raise ValueError("probe_type 不受支持")
        if (
            self.samples_per_entity <= 0
            or self.train_entities <= 0
            or self.test_entities <= 0
        ):
            raise ValueError("probe 样本预算必须大于 0")


@dataclass(frozen=True)
class ProbeResult:
    """记录 probe AUC 及方向无关可分性。"""

    auc: float
    separability: float
    random_baseline_auc: float
    majority_baseline_accuracy: float
    target_layer: int
    train_entities: int
    test_entities: int


@dataclass(frozen=True)
class RecoveryConfig:
    """冻结 TM-CP 恢复实验的 checkpoint、数据和预算。"""

    source_checkpoint_sha256: str
    method: str
    budget_tokens: int
    budget_optimizer_steps: int
    offline_data_hash: str
    seed: int
    epsilon: float = RECOVERY_EPSILON

    def __post_init__(self) -> None:
        """验证恢复预算和 epsilon 与设计契约一致。"""
        if self.budget_tokens <= 0 or self.budget_optimizer_steps <= 0:
            raise ValueError("恢复预算必须大于 0")
        if self.epsilon != RECOVERY_EPSILON:
            raise ValueError("epsilon 必须固定为 RECOVERY_EPSILON")


@dataclass(frozen=True)
class RecoveryResult:
    """记录恢复模型与 baseline 的 exact match 差异。"""

    recovered_exact_match: float
    baseline_exact_match: float
    recovery_rate: float
    budget_tokens_used: int
    budget_optimizer_steps_used: int
    epsilon: float = RECOVERY_EPSILON


def compute_recovery_rate(recovered: float, baseline: float) -> float:
    """计算允许为负且不 clamp 的归一化恢复率。"""

    if not 0.0 <= recovered <= 1.0 or not 0.0 <= baseline <= 1.0:
        raise ValueError("exact match 必须位于 [0, 1]")
    return (recovered - baseline) / max(1.0 - baseline, RECOVERY_EPSILON)


def run_probe(
    train_features: np.ndarray,
    train_labels: Sequence[int],
    test_features: np.ndarray,
    test_labels: Sequence[int],
    config: ProbeConfig,
) -> ProbeResult:
    """执行无梯度的轻量线性 probe，并返回方向无关可分性。"""

    if not isinstance(config, ProbeConfig):
        raise TypeError("config 必须是 ProbeConfig")
    x_train = np.asarray(train_features, dtype=np.float64)
    x_test = np.asarray(test_features, dtype=np.float64)
    y_train = np.asarray(train_labels, dtype=np.int64)
    y_test = np.asarray(test_labels, dtype=np.int64)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("features 必须为同维二维数组")
    if len(y_train) != len(x_train) or len(y_test) != len(x_test):
        raise ValueError("labels 与 features 长度不一致")
    classes = np.unique(y_train)
    if classes.size < 2:
        raise ValueError("probe 至少需要两个类别")
    centroids = np.stack([x_train[y_train == c].mean(axis=0) for c in classes])
    distances = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    predictions = classes[np.argmin(distances, axis=1)]
    accuracy = float(np.mean(predictions == y_test))
    majority = float(np.max(np.bincount(y_test - y_test.min())) / len(y_test))
    # 使用连续的 margin 分数计算二分类 ROC-AUC，不能把准确率冒充 AUC。
    if classes.size != 2:
        raise ValueError("当前 probe 仅支持二分类 AUC")
    positive = classes[1]
    scores = distances[:, 0] - distances[:, 1]
    labels_binary = (y_test == positive).astype(np.int64)
    positives = scores[labels_binary == 1]
    negatives = scores[labels_binary == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("test labels 必须同时包含两个类别")
    auc = float(
        np.mean(positives[:, None] > negatives[None, :])
        + 0.5 * np.mean(positives[:, None] == negatives[None, :])
    )
    # 随机标签 baseline 的解析期望固定为 0.5，避免小样本单次抽样噪声。
    random_auc = 0.5
    return ProbeResult(
        auc,
        max(auc, 1.0 - auc),
        random_auc,
        majority,
        config.target_layer,
        config.train_entities,
        config.test_entities,
    )


__all__ = [
    "ProbeConfig",
    "ProbeResult",
    "RecoveryConfig",
    "RecoveryResult",
    "compute_recovery_rate",
    "run_probe",
]
