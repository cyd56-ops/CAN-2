"""按样本累计的训练和评估指标。"""

from typing import Dict, Optional

import torch
from torch import Tensor


def _accuracy(logits: Tensor, labels: Tensor) -> Optional[float]:
    """返回空子批为 None 的样本准确率。"""

    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("logits 和 labels 必须是 Tensor")
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits/labels shape 不符合指标契约")
    if labels.numel() == 0:
        return None
    return float((logits.argmax(1) == labels).float().mean().item())


class EvaluationMetricAccumulator:
    """累计 protected/public confusion matrix 所需的计数。"""

    def __init__(self) -> None:
        self.protected_correct = self.protected_total = 0
        self.public_correct = self.public_total = 0
        self.protected_confusion = torch.zeros((10, 10), dtype=torch.long)
        self.public_confusion = torch.zeros((2, 2), dtype=torch.long)

    def update_protected(self, logits: Tensor, labels: Tensor) -> None:
        """更新 protected 样本计数。"""

        self._validate_batch(logits, labels, 10, "protected")
        if labels.numel() == 0:
            return
        self.protected_correct += int((logits.argmax(1) == labels).sum().item())
        self.protected_total += labels.numel()
        for target, prediction in zip(
            labels.detach().cpu(), logits.argmax(1).detach().cpu()
        ):
            self.protected_confusion[int(target), int(prediction)] += 1

    def update_public(self, logits: Tensor, labels: Tensor) -> None:
        """更新 public 样本计数。"""

        self._validate_batch(logits, labels, 2, "public")
        if labels.numel() == 0:
            return
        self.public_correct += int((logits.argmax(1) == labels).sum().item())
        self.public_total += labels.numel()
        for target, prediction in zip(
            labels.detach().cpu(), logits.argmax(1).detach().cpu()
        ):
            self.public_confusion[int(target), int(prediction)] += 1

    def compute(self) -> Dict[str, Optional[float]]:
        """返回按样本计数的准确率。"""

        public_recall = self.public_confusion.diag() / self.public_confusion.sum(
            1
        ).clamp_min(1)
        return {
            "protected_accuracy": (
                self.protected_correct / self.protected_total
                if self.protected_total
                else None
            ),
            "public_accuracy": (
                self.public_correct / self.public_total if self.public_total else None
            ),
            "public_balanced_accuracy": (
                float(public_recall.mean().item()) if self.public_total else None
            ),
            "public_macro_f1": self._macro_f1(),
            "protected_total": float(self.protected_total),
            "public_total": float(self.public_total),
        }

    def _macro_f1(self) -> Optional[float]:
        """根据 public confusion matrix 计算 macro-F1。"""

        if not self.public_total:
            return None
        matrix = self.public_confusion.float()
        precision = matrix.diag() / matrix.sum(0).clamp_min(1)
        recall = matrix.diag() / matrix.sum(1).clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        return float(f1.mean().item())

    @staticmethod
    def _validate_batch(
        logits: Tensor, labels: Tensor, classes: int, name: str
    ) -> None:
        """校验一批 logits 与标签，空批允许但不计入分母。"""

        if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
            raise TypeError(f"{name} logits/labels 必须是 Tensor")
        if logits.ndim != 2 or logits.shape[1] != classes:
            raise ValueError(f"{name} logits shape 必须是 [N, {classes}]")
        if labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError(f"{name} logits/labels batch 不一致")
        if logits.device != labels.device:
            raise ValueError(f"{name} logits/labels device 不一致")
        if not logits.dtype.is_floating_point:
            raise TypeError(f"{name} logits 必须使用浮点 dtype")
        if labels.dtype != torch.long:
            raise TypeError(f"{name} labels 必须是 torch.long")
        if not torch.isfinite(logits).all():
            raise ValueError(f"{name} logits 必须全部有限")
        if labels.numel() and bool(
            torch.any((labels < 0) | (labels >= classes)).item()
        ):
            raise ValueError(f"{name} labels 超出类别范围")


def training_accuracy(
    output, fine_labels: Tensor, coarse_labels: Tensor
) -> Dict[str, Optional[float]]:
    """按训练态 allow mask 计算两个 head 的准确率。"""

    allow = output.decision.allow
    return {
        "protected_accuracy": _accuracy(
            output.protected_logits[allow], fine_labels[allow]
        ),
        "public_accuracy": _accuracy(output.public_logits, coarse_labels),
    }
