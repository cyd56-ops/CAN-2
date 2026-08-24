"""Phase 2 的 masked protected loss、public CE 和知识蒸馏损失。"""

from dataclasses import dataclass
from numbers import Real
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..layers.gate_layer import AuthorizationDecision
from ..models.gated_resnet import TrainingOutput


@dataclass(frozen=True)
class LossOutput:
    """保存四项可审计损失。"""

    total: Tensor
    protected: Tensor
    public_ce: Tensor
    public_kd: Tensor


def _weight(value: float, name: str) -> float:
    """校验有限非负损失权重。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} 必须是有限实数")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"{name} 必须是有限实数")
    if result < 0:
        raise ValueError(f"{name} 不能为负")
    return result


def _temperature(value: float) -> float:
    """校验知识蒸馏温度为有限正实数。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("temperature 必须是有限正数")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result <= 0.0:
        raise ValueError("temperature 必须是有限正数")
    return result


def compute_training_loss(
    output: TrainingOutput,
    fine_labels: Tensor,
    coarse_labels: Tensor,
    teacher_fine_logits: Optional[Tensor],
    alpha: float = 1.0,
    beta_ce: float = 0.1,
    beta_kd: float = 0.9,
    temperature: float = 4.0,
) -> LossOutput:
    """计算 masked protected CE、public CE 与 10→2 类 KD。"""

    if not isinstance(output, TrainingOutput):
        raise TypeError("output 必须是 TrainingOutput")
    if not isinstance(fine_labels, Tensor) or not isinstance(coarse_labels, Tensor):
        raise TypeError("fine_labels 和 coarse_labels 必须是 Tensor")
    if not isinstance(output.protected_logits, Tensor) or not isinstance(
        output.public_logits, Tensor
    ):
        raise TypeError("protected_logits 和 public_logits 必须是 Tensor")
    b = output.public_logits.shape[0]
    if output.protected_logits.ndim != 2 or output.public_logits.ndim != 2:
        raise ValueError("logits 必须是二维 Tensor")
    if (
        not output.protected_logits.dtype.is_floating_point
        or not output.public_logits.dtype.is_floating_point
    ):
        raise TypeError("logits 必须使用浮点 dtype")
    if output.protected_logits.device != output.public_logits.device:
        raise ValueError("protected/public logits device 不一致")
    if output.protected_logits.shape[0] != b:
        raise ValueError("protected/public logits batch 不一致")
    if output.protected_logits.shape[1] != 10 or output.public_logits.shape[1] != 2:
        raise ValueError("Phase 2 logits 必须是 protected=10 类、public=2 类")
    if fine_labels.ndim != 1 or coarse_labels.ndim != 1:
        raise ValueError("labels 必须是一维 Tensor")
    if fine_labels.shape != (b,) or coarse_labels.shape != (b,):
        raise ValueError("labels batch 与 logits 不一致")
    if fine_labels.dtype != torch.long or coarse_labels.dtype != torch.long:
        raise TypeError("labels 必须是 torch.long")
    if fine_labels.device != output.public_logits.device:
        raise ValueError("labels 与 logits device 不一致")
    if not isinstance(output.decision, AuthorizationDecision):
        raise TypeError("output.decision 必须是 AuthorizationDecision")
    allow = output.decision.allow
    if (
        allow.shape != (b,)
        or allow.dtype != torch.bool
        or allow.device != fine_labels.device
    ):
        raise ValueError("decision.allow 契约非法")
    if (
        not torch.isfinite(output.protected_logits).all()
        or not torch.isfinite(output.public_logits).all()
    ):
        raise ValueError("logits 必须全部有限")
    if torch.any((fine_labels < 0) | (fine_labels >= output.protected_logits.shape[1])):
        raise ValueError("fine label 超出类别范围")
    if torch.any(
        (coarse_labels < 0) | (coarse_labels >= output.public_logits.shape[1])
    ):
        raise ValueError("coarse label 超出类别范围")
    alpha = _weight(alpha, "alpha")
    beta_ce = _weight(beta_ce, "beta_ce")
    beta_kd = _weight(beta_kd, "beta_kd")
    temperature = _temperature(temperature)
    if teacher_fine_logits is not None:
        if not isinstance(teacher_fine_logits, Tensor):
            raise TypeError("teacher logits 必须是 Tensor 或 None")
        if (
            teacher_fine_logits.shape != (b, 10)
            or teacher_fine_logits.device != fine_labels.device
        ):
            raise ValueError("teacher logits shape/device 非法")
        if teacher_fine_logits.dtype != output.public_logits.dtype:
            raise TypeError("teacher logits dtype 必须与 student public logits 一致")
        if not torch.isfinite(teacher_fine_logits).all():
            raise ValueError("teacher logits 必须全部有限")
    if beta_kd > 0.0 and teacher_fine_logits is None:
        raise ValueError("beta_kd 大于 0 时必须提供冻结 teacher logits")
    if bool(allow.any().item()):
        protected = F.cross_entropy(output.protected_logits[allow], fine_labels[allow])
    else:
        protected = output.protected_logits.sum() * 0.0
    public_ce = F.cross_entropy(output.public_logits, coarse_labels)
    if teacher_fine_logits is None or beta_kd == 0.0:
        public_kd = output.public_logits.sum() * 0.0
    else:
        vehicle = torch.logsumexp(teacher_fine_logits[:, [0, 1, 8, 9]], dim=1)
        animal = torch.logsumexp(teacher_fine_logits[:, [2, 3, 4, 5, 6, 7]], dim=1)
        teacher_coarse = torch.stack([vehicle, animal], dim=1)
        public_kd = (
            F.kl_div(
                F.log_softmax(output.public_logits / temperature, dim=1),
                F.softmax(teacher_coarse.detach() / temperature, dim=1),
                reduction="batchmean",
            )
            * temperature**2
        )
    return LossOutput(
        alpha * protected + beta_ce * public_ce + beta_kd * public_kd,
        protected,
        public_ce,
        public_kd,
    )
