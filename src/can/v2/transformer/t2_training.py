"""Phase 5.5/T2 四路 scope 的 Plain/CAN T-pretrain 训练器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..layers.gate_layer import ReasonCode
from ..training.data import CredentialGenerator
from .model import GatedDecoderTransformer
from .plain_model import PlainDecoderTransformer
from .t2_data import T2_SCOPES
from .training import count_non_padding_input_tokens, masked_causal_lm_loss


@dataclass(frozen=True)
class T2ScopeMasks:
    """保存 T2 双 head 监督与 credential 的互斥 Bool mask。"""

    protected: Tensor
    public: Tensor
    valid: Tensor
    invalid: Tensor


def build_t2_scope_masks(scopes: Sequence[str], device: torch.device) -> T2ScopeMasks:
    """验证四类 scope，并构造互斥且完整的监督与 credential mask。"""

    if isinstance(scopes, (str, bytes)) or not isinstance(scopes, Sequence):
        raise TypeError("scopes 必须是字符串序列")
    if not scopes or any(not isinstance(scope, str) for scope in scopes):
        raise ValueError("scopes 必须是非空字符串序列")
    if not isinstance(device, torch.device):
        raise TypeError("device 必须是 torch.device")
    if any(scope not in set(T2_SCOPES) for scope in scopes):
        raise ValueError("scopes 包含未知 T2 scope")
    protected = torch.tensor(
        [scope in {"protected_public", "protected_private"} for scope in scopes],
        dtype=torch.bool,
        device=device,
    )
    public = ~protected
    # T2 的 valid/invalid 路由与 protected/public head 选择逐行一致。
    valid = protected.clone()
    invalid = ~valid
    if not bool((protected ^ public).all().item()):
        raise RuntimeError("T2 protected/public mask 未形成互斥完整覆盖")
    if not torch.equal(valid, protected) or not torch.equal(invalid, public):
        raise RuntimeError("T2 credential mask 与 head mask 不一致")
    return T2ScopeMasks(protected, public, valid, invalid)


def _prepare_t2_batch(
    batch: Mapping[str, object], device: torch.device
) -> Tuple[Tensor, Tensor, Tensor, List[str], T2ScopeMasks]:
    """验证 collate 输出、四元组比例和 credential metadata。"""

    if not isinstance(batch, Mapping):
        raise TypeError("T2 batch 必须是 Mapping")
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    attention_mask = batch.get("attention_mask")
    scopes = batch.get("scopes")
    credential_classes = batch.get("credential_classes")
    if not all(
        isinstance(value, Tensor) for value in (input_ids, labels, attention_mask)
    ):
        raise TypeError("T2 batch 的 input_ids/labels/attention_mask 必须是 Tensor")
    assert isinstance(input_ids, Tensor)
    assert isinstance(labels, Tensor)
    assert isinstance(attention_mask, Tensor)
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("T2 input_ids/labels 必须是相同 shape 的二维 Tensor")
    if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
        raise ValueError("T2 attention_mask 必须是同 shape BoolTensor")
    if isinstance(scopes, (str, bytes)) or not isinstance(scopes, Sequence):
        raise TypeError("batch.scopes 必须是字符串序列")
    if isinstance(credential_classes, (str, bytes)) or not isinstance(
        credential_classes, Sequence
    ):
        raise TypeError("batch.credential_classes 必须是字符串序列")
    scope_values = list(scopes)
    credential_values = list(credential_classes)
    if len(scope_values) != input_ids.shape[0] or len(credential_values) != len(
        scope_values
    ):
        raise ValueError("T2 scope/credential metadata 与 batch 大小不一致")
    masks = build_t2_scope_masks(scope_values, device)
    expected_credentials = [
        "valid" if scope in {"protected_public", "protected_private"} else "invalid"
        for scope in scope_values
    ]
    if credential_values != expected_credentials:
        raise ValueError("credential_classes 与 T2 scope 不一致")
    counts = {scope: scope_values.count(scope) for scope in T2_SCOPES}
    if min(counts.values()) <= 0 or len(set(counts.values())) != 1:
        raise ValueError("T2 训练 batch 必须由等量完整四元组组成")
    return (
        input_ids.to(device),
        labels.to(device),
        attention_mask.to(device),
        scope_values,
        masks,
    )


def _validate_common(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    protected_weight: float,
    public_weight: float,
) -> None:
    """验证 T2 trainer 共用的优化器、设备和 loss 权重。"""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model 必须是 torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer 必须是 torch.optim.Optimizer")
    if not isinstance(device, torch.device):
        raise TypeError("device 必须是 torch.device")
    for value, name in (
        (protected_weight, "protected_weight"),
        (public_weight, "public_weight"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} 必须是有限正数")
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} 必须是有限正数")
    model_parameters = {id(parameter) for parameter in model.parameters()}
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if not model_parameters or not model_parameters.issubset(optimizer_parameters):
        raise ValueError("optimizer 必须覆盖当前模型的全部参数")


def _finish_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: Tensor,
    attention_mask: Tensor,
    global_step: int,
) -> Dict[str, float]:
    """检查 loss/梯度、执行 optimizer step 并返回 batch 指标。"""

    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("T2 T-pretrain 出现非有限 loss")
    loss.backward()
    if any(
        parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    ):
        raise FloatingPointError("T2 T-pretrain 出现非有限梯度")
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "samples": float(attention_mask.shape[0]),
        "tokens": float(count_non_padding_input_tokens(attention_mask)),
        "global_step": float(global_step + 1),
    }


class T2CanPretrainer:
    """训练 T2 CAN 双 head，并按 scope 生成真实 credential。"""

    def __init__(
        self,
        model: GatedDecoderTransformer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        credential_generator: CredentialGenerator,
        protected_weight: float = 1.0,
        public_weight: float = 1.0,
    ) -> None:
        """初始化 CAN T2 trainer 并校验模型、生成器和权重。"""

        if not isinstance(model, GatedDecoderTransformer):
            raise TypeError("model 必须是 GatedDecoderTransformer")
        if not isinstance(credential_generator, CredentialGenerator):
            raise TypeError("credential_generator 必须是 CredentialGenerator")
        _validate_common(model, optimizer, device, protected_weight, public_weight)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.credential_generator = credential_generator
        self.protected_weight = float(protected_weight)
        self.public_weight = float(public_weight)
        self.global_step = 0

    def train_batch(self, batch: Mapping[str, object]) -> Dict[str, float]:
        """训练一个完整 T2 四元组 batch，并核对 Gate 决策。"""

        self.model.train()
        input_ids, labels, attention_mask, scopes, masks = _prepare_t2_batch(
            batch, self.device
        )
        credentials = np.stack(
            [
                self.credential_generator.generate(
                    scope in {"protected_public", "protected_private"}
                )
                for scope in scopes
            ]
        )
        credential_tensor = torch.tensor(
            credentials, dtype=torch.float32, device=self.device
        )
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(input_ids, credential_tensor, attention_mask)
        if not torch.equal(output.decision.allow, masks.valid):
            raise RuntimeError("Gate 判决与 T2 credential_class 不一致")
        reason_code = output.decision.evidence.reason_code
        expected_reason = torch.where(
            masks.valid,
            torch.full_like(reason_code, int(ReasonCode.SUCCESS)),
            torch.full_like(reason_code, int(ReasonCode.LWE_VERIFICATION_FAILED)),
        )
        if not torch.equal(reason_code, expected_reason):
            raise RuntimeError("Gate reason_code 与 T2 credential_class 不一致")
        protected_loss = masked_causal_lm_loss(
            output.protected_logits, labels, masks.protected
        )
        public_loss = masked_causal_lm_loss(output.public_logits, labels, masks.public)
        loss = self.protected_weight * protected_loss + self.public_weight * public_loss
        metrics = _finish_step(
            self.model,
            self.optimizer,
            loss,
            attention_mask,
            self.global_step,
        )
        self.global_step += 1
        return metrics


class T2PlainPretrainer:
    """训练不含 Gate 的 T2 Plain 双 head oracle-head 对照。"""

    def __init__(
        self,
        model: PlainDecoderTransformer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        protected_weight: float = 1.0,
        public_weight: float = 1.0,
    ) -> None:
        """初始化 Plain T2 trainer 并校验模型和监督权重。"""

        if not isinstance(model, PlainDecoderTransformer):
            raise TypeError("model 必须是 PlainDecoderTransformer")
        _validate_common(model, optimizer, device, protected_weight, public_weight)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.protected_weight = float(protected_weight)
        self.public_weight = float(public_weight)
        self.global_step = 0

    def train_batch(self, batch: Mapping[str, object]) -> Dict[str, float]:
        """使用与 CAN 相同的四元组 mask 训练一个 Plain batch。"""

        self.model.train()
        input_ids, labels, attention_mask, _, masks = _prepare_t2_batch(
            batch, self.device
        )
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(input_ids, attention_mask)
        loss = self.protected_weight * masked_causal_lm_loss(
            output.protected_logits, labels, masks.protected
        ) + self.public_weight * masked_causal_lm_loss(
            output.public_logits, labels, masks.public
        )
        metrics = _finish_step(
            self.model,
            self.optimizer,
            loss,
            attention_mask,
            self.global_step,
        )
        self.global_step += 1
        return metrics


__all__ = [
    "T2CanPretrainer",
    "T2PlainPretrainer",
    "T2ScopeMasks",
    "build_t2_scope_masks",
]
