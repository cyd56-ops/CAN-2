"""Phase 5.5/T2 四路 scope 的 Plain/CAN T-pretrain 训练器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

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


def t2_parameter_groups(model: nn.Module) -> Dict[str, Tuple[nn.Parameter, ...]]:
    """把 T2 模型的可训练参数划分为互斥且完整的三条路径。

    参数:
        model: Plain 或 CAN Decoder Transformer。

    返回:
        ``shared_prefix``、``protected_path`` 和 ``public_path`` 参数元组。
    """

    if not isinstance(model, (GatedDecoderTransformer, PlainDecoderTransformer)):
        raise TypeError("model 必须是 T2 Plain/CAN Transformer")
    shared_modules: List[nn.Module] = [
        model.token_embedding,
        model.position_embedding,
        *list(model.blocks[: model.config.cut_layer]),
    ]
    protected_modules: List[nn.Module] = [
        *list(model.blocks[model.config.cut_layer :]),
        model.protected_norm,
        model.protected_head,
    ]
    public_modules: List[nn.Module] = [model.public_norm, model.public_head]

    def collect(modules: Sequence[nn.Module]) -> Tuple[nn.Parameter, ...]:
        """按模型注册顺序收集模块的可训练参数并去重。"""

        seen = set()
        values: List[nn.Parameter] = []
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    values.append(parameter)
        return tuple(values)

    groups = {
        "shared_prefix": collect(shared_modules),
        "protected_path": collect(protected_modules),
        "public_path": collect(public_modules),
    }
    group_ids = [{id(parameter) for parameter in values} for values in groups.values()]
    if any(
        group_ids[left] & group_ids[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise RuntimeError("T2 gradient 参数组必须互斥")
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    observed = set().union(*group_ids)
    if observed != expected:
        raise RuntimeError("T2 gradient 参数组未完整覆盖可训练参数")
    return groups


def _gradient_norms(
    groups: Mapping[str, Sequence[nn.Parameter]],
) -> Dict[str, float]:
    """计算反向传播后各互斥参数组的全局 L2 梯度范数。"""

    result: Dict[str, float] = {}
    for name, parameters in groups.items():
        squared = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError("T2 T-pretrain 出现非有限梯度")
            squared += float(torch.sum(gradient.double() ** 2).item())
        value = float(squared**0.5)
        if not np.isfinite(value):
            raise FloatingPointError("T2 gradient norm 非有限")
        result[name] = value
    return result


def _scope_observations(
    protected_logits: Tensor,
    public_logits: Tensor,
    labels: Tensor,
    scopes: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """从 detached logits 计算四 scope loss 与答案 token 数。"""

    losses: Dict[str, float] = {}
    token_counts: Dict[str, int] = {}
    with torch.no_grad():
        protected_values = protected_logits.detach()
        public_values = public_logits.detach()
        for scope in T2_SCOPES:
            mask = torch.tensor(
                [value == scope for value in scopes],
                dtype=torch.bool,
                device=labels.device,
            )
            logits = (
                protected_values
                if scope in {"protected_public", "protected_private"}
                else public_values
            )
            value = float(masked_causal_lm_loss(logits, labels, mask).item())
            answer_mask = mask[:, None] & (labels[:, 1:] != -100)
            token_count = int(answer_mask.sum().item())
            if token_count <= 0 or not np.isfinite(value):
                raise FloatingPointError("T2 出现非有限 loss 或缺少答案 token")
            losses[scope] = value
            token_counts[scope] = token_count
    return losses, token_counts


def _masked_summary(values: Tensor, mask: Tensor, name: str) -> Dict[str, float]:
    """计算非空有限 Tensor 子集的 min/mean/max。"""

    selected = values.detach()[mask]
    if selected.numel() == 0:
        raise ValueError(f"{name} 统计子集不能为空")
    if not bool(torch.isfinite(selected).all().item()):
        raise FloatingPointError(f"{name} 包含非有限值")
    return {
        "min": float(selected.min().item()),
        "mean": float(selected.mean().item()),
        "max": float(selected.max().item()),
    }


def _gate_observations(decision: object, masks: T2ScopeMasks) -> Dict[str, Any]:
    """按真实 valid/invalid 判决汇总 Gate signal 与 error norm。"""

    if not hasattr(decision, "gate_signal") or not hasattr(decision, "evidence"):
        raise TypeError("decision 缺少 Gate 观测字段")
    gate_signal = decision.gate_signal
    error_norm = decision.evidence.error_norm
    return {
        "valid": {
            "count": int(masks.valid.sum().item()),
            "signal": _masked_summary(gate_signal, masks.valid, "valid gate_signal"),
            "error_norm": _masked_summary(error_norm, masks.valid, "valid error_norm"),
        },
        "invalid": {
            "count": int(masks.invalid.sum().item()),
            "signal": _masked_summary(
                gate_signal, masks.invalid, "invalid gate_signal"
            ),
            "error_norm": _masked_summary(
                error_norm, masks.invalid, "invalid error_norm"
            ),
        },
    }


def _finish_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: Tensor,
    attention_mask: Tensor,
    global_step: int,
    *,
    observations: Optional[Mapping[str, object]] = None,
    collect_diagnostics: bool = True,
) -> Dict[str, Any]:
    """检查 loss/梯度、采集诊断、执行 optimizer step 并返回指标。"""

    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("T2 T-pretrain 出现非有限 loss")
    loss.backward()
    if any(
        parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    ):
        raise FloatingPointError("T2 T-pretrain 出现非有限梯度")
    gradient_norms = (
        _gradient_norms(t2_parameter_groups(model)) if collect_diagnostics else None
    )
    optimizer.step()
    result: Dict[str, Any] = {
        "loss": float(loss.detach().item()),
        "samples": float(attention_mask.shape[0]),
        "tokens": float(count_non_padding_input_tokens(attention_mask)),
        "global_step": float(global_step + 1),
    }
    if collect_diagnostics:
        result["total_loss"] = result["loss"]
        result["gradient_norms"] = gradient_norms
        if observations is not None:
            result.update(dict(observations))
    return result


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
        collect_diagnostics: bool = True,
    ) -> None:
        """初始化 CAN T2 trainer 并校验模型、生成器和权重。"""

        if not isinstance(model, GatedDecoderTransformer):
            raise TypeError("model 必须是 GatedDecoderTransformer")
        if not isinstance(credential_generator, CredentialGenerator):
            raise TypeError("credential_generator 必须是 CredentialGenerator")
        _validate_common(model, optimizer, device, protected_weight, public_weight)
        if not isinstance(collect_diagnostics, bool):
            raise TypeError("collect_diagnostics 必须是 bool")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.credential_generator = credential_generator
        self.protected_weight = float(protected_weight)
        self.public_weight = float(public_weight)
        self.collect_diagnostics = collect_diagnostics
        self.global_step = 0

    def train_batch(self, batch: Mapping[str, object]) -> Dict[str, Any]:
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
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("T2 T-pretrain 出现非有限 loss")
        observations: Optional[Dict[str, object]] = None
        if self.collect_diagnostics:
            scope_losses, scope_tokens = _scope_observations(
                output.protected_logits,
                output.public_logits,
                labels,
                scopes,
            )
            observations = {
                "protected_head_loss": float(protected_loss.detach().item()),
                "public_head_loss": float(public_loss.detach().item()),
                "scope_losses": scope_losses,
                "scope_answer_tokens": scope_tokens,
                "gate": _gate_observations(output.decision, masks),
            }
        metrics = _finish_step(
            self.model,
            self.optimizer,
            loss,
            attention_mask,
            self.global_step,
            observations=observations,
            collect_diagnostics=self.collect_diagnostics,
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
        collect_diagnostics: bool = True,
    ) -> None:
        """初始化 Plain T2 trainer 并校验模型和监督权重。"""

        if not isinstance(model, PlainDecoderTransformer):
            raise TypeError("model 必须是 PlainDecoderTransformer")
        _validate_common(model, optimizer, device, protected_weight, public_weight)
        if not isinstance(collect_diagnostics, bool):
            raise TypeError("collect_diagnostics 必须是 bool")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.protected_weight = float(protected_weight)
        self.public_weight = float(public_weight)
        self.collect_diagnostics = collect_diagnostics
        self.global_step = 0

    def train_batch(self, batch: Mapping[str, object]) -> Dict[str, Any]:
        """使用与 CAN 相同的四元组 mask 训练一个 Plain batch。"""

        self.model.train()
        input_ids, labels, attention_mask, scopes, masks = _prepare_t2_batch(
            batch, self.device
        )
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(input_ids, attention_mask)
        protected_loss = masked_causal_lm_loss(
            output.protected_logits, labels, masks.protected
        )
        public_loss = masked_causal_lm_loss(output.public_logits, labels, masks.public)
        loss = self.protected_weight * protected_loss + self.public_weight * public_loss
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("T2 T-pretrain 出现非有限 loss")
        observations = None
        if self.collect_diagnostics:
            scope_losses, scope_tokens = _scope_observations(
                output.protected_logits,
                output.public_logits,
                labels,
                scopes,
            )
            observations = {
                "protected_head_loss": float(protected_loss.detach().item()),
                "public_head_loss": float(public_loss.detach().item()),
                "scope_losses": scope_losses,
                "scope_answer_tokens": scope_tokens,
                "gate": None,
            }
        metrics = _finish_step(
            self.model,
            self.optimizer,
            loss,
            attention_mask,
            self.global_step,
            observations=observations,
            collect_diagnostics=self.collect_diagnostics,
        )
        self.global_step += 1
        return metrics


__all__ = [
    "T2CanPretrainer",
    "T2PlainPretrainer",
    "T2ScopeMasks",
    "build_t2_scope_masks",
    "t2_parameter_groups",
]
