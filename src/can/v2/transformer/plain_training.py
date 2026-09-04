"""Plain decoder-only Transformer 的 exploratory 训练循环。"""

from typing import Dict, List, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .plain_model import PlainDecoderTransformer
from .training import count_non_padding_input_tokens, masked_causal_lm_loss


class PlainDecoderTrainer:
    """训练不含 Gate/credential 的同构 Plain 双 head 模型。"""

    def __init__(
        self,
        model: PlainDecoderTransformer,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        protected_weight: float = 1.0,
        public_weight: float = 1.0,
    ) -> None:
        """初始化训练器并验证数据、设备和监督权重。"""
        if not isinstance(model, PlainDecoderTransformer):
            raise TypeError("model 必须是 PlainDecoderTransformer")
        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader 必须是 DataLoader")
        if len(train_loader) == 0:
            raise ValueError("train_loader 不能为空")
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
            if not torch.isfinite(torch.tensor(float(value))) or float(value) <= 0.0:
                raise ValueError(f"{name} 必须是有限正数")
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.protected_weight = float(protected_weight)
        self.public_weight = float(public_weight)
        self.global_step = 0
        self.current_epoch = 0

    def train_epoch(
        self, progress: bool = False, description: str = "Plain T-pretrain"
    ) -> Dict[str, float]:
        """训练一个完整 epoch 并返回 loss、token 和步数统计。"""
        if not isinstance(progress, bool):
            raise TypeError("progress 必须是 bool")
        batch_sampler = getattr(self.train_loader, "batch_sampler", None)
        set_epoch = getattr(batch_sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self.current_epoch)
        self.model.train()
        batches = self.train_loader
        if progress:
            try:
                from tqdm.auto import tqdm

                batches = tqdm(
                    batches,
                    total=len(self.train_loader),
                    desc=description,
                    leave=False,
                )
            except ImportError:
                pass
        total_loss = 0.0
        total_samples = 0
        total_tokens = 0
        for batch in batches:
            input_ids, labels, attention_mask, scopes = self._prepare_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(input_ids, attention_mask)
            protected_mask = torch.tensor(
                [scope != "refusal" for scope in scopes],
                dtype=torch.bool,
                device=self.device,
            )
            public_mask = torch.tensor(
                [scope != "private" for scope in scopes],
                dtype=torch.bool,
                device=self.device,
            )
            loss = self.protected_weight * masked_causal_lm_loss(
                output.protected_logits, labels, protected_mask
            ) + self.public_weight * masked_causal_lm_loss(
                output.public_logits, labels, public_mask
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Plain T-pretrain 出现非有限 loss")
            loss.backward()
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in self.model.parameters()
            ):
                raise FloatingPointError("Plain T-pretrain 出现非有限梯度")
            self.optimizer.step()
            batch_size = int(input_ids.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            total_tokens += count_non_padding_input_tokens(attention_mask)
            self.global_step += 1
        self.current_epoch += 1
        return {
            "loss": total_loss / total_samples,
            "samples": float(total_samples),
            "tokens": float(total_tokens),
            "global_step": float(self.global_step),
            "epoch": float(self.current_epoch),
        }

    def _prepare_batch(
        self, batch: Dict[str, object]
    ) -> tuple[Tensor, Tensor, Tensor, List[str]]:
        """校验 collate batch 并把 Tensor 移动到目标设备。"""
        if not isinstance(batch, dict):
            raise TypeError("batch 必须是字典")
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")
        scopes = batch.get("scopes")
        if not all(
            isinstance(value, Tensor) for value in (input_ids, labels, attention_mask)
        ):
            raise TypeError("batch Tensor 字段类型错误")
        if not isinstance(scopes, Sequence) or isinstance(scopes, (str, bytes)):
            raise TypeError("batch.scopes 必须是字符串序列")
        scope_values = list(scopes)
        if len(scope_values) != input_ids.shape[0] or any(
            value not in {"public", "private", "refusal"} for value in scope_values
        ):
            raise ValueError("batch.scopes 与 batch 大小或允许集合不一致")
        return (
            input_ids.to(self.device),
            labels.to(self.device),
            attention_mask.to(self.device),
            scope_values,
        )


__all__ = [
    "PlainDecoderTrainer",
]
