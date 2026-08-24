"""Phase 2 的可复现三阶段训练器与 checkpoint 工具。"""

import hashlib
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CredentialGenerator
from .loss import compute_training_loss
from .metrics import EvaluationMetricAccumulator


class GatedResNetTrainer:
    """执行 Stage A/B/C 训练、验证、早停和原子 checkpoint 保存。"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        credential_generator: CredentialGenerator,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        stage: str = "A",
        teacher: Optional[nn.Module] = None,
        valid_ratio: float = 1.0,
        alpha: float = 1.0,
        beta_ce: float = 0.0,
        beta_kd: float = 0.0,
        temperature: float = 4.0,
        scheduler: Optional[object] = None,
        max_grad_norm: Optional[float] = None,
        progress: bool = True,
        teacher_identity: Optional[Dict[str, Any]] = None,
        protected_baseline: Optional[float] = None,
        max_protected_drop: Optional[float] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """初始化训练器并校验阶段、teacher 和损失配置。

        参数:
            model: 待训练的 Gated ResNet 模型。
            train_loader: 训练 DataLoader。
            val_loader: 验证 DataLoader。
            credential_generator: 负责生成并验证 credential 的采样器。
            optimizer: 只包含当前阶段可训练参数的优化器。
            device: 模型和 batch 所在设备。
            stage: ``A``、``B`` 或 ``C``。
            teacher: Stage A best 的冻结副本；B/C 必须提供。
            valid_ratio: 训练 batch 中 valid credential 比例。
            alpha/beta_ce/beta_kd/temperature: 三项损失权重和 KD 温度。
            scheduler: 可选的 PyTorch 学习率调度器。
            max_grad_norm: 可选梯度裁剪上限。
            progress: 是否显示唯一的 epoch 进度条。
            teacher_identity: 写入 checkpoint 的 teacher 路径/hash 元数据。
            protected_baseline: Stage C 的 Stage A protected 验证基线。
            max_protected_drop: Stage C 允许的绝对下降上限。
            checkpoint_metadata: 写入每个 checkpoint 的实验元数据。
        """

        if not isinstance(model, nn.Module):
            raise TypeError("model 必须是 nn.Module")
        if not isinstance(train_loader, DataLoader) or not isinstance(
            val_loader, DataLoader
        ):
            raise TypeError("train_loader 和 val_loader 必须是 DataLoader")
        if not isinstance(credential_generator, CredentialGenerator):
            raise TypeError("credential_generator 类型非法")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer 必须是 torch.optim.Optimizer")
        if not isinstance(device, torch.device):
            raise TypeError("device 必须是 torch.device")
        if stage not in {"A", "B", "C"}:
            raise ValueError("stage 必须是 A、B 或 C")
        if isinstance(valid_ratio, bool) or not isinstance(valid_ratio, (int, float)):
            raise TypeError("valid_ratio 必须是有限实数")
        if not np.isfinite(float(valid_ratio)) or not 0.0 <= float(valid_ratio) <= 1.0:
            raise ValueError("valid_ratio 必须位于 [0, 1]")
        if stage in {"B", "C"} and teacher is None:
            raise ValueError("Stage B/C 必须提供冻结 teacher")
        if stage in {"B", "C"} and not teacher_identity:
            raise ValueError("Stage B/C 必须提供 teacher_identity")
        if stage == "C" and protected_baseline is None:
            raise ValueError("Stage C 必须提供 protected_baseline")
        if stage == "C" and max_protected_drop is None:
            raise ValueError("Stage C 必须提供 max_protected_drop")
        if teacher is not None and not isinstance(teacher, nn.Module):
            raise TypeError("teacher 必须是 nn.Module")
        if not isinstance(progress, bool):
            raise TypeError("progress 必须是 bool")
        if max_grad_norm is not None:
            if isinstance(max_grad_norm, bool) or not isinstance(
                max_grad_norm, (int, float)
            ):
                raise TypeError("max_grad_norm 必须是有限正数或 None")
            if not np.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0.0:
                raise ValueError("max_grad_norm 必须是有限正数")
        if protected_baseline is not None and not np.isfinite(
            float(protected_baseline)
        ):
            raise ValueError("protected_baseline 必须是有限数")
        if max_protected_drop is not None:
            if (
                not np.isfinite(float(max_protected_drop))
                or float(max_protected_drop) < 0.0
            ):
                raise ValueError("max_protected_drop 必须是有限非负数")

        self.model = model.to(device)
        self.teacher = teacher.to(device) if teacher is not None else None
        if self.teacher is not None:
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.credential_generator = credential_generator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.stage = stage
        self.valid_ratio = float(valid_ratio)
        self.loss_weights = {
            "alpha": alpha,
            "beta_ce": beta_ce,
            "beta_kd": beta_kd,
            "temperature": temperature,
        }
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        self.progress = progress
        self.teacher_identity = dict(teacher_identity or {})
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.protected_baseline = (
            float(protected_baseline) if protected_baseline is not None else None
        )
        self.max_protected_drop = (
            float(max_protected_drop) if max_protected_drop is not None else None
        )
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric: Optional[float] = None
        self.best_metrics: Dict[str, Optional[float]] = {}
        self._validate_teacher_identity()

    def _validate_teacher_identity(self) -> None:
        """验证 teacher checkpoint 的存在性和 SHA-256 绑定。"""

        if not self.teacher_identity:
            return
        path_value = self.teacher_identity.get("path")
        expected_hash = self.teacher_identity.get("sha256")
        if not isinstance(path_value, str) or not Path(path_value).is_file():
            raise FileNotFoundError("teacher checkpoint 路径不存在")
        if (
            not isinstance(expected_hash, str)
            or checkpoint_sha256(Path(path_value)) != expected_hash
        ):
            raise ValueError("teacher checkpoint SHA-256 不匹配")

    def _configure_stage(self) -> None:
        """设置阶段冻结策略，并阻止 Stage B 的 BatchNorm 统计漂移。"""

        self.model.train()
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        if self.stage == "A":
            # Stage A 只训练 protected 路径，public head 保持未训练状态。
            public_head = getattr(self.model, "public_fc", None)
            if public_head is not None:
                public_head.eval()
                for parameter in public_head.parameters():
                    parameter.requires_grad_(False)
            return
        if self.stage == "C":
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            return

        # 顶层保持 train()，让模型返回 TrainingOutput；各冻结子模块显式 eval()。
        for module in self.model.modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        public_head = getattr(self.model, "public_fc", None)
        if public_head is None:
            raise AttributeError("模型缺少 Stage B 所需的 public_fc")
        public_head.train()
        for parameter in public_head.parameters():
            parameter.requires_grad_(True)
        self.model.training = True

    def _trainable_parameters(self):
        """返回当前阶段仍允许更新的参数。"""

        return [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

    def _teacher_logits(self, images: Tensor) -> Optional[Tensor]:
        """用 valid credential 调用冻结 teacher，并严格校验稀疏 indices。"""

        if self.teacher is None:
            return None
        if images.ndim != 4:
            raise ValueError("images 必须是四维 batch Tensor")
        with torch.inference_mode():
            credentials = self.credential_generator.all_valid(images.shape[0])
            output = self.teacher(images, credentials.values)
            if not hasattr(output, "protected_logits"):
                raise TypeError("teacher 输出缺少 protected_logits")
            protected_logits = output.protected_logits
            if hasattr(output, "protected_indices"):
                expected = torch.arange(images.shape[0], device=images.device)
                if not torch.equal(output.protected_indices, expected):
                    raise RuntimeError("teacher valid indices 不完整或乱序")
            else:
                raise TypeError("teacher 推理输出缺少 protected_indices")
            if protected_logits.shape != (images.shape[0], 10):
                raise ValueError("teacher protected logits 必须是 [B, 10]")
            if not torch.isfinite(protected_logits).all():
                raise ValueError("teacher logits 必须全部有限")
            return protected_logits.detach()

    def train_epoch(
        self, epoch: int = 1, total_epochs: int = 1
    ) -> Dict[str, Optional[float]]:
        """训练一个 epoch，并使用唯一 tqdm 进度条显示实时指标。"""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("epoch 必须是正整数")
        if (
            isinstance(total_epochs, bool)
            or not isinstance(total_epochs, int)
            or total_epochs <= 0
        ):
            raise ValueError("total_epochs 必须是正整数")
        if len(self.train_loader) == 0:
            raise ValueError(
                "训练 DataLoader 没有 batch；drop_last=True 时数据量必须不少于 batch_size"
            )
        self._configure_stage()
        total_loss = 0.0
        sample_count = 0
        protected_correct = protected_total = 0
        public_correct = public_total = 0
        iterator = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{total_epochs}",
            unit="batch",
            leave=True,
            ncols=120,
            disable=not self.progress,
        )
        try:
            for images, fine_labels, coarse_labels in iterator:
                if not isinstance(images, Tensor):
                    raise TypeError("DataLoader images 必须是 Tensor")
                if not isinstance(fine_labels, Tensor) or not isinstance(
                    coarse_labels, Tensor
                ):
                    raise TypeError("DataLoader labels 必须是 Tensor")
                if fine_labels.dtype != torch.long or coarse_labels.dtype != torch.long:
                    raise TypeError("DataLoader labels 必须是 torch.long")
                images = images.to(self.device, non_blocking=True)
                fine_labels = fine_labels.to(
                    self.device, dtype=torch.long, non_blocking=True
                )
                coarse_labels = coarse_labels.to(
                    self.device, dtype=torch.long, non_blocking=True
                )
                min_valid = (
                    2 if self.stage in {"A", "C"} and self.valid_ratio > 0.0 else 0
                )
                credential_batch = self.credential_generator.batch_generate(
                    images.shape[0], self.valid_ratio, min_valid=min_valid
                )
                output = self.model(images, credential_batch.values)
                teacher_logits = self._teacher_logits(images)
                losses = compute_training_loss(
                    output,
                    fine_labels,
                    coarse_labels,
                    teacher_logits,
                    **self.loss_weights,
                )
                if not torch.isfinite(losses.total).item():
                    raise FloatingPointError("训练 loss 出现 NaN 或 Inf")
                self.optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._trainable_parameters(), self.max_grad_norm
                    )
                for parameter in self._trainable_parameters():
                    if (
                        parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all().item()
                    ):
                        raise FloatingPointError("训练梯度出现 NaN 或 Inf")
                self.optimizer.step()
                self.global_step += 1

                batch_size = images.shape[0]
                sample_count += batch_size
                total_loss += float(losses.total.detach().item()) * batch_size
                allow = output.decision.allow
                protected_total += int(allow.sum().item())
                if bool(allow.any().item()):
                    protected_correct += int(
                        (output.protected_logits[allow].argmax(1) == fine_labels[allow])
                        .sum()
                        .item()
                    )
                public_total += batch_size
                public_correct += int(
                    (output.public_logits.argmax(1) == coarse_labels).sum().item()
                )
                if self.progress:
                    iterator.set_postfix(
                        loss=f"{losses.total.item():.4f}",
                        acc_p=(
                            f"{100.0 * protected_correct / protected_total:.2f}%"
                            if protected_total
                            else "n/a"
                        ),
                        acc_c=f"{100.0 * public_correct / public_total:.2f}%",
                    )
        finally:
            iterator.close()
        if sample_count == 0:
            raise RuntimeError("训练 epoch 未处理任何样本")
        self.current_epoch = epoch
        return {
            "loss": total_loss / sample_count if sample_count else None,
            "protected_accuracy": (
                protected_correct / protected_total if protected_total else None
            ),
            "public_accuracy": public_correct / public_total if public_total else None,
            "protected_total": float(protected_total),
            "public_total": float(public_total),
        }

    def validate(self) -> Dict[str, Optional[float]]:
        """在同一 validation split 上分别评估全 valid 和全 invalid 路径。"""

        self.model.eval()
        metrics = EvaluationMetricAccumulator()
        with torch.inference_mode():
            for images, fine_labels, coarse_labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                fine_labels = fine_labels.to(
                    self.device, dtype=torch.long, non_blocking=True
                )
                coarse_labels = coarse_labels.to(
                    self.device, dtype=torch.long, non_blocking=True
                )

                valid = self.model(
                    images, self.credential_generator.all_valid(images.shape[0]).values
                )
                if not hasattr(valid, "protected_indices"):
                    raise TypeError("验证输出缺少 protected_indices")
                metrics.update_protected(
                    valid.protected_logits,
                    fine_labels.index_select(0, valid.protected_indices),
                )

                invalid = self.model(
                    images,
                    self.credential_generator.all_invalid(images.shape[0]).values,
                )
                if not hasattr(invalid, "public_indices"):
                    raise TypeError("验证输出缺少 public_indices")
                metrics.update_public(
                    invalid.public_logits,
                    coarse_labels.index_select(0, invalid.public_indices),
                )
        result = metrics.compute()
        self.best_metrics.update(result)
        return result

    def _monitor_value(self, metrics: Dict[str, Optional[float]]) -> Optional[float]:
        """按阶段返回固定的 checkpoint 选择指标。"""

        if self.stage == "A":
            return metrics.get("protected_accuracy")
        public_metric = metrics.get("public_balanced_accuracy")
        if self.stage == "B":
            return public_metric
        if self.protected_baseline is None or self.max_protected_drop is None:
            raise RuntimeError("Stage C protected 约束未初始化")
        protected = metrics.get("protected_accuracy")
        if protected is None or public_metric is None:
            return None
        if (
            self.max_protected_drop is not None
            and protected < self.protected_baseline - self.max_protected_drop
        ):
            return None
        return public_metric

    def save_checkpoint(
        self,
        path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        best_metrics: Optional[Dict[str, Optional[float]]] = None,
    ) -> None:
        """原子保存模型、优化器、teacher identity、配置元数据和全部 RNG 状态。"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "schema_version": 2,
            "stage": self.stage,
            "epoch": int(self.current_epoch if epoch is None else epoch),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_metrics": dict(best_metrics or self.best_metrics),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "teacher_identity": dict(self.teacher_identity),
            "metadata": {**self.checkpoint_metadata, **dict(metadata or {})},
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "credential": self.credential_generator.rng_state(),
                "loader": self._loader_rng_state(),
            },
        }
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_checkpoint(
        self, path: Path, map_location: Optional[object] = None
    ) -> Dict[str, Any]:
        """加载受信任 checkpoint，验证 stage/teacher identity 后恢复状态。"""

        payload = torch.load(
            Path(path),
            map_location=map_location or self.device,
            weights_only=False,
        )
        if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
            raise ValueError("checkpoint schema 不受支持")
        if payload.get("stage") != self.stage:
            raise ValueError("checkpoint stage 不匹配")
        saved_identity = payload.get("teacher_identity", {})
        if self.stage in {"B", "C"} and saved_identity != self.teacher_identity:
            raise ValueError("checkpoint teacher identity 不匹配")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata 非法")
        self._validate_checkpoint_metadata(metadata)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler") is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.current_epoch = int(payload.get("epoch", 0))
        self.global_step = int(payload["global_step"])
        self.best_metric = payload.get("best_metric")
        self.best_metrics = dict(payload.get("best_metrics", {}))
        rng = payload.get("rng", {})
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("credential") is not None:
            self.credential_generator.set_rng_state(rng["credential"])
        if rng.get("loader") is not None:
            self._set_loader_rng_state(rng["loader"])
        return dict(metadata)

    def _validate_checkpoint_metadata(self, metadata: Dict[str, Any]) -> None:
        """校验恢复 checkpoint 的配置、split 和 LWE 公共参数。"""

        for key in ("mapping_version", "split"):
            expected = self.checkpoint_metadata.get(key)
            if expected is not None and metadata.get(key) != expected:
                raise ValueError(f"checkpoint {key} 与当前实验不匹配")
        expected_signature = self.checkpoint_metadata.get("config_signature")
        if (
            expected_signature is not None
            and metadata.get("config_signature") != expected_signature
        ):
            raise ValueError("checkpoint config_signature 与当前实验不匹配")
        expected_lwe = self.checkpoint_metadata.get("lwe")
        if expected_lwe is not None and metadata.get("lwe") != expected_lwe:
            raise ValueError("checkpoint LWEParams 与当前实验不匹配")
        expected_A = self.checkpoint_metadata.get("A")
        expected_b = self.checkpoint_metadata.get("b")
        saved_A = metadata.get("A")
        saved_b = metadata.get("b")
        if expected_A is not None:
            if not isinstance(saved_A, np.ndarray) or not np.array_equal(
                saved_A, expected_A
            ):
                raise ValueError("checkpoint LWE A 与当前实验不匹配")
        if expected_b is not None:
            if not isinstance(saved_b, np.ndarray) or not np.array_equal(
                saved_b, expected_b
            ):
                raise ValueError("checkpoint LWE b 与当前实验不匹配")

    def _loader_rng_state(self) -> Optional[Tensor]:
        """读取训练 DataLoader 的显式 generator 状态。"""

        generator = getattr(self.train_loader, "generator", None)
        return generator.get_state() if generator is not None else None

    def _set_loader_rng_state(self, state: Tensor) -> None:
        """恢复训练 DataLoader 的显式 generator 状态。"""

        generator = getattr(self.train_loader, "generator", None)
        if generator is not None:
            generator.set_state(state)

    def fit(
        self,
        epochs: int,
        patience: int = 0,
        min_delta: float = 0.0,
        checkpoint_dir: Optional[Path] = None,
        start_epoch: Optional[int] = None,
    ) -> Dict[str, Optional[float]]:
        """训练指定阶段，按固定 validation 指标早停并保存 last/best。"""

        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs 必须是正整数")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
            raise ValueError("patience 必须是非负整数")
        if not np.isfinite(float(min_delta)) or float(min_delta) < 0.0:
            raise ValueError("min_delta 必须是有限非负数")
        stale = 0
        history: Dict[str, Optional[float]] = {}
        first_epoch = (
            self.current_epoch + 1 if start_epoch is None else int(start_epoch)
        )
        if first_epoch < 1 or first_epoch > epochs + 1:
            raise ValueError("start_epoch 超出训练范围")
        for epoch in range(first_epoch, epochs + 1):
            train_metrics = self.train_epoch(epoch, epochs)
            val_metrics = self.validate()
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    monitor_for_scheduler = val_metrics.get("public_balanced_accuracy")
                    if monitor_for_scheduler is not None:
                        self.scheduler.step(monitor_for_scheduler)
                else:
                    self.scheduler.step()
            history = {"epoch": float(epoch), **train_metrics, **val_metrics}
            if checkpoint_dir is not None:
                directory = Path(checkpoint_dir)
                self.save_checkpoint(directory / "last.ckpt", history, epoch, history)
            monitor = self._monitor_value(val_metrics)
            improved = monitor is not None and (
                self.best_metric is None
                or monitor > self.best_metric + float(min_delta)
            )
            if improved:
                self.best_metric = float(monitor)
                stale = 0
                if checkpoint_dir is not None:
                    self.save_checkpoint(
                        Path(checkpoint_dir) / "best.ckpt", history, epoch, history
                    )
            else:
                stale += 1
            if self.progress:
                tqdm.write(
                    f"Epoch {epoch}/{epochs} | loss={train_metrics['loss']} | "
                    f"protected={val_metrics.get('protected_accuracy')} | "
                    f"public_balanced={val_metrics.get('public_balanced_accuracy')}"
                )
            if patience and stale >= patience:
                break
        return history

    def train(self, num_epochs: int) -> Dict[str, Optional[float]]:
        """兼容设计文档命名，调用 ``fit`` 完成当前阶段训练。"""

        return self.fit(num_epochs)


def checkpoint_sha256(path: Path) -> str:
    """计算 checkpoint 文件 SHA-256，用于 teacher identity 绑定。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["GatedResNetTrainer", "checkpoint_sha256"]
