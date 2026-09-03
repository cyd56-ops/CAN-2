"""Phase 5 T0 的损失、冻结策略、训练循环与质量门。"""

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..layers.gate_layer import AuthorizationDecision, ReasonCode
from ..training.data import CredentialGenerator
from .model import GatedDecoderTransformer


@dataclass(frozen=True)
class PretrainMetrics:
    """保存 T-pretrain validation 的三项 go/no-go 指标。"""

    public_exact_match: float
    private_exact_match: float
    refusal_rate: float


def pretrain_go_no_go(metrics: PretrainMetrics) -> bool:
    """判断 T-pretrain 是否达到进入 Stage A/B/C 的绝对门槛。"""

    if not isinstance(metrics, PretrainMetrics):
        raise TypeError("metrics 必须是 PretrainMetrics")
    values = (
        metrics.public_exact_match,
        metrics.private_exact_match,
        metrics.refusal_rate,
    )
    if any(not torch.isfinite(torch.tensor(value)).item() for value in values):
        raise ValueError("pretrain 指标必须是有限数")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("pretrain 指标必须位于 [0, 1]")
    return (
        metrics.public_exact_match >= 0.80
        and metrics.private_exact_match >= 0.80
        and metrics.refusal_rate >= 0.90
    )


def count_non_padding_input_tokens(attention_mask: Tensor) -> int:
    """统计 prompt 与 target 中全部非 padding token。"""

    if not isinstance(attention_mask, Tensor):
        raise TypeError("attention_mask 必须是 Tensor")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask 必须是 [B,T]")
    if attention_mask.dtype not in {torch.bool, torch.int32, torch.int64}:
        raise TypeError("attention_mask 必须是 bool 或整数 Tensor")
    if bool(((attention_mask != 0) & (attention_mask != 1)).any().item()):
        raise ValueError("attention_mask 只能包含 0/1")
    return int(attention_mask.sum().item())


def masked_causal_lm_loss(
    logits: Tensor, labels: Tensor, sample_mask: Optional[Tensor] = None
) -> Tensor:
    """计算支持逐样本路由 mask 的 shifted causal LM loss。

    参数:
        logits: 模型 logits，shape 为 ``[B, T, V]``。
        labels: 目标 token，shape 为 ``[B, T]``；忽略位置使用 ``-100``。
        sample_mask: 可选 BoolTensor ``[B]``，只选择指定路由样本。

    返回:
        标量交叉熵；空选择返回与计算图连接的零值。
    """

    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("logits 和 labels 必须是 Tensor")
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits/labels 必须分别为 [B,T,V] 和 [B,T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits 和 labels 的 batch/序列维必须一致")
    if labels.dtype != torch.long:
        raise TypeError("labels 必须是 torch.long")
    if logits.shape[1] < 2:
        raise ValueError("因果 LM loss 至少需要两个 token")
    if sample_mask is None:
        sample_mask = torch.ones(
            logits.shape[0], dtype=torch.bool, device=logits.device
        )
    if not isinstance(sample_mask, Tensor) or sample_mask.dtype != torch.bool:
        raise TypeError("sample_mask 必须是 BoolTensor")
    if sample_mask.shape != (logits.shape[0],) or sample_mask.device != logits.device:
        raise ValueError("sample_mask 必须与 logits batch/device 对齐")
    if not bool(sample_mask.any().item()):
        return logits.sum() * 0.0

    selected_logits = logits[sample_mask, :-1, :].reshape(-1, logits.shape[-1])
    selected_labels = labels[sample_mask, 1:].reshape(-1)
    if not bool((selected_labels != -100).any().item()):
        return selected_logits.sum() * 0.0
    return F.cross_entropy(selected_logits, selected_labels, ignore_index=-100)


def causal_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    temperature: float = 4.0,
) -> Tensor:
    """在答案 token 位置计算冻结 teacher 的因果 KL 蒸馏损失。"""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student 与 teacher logits shape 必须一致")
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("labels 必须与 logits 的 batch/序列维一致")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature 必须是有限正数")
    temperature = float(temperature)
    if not torch.isfinite(torch.tensor(temperature)).item() or temperature <= 0.0:
        raise ValueError("temperature 必须是有限正数")
    answer_mask = labels[:, 1:] != -100
    if not bool(answer_mask.any().item()):
        return student_logits.sum() * 0.0
    student = student_logits[:, :-1, :][answer_mask] / temperature
    teacher = teacher_logits[:, :-1, :][answer_mask] / temperature
    return (
        F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


def configure_stage(model: GatedDecoderTransformer, stage: str) -> Dict[str, int]:
    """配置 T-pretrain/A/B/C 的可训练参数，并返回参数量摘要。"""

    if not isinstance(model, GatedDecoderTransformer):
        raise TypeError("model 必须是 GatedDecoderTransformer")
    if stage not in {"T-pretrain", "A", "B", "C"}:
        raise ValueError("stage 必须是 T-pretrain、A、B 或 C")
    for parameter in model.parameters():
        parameter.requires_grad_(stage in {"T-pretrain", "C"})
    if stage in {"A", "B"}:
        for module in (model.public_norm, model.public_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable": trainable, "total": total}


def freeze_teacher(teacher: nn.Module) -> None:
    """把 T-pretrain teacher 固定为 eval 且不可训练的只读模型。"""

    if not isinstance(teacher, nn.Module):
        raise TypeError("teacher 必须是 nn.Module")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


def validate_mixed_routing(decision: AuthorizationDecision) -> None:
    """验证 Stage C mixed batch 至少包含 2 valid 和 1 parsed invalid。"""

    if not isinstance(decision, AuthorizationDecision):
        raise TypeError("decision 必须是 AuthorizationDecision")
    valid_count = int(decision.allow.sum().item())
    invalid_count = int(
        (decision.evidence.reason_code == int(ReasonCode.LWE_VERIFICATION_FAILED))
        .sum()
        .item()
    )
    malformed_count = decision.allow.shape[0] - valid_count - invalid_count
    if malformed_count:
        raise ValueError("Stage C mixed batch 不允许格式错误 credential")
    if valid_count < 2 or invalid_count < 1:
        raise ValueError("Stage C mixed batch 至少需要 2 valid 和 1 invalid")


class Phase5Trainer:
    """执行 T-pretrain 和 Stage A/B/C 的最小可恢复训练循环。"""

    def __init__(
        self,
        model: GatedDecoderTransformer,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        stage: str,
        credential_generator: Optional[CredentialGenerator] = None,
        teacher: Optional[GatedDecoderTransformer] = None,
        teacher_identity: Optional[Dict[str, str]] = None,
        ce_weight: float = 1.0,
        kd_weight: float = 1.0,
        temperature: float = 4.0,
        pretrain_protected_weight: float = 1.0,
        pretrain_public_weight: float = 1.0,
    ) -> None:
        """初始化训练器并验证阶段依赖。

        参数:
            model: 待训练的 gated Transformer student。
            train_loader: 返回 Phase 5 collate batch 的 DataLoader。
            optimizer: 当前阶段使用的优化器。
            device: 模型和 batch 所在设备。
            stage: ``T-pretrain``、``A``、``B`` 或 ``C``。
            credential_generator: A/B/C 的确定性 credential 生成器。
            teacher: Stage B/C 使用的冻结 T-pretrain teacher。
            teacher_identity: Stage B/C 必需的 teacher checkpoint/manifest 摘要。
            ce_weight: 监督交叉熵权重。
            kd_weight: Stage B/C 蒸馏权重。
            temperature: 蒸馏温度。
            pretrain_protected_weight: T-pretrain protected head 监督权重。
            pretrain_public_weight: T-pretrain public head 监督权重。
        """

        if not isinstance(model, GatedDecoderTransformer):
            raise TypeError("model 必须是 GatedDecoderTransformer")
        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader 必须是 DataLoader")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer 必须是 torch.optim.Optimizer")
        if not isinstance(device, torch.device):
            raise TypeError("device 必须是 torch.device")
        if stage not in {"T-pretrain", "A", "B", "C"}:
            raise ValueError("stage 必须是 T-pretrain、A、B 或 C")
        if stage != "T-pretrain" and not isinstance(
            credential_generator, CredentialGenerator
        ):
            raise ValueError("Stage A/B/C 必须提供 CredentialGenerator")
        if stage in {"B", "C"} and not isinstance(teacher, GatedDecoderTransformer):
            raise ValueError("Stage B/C 必须提供冻结的 T-pretrain teacher")
        if stage in {"B", "C"}:
            self._validate_teacher_identity(teacher_identity)
        for value, name in (
            (ce_weight, "ce_weight"),
            (kd_weight, "kd_weight"),
            (temperature, "temperature"),
            (pretrain_protected_weight, "pretrain_protected_weight"),
            (pretrain_public_weight, "pretrain_public_weight"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} 必须是有限正数")
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} 必须是有限正数")

        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.stage = stage
        self.credential_generator = credential_generator
        self.teacher = teacher.to(device) if teacher is not None else None
        self.teacher_identity = dict(teacher_identity or {})
        if self.teacher is not None:
            freeze_teacher(self.teacher)
        self.ce_weight = float(ce_weight)
        self.kd_weight = float(kd_weight)
        self.temperature = float(temperature)
        self.pretrain_protected_weight = float(pretrain_protected_weight)
        self.pretrain_public_weight = float(pretrain_public_weight)
        self.global_step = 0
        self.current_epoch = 0
        configure_stage(self.model, stage)

    @staticmethod
    def _validate_teacher_identity(identity: Optional[Dict[str, str]]) -> None:
        """验证 teacher checkpoint 与 manifest 的 SHA-256 identity。"""

        if not isinstance(identity, dict):
            raise ValueError("Stage B/C 必须提供 teacher_identity")
        for key in ("checkpoint_sha256", "manifest_sha256"):
            value = identity.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"teacher_identity.{key} 必须是小写 SHA-256")

    def train_epoch(
        self, progress: bool = False, description: Optional[str] = None
    ) -> Dict[str, float]:
        """训练一个 epoch，并返回样本加权的 loss 与 token 数。

        参数:
            progress: 是否显示 batch 级进度条；缺少 tqdm 时自动退化为普通迭代。
            description: 进度条描述文本。
        """

        if not isinstance(progress, bool):
            raise TypeError("progress 必须是 bool")

        if len(self.train_loader) == 0:
            raise RuntimeError("train_loader 为空，无法执行 Phase 5 训练")
        batch_sampler = getattr(self.train_loader, "batch_sampler", None)
        set_epoch = getattr(batch_sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self.current_epoch)
        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()
        total_loss = 0.0
        total_samples = 0
        total_tokens = 0
        batches = self.train_loader
        if progress:
            try:
                from tqdm.auto import tqdm

                batches = tqdm(
                    batches,
                    total=len(self.train_loader),
                    desc=description
                    or f"Stage {self.stage} epoch {self.current_epoch + 1}",
                    leave=False,
                )
            except ImportError:
                pass
        for batch_index, batch in enumerate(batches):
            input_ids, labels, attention_mask, scopes = self._prepare_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._batch_loss(input_ids, labels, attention_mask, scopes)
            if not bool(torch.isfinite(loss).item()):
                scope_counts = {
                    scope: scopes.count(scope)
                    for scope in ("public", "private", "refusal")
                }
                raise FloatingPointError(
                    "Phase 5 训练出现非有限 loss: "
                    f"stage={self.stage}, epoch={self.current_epoch}, "
                    f"global_step={self.global_step}, batch={batch_index}, "
                    f"scopes={scope_counts}"
                )
            loss.backward()
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in self.model.parameters()
            ):
                raise FloatingPointError(
                    "Phase 5 训练出现非有限梯度: "
                    f"stage={self.stage}, epoch={self.current_epoch}, "
                    f"global_step={self.global_step}, batch={batch_index}"
                )
            self.optimizer.step()
            batch_size = input_ids.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            total_tokens += count_non_padding_input_tokens(attention_mask)
            self.global_step += 1
            if progress and hasattr(batches, "set_postfix"):
                batches.set_postfix(loss=f"{float(loss.detach().item()):.4f}")
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
        """验证 DataLoader batch 并移动 Tensor 到训练设备。"""

        if not isinstance(batch, dict):
            raise TypeError("Phase 5 batch 必须是字典")
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")
        scopes = batch.get("scopes")
        if not all(
            isinstance(value, Tensor) for value in (input_ids, labels, attention_mask)
        ):
            raise TypeError("batch 的 input_ids/labels/attention_mask 必须是 Tensor")
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

    def _batch_loss(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Tensor,
        scopes: List[str],
    ) -> Tensor:
        """按当前阶段计算监督和蒸馏损失。"""

        if not self.model.training:
            raise RuntimeError("_batch_loss 要求 student 处于 training 模式")

        if self.stage == "T-pretrain":
            protected_logits = self.model.direct_protected_logits(
                input_ids, attention_mask
            )
            protected_mask = torch.tensor(
                [scope != "refusal" for scope in scopes],
                dtype=torch.bool,
                device=self.device,
            )
            public_logits = self.model.direct_public_logits(input_ids, attention_mask)
            public_mask = torch.tensor(
                [scope != "private" for scope in scopes],
                dtype=torch.bool,
                device=self.device,
            )
            # 两个 head 分别学习 protected 知识和 public/refusal 行为，
            # 避免相同 private prompt 的答案与拒答 target 在同一 head 冲突。
            protected_loss = masked_causal_lm_loss(
                protected_logits, labels, protected_mask
            )
            public_loss = masked_causal_lm_loss(public_logits, labels, public_mask)
            return (
                protected_loss * self.pretrain_protected_weight
                + public_loss * self.pretrain_public_weight
            )

        assert self.credential_generator is not None
        credentials = self._credentials_for_scopes(scopes)
        output = self.model(input_ids, credentials, attention_mask)
        public_mask = torch.tensor(
            [scope in {"public", "refusal"} for scope in scopes],
            dtype=torch.bool,
            device=self.device,
        )
        private_mask = ~public_mask
        if self.stage == "C":
            validate_mixed_routing(output.decision)
        public_indices = torch.nonzero(~output.decision.allow, as_tuple=False).flatten()
        protected_indices = torch.nonzero(
            output.decision.allow, as_tuple=False
        ).flatten()
        public_labels = labels.index_select(0, public_indices)
        public_logits = output.public_logits.index_select(0, public_indices)
        supervised = masked_causal_lm_loss(
            public_logits,
            public_labels,
            torch.ones(public_logits.shape[0], dtype=torch.bool, device=self.device),
        )
        if self.stage == "C":
            protected_labels = labels.index_select(0, protected_indices)
            protected_logits = output.protected_logits.index_select(
                0, protected_indices
            )
            supervised = supervised + masked_causal_lm_loss(
                protected_logits,
                protected_labels,
                torch.ones(
                    protected_logits.shape[0],
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
        loss = supervised * self.ce_weight
        if self.stage in {"B", "C"}:
            assert self.teacher is not None
            with torch.inference_mode():
                teacher_logits = self.teacher.direct_protected_logits(
                    input_ids, attention_mask
                )
            kd_public_mask = torch.tensor(
                [scope == "public" for scope in scopes],
                dtype=torch.bool,
                device=self.device,
            )
            kd_masks = [kd_public_mask]
            student_logits = [output.public_logits.index_select(0, public_indices)]
            kd_labels = [labels.index_select(0, public_indices)]
            if self.stage == "C":
                kd_masks.append(private_mask)
                student_logits.append(
                    output.protected_logits.index_select(0, protected_indices)
                )
                kd_labels.append(labels.index_select(0, protected_indices))
            kd_total = output.public_logits.sum() * 0.0
            kd_teacher = [teacher_logits.index_select(0, public_indices)]
            if self.stage == "C":
                kd_teacher.append(teacher_logits.index_select(0, protected_indices))
            for mask, logits, teacher_target, target_labels in zip(
                kd_masks, student_logits, kd_teacher, kd_labels
            ):
                if logits.shape[0] > 0:
                    kd_total = kd_total + causal_distillation_loss(
                        logits,
                        teacher_target,
                        target_labels,
                        self.temperature,
                    )
            loss = loss + kd_total * self.kd_weight
        return loss

    def _credentials_for_scopes(self, scopes: List[str]) -> Tensor:
        """按 private=valid、public/refusal=invalid 生成逐样本 credential。"""

        assert self.credential_generator is not None
        values = [
            self.credential_generator.generate(scope == "private") for scope in scopes
        ]
        return torch.tensor(np.stack(values), dtype=torch.float32, device=self.device)

    def save_checkpoint(self, path: Path) -> None:
        """原子保存模型、优化器、RNG 与 credential RNG 状态。"""

        if not isinstance(path, Path):
            raise TypeError("path 必须是 pathlib.Path")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "stage": self.stage,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "credential_rng": (
                self.credential_generator.rng_state()
                if self.credential_generator is not None
                else None
            ),
            "config": self.model.config.__dict__,
            "lwe_identity": self._lwe_identity(),
            "lwe_params": self._lwe_params_payload(),
            "A": self.model.gate_layer.verifier.A.detach().cpu().numpy(),
            "b": self.model.gate_layer.verifier.b.detach().cpu().numpy(),
            "teacher_identity": self.teacher_identity,
            "pretrain_protected_weight": self.pretrain_protected_weight,
            "pretrain_public_weight": self.pretrain_public_weight,
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, path: Path) -> None:
        """从受信 checkpoint 恢复训练状态与全部 RNG。"""

        if not isinstance(path, Path) or not path.is_file():
            raise FileNotFoundError("Phase 5 checkpoint 不存在")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("stage") != self.stage:
            raise ValueError("Phase 5 checkpoint schema 或 stage 不匹配")
        if payload.get("config") != self.model.config.__dict__:
            raise ValueError("Phase 5 checkpoint 模型配置不匹配")
        if payload.get("lwe_identity") != self._lwe_identity():
            raise ValueError("Phase 5 checkpoint LWE identity 不匹配")
        if payload.get("teacher_identity") != self.teacher_identity:
            raise ValueError("Phase 5 checkpoint teacher identity 不匹配")
        for key, current in (
            ("pretrain_protected_weight", self.pretrain_protected_weight),
            ("pretrain_public_weight", self.pretrain_public_weight),
        ):
            saved = float(payload.get(key, 1.0))
            if saved != current:
                raise ValueError(f"Phase 5 checkpoint {key} 不匹配")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload["global_step"])
        self.current_epoch = int(payload["current_epoch"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        cuda_rng = payload.get("cuda_rng")
        if cuda_rng is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("checkpoint 含 CUDA RNG，但当前环境无 CUDA")
            torch.cuda.set_rng_state_all(cuda_rng)
        if self.credential_generator is not None:
            credential_rng = payload.get("credential_rng")
            if not isinstance(credential_rng, dict):
                raise ValueError("checkpoint 缺少 credential RNG 状态")
            self.credential_generator.set_rng_state(credential_rng)

    def _lwe_identity(self) -> Dict[str, object]:
        """返回当前 Gate 公共参数与阈值的稳定 identity。"""

        verifier = self.model.gate_layer.verifier
        return {
            "n": verifier.n,
            "m": verifier.m,
            "error_threshold": verifier.error_threshold,
            "A_sha256": hashlib.sha256(
                verifier.A.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            "b_sha256": hashlib.sha256(
                verifier.b.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        }

    def _lwe_params_payload(self) -> Dict[str, object]:
        """返回可重建 Gate 的完整 LWE 参数（不包含 secret）。"""
        params = self.model.gate_layer.verifier.params
        return {
            "n": int(params.n),
            "m": int(params.m),
            "q": float(params.q),
            "sigma": float(params.sigma),
            "secret_bound": float(params.secret_bound),
            "error_threshold": float(params.error_threshold),
        }


__all__ = [
    "PretrainMetrics",
    "Phase5Trainer",
    "causal_distillation_loss",
    "count_non_padding_input_tokens",
    "configure_stage",
    "freeze_teacher",
    "masked_causal_lm_loss",
    "pretrain_go_no_go",
    "validate_mixed_routing",
]
