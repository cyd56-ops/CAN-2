"""Phase 5.5/T2 单四元组过拟合诊断工具。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..layers.gate_layer import ReasonCode
from ..training.data import CredentialGenerator
from .model import GatedDecoderTransformer
from .plain_model import PlainDecoderTransformer
from .t2_checkpoint import array_sha256
from .t2_data import T2_SCOPES, T2Example, generate_t2_split
from .t2_training import (
    _finish_step,
    _gate_observations,
    _prepare_t2_batch,
    _scope_observations,
    _validate_common,
)
from .training import masked_causal_lm_loss

T2_DIAGNOSTIC_PROTOCOL = "t2-single-quartet-overfit-v1"
T2_DIAGNOSTIC_VARIANTS = ("plain", "can_soft", "can_direct")


def materialize_t2_diagnostic_quartet(
    split_generator: Callable[..., List[T2Example]] = generate_t2_split,
) -> Tuple[T2Example, ...]:
    """仅物化固定 CAP/C0 train split 并选择诊断四元组。"""

    if not callable(split_generator):
        raise TypeError("split_generator 必须可调用")
    rows = split_generator(
        "t2_nl_cap",
        "train",
        20260903,
        {"train": 12, "dev": 4, "validation": 4, "test": 4},
        prompt_group="C0",
    )
    return select_first_t2_quartet(rows)


def select_first_t2_quartet(examples: Sequence[T2Example]) -> Tuple[T2Example, ...]:
    """从 train split 规范排序后选择第一个完整四元组。

    参数:
        examples: 同一 T2 train split 的样本。

    返回:
        按冻结 scope 顺序排列的四条样本。
    """

    if (
        isinstance(examples, (str, bytes))
        or not isinstance(examples, Sequence)
        or not examples
    ):
        raise ValueError("examples 必须是非空 T2Example 序列")
    if any(not isinstance(row, T2Example) for row in examples):
        raise TypeError("examples 只能包含 T2Example")
    if any(row.split != "train" for row in examples):
        raise ValueError("单四元组诊断只允许 train split")
    grouped: Dict[Tuple[str, str], Dict[str, T2Example]] = {}
    for row in examples:
        key = (row.source_id, row.prompt_template_id)
        scoped = grouped.setdefault(key, {})
        if row.scope in scoped:
            raise ValueError("source/template/scope 不能重复")
        scoped[row.scope] = row
    complete = [key for key, scoped in grouped.items() if set(scoped) == set(T2_SCOPES)]
    if not complete:
        raise ValueError("train split 不含完整 T2 四元组")
    key = sorted(complete)[0]
    quartet = tuple(grouped[key][scope] for scope in T2_SCOPES)
    if [row.credential_class for row in quartet].count("valid") != 2:
        raise RuntimeError("诊断四元组必须包含恰好两个 valid credential")
    return quartet


def compact_t2_evaluation(report: Mapping[str, Any]) -> Dict[str, Any]:
    """返回不含逐样本明文的 evaluator 摘要。"""

    if not isinstance(report, Mapping) or "diagnostics" not in report:
        raise ValueError("evaluator report 缺少 diagnostics")
    return {key: value for key, value in report.items() if key != "diagnostics"}


def _finite_tree(value: object) -> bool:
    """递归判断诊断结构中的所有数值是否有限。"""

    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite_tree(item) for item in value)
    return False


def _complete_training_observations(
    metrics: Mapping[str, Any], *, require_gate: bool
) -> bool:
    """验证成功判定所需 loss、token、gradient 与 Gate 观测均存在。"""

    scalar_fields = ("total_loss", "public_head_loss", "protected_head_loss")
    if any(field not in metrics for field in scalar_fields):
        return False
    scopes = metrics.get("scope_losses")
    answer_tokens = metrics.get("scope_answer_tokens")
    gradients = metrics.get("gradient_norms")
    if not isinstance(scopes, Mapping) or set(scopes) != set(T2_SCOPES):
        return False
    if not isinstance(answer_tokens, Mapping) or set(answer_tokens) != set(T2_SCOPES):
        return False
    if not isinstance(gradients, Mapping) or set(gradients) != {
        "shared_prefix",
        "protected_path",
        "public_path",
    }:
        return False
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in answer_tokens.values()
    ):
        return False
    gate = metrics.get("gate")
    if require_gate:
        if not isinstance(gate, Mapping) or set(gate) != {"valid", "invalid"}:
            return False
        for credential_class in ("valid", "invalid"):
            values = gate[credential_class]
            if not isinstance(values, Mapping) or values.get("count") != 2:
                return False
            if not isinstance(values.get("signal"), Mapping) or not isinstance(
                values.get("error_norm"), Mapping
            ):
                return False
    elif gate is not None:
        return False
    return _finite_tree(metrics)


def overfit_snapshot(
    report: Mapping[str, Any], train_metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    """按预注册门槛把一次 evaluator 报告转换为判定快照。"""

    try:
        by_scope = report["text_metrics"]["by_scope"]
        teacher = report["teacher_forced_by_scope"]
        exact = {scope: float(by_scope[scope]["exact_match"]) for scope in T2_SCOPES}
        f1 = {scope: float(by_scope[scope]["token_f1"]) for scope in T2_SCOPES}
        teacher_accuracy = {
            scope: float(teacher[scope]["token_accuracy"]) for scope in T2_SCOPES
        }
        invalid_sequences = int(report["generation_safety"]["invalid_sequences"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("T2 evaluator report 不满足诊断 schema") from exc
    routing = report.get("routing")
    routing_ok = True
    if report.get("model_kind") == "can":
        routing_ok = isinstance(routing, Mapping) and all(
            (
                routing.get("route_calls") == 4,
                routing.get("valid") == 2,
                routing.get("invalid") == 2,
                routing.get("protected_indices") == 2,
                routing.get("public_indices") == 2,
                routing.get("rejected_indices") == 0,
                routing.get("invalid_protected_block_calls") == 0,
            )
        )
    complete_observations = _complete_training_observations(
        train_metrics, require_gate=report.get("model_kind") == "can"
    )
    finite = complete_observations and _finite_tree(report)
    if not finite:
        raise FloatingPointError("诊断缺少完整观测或包含非有限值")
    if not routing_ok:
        raise RuntimeError("诊断 CAN routing 不满足真实授权/zero-call 约束")
    pass_once = (
        finite
        and invalid_sequences == 0
        and routing_ok
        and all(value == 1.0 for value in exact.values())
        and all(value == 1.0 for value in f1.values())
        and all(value == 1.0 for value in teacher_accuracy.values())
    )
    partial_once = (
        finite
        and all(value >= 0.95 for value in f1.values())
        and all(value >= 0.95 for value in teacher_accuracy.values())
    )
    return {
        "exact_match": exact,
        "token_f1": f1,
        "teacher_forced_token_accuracy": teacher_accuracy,
        "generation_safety_ok": invalid_sequences == 0,
        "routing_ok": routing_ok,
        "finite": finite,
        "complete_training_observations": complete_observations,
        "pass_once": pass_once,
        "partial_once": partial_once,
    }


def classify_t2_overfit_outcome(statuses: Mapping[str, str]) -> str:
    """按预注册跨变体决策表返回下一轮工程调查方向。"""

    if set(statuses) != set(T2_DIAGNOSTIC_VARIANTS):
        raise ValueError("statuses 必须完整包含三个诊断变体")
    allowed = {"passed", "partial_progress", "failed_to_overfit", "invalid_run"}
    if any(value not in allowed for value in statuses.values()):
        raise ValueError("statuses 包含未知诊断状态")
    if "invalid_run" in statuses.values():
        return "invalid_protocol"
    plain = statuses["plain"] == "passed"
    soft = statuses["can_soft"] == "passed"
    direct = statuses["can_direct"] == "passed"
    if not plain:
        return "investigate_plain_baseline"
    if soft and direct:
        return "investigate_multi_source_and_objective"
    if not soft and direct:
        return "investigate_soft_gate_semantics"
    if not soft and not direct:
        return "investigate_can_routing_and_credential"
    return "invalid_protocol"


@dataclass
class T2OverfitTracker:
    """跟踪连续成功、partial progress 和每 scope 首次 EM 时点。"""

    max_updates: int = 512
    required_streak: int = 3
    current_streak: int = 0
    first_pass_update: Optional[int] = None
    first_pass_tokens: Optional[int] = None
    passed_at_update: Optional[int] = None
    passed_at_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        """验证停止参数并初始化描述性状态。"""

        if any(
            type(value) is not int or value <= 0
            for value in (self.max_updates, self.required_streak)
        ):
            raise ValueError("max_updates 和 required_streak 必须为正整数")
        self.snapshots: List[Dict[str, Any]] = []
        self.first_scope_em_update: Dict[str, Optional[int]] = {
            scope: None for scope in T2_SCOPES
        }

    def observe(self, update: int, tokens: int, snapshot: Mapping[str, Any]) -> bool:
        """记录一个评估点；达到连续成功门槛时返回 True。"""

        if update <= 0 or tokens <= 0 or update > self.max_updates:
            raise ValueError("诊断 update/tokens 超出预注册范围")
        if self.snapshots and update <= int(self.snapshots[-1]["update"]):
            raise ValueError("诊断 update 必须严格递增")
        if update % 16 or (
            self.snapshots and update - self.snapshots[-1]["update"] != 16
        ):
            raise ValueError("评估点必须间隔 16 updates")
        if self.snapshots and tokens <= self.snapshots[-1]["tokens"]:
            raise ValueError("诊断 tokens 必须严格递增")
        if not _finite_tree(snapshot):
            raise FloatingPointError("诊断 snapshot 包含非有限值")
        entry = {"update": update, "tokens": tokens, **dict(snapshot)}
        self.snapshots.append(entry)
        exact = snapshot.get("exact_match")
        if not isinstance(exact, Mapping):
            raise ValueError("snapshot 缺少 exact_match")
        for scope in T2_SCOPES:
            if exact.get(scope) == 1.0 and self.first_scope_em_update[scope] is None:
                self.first_scope_em_update[scope] = update
        if snapshot.get("pass_once") is True:
            if self.first_pass_update is None:
                self.first_pass_update = update
                self.first_pass_tokens = tokens
            self.current_streak += 1
        else:
            self.current_streak = 0
        if self.current_streak >= self.required_streak:
            self.passed_at_update = update
            self.passed_at_tokens = tokens
            return True
        return False

    def status(self, completed_updates: int, invalid: bool = False) -> str:
        """按预注册优先级返回 passed/partial/failed/invalid 状态。"""

        if invalid:
            return "invalid_run"
        if self.passed_at_update is not None:
            return "passed"
        if completed_updates < self.max_updates:
            return "running"
        recent = self.snapshots[-self.required_streak :]
        if len(recent) == self.required_streak and all(
            row.get("partial_once") is True for row in recent
        ):
            return "partial_progress"
        return "failed_to_overfit"


class T2CanDirectPretrainer:
    """仅供诊断的 CAN trainer：真实授权后绕过 soft 幅度缩放。"""

    def __init__(
        self,
        model: GatedDecoderTransformer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        credential_generator: CredentialGenerator,
        protected_weight: float = 1.0,
        public_weight: float = 1.0,
    ) -> None:
        """初始化 direct 消融，并保留真实 Gate 与 credential 生成器。"""

        if not isinstance(model, GatedDecoderTransformer):
            raise TypeError("model 必须是 GatedDecoderTransformer")
        if not isinstance(credential_generator, CredentialGenerator):
            raise TypeError("credential_generator 必须是 CredentialGenerator")
        if not isinstance(device, torch.device):
            raise TypeError("device 必须是 torch.device")
        _validate_common(model, optimizer, device, protected_weight, public_weight)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.credential_generator = credential_generator
        self.protected_weight = float(protected_weight)
        self.public_weight = float(public_weight)
        self.global_step = 0

    def train_batch(self, batch: Mapping[str, object]) -> Dict[str, Any]:
        """使用真实 allow 索引训练一个 direct-protected 诊断 batch。"""

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
        credential = torch.tensor(credentials, dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)

        # 先执行真实 verifier/coordinator；标签只用于随后交叉验证，不能授予权限。
        prefix = self.model._forward_prefix(input_ids, attention_mask)
        _, decision = self.model.gate_layer(prefix.unsqueeze(-1), credential)
        if not torch.equal(decision.allow, masks.valid):
            raise RuntimeError("can_direct 的真实 Gate allow 与 T2 schema 不一致")
        expected_reason = torch.where(
            masks.valid,
            torch.full_like(decision.evidence.reason_code, int(ReasonCode.SUCCESS)),
            torch.full_like(
                decision.evidence.reason_code,
                int(ReasonCode.LWE_VERIFICATION_FAILED),
            ),
        )
        if not torch.equal(decision.evidence.reason_code, expected_reason):
            raise RuntimeError("can_direct 的 reason_code 与 T2 schema 不一致")
        valid_indices = torch.nonzero(decision.allow, as_tuple=False).flatten()
        linked_zero = prefix.sum(dim=2, keepdim=True) * 0.0
        protected_logits = linked_zero.expand(
            -1, -1, self.model.config.vocab_size
        ).clone()
        if valid_indices.numel() > 0:
            protected_values = self.model._forward_protected(
                prefix.index_select(0, valid_indices),
                attention_mask.index_select(0, valid_indices),
            )
            protected_logits = protected_logits.index_copy(
                0, valid_indices, protected_values
            )
        public_logits = self.model._forward_public(prefix)
        protected_loss = masked_causal_lm_loss(
            protected_logits, labels, masks.protected
        )
        public_loss = masked_causal_lm_loss(public_logits, labels, masks.public)
        loss = self.protected_weight * protected_loss + self.public_weight * public_loss
        scope_losses, scope_tokens = _scope_observations(
            protected_logits, public_logits, labels, scopes
        )
        observations = {
            "protected_head_loss": float(protected_loss.detach().item()),
            "public_head_loss": float(public_loss.detach().item()),
            "scope_losses": scope_losses,
            "scope_answer_tokens": scope_tokens,
            "gate": _gate_observations(decision, masks),
            "direct_routing": {
                "valid": int(valid_indices.numel()),
                "invalid": int(masks.invalid.sum().item()),
                "protected_rows_executed": int(valid_indices.numel()),
                "invalid_protected_block_calls": 0,
            },
        }
        metrics = _finish_step(
            self.model,
            self.optimizer,
            loss,
            attention_mask,
            self.global_step,
            observations=observations,
            collect_diagnostics=True,
        )
        self.global_step += 1
        return metrics


def save_t2_diagnostic_checkpoint(
    path: Path,
    model: nn.Module,
    variant: str,
    update: int,
    tokens: int,
) -> None:
    """保存不含 secret、credential 与 RNG 状态的诊断权重。"""

    if variant not in set(T2_DIAGNOSTIC_VARIANTS):
        raise ValueError("variant 不受支持")
    if path.exists():
        raise FileExistsError("诊断 checkpoint 已存在")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("gate_layer.verifier.")
    }
    public_identity: Optional[Dict[str, str]] = None
    if isinstance(model, GatedDecoderTransformer):
        verifier = model.gate_layer.verifier
        public_identity = {
            "A_sha256": array_sha256(verifier.A.detach().cpu().numpy()),
            "b_sha256": array_sha256(verifier.b.detach().cpu().numpy()),
        }
    payload = {
        "schema_version": 1,
        "diagnostic_only": True,
        "protocol": T2_DIAGNOSTIC_PROTOCOL,
        "seed": 20260903,
        "keypair_seed_offset": 10000,
        "variant": variant,
        "update": update,
        "tokens": tokens,
        "model_config": asdict(model.config),
        "model_state_without_lwe_public": state,
        "lwe_public_identity": public_identity,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "T2CanDirectPretrainer",
    "T2OverfitTracker",
    "T2_DIAGNOSTIC_PROTOCOL",
    "T2_DIAGNOSTIC_VARIANTS",
    "compact_t2_evaluation",
    "classify_t2_overfit_outcome",
    "materialize_t2_diagnostic_quartet",
    "overfit_snapshot",
    "save_t2_diagnostic_checkpoint",
    "select_first_t2_quartet",
]
