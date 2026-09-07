"""Phase 5.5/T2 Plain/CAN 的统一生成与 teacher-forced evaluator。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..layers.gate_layer import ReasonCode
from ..training.data import CredentialGenerator
from .model import GatedDecoderTransformer
from .plain_model import PlainDecoderTransformer
from .t2_data import T2_SCOPES, T2Example
from .t2_metrics import T2Prediction, evaluate_t2_predictions
from .tokenizer import ByteTokenizer

T2Model = Union[GatedDecoderTransformer, PlainDecoderTransformer]


def _validate_examples(examples: Sequence[T2Example]) -> Tuple[str, str]:
    """验证评估样本来自同一 suite/split，并返回其身份。"""

    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise TypeError("examples 必须是 T2Example 序列")
    if not examples or any(not isinstance(row, T2Example) for row in examples):
        raise ValueError("examples 必须是非空 T2Example 序列")
    suites = {row.suite_id for row in examples}
    splits = {row.split for row in examples}
    if len(suites) != 1 or len(splits) != 1:
        raise ValueError("一次评估只能包含一个 suite 和一个 split")
    if len({row.sample_id for row in examples}) != len(examples):
        raise ValueError("sample_id 不能重复")
    return next(iter(suites)), next(iter(splits))


def _pad_prompts(
    rows: Sequence[T2Example],
    tokenizer: ByteTokenizer,
    device: torch.device,
    max_length: int,
) -> Tuple[Tensor, Tensor, List[int]]:
    """编码并右侧填充一批 prompt，返回张量、mask 和原长度。"""

    encoded = [
        tokenizer.encode(row.prompt, add_bos=True, add_eos=False, max_length=max_length)
        for row in rows
    ]
    lengths = [len(values) for values in encoded]
    width = max(lengths)
    input_ids = torch.full(
        (len(rows), width),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, values in enumerate(encoded):
        input_ids[index, : len(values)] = torch.tensor(
            values, dtype=torch.long, device=device
        )
        attention_mask[index, : len(values)] = True
    return input_ids, attention_mask, lengths


def _continuation(
    sequence: Sequence[int], prompt_length: int, eos_token_id: int
) -> Tuple[List[int], bool]:
    """提取生成 continuation，并报告是否观察到 EOS。"""

    values = list(sequence[prompt_length:])
    if eos_token_id in values:
        values = values[: values.index(eos_token_id)]
        return values, True
    return values, False


def _first_divergence(generated: Sequence[int], target: Sequence[int]) -> Optional[int]:
    """返回生成 token 与目标 token 的首个分叉位置。"""

    for index, (left, right) in enumerate(zip(generated, target)):
        if left != right:
            return index
    if len(generated) != len(target):
        return min(len(generated), len(target))
    return None


class T2Evaluator:
    """对 T2 Plain/CAN 执行统一、可审计且 fail-closed 的评估。"""

    def __init__(
        self,
        model: T2Model,
        tokenizer: ByteTokenizer,
        device: torch.device,
        credential_generator: Optional[CredentialGenerator] = None,
        max_new_tokens: int = 32,
        cache_mode: str = "none",
        batch_size: int = 16,
    ) -> None:
        """初始化 evaluator，并锁定模型种类与 credential 边界。"""

        if not isinstance(model, (GatedDecoderTransformer, PlainDecoderTransformer)):
            raise TypeError(
                "model 必须是 PlainDecoderTransformer 或 GatedDecoderTransformer"
            )
        if not isinstance(tokenizer, ByteTokenizer):
            raise TypeError("tokenizer 必须是 ByteTokenizer")
        if not isinstance(device, torch.device):
            raise TypeError("device 必须是 torch.device")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens 必须是整数")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size 必须是整数")
        if batch_size < 4 or batch_size % 4 != 0:
            raise ValueError("T2 evaluator batch_size 必须至少为 4 且能被 4 整除")
        if cache_mode not in {"none", "kv"}:
            raise ValueError("cache_mode 必须为 none 或 kv")
        is_can = isinstance(model, GatedDecoderTransformer)
        if is_can and not isinstance(credential_generator, CredentialGenerator):
            raise ValueError("CAN 完整评估必须提供 CredentialGenerator")
        if not is_can and credential_generator is not None:
            raise ValueError("Plain evaluator 禁止接收 credential")
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.credential_generator = credential_generator
        self.max_new_tokens = max_new_tokens
        self.cache_mode = cache_mode
        self.batch_size = batch_size
        self.model_kind = "can" if is_can else "plain"

    def evaluate(self, examples: Sequence[T2Example]) -> Dict[str, Any]:
        """生成逐样本输出并汇总文本、访问、token 和路由指标。"""

        suite_id, split = _validate_examples(examples)
        predictions: List[T2Prediction] = []
        diagnostics: List[Dict[str, Any]] = []
        route_totals = {
            "route_calls": 0,
            "valid": 0,
            "invalid": 0,
            "protected_indices": 0,
            "public_indices": 0,
            "rejected_indices": 0,
            "invalid_protected_block_calls": 0,
        }
        # 每个 chunk 保持四元组边界，CAN 在一次 route 中覆盖 valid/invalid 两侧。
        for start in range(0, len(examples), self.batch_size):
            rows = list(examples[start : start + self.batch_size])
            if len(rows) % 4 != 0:
                raise ValueError("评估 batch 必须保持完整 T2 四元组")
            generated, batch_route = self._generate_batch(rows)
            for key, value in batch_route.items():
                route_totals[key] += value
            for row, sequence, stop_reason, prompt_length in generated:
                continuation, saw_eos = _continuation(
                    sequence, prompt_length, self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(continuation)
                target_tokens = self.tokenizer.encode(
                    row.target, add_bos=False, add_eos=False
                )
                token_loss, token_accuracy = self._teacher_forced(row)
                predictions.append(T2Prediction(row.sample_id, generated_text))
                diagnostics.append(
                    {
                        "sample_id": row.sample_id,
                        "scope": row.scope,
                        "credential_class": row.credential_class,
                        "head": (
                            "protected"
                            if row.scope in {"protected_public", "protected_private"}
                            else "public"
                        ),
                        "generated_text": generated_text,
                        "target": row.target,
                        "first_divergence_position": _first_divergence(
                            continuation, target_tokens
                        ),
                        "teacher_forced_token_loss": token_loss,
                        "teacher_forced_token_accuracy": token_accuracy,
                        "stop_reason": stop_reason,
                        "saw_eos": saw_eos,
                    }
                )
        report = evaluate_t2_predictions(examples, predictions)
        scope_token_metrics = self._scope_token_metrics(diagnostics)
        return {
            "schema_version": 1,
            "status": "ok",
            "model_kind": self.model_kind,
            "suite_id": suite_id,
            "split": split,
            "route_mode": (
                "credential_gate" if self.model_kind == "can" else "oracle_head"
            ),
            "gate_or_credential": self.model_kind == "can",
            "text_metrics": asdict(report),
            "teacher_forced_by_scope": scope_token_metrics,
            "routing": route_totals if self.model_kind == "can" else None,
            "diagnostics": diagnostics,
        }

    def _generate_batch(
        self, rows: Sequence[T2Example]
    ) -> Tuple[List[Tuple[T2Example, Sequence[int], str, int]], Dict[str, int]]:
        """执行一批 Plain oracle-head 或 CAN credential-gated 生成。"""

        input_ids, mask, lengths = _pad_prompts(
            rows, self.tokenizer, self.device, self.model.config.max_seq_len
        )
        empty_route = {
            "route_calls": 0,
            "valid": 0,
            "invalid": 0,
            "protected_indices": 0,
            "public_indices": 0,
            "rejected_indices": 0,
            "invalid_protected_block_calls": 0,
        }
        if self.model_kind == "can":
            assert isinstance(self.model, GatedDecoderTransformer)
            assert self.credential_generator is not None
            valid = torch.tensor(
                [row.credential_class == "valid" for row in rows],
                dtype=torch.bool,
                device=self.device,
            )
            credentials = np.stack(
                [
                    self.credential_generator.generate(bool(value))
                    for value in valid.tolist()
                ]
            )
            output = self.model.generate(
                input_ids,
                torch.tensor(credentials, dtype=torch.float32, device=self.device),
                mask,
                max_new_tokens=self.max_new_tokens,
                cache_mode=self.cache_mode,
            )
            expected_reason = torch.where(
                valid,
                torch.full_like(
                    output.decision.evidence.reason_code, int(ReasonCode.SUCCESS)
                ),
                torch.full_like(
                    output.decision.evidence.reason_code,
                    int(ReasonCode.LWE_VERIFICATION_FAILED),
                ),
            )
            if not torch.equal(output.decision.allow, valid):
                raise RuntimeError("CAN evaluator 的 Gate 判决与 scope 不一致")
            if not torch.equal(output.decision.evidence.reason_code, expected_reason):
                raise RuntimeError("CAN evaluator 的 reason_code 与 scope 不一致")
            if not bool((output.route_call_count == 1).all().item()):
                raise RuntimeError("CAN 每条序列必须且只能提交一次 route")
            protected = set(torch.nonzero(valid, as_tuple=False).flatten().tolist())
            public = set(torch.nonzero(~valid, as_tuple=False).flatten().tolist())
            if protected & public or protected | public != set(range(len(rows))):
                raise RuntimeError("CAN mixed routing 索引未互斥完整覆盖")
            route = {
                "route_calls": int(output.route_call_count.sum().item()),
                "valid": len(protected),
                "invalid": len(public),
                "protected_indices": len(protected),
                "public_indices": len(public),
                "rejected_indices": 0,
                "invalid_protected_block_calls": 0,
            }
            return [
                (
                    row,
                    output.token_ids[index],
                    output.stop_reasons[index],
                    lengths[index],
                )
                for index, row in enumerate(rows)
            ], route

        assert isinstance(self.model, PlainDecoderTransformer)
        collected: Dict[int, Tuple[Sequence[int], str]] = {}
        for head in ("public", "protected"):
            indices = [
                index
                for index, row in enumerate(rows)
                if (row.scope in {"protected_public", "protected_private"})
                == (head == "protected")
            ]
            if not indices:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
            output = self.model.generate(
                input_ids.index_select(0, index_tensor),
                head,
                mask.index_select(0, index_tensor),
                max_new_tokens=self.max_new_tokens,
                cache_mode=self.cache_mode,
            )
            for local, original in enumerate(indices):
                collected[original] = (
                    output.token_ids[local],
                    output.stop_reasons[local],
                )
        if set(collected) != set(range(len(rows))):
            raise RuntimeError("Plain oracle-head 索引未完整覆盖")
        return [
            (row, collected[index][0], collected[index][1], lengths[index])
            for index, row in enumerate(rows)
        ], empty_route

    @torch.inference_mode()
    def _teacher_forced(self, row: T2Example) -> Tuple[float, float]:
        """计算单条答案的 teacher-forced token loss 与准确率。"""

        prompt = self.tokenizer.encode(
            row.prompt,
            add_bos=True,
            add_eos=False,
            max_length=self.model.config.max_seq_len,
        )
        target = self.tokenizer.encode(
            row.target,
            add_bos=False,
            add_eos=True,
            max_length=self.model.config.max_seq_len,
        )
        if len(prompt) + len(target) > self.model.config.max_seq_len:
            raise ValueError("prompt + target 超过模型 max_seq_len")
        values = torch.tensor([prompt + target], dtype=torch.long, device=self.device)
        mask = torch.ones_like(values, dtype=torch.bool)
        protected = row.scope in {"protected_public", "protected_private"}
        if isinstance(self.model, GatedDecoderTransformer):
            logits = (
                self.model.direct_protected_logits(values, mask)
                if protected
                else self.model.direct_public_logits(values, mask)
            )
        else:
            logits = self.model.logits(
                values, "protected" if protected else "public", mask
            )
        answer_logits = logits[0, len(prompt) - 1 : -1]
        labels = torch.tensor(target, dtype=torch.long, device=self.device)
        loss = float(F.cross_entropy(answer_logits, labels).item())
        accuracy = float((answer_logits.argmax(dim=-1) == labels).float().mean().item())
        if not np.isfinite(loss) or not np.isfinite(accuracy):
            raise FloatingPointError("T2 evaluator 出现非有限 teacher-forced 指标")
        return loss, accuracy

    @staticmethod
    def _scope_token_metrics(
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """按 scope 汇总逐样本 teacher-forced 指标。"""

        result: Dict[str, Dict[str, Any]] = {}
        for scope in T2_SCOPES:
            rows = [row for row in diagnostics if row["scope"] == scope]
            if not rows:
                result[scope] = {
                    "status": "not_applicable",
                    "token_loss": None,
                    "token_accuracy": None,
                    "total_sequences": 0,
                }
                continue
            result[scope] = {
                "status": "ok",
                "token_loss": float(
                    np.mean([row["teacher_forced_token_loss"] for row in rows])
                ),
                "token_accuracy": float(
                    np.mean([row["teacher_forced_token_accuracy"] for row in rows])
                ),
                "total_sequences": len(rows),
            }
        return result


__all__ = ["T2Evaluator"]
