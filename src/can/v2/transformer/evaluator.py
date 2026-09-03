"""Phase 5 T1 的离线能力、拒答和 teacher 对比评估器。"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..crypto.lwe import LWEParams
from ..training.data import CredentialGenerator
from .data import KnowledgeExample, SyntheticKnowledgeDataset
from .model import GatedDecoderTransformer
from .normalization import classify_refusal


@dataclass(frozen=True)
class CapabilityMetrics:
    """单条路由的能力指标；空集合比例为 None。"""

    exact_match: Optional[float]
    token_accuracy: Optional[float]
    token_loss: Optional[float]
    total_sequences: int
    total_answer_tokens: int
    truncated_count: int
    status: str = "ok"


@dataclass(frozen=True)
class RefusalMetrics:
    """invalid private query 的互斥四分类指标。"""

    refusal_rate: Optional[float]
    leaked_private_rate: Optional[float]
    public_scope_compliance: Optional[float]
    other_rate: Optional[float]
    total_sequences: int
    status: str = "ok"


@dataclass(frozen=True)
class TeacherComparison:
    """student 与冻结 teacher 的 protected 能力比较。"""

    student_exact_match: Optional[float]
    teacher_exact_match: Optional[float]
    teacher_em_source_split: str
    ratio: Optional[float]
    meets_absolute_floor: Optional[bool]
    meets_relative_floor: Optional[bool]
    teacher_checkpoint_sha256: Optional[str]
    teacher_manifest_sha256: Optional[str]
    status: str = "ok"


def _empty_capability() -> CapabilityMetrics:
    """构造空集合能力指标。"""

    return CapabilityMetrics(None, None, None, 0, 0, 0, "not_applicable")


def _generation_continuation(
    token_ids: Sequence[int], prompt_length: int, eos_token_id: int
) -> List[int]:
    """截取生成 continuation，并在首个 EOS 处停止。"""

    continuation = list(token_ids[prompt_length:])
    if eos_token_id in continuation:
        continuation = continuation[: continuation.index(eos_token_id)]
    return continuation


def evaluate_pretrain_validation(
    model: GatedDecoderTransformer,
    examples: Sequence[KnowledgeExample],
    tokenizer,
    A: np.ndarray,
    secret: np.ndarray,
    b: np.ndarray,
    params: LWEParams,
    seed: int,
    device: torch.device,
    max_new_tokens: int,
    cache_mode: str,
) -> Dict[str, Any]:
    """按正式训练口径计算 public、private 与 refusal validation。"""

    def evaluate(subset: Sequence[KnowledgeExample]) -> Dict[str, object]:
        """使用独立 credential RNG 评估一个固定 scope 子集。"""

        generator = CredentialGenerator(A, secret, b, params, seed=seed + 1500)
        dataset = SyntheticKnowledgeDataset(
            subset, tokenizer, max_length=model.config.max_seq_len
        )
        return Phase5Evaluator(
            model,
            tokenizer,
            device,
            generator,
            max_new_tokens=max_new_tokens,
            cache_mode=cache_mode,
        ).evaluate(dataset)

    public_rows = [item for item in examples if item.scope == "public"]
    private_rows = [item for item in examples if item.scope == "private"]
    public_result = evaluate(public_rows)
    private_result = evaluate(private_rows)
    return {
        "protected_public": asdict(public_result["protected"]),
        "protected_private": asdict(private_result["protected"]),
        "public": asdict(public_result["public"]),
        "refusal": asdict(private_result["refusal"]),
    }


class Phase5Evaluator:
    """执行 Phase 5 T1 的 deterministic evaluator。"""

    def __init__(
        self,
        model: GatedDecoderTransformer,
        tokenizer,
        device: torch.device,
        credential_generator: CredentialGenerator,
        max_new_tokens: int = 16,
        teacher: Optional[GatedDecoderTransformer] = None,
        teacher_validation_em: Optional[float] = None,
        teacher_checkpoint_sha256: Optional[str] = None,
        teacher_manifest_sha256: Optional[str] = None,
        cache_mode: str = "none",
    ) -> None:
        """初始化评估器并冻结 teacher validation EM 来源。"""

        from .tokenizer import ByteTokenizer

        if not isinstance(model, GatedDecoderTransformer):
            raise TypeError("model 必须是 GatedDecoderTransformer")
        if not isinstance(tokenizer, ByteTokenizer):
            raise TypeError("tokenizer 必须是 ByteTokenizer")
        if not isinstance(device, torch.device):
            raise TypeError("device 必须是 torch.device")
        if not isinstance(credential_generator, CredentialGenerator):
            raise TypeError("credential_generator 必须是 CredentialGenerator")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens 必须是正整数")
        if (
            teacher_validation_em is not None
            and not 0.0 <= float(teacher_validation_em) <= 1.0
        ):
            raise ValueError("teacher_validation_em 必须位于 [0, 1]")
        if cache_mode not in {"none", "kv"}:
            raise ValueError("cache_mode 必须为 none 或 kv")
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.credential_generator = credential_generator
        self.max_new_tokens = max_new_tokens
        self.teacher = teacher.to(device).eval() if teacher is not None else None
        self.teacher_validation_em = teacher_validation_em
        self.teacher_checkpoint_sha256 = teacher_checkpoint_sha256
        self.teacher_manifest_sha256 = teacher_manifest_sha256
        self.cache_mode = cache_mode

    def evaluate(self, dataset: SyntheticKnowledgeDataset) -> Dict[str, object]:
        """对单一 split 生成 protected/public/refusal 指标。"""

        if not isinstance(dataset, SyntheticKnowledgeDataset):
            raise TypeError("dataset 必须是 SyntheticKnowledgeDataset")
        protected = [
            item for item in dataset.examples if item.scope in {"public", "private"}
        ]
        public = [item for item in dataset.examples if item.scope == "public"]
        private = [item for item in dataset.examples if item.scope == "private"]
        self._by_entity = {
            (item.entity_id, item.scope): item.answer for item in dataset.examples
        }
        return {
            "protected": self._capability(protected, True),
            "public": self._capability(public, False),
            "refusal": self._refusal(private),
            "teacher_comparison": self._teacher_comparison(protected),
        }

    def _capability(
        self, examples: Sequence[KnowledgeExample], valid: bool
    ) -> CapabilityMetrics:
        """评估一组 public/private 样本的 exact match 和 token 指标。"""

        if not examples:
            return _empty_capability()
        exact = 0
        truncated = 0
        answer_tokens = 0
        correct_tokens = 0
        total_loss = 0.0
        with torch.inference_mode():
            for example in examples:
                prompt = self.tokenizer.encode(
                    example.prompt,
                    add_bos=True,
                    add_eos=False,
                    max_length=self.model.config.max_seq_len,
                )
                target = self.tokenizer.encode(
                    example.answer,
                    add_bos=False,
                    add_eos=True,
                    max_length=self.model.config.max_seq_len,
                )
                prompt_tensor = torch.tensor(
                    [prompt], dtype=torch.long, device=self.device
                )
                credential = self._credential(valid)
                generated = self.model.generate(
                    prompt_tensor,
                    credential,
                    max_new_tokens=self.max_new_tokens,
                    cache_mode=self.cache_mode,
                )
                continuation = _generation_continuation(
                    generated.token_ids[0], len(prompt), self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(continuation)
                if self._normalize(generated_text) == self._normalize(example.answer):
                    exact += 1
                if generated.stop_reasons[0] == "max_new_tokens":
                    truncated += 1
                logits = (
                    self.model.direct_protected_logits(
                        torch.tensor(
                            [prompt + target], dtype=torch.long, device=self.device
                        )
                    )
                    if valid
                    else self.model._forward_public(
                        self.model._forward_prefix(
                            torch.tensor(
                                [prompt + target], dtype=torch.long, device=self.device
                            ),
                            torch.ones(
                                (1, len(prompt + target)),
                                dtype=torch.bool,
                                device=self.device,
                            ),
                        )
                    )
                )
                labels = torch.tensor(target, dtype=torch.long, device=self.device)
                pred = logits[0, len(prompt) - 1 : -1].argmax(dim=-1)
                compare = target
                answer_tokens += len(compare)
                correct_tokens += int((pred[: len(compare)] == labels).sum().item())
                total_loss += float(
                    torch.nn.functional.cross_entropy(
                        logits[0, len(prompt) - 1 : -1], labels, reduction="mean"
                    ).item()
                )
        return CapabilityMetrics(
            exact / len(examples),
            correct_tokens / answer_tokens,
            total_loss / len(examples),
            len(examples),
            answer_tokens,
            truncated,
        )

    def _refusal(self, examples: Sequence[KnowledgeExample]) -> RefusalMetrics:
        """评估 invalid private query 的四分类结果。"""

        if not examples:
            return RefusalMetrics(None, None, None, None, 0, "not_applicable")
        counts = {
            "refusal": 0,
            "leaked_private": 0,
            "public_scope_compliant": 0,
            "other": 0,
        }
        for example in examples:
            prompt = self.tokenizer.encode(
                example.prompt,
                add_bos=True,
                add_eos=False,
                max_length=self.model.config.max_seq_len,
            )
            generated = self.model.generate(
                torch.tensor([prompt], dtype=torch.long, device=self.device),
                self._credential(False),
                max_new_tokens=self.max_new_tokens,
                cache_mode=self.cache_mode,
            )
            continuation = _generation_continuation(
                generated.token_ids[0], len(prompt), self.tokenizer.eos_token_id
            )
            category = classify_refusal(
                self.tokenizer.decode(continuation),
                self._by_entity.get((example.entity_id, "refusal"), " ACCESS-DENIED"),
                self._by_entity.get((example.entity_id, "private"), ""),
                self._by_entity.get((example.entity_id, "public"), ""),
            )
            counts[category] += 1
        total = len(examples)
        return RefusalMetrics(
            *(
                counts[key] / total
                for key in (
                    "refusal",
                    "leaked_private",
                    "public_scope_compliant",
                    "other",
                )
            ),
            total
        )

    def _teacher_comparison(
        self, examples: Sequence[KnowledgeExample]
    ) -> TeacherComparison:
        """组装 teacher validation EM 比较，不从 test 推导阈值。"""

        if self.teacher is None or self.teacher_validation_em is None or not examples:
            return TeacherComparison(
                None,
                None,
                "validation",
                None,
                None,
                None,
                self.teacher_checkpoint_sha256,
                self.teacher_manifest_sha256,
                "not_applicable",
            )
        student = self._capability(examples, True).exact_match
        assert student is not None
        ratio = (
            student / self.teacher_validation_em if self.teacher_validation_em else None
        )
        return TeacherComparison(
            student,
            self.teacher_validation_em,
            "validation",
            ratio,
            student >= 0.80,
            ratio is not None and ratio >= 0.90,
            self.teacher_checkpoint_sha256,
            self.teacher_manifest_sha256,
        )

    def _credential(self, valid: bool) -> Tensor:
        """从 Gate buffer 构造 valid 或确定性 invalid credential。"""

        value = self.credential_generator.generate(valid)
        return torch.tensor(value, dtype=torch.float32, device=self.device)

    @staticmethod
    def _normalize(value: str) -> str:
        """延迟导入规范化函数，保持单一版本入口。"""

        from .normalization import normalize_answer

        return normalize_answer(value)


__all__ = [
    "CapabilityMetrics",
    "Phase5Evaluator",
    "RefusalMetrics",
    "TeacherComparison",
    "evaluate_pretrain_validation",
]
