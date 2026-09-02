"""Phase 5 合成知识数据与因果语言模型 batch 工具。"""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class KnowledgeExample:
    """描述一条版本化的公开、私有或拒答样本。"""

    sample_id: str
    scope: str
    entity_id: str
    prompt_type: str
    prompt: str
    answer: str
    split: str
    seed: int
    generator_version: str = "phase5-t1-private-query-v2"


def generate_synthetic_corpus(
    seed: int,
    train_entities: int = 24,
    validation_entities: int = 8,
    test_entities: int = 8,
) -> Dict[str, List[KnowledgeExample]]:
    """生成实体互斥的 public/private/refusal 合成语料。

    参数:
        seed: 控制实体和值生成的显式随机种子。
        train_entities: 训练 split 的实体数量。
        validation_entities: 验证 split 的实体数量。
        test_entities: 测试 split 的实体数量。

    返回:
        以 ``train``、``validation``、``test`` 为键的样本字典。
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed 必须是整数")
    counts = {
        "train": train_entities,
        "validation": validation_entities,
        "test": test_entities,
    }
    for name, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{name}_entities 必须是整数")
        if count <= 0:
            raise ValueError(f"{name}_entities 必须大于 0")

    rng = np.random.default_rng(seed)
    corpus: Dict[str, List[KnowledgeExample]] = {}
    offset = 0
    for split, count in counts.items():
        examples: List[KnowledgeExample] = []
        for local_index in range(count):
            entity_id = f"entity-{offset + local_index:05d}"
            public_value = int(rng.integers(1000, 9999))
            private_value = int(rng.integers(100000, 999999))
            public_prompt = (
                f"Record: {entity_id} public code is PUBLIC-{public_value}. "
                f"Question: What is the public code for {entity_id}? Answer:"
            )
            # 私有答案只作为监督 target，不放入 prompt，避免模型通过复制上下文作弊。
            private_prompt = (
                f"Question: What is the private code for {entity_id}? Answer:"
            )
            examples.extend(
                [
                    KnowledgeExample(
                        sample_id=f"{split}-{entity_id}-public",
                        scope="public",
                        entity_id=entity_id,
                        prompt_type="code_lookup",
                        prompt=public_prompt,
                        answer=f" PUBLIC-{public_value}",
                        split=split,
                        seed=seed,
                    ),
                    KnowledgeExample(
                        sample_id=f"{split}-{entity_id}-private",
                        scope="private",
                        entity_id=entity_id,
                        prompt_type="code_lookup",
                        prompt=private_prompt,
                        answer=f" PRIVATE-{private_value}",
                        split=split,
                        seed=seed,
                    ),
                    KnowledgeExample(
                        sample_id=f"{split}-{entity_id}-refusal",
                        scope="refusal",
                        entity_id=entity_id,
                        prompt_type="code_lookup",
                        prompt=private_prompt,
                        answer=" ACCESS-DENIED",
                        split=split,
                        seed=seed,
                    ),
                ]
            )
        corpus[split] = examples
        offset += count
    _validate_entity_disjoint(corpus)
    return corpus


def _validate_entity_disjoint(corpus: Mapping[str, Sequence[KnowledgeExample]]) -> None:
    """验证三个 split 的实体集合完全不重叠。"""

    expected = {"train", "validation", "test"}
    if set(corpus) != expected:
        raise ValueError("corpus 必须恰好包含 train/validation/test")
    entity_sets = {
        split: {example.entity_id for example in examples}
        for split, examples in corpus.items()
    }
    if entity_sets["train"] & entity_sets["validation"]:
        raise ValueError("train 与 validation 实体发生重叠")
    if entity_sets["train"] & entity_sets["test"]:
        raise ValueError("train 与 test 实体发生重叠")
    if entity_sets["validation"] & entity_sets["test"]:
        raise ValueError("validation 与 test 实体发生重叠")


class SyntheticKnowledgeDataset(Dataset):
    """把合成知识样本转换为 teacher-forced 因果 LM 样本。"""

    def __init__(
        self,
        examples: Sequence[KnowledgeExample],
        tokenizer: ByteTokenizer,
        max_length: int = 256,
    ) -> None:
        """初始化数据集并验证样本 ID 和长度。

        参数:
            examples: 同一 split 的知识样本。
            tokenizer: 固定 byte-level tokenizer。
            max_length: prompt 与答案合并后的最大长度。
        """

        if not isinstance(tokenizer, ByteTokenizer):
            raise TypeError("tokenizer 必须是 ByteTokenizer")
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length 必须是整数")
        if max_length <= 1:
            raise ValueError("max_length 必须大于 1")
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("examples 不能为空")
        if any(not isinstance(item, KnowledgeExample) for item in self.examples):
            raise TypeError("examples 必须只包含 KnowledgeExample")
        sample_ids = [item.sample_id for item in self.examples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_id 不能重复")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """返回样本数量。"""

        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """返回一条带 prompt mask 的因果 LM 样本。"""

        example = self.examples[index]
        prompt_ids = self.tokenizer.encode(
            example.prompt, add_bos=True, add_eos=False, max_length=self.max_length
        )
        answer_ids = self.tokenizer.encode(
            example.answer, add_bos=False, add_eos=True, max_length=self.max_length
        )
        token_ids = prompt_ids + answer_ids
        if len(token_ids) > self.max_length:
            raise ValueError(f"样本 {example.sample_id} 超过 max_length")
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "scope": example.scope,
            "sample_id": example.sample_id,
        }


class EntityTripletBatchSampler(Sampler[List[int]]):
    """按完整 entity triplet 组成满足 2-valid/1-invalid 下限的 batch。"""

    def __init__(
        self,
        examples: Sequence[KnowledgeExample],
        batch_size: int,
        seed: int,
    ) -> None:
        """初始化确定性实体 batch sampler。

        参数:
            examples: 每个实体恰含 public/private/refusal 三条记录的序列。
            batch_size: 必须是至少 6 且能被 3 整除的 batch 大小。
            seed: 控制实体顺序的显式 seed。
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size 必须是整数")
        if batch_size < 6 or batch_size % 3 != 0:
            raise ValueError("batch_size 必须至少为 6 且能被 3 整除")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed 必须是整数")
        by_entity: Dict[str, List[int]] = {}
        scopes_by_entity: Dict[str, set] = {}
        for index, example in enumerate(examples):
            if not isinstance(example, KnowledgeExample):
                raise TypeError("examples 必须只包含 KnowledgeExample")
            by_entity.setdefault(example.entity_id, []).append(index)
            scopes_by_entity.setdefault(example.entity_id, set()).add(example.scope)
        required_scopes = {"public", "private", "refusal"}
        if any(
            len(by_entity[entity]) != 3 or scopes_by_entity[entity] != required_scopes
            for entity in by_entity
        ):
            raise ValueError("每个实体必须恰含 public/private/refusal 三条记录")
        self.entity_rows = list(by_entity.values())
        self.entities_per_batch = batch_size // 3
        if len(self.entity_rows) < self.entities_per_batch:
            raise ValueError("实体数量不足以形成一个完整 mixed batch")
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        """按当前 epoch seed 产生完整 entity-triplet batch。"""

        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.entity_rows)).tolist()
        usable = len(order) - len(order) % self.entities_per_batch
        for start in range(0, usable, self.entities_per_batch):
            rows: List[int] = []
            for entity_index in order[start : start + self.entities_per_batch]:
                rows.extend(self.entity_rows[entity_index])
            yield rows

    def __len__(self) -> int:
        """返回丢弃不完整 entity group 后的 batch 数。"""

        return len(self.entity_rows) // self.entities_per_batch

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch，以便恢复时确定性重建实体顺序。"""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch 必须是非负整数")
        self.epoch = epoch


def collate_causal_lm_batch(
    items: Sequence[Mapping[str, object]], pad_token_id: int = 258
) -> Dict[str, object]:
    """右侧填充因果 LM 样本并生成 attention mask。"""

    if not items:
        raise ValueError("items 不能为空")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise TypeError("pad_token_id 必须是整数")
    validated: List[tuple[Tensor, Tensor, str, str]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("每个 collate item 必须是 Mapping")
        tokens = item.get("input_ids")
        targets = item.get("labels")
        scope = item.get("scope")
        sample_id = item.get("sample_id")
        if not isinstance(tokens, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("input_ids 和 labels 必须是 Tensor")
        if tokens.ndim != 1 or targets.shape != tokens.shape:
            raise ValueError("input_ids 和 labels 必须是对齐的一维 Tensor")
        if tokens.dtype != torch.long or targets.dtype != torch.long:
            raise TypeError("input_ids 和 labels 必须是 torch.long")
        if not isinstance(scope, str) or not isinstance(sample_id, str):
            raise TypeError("scope 和 sample_id 必须是 str")
        validated.append((tokens, targets, scope, sample_id))
    max_length = max(int(tokens.shape[0]) for tokens, _, _, _ in validated)
    batch_size = len(items)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    scopes: List[str] = []
    sample_ids: List[str] = []
    for row, (tokens, targets, scope, sample_id) in enumerate(validated):
        length = tokens.shape[0]
        input_ids[row, :length] = tokens
        labels[row, :length] = targets
        attention_mask[row, :length] = True
        scopes.append(scope)
        sample_ids.append(sample_id)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "scopes": scopes,
        "sample_ids": sample_ids,
    }


__all__ = [
    "KnowledgeExample",
    "EntityTripletBatchSampler",
    "SyntheticKnowledgeDataset",
    "collate_causal_lm_batch",
    "generate_synthetic_corpus",
]
