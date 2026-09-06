"""Phase 5.5/T2 自然语言 CAP/MEM 数据协议与完整性检查。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .data import collate_causal_lm_batch
from .tokenizer import ByteTokenizer

T2_CAP_SUITE = "t2_nl_cap"
T2_MEM_SUITE = "t2_nl_mem"
T2_GENERATOR_VERSION = "phase5-t2-natural-language-pilot-v1"
T2_SPLITS = ("train", "dev", "validation", "test")
T2_SCOPES = ("public", "protected_public", "protected_private", "refusal")
T2_PROMPT_GROUPS = ("C0", "C1", "C2")
T2_REFUSAL_TARGET = " ACCESS-DENIED"

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_ENTITY_PREFIXES = (
    "Meridian",
    "Aurora",
    "Cedar",
    "Harbor",
    "Lattice",
    "Orchid",
    "Pioneer",
    "Summit",
)
_ENTITY_KINDS = ("Observatory", "Archive", "Clinic", "Library")
_LOCATIONS = (
    "North Harbor",
    "Cedar Valley",
    "Silver Ridge",
    "East Meadow",
    "Lake Junction",
    "West Terrace",
)
_DISTRICTS = ("amber", "indigo", "willow", "coral", "silver", "maple")
_TIERS = ("one", "two", "three", "four", "five", "six", "seven", "eight")

_CAP_TEMPLATES = {
    "cap-v1": (
        "Context: {context} Question: Where is {entity} located? Answer:",
        "Context: {context} Question: Which access tier applies to {entity}? Answer:",
    ),
    "cap-v2": (
        "Read this record: {context} Query: Name {entity}'s location. Response:",
        "Read this record: {context} Query: Infer the access tier for {entity}. Response:",
    ),
    "cap-v3": (
        "Evidence: {context} Task: Return the place containing {entity}. Result:",
        "Evidence: {context} Task: Follow the district rule and return {entity}'s tier. Result:",
    ),
    "cap-dev-heldout": (
        "Passage: {context} Based on the passage, where can {entity} be found? Answer:",
        "Passage: {context} Based on the passage, what tier governs {entity}? Answer:",
    ),
    "cap-validation-heldout": (
        "Source: {context} State the location of {entity}. Reply:",
        "Source: {context} Combine the facts to identify the tier for {entity}. Reply:",
    ),
    "cap-test-heldout": (
        "Document: {context} Give {entity}'s location. Output:",
        "Document: {context} Determine {entity}'s applicable access tier. Output:",
    ),
}

_MEM_TEMPLATES = {
    "mem-v1": (
        "Question: Where is {entity} located? Answer:",
        "Question: Which access tier belongs to {entity}? Answer:",
    ),
    "mem-v2": (
        "Query: Recall the location of {entity}. Response:",
        "Query: Recall the private tier of {entity}. Response:",
    ),
    "mem-v3": (
        "Memory check: Give {entity}'s location. Result:",
        "Memory check: Give {entity}'s access tier. Result:",
    ),
    "mem-dev-heldout": (
        "From memory, where can {entity} be found? Answer:",
        "From memory, what protected tier is assigned to {entity}? Answer:",
    ),
    "mem-validation-heldout": (
        "State the remembered location for {entity}. Reply:",
        "State the remembered access tier for {entity}. Reply:",
    ),
    "mem-test-heldout": (
        "What place is associated with {entity}? Output:",
        "What confidential tier is associated with {entity}? Output:",
    ),
}


@dataclass(frozen=True)
class T2Example:
    """描述一条严格版本化的 T2 自然语言样本。"""

    sample_id: str
    entity_id: str
    question_id: str
    suite_id: str
    scope: str
    credential_class: str
    prompt: str
    target: str
    answer_id: str
    prompt_template_id: str
    source_id: str
    source_sha256: str
    reasoning_depth: Optional[int]
    generator_version: str
    seed: int
    split: str
    prompt_group: str
    answer_aliases: Tuple[str, ...] = ()
    public_fallback: str = ""

    def __post_init__(self) -> None:
        """在对象创建时拒绝未知枚举、错误类型和非规范标识符。"""

        identifier_fields = (
            self.sample_id,
            self.entity_id,
            self.question_id,
            self.answer_id,
            self.prompt_template_id,
            self.source_id,
            self.generator_version,
        )
        if any(
            not isinstance(value, str) or not _IDENTIFIER.fullmatch(value)
            for value in identifier_fields
        ):
            raise ValueError("T2 标识符必须使用规范的小写 ASCII 编码")
        if self.suite_id not in {T2_CAP_SUITE, T2_MEM_SUITE}:
            raise ValueError("suite_id 必须为 t2_nl_cap 或 t2_nl_mem")
        if self.scope not in set(T2_SCOPES):
            raise ValueError("scope 不受支持")
        expected_credentials = {
            "public": "invalid",
            "protected_public": "valid",
            "protected_private": "valid",
            "refusal": "invalid",
        }
        if self.credential_class != expected_credentials[self.scope]:
            raise ValueError("credential_class 与 scope 不一致")
        if self.split not in set(T2_SPLITS):
            raise ValueError("split 必须为 train/dev/validation/test")
        if self.prompt_group not in set(T2_PROMPT_GROUPS):
            raise ValueError("prompt_group 必须为 C0、C1 或 C2")
        if (
            not isinstance(self.prompt, str)
            or not self.prompt.strip()
            or "\x00" in self.prompt
        ):
            raise ValueError("prompt 必须是非空且不含 NUL 的字符串")
        if (
            not isinstance(self.target, str)
            or not self.target.strip()
            or "\x00" in self.target
        ):
            raise ValueError("target 必须是非空且不含 NUL 的字符串")
        if not isinstance(self.source_sha256, str) or not _SHA256.fullmatch(
            self.source_sha256
        ):
            raise ValueError("source_sha256 必须是小写 64 位 SHA-256")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed 必须是整数")
        if self.suite_id == T2_CAP_SUITE:
            if isinstance(self.reasoning_depth, bool) or not isinstance(
                self.reasoning_depth, int
            ):
                raise TypeError("CAP reasoning_depth 必须是整数")
            if self.reasoning_depth <= 0:
                raise ValueError("CAP reasoning_depth 必须大于 0")
        elif self.reasoning_depth is not None:
            raise ValueError("MEM reasoning_depth 必须为 None")
        if not isinstance(self.answer_aliases, tuple) or any(
            not isinstance(alias, str) or not alias.strip()
            for alias in self.answer_aliases
        ):
            raise TypeError("answer_aliases 必须是非空字符串组成的 tuple")
        if not isinstance(self.public_fallback, str):
            raise TypeError("public_fallback 必须是字符串")
        if self.scope == "refusal":
            if self.target.strip() != T2_REFUSAL_TARGET.strip():
                raise ValueError("refusal target 必须固定为 ACCESS-DENIED")
            if not self.public_fallback.strip():
                raise ValueError("refusal 样本必须声明 public_fallback")
        elif self.public_fallback:
            raise ValueError("只有 refusal 样本可以声明 public_fallback")


def _validate_seed_and_counts(seed: int, counts: Mapping[str, int]) -> None:
    """验证生成器的 seed 和四路 split 数量。"""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed 必须是整数")
    if set(counts) != set(T2_SPLITS):
        raise ValueError("counts 必须恰好包含 train/dev/validation/test")
    for split, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{split} 数量必须是整数")
        if count <= 0:
            raise ValueError(f"{split} 数量必须大于 0")


def _source_record(index: int, rng: np.random.Generator) -> Dict[str, str]:
    """按显式 RNG 生成一条有语义且可复现的事实卡片。"""

    prefix = _ENTITY_PREFIXES[index % len(_ENTITY_PREFIXES)]
    kind = _ENTITY_KINDS[(index // len(_ENTITY_PREFIXES)) % len(_ENTITY_KINDS)]
    entity = f"{prefix} {kind} {index:03d}"
    location = str(rng.choice(_LOCATIONS))
    district = str(rng.choice(_DISTRICTS))
    tier = str(rng.choice(_TIERS))
    return {
        "entity": entity,
        "location": location,
        "district": district,
        "tier": tier,
    }


def _record_sha256(record: Mapping[str, str]) -> str:
    """计算事实卡片的规范 JSON SHA-256。"""

    payload = json.dumps(
        record, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _template_ids(prompt_group: str, split: str, prefix: str) -> Tuple[str, ...]:
    """根据 C0/C1/C2 和 split 返回冻结的模板集合。"""

    if prompt_group not in set(T2_PROMPT_GROUPS):
        raise ValueError("prompt_group 必须为 C0、C1 或 C2")
    if split not in set(T2_SPLITS):
        raise ValueError("split 不受支持")
    if prompt_group == "C0":
        return (f"{prefix}-v1",)
    if split == "train":
        return (
            (f"{prefix}-v1", f"{prefix}-v2", f"{prefix}-v3")
            if prompt_group == "C2"
            else (f"{prefix}-v1",)
        )
    return (f"{prefix}-{split}-heldout",)


def _build_rows(
    *,
    record: Mapping[str, str],
    suite_id: str,
    split: str,
    template_id: str,
    prompt_group: str,
    seed: int,
    source_id: str,
) -> List[T2Example]:
    """从一条事实卡片构造 public/protected/refusal 四元组。"""

    templates = _CAP_TEMPLATES if suite_id == T2_CAP_SUITE else _MEM_TEMPLATES
    public_template, protected_template = templates[template_id]
    context = (
        f"{record['entity']} is located in {record['location']}. "
        f"{record['location']} belongs to the {record['district']} district. "
        f"The {record['district']} district uses access tier {record['tier']}."
    )
    format_values = {"context": context, "entity": record["entity"]}
    public_prompt = public_template.format(**format_values)
    protected_prompt = protected_template.format(**format_values)
    source_sha256 = _record_sha256(record)
    entity_slug = source_id.rsplit("-", 1)[-1]
    template_slug = template_id.replace("-", ".")
    public_target = f" {record['location']}"
    protected_target = f" Tier {record['tier']}"
    common = {
        "entity_id": f"t2-entity-{entity_slug}",
        "suite_id": suite_id,
        "prompt_template_id": template_id,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "reasoning_depth": 1 if suite_id == T2_CAP_SUITE else None,
        "generator_version": T2_GENERATOR_VERSION,
        "seed": seed,
        "split": split,
        "prompt_group": prompt_group,
    }
    public_question = f"{source_id}-public"
    protected_question = f"{source_id}-protected"
    rows = [
        T2Example(
            sample_id=f"{suite_id}-{split}-{entity_slug}-{template_slug}-public",
            question_id=public_question,
            scope="public",
            credential_class="invalid",
            prompt=public_prompt,
            target=public_target,
            answer_id=f"{source_id}-location",
            answer_aliases=(record["location"],),
            **common,
        ),
        T2Example(
            sample_id=f"{suite_id}-{split}-{entity_slug}-{template_slug}-protected.public",
            question_id=public_question,
            scope="protected_public",
            credential_class="valid",
            prompt=public_prompt,
            target=public_target,
            answer_id=f"{source_id}-location",
            answer_aliases=(record["location"],),
            **common,
        ),
        T2Example(
            sample_id=f"{suite_id}-{split}-{entity_slug}-{template_slug}-protected.private",
            question_id=protected_question,
            scope="protected_private",
            credential_class="valid",
            prompt=protected_prompt,
            target=protected_target,
            answer_id=f"{source_id}-tier",
            answer_aliases=(f"tier {record['tier']}", record["tier"]),
            **{
                **common,
                "reasoning_depth": 3 if suite_id == T2_CAP_SUITE else None,
            },
        ),
        T2Example(
            sample_id=f"{suite_id}-{split}-{entity_slug}-{template_slug}-refusal",
            question_id=protected_question,
            scope="refusal",
            credential_class="invalid",
            prompt=protected_prompt,
            target=T2_REFUSAL_TARGET,
            answer_id="t2-access-denied",
            public_fallback=public_target,
            **{
                **common,
                "reasoning_depth": 3 if suite_id == T2_CAP_SUITE else None,
            },
        ),
    ]
    return rows


def generate_t2_cap_corpus(
    seed: int,
    train_sources: int = 12,
    dev_sources: int = 4,
    validation_sources: int = 4,
    test_sources: int = 4,
    *,
    prompt_group: str = "C0",
) -> Dict[str, List[T2Example]]:
    """生成 source-disjoint 的上下文内自然语言能力分级语料。"""

    counts = {
        "train": train_sources,
        "dev": dev_sources,
        "validation": validation_sources,
        "test": test_sources,
    }
    _validate_seed_and_counts(seed, counts)
    if prompt_group not in set(T2_PROMPT_GROUPS):
        raise ValueError("prompt_group 必须为 C0、C1 或 C2")
    rng = np.random.default_rng(seed)
    corpus: Dict[str, List[T2Example]] = {}
    offset = 0
    for split in T2_SPLITS:
        rows: List[T2Example] = []
        for local_index in range(counts[split]):
            source_index = offset + local_index
            source_id = f"t2-cap-source-{source_index:05d}"
            record = _source_record(source_index, rng)
            for template_id in _template_ids(prompt_group, split, "cap"):
                rows.extend(
                    _build_rows(
                        record=record,
                        suite_id=T2_CAP_SUITE,
                        split=split,
                        template_id=template_id,
                        prompt_group=prompt_group,
                        seed=seed,
                        source_id=source_id,
                    )
                )
        corpus[split] = rows
        offset += counts[split]
    validate_t2_corpus(corpus, expected_suite=T2_CAP_SUITE)
    return corpus


def generate_t2_mem_corpus(
    seed: int,
    entity_count: int = 12,
    *,
    prompt_group: str = "C0",
) -> Dict[str, List[T2Example]]:
    """生成已见事实、held-out 问法的闭卷记忆与泄漏语料。"""

    _validate_seed_and_counts(seed, {split: entity_count for split in T2_SPLITS})
    if prompt_group not in set(T2_PROMPT_GROUPS):
        raise ValueError("prompt_group 必须为 C0、C1 或 C2")
    rng = np.random.default_rng(seed)
    records = [_source_record(index, rng) for index in range(entity_count)]
    corpus: Dict[str, List[T2Example]] = {}
    for split in T2_SPLITS:
        rows: List[T2Example] = []
        for index, record in enumerate(records):
            source_id = f"t2-mem-source-{index:05d}"
            for template_id in _template_ids(prompt_group, split, "mem"):
                rows.extend(
                    _build_rows(
                        record=record,
                        suite_id=T2_MEM_SUITE,
                        split=split,
                        template_id=template_id,
                        prompt_group=prompt_group,
                        seed=seed,
                        source_id=source_id,
                    )
                )
        corpus[split] = rows
    validate_t2_corpus(corpus, expected_suite=T2_MEM_SUITE)
    return corpus


def validate_t2_corpus(
    corpus: Mapping[str, Sequence[T2Example]], *, expected_suite: Optional[str] = None
) -> None:
    """验证 T2 corpus 的 schema、配对、hash 和 CAP/MEM 隔离契约。"""

    if not isinstance(corpus, Mapping) or set(corpus) != set(T2_SPLITS):
        raise ValueError("T2 corpus 必须恰好包含 train/dev/validation/test")
    examples: List[T2Example] = []
    for split in T2_SPLITS:
        rows = corpus[split]
        if not isinstance(rows, Sequence) or not rows:
            raise ValueError(f"{split} 样本不能为空")
        for example in rows:
            if not isinstance(example, T2Example):
                raise TypeError("corpus 只能包含 T2Example")
            if example.split != split:
                raise ValueError("样本 split 与 corpus key 不一致")
            examples.append(example)
    sample_ids = [example.sample_id for example in examples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id 必须全局唯一")
    suites = {example.suite_id for example in examples}
    if len(suites) != 1 or (expected_suite is not None and suites != {expected_suite}):
        raise ValueError("单个 corpus 必须只包含指定 CAP 或 MEM suite")
    groups = {example.prompt_group for example in examples}
    if len(groups) != 1:
        raise ValueError("单个 corpus 不能混合 prompt_group")
    if len({example.generator_version for example in examples}) != 1:
        raise ValueError("单个 corpus 不能混合 generator_version")
    if len({example.seed for example in examples}) != 1:
        raise ValueError("单个 corpus 不能混合 seed")

    # 同一 source 的事实摘要必须在所有配对和 split 中保持一致。
    source_hashes: Dict[str, str] = {}
    protected_targets = {
        (example.split, example.question_id): (example.target,) + example.answer_aliases
        for example in examples
        if example.scope == "protected_private"
    }
    for example in examples:
        previous = source_hashes.setdefault(example.source_id, example.source_sha256)
        if previous != example.source_sha256:
            raise ValueError("同一 source_id 的 source_sha256 不一致")
        answers = protected_targets.get((example.split, example.question_id), ())
        if (
            example.suite_id == T2_MEM_SUITE
            and example.scope
            in {
                "protected_private",
                "refusal",
            }
            and any(
                answer.strip().casefold() in example.prompt.casefold()
                for answer in answers
            )
        ):
            raise ValueError("MEM protected answer 或 alias 不得出现在 prompt 中")

    # 每个 source/template 必须形成完整四元组并保持成对 prompt 一致。
    paired: Dict[Tuple[str, str, str], Dict[str, T2Example]] = {}
    for example in examples:
        key = (example.split, example.source_id, example.prompt_template_id)
        scope_rows = paired.setdefault(key, {})
        if example.scope in scope_rows:
            raise ValueError("同一 source/template/scope 不能重复")
        scope_rows[example.scope] = example
    for scope_rows in paired.values():
        if set(scope_rows) != set(T2_SCOPES):
            raise ValueError("每个 source/template 必须包含完整 T2 四元组")
        if scope_rows["public"].prompt != scope_rows["protected_public"].prompt:
            raise ValueError("public 与 protected_public prompt 必须一致")
        if scope_rows["protected_private"].prompt != scope_rows["refusal"].prompt:
            raise ValueError("protected_private 与 refusal prompt 必须一致")
        if scope_rows["public"].target != scope_rows["protected_public"].target:
            raise ValueError("public 与 protected_public target 必须一致")
        if (
            scope_rows["public"].question_id
            != scope_rows["protected_public"].question_id
        ):
            raise ValueError("public 与 protected_public question_id 必须一致")
        if (
            scope_rows["protected_private"].question_id
            != scope_rows["refusal"].question_id
        ):
            raise ValueError("protected_private 与 refusal question_id 必须一致")

    suite_id = next(iter(suites))
    entity_sets = {
        split: {example.entity_id for example in corpus[split]} for split in T2_SPLITS
    }
    source_sets = {
        split: {example.source_id for example in corpus[split]} for split in T2_SPLITS
    }
    if suite_id == T2_CAP_SUITE:
        for left_index, left in enumerate(T2_SPLITS):
            for right in T2_SPLITS[left_index + 1 :]:
                if (
                    entity_sets[left] & entity_sets[right]
                    or source_sets[left] & source_sets[right]
                ):
                    raise ValueError("CAP 的 entity/source 必须跨 split 隔离")
    else:
        if any(entity_sets[split] != entity_sets["train"] for split in T2_SPLITS[1:]):
            raise ValueError("MEM 主评估必须复用训练阶段已见实体")
        if any(source_sets[split] != source_sets["train"] for split in T2_SPLITS[1:]):
            raise ValueError("MEM 主评估必须复用训练阶段已见事实")

    prompt_group = next(iter(groups))
    templates_by_split = {
        split: {example.prompt_template_id for example in corpus[split]}
        for split in T2_SPLITS
    }
    expected_train_template_count = 3 if prompt_group == "C2" else 1
    if len(templates_by_split["train"]) != expected_train_template_count or any(
        len(templates_by_split[split]) != 1 for split in T2_SPLITS[1:]
    ):
        raise ValueError("prompt_group 的训练/评估模板数量不符合协议")
    if prompt_group == "C0":
        if any(
            templates_by_split[split] != templates_by_split["train"]
            for split in T2_SPLITS[1:]
        ):
            raise ValueError("C0 必须在四路 split 使用同一模板")
    else:
        train_templates = templates_by_split["train"]
        for split in T2_SPLITS[1:]:
            split_templates = templates_by_split[split]
            if train_templates & split_templates:
                raise ValueError("C1/C2 held-out 模板不得进入训练模板集合")
        eval_templates = [
            next(iter(templates_by_split[split])) for split in T2_SPLITS[1:]
        ]
        if len(eval_templates) != len(set(eval_templates)):
            raise ValueError("C1/C2 的 dev/validation/test 模板必须彼此隔离")


def t2_corpus_sha256(corpus: Mapping[str, Sequence[T2Example]]) -> str:
    """验证并计算与输入映射/列表顺序无关的 corpus SHA-256。"""

    validate_t2_corpus(corpus)
    payload = {
        split: [
            asdict(row)
            for row in sorted(corpus[split], key=lambda item: item.sample_id)
        ]
        for split in T2_SPLITS
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class T2CausalLMDataset(Dataset):
    """把 T2 自然语言样本转换为 answer-only 因果语言模型输入。"""

    def __init__(
        self,
        examples: Sequence[T2Example],
        tokenizer: ByteTokenizer,
        max_length: int = 256,
    ) -> None:
        """验证单 split 样本、tokenizer 和最大序列长度。"""

        if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
            raise TypeError("examples 必须是 T2Example 序列")
        self.examples = list(examples)
        if not self.examples or any(
            not isinstance(example, T2Example) for example in self.examples
        ):
            raise ValueError("examples 必须是非空 T2Example 序列")
        if len({example.sample_id for example in self.examples}) != len(self.examples):
            raise ValueError("dataset sample_id 不能重复")
        if len({example.split for example in self.examples}) != 1:
            raise ValueError("单个 dataset 只能包含一个 split")
        if len({example.suite_id for example in self.examples}) != 1:
            raise ValueError("单个 dataset 只能包含一个 suite")
        if not isinstance(tokenizer, ByteTokenizer):
            raise TypeError("tokenizer 必须是 ByteTokenizer")
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length 必须是整数")
        if max_length <= 1:
            raise ValueError("max_length 必须大于 1")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """返回 T2 样本数量。"""

        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """返回 prompt mask 后的因果 LM 张量与审计 metadata。"""

        example = self.examples[index]
        prompt_ids = self.tokenizer.encode(
            example.prompt,
            add_bos=True,
            add_eos=False,
            max_length=self.max_length,
        )
        target_ids = self.tokenizer.encode(
            example.target,
            add_bos=False,
            add_eos=True,
            max_length=self.max_length,
        )
        token_ids = prompt_ids + target_ids
        if len(token_ids) > self.max_length:
            raise ValueError(f"T2 样本 {example.sample_id} 超过 max_length")
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(
                [-100] * len(prompt_ids) + target_ids, dtype=torch.long
            ),
            "scope": example.scope,
            "sample_id": example.sample_id,
            "suite_id": example.suite_id,
            "credential_class": example.credential_class,
            "source_id": example.source_id,
            "entity_id": example.entity_id,
            "question_id": example.question_id,
            "answer_id": example.answer_id,
            "prompt_template_id": example.prompt_template_id,
            "split": example.split,
            "prompt_group": example.prompt_group,
        }


def collate_t2_causal_lm_batch(
    items: Sequence[Mapping[str, object]], pad_token_id: int = 258
) -> Dict[str, object]:
    """右填充 T2 batch，并保留 suite、credential 和 source metadata。"""

    base = collate_causal_lm_batch(items, pad_token_id=pad_token_id)
    metadata: Dict[str, List[str]] = {
        "suite_ids": [],
        "credential_classes": [],
        "source_ids": [],
        "entity_ids": [],
        "question_ids": [],
        "answer_ids": [],
        "prompt_template_ids": [],
        "splits": [],
        "prompt_groups": [],
    }
    source_keys = {
        "suite_ids": "suite_id",
        "credential_classes": "credential_class",
        "source_ids": "source_id",
        "entity_ids": "entity_id",
        "question_ids": "question_id",
        "answer_ids": "answer_id",
        "prompt_template_ids": "prompt_template_id",
        "splits": "split",
        "prompt_groups": "prompt_group",
    }
    for item in items:
        for output_key, source_key in source_keys.items():
            value = item.get(source_key)
            if not isinstance(value, str):
                raise TypeError(f"{source_key} 必须是 str")
            metadata[output_key].append(value)
    if len(set(metadata["suite_ids"])) != 1:
        raise ValueError("单个 batch 不能混合 CAP/MEM suite")
    return {**base, **metadata}


class T2QuadrupletBatchSampler(Sampler[List[int]]):
    """按 source/template 四元组构造确定性的 mixed-routing batch。"""

    def __init__(
        self, examples: Sequence[T2Example], batch_size: int, seed: int
    ) -> None:
        """验证每组 2 valid/2 invalid，并配置按 epoch 派生的确定性顺序。"""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size 必须是整数")
        if batch_size < 4 or batch_size % 4 != 0:
            raise ValueError("T2 batch_size 必须至少为 4 且能被 4 整除")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed 必须是整数")
        groups: Dict[Tuple[str, str], List[int]] = {}
        scopes: Dict[Tuple[str, str], set] = {}
        credentials: Dict[Tuple[str, str], List[str]] = {}
        for index, example in enumerate(examples):
            if not isinstance(example, T2Example):
                raise TypeError("examples 必须只包含 T2Example")
            key = (example.source_id, example.prompt_template_id)
            groups.setdefault(key, []).append(index)
            scopes.setdefault(key, set()).add(example.scope)
            credentials.setdefault(key, []).append(example.credential_class)
        if any(
            len(groups[key]) != 4
            or scopes[key] != set(T2_SCOPES)
            or credentials[key].count("valid") != 2
            or credentials[key].count("invalid") != 2
            for key in groups
        ):
            raise ValueError(
                "每个 T2 source/template 必须恰含 2 valid/2 invalid 四元组"
            )
        self.group_rows = list(groups.values())
        self.groups_per_batch = batch_size // 4
        if len(self.group_rows) < self.groups_per_batch:
            raise ValueError("T2 四元组数量不足以形成一个完整 batch")
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        """按当前 epoch 的派生 seed 产生完整四元组 batch。"""

        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.group_rows)).tolist()
        usable = len(order) - len(order) % self.groups_per_batch
        for start in range(0, usable, self.groups_per_batch):
            rows: List[int] = []
            for group_index in order[start : start + self.groups_per_batch]:
                rows.extend(self.group_rows[group_index])
            yield rows

    def __len__(self) -> int:
        """返回丢弃不完整四元组组后的 batch 数。"""

        return len(self.group_rows) // self.groups_per_batch

    def set_epoch(self, epoch: int) -> None:
        """设置非负 epoch，以便恢复时重建完全相同的数据顺序。"""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch 必须是非负整数")
        self.epoch = epoch


__all__ = [
    "T2_CAP_SUITE",
    "T2_MEM_SUITE",
    "T2_GENERATOR_VERSION",
    "T2_PROMPT_GROUPS",
    "T2_REFUSAL_TARGET",
    "T2_SCOPES",
    "T2_SPLITS",
    "T2Example",
    "T2CausalLMDataset",
    "T2QuadrupletBatchSampler",
    "collate_t2_causal_lm_batch",
    "generate_t2_cap_corpus",
    "generate_t2_mem_corpus",
    "t2_corpus_sha256",
    "validate_t2_corpus",
]
