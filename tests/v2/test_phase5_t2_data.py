"""Phase 5.5/T2 自然语言数据协议专项测试。"""

from dataclasses import replace

import pytest

from src.can.v2.transformer import (
    T2_CAP_SUITE,
    T2_MEM_SUITE,
    T2_REFUSAL_TARGET,
    ByteTokenizer,
    T2CausalLMDataset,
    T2QuadrupletBatchSampler,
    collate_t2_causal_lm_batch,
    generate_t2_cap_corpus,
    generate_t2_mem_corpus,
    generate_t2_split,
    t2_corpus_sha256,
    t2_split_sha256,
    validate_t2_corpus,
)


def _replace_all(corpus, split, **changes):
    """复制 corpus，并统一替换指定 split 的字段。"""

    result = {name: list(rows) for name, rows in corpus.items()}
    result[split] = [replace(row, **changes) for row in result[split]]
    return result


def test_cap_corpus_is_deterministic_and_seeded() -> None:
    """CAP corpus 在相同 seed 下逐项一致，不同 seed 的 hash 不同。"""

    first = generate_t2_cap_corpus(101, 3, 2, 2, 2)
    second = generate_t2_cap_corpus(101, 3, 2, 2, 2)
    changed = generate_t2_cap_corpus(102, 3, 2, 2, 2)
    assert first == second
    assert t2_corpus_sha256(first) == t2_corpus_sha256(second)
    assert t2_corpus_sha256(first) != t2_corpus_sha256(changed)


@pytest.mark.parametrize(
    "prompt_group,expected_hash",
    [
        ("C0", "18b0ba93c354e1d6bc2f314e0a19601d62172055426d1207edcfcd02f3752b20"),
        ("C1", "b9c173052c9de2b7de8e9aec09f343fb6bf785085fe39204cfea59de2ad617c3"),
        ("C2", "c272c7684593bb93999fc822b0ca26976e99126d777b0f40cd865b1f491093c3"),
    ],
)
def test_cap_lazy_split_preserves_frozen_corpus_hash(
    prompt_group: str, expected_hash: str
) -> None:
    """CAP 改为延迟 split 组合后不得改变既有 corpus 内容。"""

    corpus = generate_t2_cap_corpus(151, 3, 2, 2, 2, prompt_group=prompt_group)
    assert t2_corpus_sha256(corpus) == expected_hash
    counts = {"train": 3, "dev": 2, "validation": 2, "test": 2}
    for split, rows in corpus.items():
        generated = generate_t2_split(
            T2_CAP_SUITE, split, 151, counts, prompt_group=prompt_group
        )
        assert generated == rows
        assert t2_split_sha256(generated) == t2_split_sha256(rows)


@pytest.mark.parametrize(
    "prompt_group,expected_hash",
    [
        ("C0", "52cee573cb40242b209ad6b68c426924e82ea43d9a4c828cd054adfc21a9feee"),
        ("C1", "e3a04d2449387f5b0ab55379f02c30fad85fd76e7ec45638540c05dbb2d86227"),
        ("C2", "2bed615e2864e5cc6adb2272e5ae40c5214b15a3f5233dea58a79fff5779f0fc"),
    ],
)
def test_mem_lazy_split_preserves_frozen_corpus_hash(
    prompt_group: str, expected_hash: str
) -> None:
    """MEM 延迟 split 必须复用既有事实并保持 corpus hash。"""

    corpus = generate_t2_mem_corpus(151, 3, prompt_group=prompt_group)
    assert t2_corpus_sha256(corpus) == expected_hash
    counts = {split: 3 for split in ("train", "dev", "validation", "test")}
    for split, rows in corpus.items():
        assert (
            generate_t2_split(
                T2_MEM_SUITE, split, 151, counts, prompt_group=prompt_group
            )
            == rows
        )


def test_lazy_split_rejects_invalid_identity_and_mem_counts() -> None:
    """延迟生成必须拒绝错误 suite/split 和不等长 MEM 实体规划。"""

    counts = {"train": 2, "dev": 1, "validation": 1, "test": 1}
    with pytest.raises(ValueError, match="suite_id"):
        generate_t2_split("unknown", "train", 1, counts)
    with pytest.raises(ValueError, match="split"):
        generate_t2_split(T2_CAP_SUITE, "holdout", 1, counts)
    with pytest.raises(ValueError, match="相同实体数量"):
        generate_t2_split(T2_MEM_SUITE, "train", 1, counts)


def test_cap_splits_are_source_and_entity_disjoint() -> None:
    """CAP 四路 split 的 source 与 entity 必须完全隔离。"""

    corpus = generate_t2_cap_corpus(103, 3, 2, 2, 2)
    for left_index, left in enumerate(("train", "dev", "validation", "test")):
        left_sources = {row.source_id for row in corpus[left]}
        left_entities = {row.entity_id for row in corpus[left]}
        for right in ("train", "dev", "validation", "test")[left_index + 1 :]:
            assert left_sources.isdisjoint({row.source_id for row in corpus[right]})
            assert left_entities.isdisjoint({row.entity_id for row in corpus[right]})


def test_cap_quadruplet_has_paired_prompts_and_reasoning_depth() -> None:
    """CAP 每个 source/template 都包含对齐的四类请求。"""

    rows = generate_t2_cap_corpus(105, 1, 1, 1, 1)["train"]
    by_scope = {row.scope: row for row in rows}
    assert set(by_scope) == {
        "public",
        "protected_public",
        "protected_private",
        "refusal",
    }
    assert by_scope["public"].prompt == by_scope["protected_public"].prompt
    assert by_scope["protected_private"].prompt == by_scope["refusal"].prompt
    assert by_scope["public"].reasoning_depth == 1
    assert by_scope["protected_private"].reasoning_depth == 3
    assert by_scope["refusal"].target == T2_REFUSAL_TARGET
    assert (
        by_scope["protected_private"].target.strip().casefold()
        in by_scope["protected_private"].prompt.casefold()
    )


@pytest.mark.parametrize("prompt_group", ["C1", "C2"])
def test_cap_heldout_templates_do_not_enter_training(prompt_group: str) -> None:
    """C1/C2 的 dev/validation/test 模板均不得进入训练集合。"""

    corpus = generate_t2_cap_corpus(107, 2, 1, 1, 1, prompt_group=prompt_group)
    train_templates = {row.prompt_template_id for row in corpus["train"]}
    for split in ("dev", "validation", "test"):
        assert train_templates.isdisjoint(
            {row.prompt_template_id for row in corpus[split]}
        )


def test_cap_c2_has_three_training_templates() -> None:
    """C2 为每个训练 source 生成三套完整四元组。"""

    corpus = generate_t2_cap_corpus(109, 2, 1, 1, 1, prompt_group="C2")
    assert len(corpus["train"]) == 2 * 3 * 4
    assert {row.prompt_template_id for row in corpus["train"]} == {
        "cap-v1",
        "cap-v2",
        "cap-v3",
    }


def test_mem_reuses_seen_facts_but_hides_protected_answer() -> None:
    """MEM 四路 split 复用已见事实，且查询不复制 protected answer。"""

    corpus = generate_t2_mem_corpus(111, 3, prompt_group="C1")
    train_entities = {row.entity_id for row in corpus["train"]}
    train_sources = {row.source_id for row in corpus["train"]}
    for split in ("dev", "validation", "test"):
        assert {row.entity_id for row in corpus[split]} == train_entities
        assert {row.source_id for row in corpus[split]} == train_sources
        for row in corpus[split]:
            if row.scope == "protected_private":
                assert row.target.strip().casefold() not in row.prompt.casefold()
                assert row.reasoning_depth is None


def test_mem_c2_uses_seen_facts_and_heldout_prompts() -> None:
    """MEM C2 只扩展训练问法，不改变事实或答案。"""

    corpus = generate_t2_mem_corpus(113, 2, prompt_group="C2")
    train_templates = {row.prompt_template_id for row in corpus["train"]}
    validation_templates = {row.prompt_template_id for row in corpus["validation"]}
    assert train_templates == {"mem-v1", "mem-v2", "mem-v3"}
    assert train_templates.isdisjoint(validation_templates)
    train_targets = {(row.source_id, row.scope, row.target) for row in corpus["train"]}
    validation_targets = {
        (row.source_id, row.scope, row.target) for row in corpus["validation"]
    }
    assert {
        (source, scope, target) for source, scope, target in validation_targets
    }.issubset(train_targets)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (generate_t2_cap_corpus, {"seed": True}),
        (generate_t2_cap_corpus, {"seed": 1, "train_sources": 0}),
        (generate_t2_cap_corpus, {"seed": 1, "prompt_group": "C3"}),
        (generate_t2_mem_corpus, {"seed": 1, "entity_count": 0}),
    ],
)
def test_generators_reject_invalid_configuration(factory, kwargs) -> None:
    """非法 seed、数量和 prompt group 必须 fail-fast。"""

    with pytest.raises((TypeError, ValueError)):
        factory(**kwargs)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("sample_id", "Invalid ID", ValueError),
        ("suite_id", "unknown_suite", ValueError),
        ("scope", "unknown", ValueError),
        ("credential_class", "valid", ValueError),
        ("split", "holdout", ValueError),
        ("prompt_group", "C3", ValueError),
        ("prompt", "\x00", ValueError),
        ("target", "", ValueError),
        ("source_sha256", "A" * 64, ValueError),
        ("seed", True, TypeError),
        ("reasoning_depth", 0, ValueError),
        ("answer_aliases", ["alias"], TypeError),
        ("public_fallback", 1, TypeError),
        ("public_fallback", " North Harbor", ValueError),
    ],
)
def test_t2_example_rejects_invalid_schema(field, value, error) -> None:
    """T2Example 必须在对象边界拒绝非规范字段。"""

    example = generate_t2_cap_corpus(114, 1, 1, 1, 1)["train"][0]
    with pytest.raises(error):
        replace(example, **{field: value})


def test_t2_example_rejects_invalid_suite_specific_fields() -> None:
    """CAP/MEM reasoning 与 refusal 专用字段不得互相混用。"""

    cap_rows = generate_t2_cap_corpus(116, 1, 1, 1, 1)["train"]
    refusal = next(row for row in cap_rows if row.scope == "refusal")
    with pytest.raises(ValueError, match="refusal target"):
        replace(refusal, target=" unexpected")
    with pytest.raises(ValueError, match="public_fallback"):
        replace(refusal, public_fallback="")

    mem = generate_t2_mem_corpus(116, 1)["train"][0]
    with pytest.raises(ValueError, match="reasoning_depth"):
        replace(mem, reasoning_depth=1)


def test_validator_rejects_source_hash_tampering() -> None:
    """同一 MEM source 的摘要漂移必须被检测。"""

    corpus = generate_t2_mem_corpus(115, 2)
    tampered = {split: list(rows) for split, rows in corpus.items()}
    tampered["validation"][0] = replace(
        tampered["validation"][0], source_sha256="0" * 64
    )
    with pytest.raises(ValueError, match="source_sha256"):
        validate_t2_corpus(tampered, expected_suite=T2_MEM_SUITE)


def test_validator_rejects_duplicate_sample_id() -> None:
    """跨 split 重复 sample ID 必须被拒绝。"""

    corpus = generate_t2_cap_corpus(117, 2, 1, 1, 1)
    tampered = {split: list(rows) for split, rows in corpus.items()}
    tampered["dev"][0] = replace(
        tampered["dev"][0], sample_id=tampered["train"][0].sample_id
    )
    with pytest.raises(ValueError, match="sample_id"):
        validate_t2_corpus(tampered, expected_suite=T2_CAP_SUITE)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("seed", 999, "混合 seed"),
        ("generator_version", "phase5-t2-tampered-v1", "generator_version"),
        ("prompt_group", "C1", "prompt_group"),
    ],
)
def test_validator_rejects_mixed_corpus_provenance(field, value, match) -> None:
    """单个 corpus 不得混入其他 seed、生成器版本或 prompt 组。"""

    corpus = generate_t2_cap_corpus(118, 2, 1, 1, 1)
    tampered = {split: list(rows) for split, rows in corpus.items()}
    tampered["train"][0] = replace(tampered["train"][0], **{field: value})
    with pytest.raises(ValueError, match=match):
        validate_t2_corpus(tampered, expected_suite=T2_CAP_SUITE)


def test_validator_rejects_incomplete_quadruplet() -> None:
    """删除四元组任一 scope 后，corpus 校验必须失败。"""

    corpus = generate_t2_cap_corpus(120, 2, 1, 1, 1)
    tampered = {split: list(rows) for split, rows in corpus.items()}
    tampered["train"].pop()
    with pytest.raises(ValueError, match="完整 T2 四元组"):
        validate_t2_corpus(tampered, expected_suite=T2_CAP_SUITE)


@pytest.mark.parametrize("prompt_group", ["C0", "C1", "C2"])
def test_validator_rejects_prompt_template_protocol_tampering(
    prompt_group: str,
) -> None:
    """C0/C1/C2 被替换为错误训练模板时必须 fail-closed。"""

    corpus = generate_t2_cap_corpus(122, 2, 1, 1, 1, prompt_group=prompt_group)
    train_template = corpus["train"][0].prompt_template_id
    replacement = "cap-v2" if prompt_group == "C0" else train_template
    tampered = _replace_all(corpus, "dev", prompt_template_id=replacement)
    expected = "C0 必须" if prompt_group == "C0" else "held-out"
    with pytest.raises(ValueError, match=expected):
        validate_t2_corpus(tampered, expected_suite=T2_CAP_SUITE)


def test_validator_rejects_cap_source_overlap() -> None:
    """CAP source 被复制到另一个 split 时必须 fail-closed。"""

    corpus = generate_t2_cap_corpus(119, 2, 1, 1, 1)
    tampered = {split: list(rows) for split, rows in corpus.items()}
    source = tampered["train"][0]
    replacement = tampered["dev"][0]
    tampered["dev"][0] = replace(
        replacement,
        source_id=source.source_id,
        source_sha256=source.source_sha256,
        entity_id=source.entity_id,
    )
    with pytest.raises(ValueError):
        validate_t2_corpus(tampered, expected_suite=T2_CAP_SUITE)


def test_corpus_hash_is_independent_of_row_order() -> None:
    """规范 corpus hash 不受 split 内列表顺序影响。"""

    corpus = generate_t2_cap_corpus(121, 2, 1, 1, 1)
    reversed_rows = {split: list(reversed(rows)) for split, rows in corpus.items()}
    assert t2_corpus_sha256(corpus) == t2_corpus_sha256(reversed_rows)


def test_t2_dataset_masks_prompt_and_preserves_metadata() -> None:
    """T2 dataset 只监督 target，并保留路由所需 metadata。"""

    example = generate_t2_cap_corpus(123, 1, 1, 1, 1)["train"][0]
    tokenizer = ByteTokenizer()
    dataset = T2CausalLMDataset([example], tokenizer)
    item = dataset[0]
    prompt_length = len(tokenizer.encode(example.prompt, add_bos=True, add_eos=False))
    assert item["labels"][:prompt_length].tolist() == [-100] * prompt_length
    assert item["scope"] == example.scope
    assert item["credential_class"] == example.credential_class
    assert item["source_id"] == example.source_id


def test_t2_dataset_and_collate_produce_aligned_batch() -> None:
    """T2 collate 必须填充张量并保持每行 metadata 对齐。"""

    examples = generate_t2_cap_corpus(125, 1, 1, 1, 1)["train"]
    dataset = T2CausalLMDataset(examples, ByteTokenizer())
    batch = collate_t2_causal_lm_batch(
        [dataset[index] for index in range(len(dataset))]
    )
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["scopes"] == [example.scope for example in examples]
    assert batch["credential_classes"] == [
        example.credential_class for example in examples
    ]
    assert batch["suite_ids"] == [T2_CAP_SUITE] * len(examples)


def test_t2_dataset_rejects_mixed_split_and_overlength_sample() -> None:
    """混合 split 或超过 max length 的样本必须在训练前失败。"""

    corpus = generate_t2_cap_corpus(127, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="一个 split"):
        T2CausalLMDataset([corpus["train"][0], corpus["dev"][0]], ByteTokenizer())
    dataset = T2CausalLMDataset([corpus["train"][0]], ByteTokenizer(), max_length=32)
    with pytest.raises(ValueError, match="超过 max_length"):
        _ = dataset[0]


@pytest.mark.parametrize(
    "examples,tokenizer,max_length,error",
    [
        ([], ByteTokenizer(), 256, ValueError),
        ("not-a-sequence", ByteTokenizer(), 256, TypeError),
        (["not-an-example"], ByteTokenizer(), 256, ValueError),
        (None, ByteTokenizer(), 256, TypeError),
    ],
)
def test_t2_dataset_rejects_invalid_examples(
    examples, tokenizer, max_length, error
) -> None:
    """dataset 必须拒绝空集合、裸字符串和错误元素类型。"""

    with pytest.raises(error):
        T2CausalLMDataset(examples, tokenizer, max_length=max_length)


@pytest.mark.parametrize(
    "tokenizer,max_length,error",
    [
        (object(), 256, TypeError),
        (ByteTokenizer(), True, TypeError),
        (ByteTokenizer(), 1, ValueError),
    ],
)
def test_t2_dataset_rejects_invalid_tokenizer_or_length(
    tokenizer, max_length, error
) -> None:
    """dataset 必须拒绝错误 tokenizer 和非法长度边界。"""

    example = generate_t2_cap_corpus(126, 1, 1, 1, 1)["train"][0]
    with pytest.raises(error):
        T2CausalLMDataset([example], tokenizer, max_length=max_length)


def test_t2_collate_rejects_mixed_suite() -> None:
    """单个 batch 混合 CAP/MEM 时必须 fail-fast。"""

    cap = generate_t2_cap_corpus(129, 1, 1, 1, 1)["train"][0]
    mem = generate_t2_mem_corpus(129, 1)["train"][0]
    cap_item = T2CausalLMDataset([cap], ByteTokenizer())[0]
    mem_item = T2CausalLMDataset([mem], ByteTokenizer())[0]
    with pytest.raises(ValueError, match="混合 CAP/MEM"):
        collate_t2_causal_lm_batch([cap_item, mem_item])


def test_t2_collate_rejects_non_string_metadata() -> None:
    """collate 必须在填充前拒绝类型混淆的审计 metadata。"""

    example = generate_t2_cap_corpus(130, 1, 1, 1, 1)["train"][0]
    item = T2CausalLMDataset([example], ByteTokenizer())[0]
    item["source_id"] = None
    with pytest.raises(TypeError, match="source_id 必须是 str"):
        collate_t2_causal_lm_batch([item])


def test_t2_quadruplet_sampler_preserves_mixed_routing_groups() -> None:
    """T2 sampler 每个 batch 必须包含完整四元组和相等 valid/invalid 数量。"""

    examples = generate_t2_cap_corpus(131, 4, 1, 1, 1)["train"]
    sampler = T2QuadrupletBatchSampler(examples, batch_size=8, seed=131)
    batches = list(sampler)
    assert len(batches) == 2
    for indices in batches:
        rows = [examples[index] for index in indices]
        assert len(rows) == 8
        assert sum(row.credential_class == "valid" for row in rows) == 4
        assert sum(row.credential_class == "invalid" for row in rows) == 4
        assert {row.scope for row in rows} == {
            "public",
            "protected_public",
            "protected_private",
            "refusal",
        }


def test_t2_quadruplet_sampler_is_epoch_deterministic() -> None:
    """相同 seed/epoch 顺序可复现，修改 epoch 后使用新派生顺序。"""

    examples = generate_t2_cap_corpus(133, 6, 1, 1, 1)["train"]
    first = T2QuadrupletBatchSampler(examples, batch_size=8, seed=133)
    second = T2QuadrupletBatchSampler(examples, batch_size=8, seed=133)
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)


@pytest.mark.parametrize("batch_size", [1, 6, True])
def test_t2_quadruplet_sampler_rejects_invalid_batch_size(batch_size) -> None:
    """非四倍数、过小或 bool batch size 必须拒绝。"""

    examples = generate_t2_cap_corpus(135, 2, 1, 1, 1)["train"]
    with pytest.raises((TypeError, ValueError)):
        T2QuadrupletBatchSampler(examples, batch_size=batch_size, seed=135)


def test_t2_quadruplet_sampler_rejects_incomplete_groups_and_bad_seed() -> None:
    """sampler 必须拒绝不完整四元组、错误 seed 和不足一批的输入。"""

    examples = generate_t2_cap_corpus(137, 2, 1, 1, 1)["train"]
    with pytest.raises(ValueError, match="四元组"):
        T2QuadrupletBatchSampler(examples[:-1], batch_size=4, seed=137)
    with pytest.raises(TypeError, match="seed"):
        T2QuadrupletBatchSampler(examples, batch_size=4, seed=True)
    with pytest.raises(ValueError, match="不足"):
        T2QuadrupletBatchSampler(examples[:4], batch_size=8, seed=137)


@pytest.mark.parametrize("epoch", [-1, True, 1.5])
def test_t2_quadruplet_sampler_rejects_invalid_epoch(epoch) -> None:
    """恢复顺序使用的 epoch 必须是非负整数。"""

    examples = generate_t2_cap_corpus(139, 2, 1, 1, 1)["train"]
    sampler = T2QuadrupletBatchSampler(examples, batch_size=4, seed=139)
    with pytest.raises(ValueError, match="非负整数"):
        sampler.set_epoch(epoch)
