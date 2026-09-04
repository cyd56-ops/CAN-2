"""Phase 5 E2 exploratory 数据协议专项测试。"""

import pytest

from src.can.v2.transformer import (
    build_same_template_validation,
    generate_e2_corpus,
)


def test_structured_protocol_is_deterministic_and_private_answer_not_prompt() -> None:
    """结构化协议在相同 seed 下稳定，private 答案不进入 prompt。"""
    first = generate_e2_corpus(7, 12, 2, 2, protocol="structured")
    second = generate_e2_corpus(7, 12, 2, 2, protocol="structured")
    assert first == second
    for example in first["train"]:
        if example.scope == "private":
            assert example.answer.strip() not in example.prompt


def test_random_short_protocol_is_seeded_and_has_three_digit_codes() -> None:
    """短随机 code 可复现且格式固定。"""
    corpus = generate_e2_corpus(9, 12, 1, 1, protocol="random-short")
    private = [item for item in corpus["train"] if item.scope == "private"]
    assert all(item.answer.strip().startswith("CODE-") for item in private)
    assert all(len(item.answer.strip().split("-", 1)[1]) == 3 for item in private)


def test_same_template_validation_reuses_training_entities_and_prompts() -> None:
    """同模板 validation 保留训练实体、prompt 和答案映射。"""
    corpus = generate_e2_corpus(11, 12, 1, 1)
    validation = build_same_template_validation(corpus["train"], 4)
    train = {(item.entity_id, item.scope): item for item in corpus["train"]}
    assert len(validation) == 12
    for item in validation:
        source = train[(item.entity_id, item.scope)]
        assert item.prompt == source.prompt
        assert item.answer == source.answer


def test_paraphrase_validation_does_not_leak_answers() -> None:
    """改写 prompt 不应把 public/private 答案复制进输入。"""
    corpus = generate_e2_corpus(13, 12, 1, 1)
    validation = build_same_template_validation(
        corpus["train"], 2, prompt_mode="paraphrase"
    )
    for item in validation:
        assert item.answer.strip() not in item.prompt


def test_e2_protocol_rejects_unknown_values() -> None:
    """未知协议和 prompt 模式必须 fail fast。"""
    with pytest.raises(ValueError):
        generate_e2_corpus(1, protocol="unknown")
    with pytest.raises(ValueError):
        generate_e2_corpus(1, prompt_mode="unknown")
