"""Phase 5.5/T2 自然语言指标专项测试。"""

from dataclasses import replace

import pytest

from src.can.v2.transformer import (
    T2_CAP_SUITE,
    T2_MEM_SUITE,
    T2Prediction,
    evaluate_t2_predictions,
    generate_t2_cap_corpus,
    generate_t2_mem_corpus,
    normalize_t2_answer,
    score_t2_text,
)


def _perfect_predictions(examples):
    """为给定样本构造逐项正确的确定性输出。"""

    return [T2Prediction(row.sample_id, row.target) for row in examples]


def test_normalizer_handles_nfkc_articles_case_and_punctuation() -> None:
    """T2 normalizer 统一兼容字符、冠词、大小写和标点。"""

    assert normalize_t2_answer("  The Ｔier-Seven! ") == "tier seven"


def test_text_score_reports_partial_overlap_and_edit_similarity() -> None:
    """非完全匹配仍应提供可解释的 F1 与 edit similarity。"""

    result = score_t2_text("North Harbor district", ["North Harbor"])
    assert result.exact_match == 0.0
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == 1.0
    assert result.token_f1 == pytest.approx(0.8)
    assert 0.0 < result.edit_similarity < 1.0


def test_text_score_uses_preregistered_aliases() -> None:
    """多个参考答案中任一规范匹配都应得到满分。"""

    result = score_t2_text("seven", ["Tier seven", "seven"])
    assert result.exact_match == 1.0
    assert result.token_f1 == 1.0
    assert result.edit_similarity == 1.0


@pytest.mark.parametrize("references", [[], "answer"])
def test_text_score_rejects_invalid_reference_container(references) -> None:
    """空 references 或裸字符串必须被拒绝。"""

    with pytest.raises(ValueError):
        score_t2_text("answer", references)


def test_cap_perfect_report_separates_unauthorized_and_leakage_fields() -> None:
    """CAP 完美拒答时只定义未授权答案率，不定义 private leakage。"""

    examples = generate_t2_cap_corpus(201, 1, 1, 1, 1)["dev"]
    report = evaluate_t2_predictions(examples, _perfect_predictions(examples))
    assert report.suite_id == T2_CAP_SUITE
    assert report.by_scope["public"].exact_match == 1.0
    assert report.by_scope["protected_private"].token_f1 == 1.0
    assert report.access.refusal_rate == 1.0
    assert report.access.unauthorized_protected_answer_rate == 0.0
    assert report.access.private_fact_leakage_rate is None


def test_cap_protected_answer_is_counted_as_unauthorized_capability() -> None:
    """CAP invalid 请求复述 protected 答案时计入未授权能力执行。"""

    examples = generate_t2_cap_corpus(203, 1, 1, 1, 1)["dev"]
    protected = next(row for row in examples if row.scope == "protected_private")
    predictions = _perfect_predictions(examples)
    predictions = [
        (
            replace(item, generated_text=protected.target)
            if next(row for row in examples if row.sample_id == item.sample_id).scope
            == "refusal"
            else item
        )
        for item in predictions
    ]
    report = evaluate_t2_predictions(examples, predictions)
    assert report.access.refusal_rate == 0.0
    assert report.access.unauthorized_protected_answer_rate == 1.0
    assert report.access.private_fact_leakage_rate is None


def test_mem_protected_answer_is_counted_as_private_fact_leakage() -> None:
    """MEM invalid 请求复述已记忆 protected fact 时计入泄漏。"""

    examples = generate_t2_mem_corpus(205, 1)["validation"]
    protected = next(row for row in examples if row.scope == "protected_private")
    predictions = _perfect_predictions(examples)
    refusal_id = next(row.sample_id for row in examples if row.scope == "refusal")
    predictions = [
        (
            replace(item, generated_text=protected.target)
            if item.sample_id == refusal_id
            else item
        )
        for item in predictions
    ]
    report = evaluate_t2_predictions(examples, predictions)
    assert report.suite_id == T2_MEM_SUITE
    assert report.access.unauthorized_protected_answer_rate is None
    assert report.access.private_fact_leakage_rate == 1.0


def test_public_fallback_is_classified_separately() -> None:
    """公开范围降级不得计入拒答、泄漏或 other。"""

    examples = generate_t2_cap_corpus(207, 1, 1, 1, 1)["dev"]
    refusal = next(row for row in examples if row.scope == "refusal")
    predictions = _perfect_predictions(examples)
    predictions = [
        (
            replace(item, generated_text=refusal.public_fallback)
            if item.sample_id == refusal.sample_id
            else item
        )
        for item in predictions
    ]
    report = evaluate_t2_predictions(examples, predictions)
    assert report.access.public_scope_compliance == 1.0
    assert report.access.refusal_rate == 0.0
    assert report.access.other_rate == 0.0


def test_unrecognized_invalid_output_is_other() -> None:
    """无法归入前三类的 invalid 输出必须稳定计入 other。"""

    examples = generate_t2_cap_corpus(209, 1, 1, 1, 1)["dev"]
    predictions = _perfect_predictions(examples)
    refusal_id = next(row.sample_id for row in examples if row.scope == "refusal")
    predictions = [
        (
            replace(item, generated_text="unrelated response")
            if item.sample_id == refusal_id
            else item
        )
        for item in predictions
    ]
    report = evaluate_t2_predictions(examples, predictions)
    assert report.access.other_rate == 1.0


def test_private_answer_detection_requires_token_boundaries() -> None:
    """protected answer 短别名不得命中较长单词内部的子串。"""

    examples = generate_t2_mem_corpus(210, 1)["validation"]
    protected = next(row for row in examples if row.scope == "protected_private")
    embedded_alias = f"prefix{protected.answer_aliases[-1]}suffix"
    refusal_id = next(row.sample_id for row in examples if row.scope == "refusal")
    predictions = _perfect_predictions(examples)
    predictions = [
        (
            replace(item, generated_text=embedded_alias)
            if item.sample_id == refusal_id
            else item
        )
        for item in predictions
    ]
    report = evaluate_t2_predictions(examples, predictions)
    assert report.access.private_fact_leakage_rate == 0.0
    assert report.access.other_rate == 1.0


def test_prediction_alignment_rejects_missing_and_extra_ids() -> None:
    """预测必须完整覆盖样本且不能带入额外 ID。"""

    examples = generate_t2_cap_corpus(211, 1, 1, 1, 1)["dev"]
    predictions = _perfect_predictions(examples)
    with pytest.raises(ValueError, match="完全覆盖"):
        evaluate_t2_predictions(examples, predictions[:-1])
    with pytest.raises(ValueError, match="完全覆盖"):
        evaluate_t2_predictions(
            examples, predictions + [T2Prediction("extra-sample", "answer")]
        )


def test_prediction_alignment_rejects_duplicate_ids() -> None:
    """重复 prediction ID 必须 fail-fast。"""

    examples = generate_t2_cap_corpus(213, 1, 1, 1, 1)["dev"]
    predictions = _perfect_predictions(examples)
    with pytest.raises(ValueError, match="不能重复"):
        evaluate_t2_predictions(examples, predictions + [predictions[0]])


def test_report_rejects_mixed_suite_or_split() -> None:
    """CAP/MEM 或不同 split 不能在一次报告中混合。"""

    cap = generate_t2_cap_corpus(215, 1, 1, 1, 1)
    mem = generate_t2_mem_corpus(215, 1)
    mixed_suite = [cap["dev"][0], mem["dev"][0]]
    with pytest.raises(ValueError, match="一个 suite"):
        evaluate_t2_predictions(mixed_suite, _perfect_predictions(mixed_suite))
    mixed_split = [cap["dev"][0], cap["validation"][0]]
    with pytest.raises(ValueError, match="一个 suite 和一个 split"):
        evaluate_t2_predictions(mixed_split, _perfect_predictions(mixed_split))


def test_scope_without_rows_is_not_applicable() -> None:
    """显式边界 fixture 中缺少的 scope 必须返回 null/not_applicable。"""

    example = generate_t2_cap_corpus(217, 1, 1, 1, 1)["dev"][0]
    report = evaluate_t2_predictions(
        [example], [T2Prediction(example.sample_id, example.target)]
    )
    assert report.by_scope["protected_private"].status == "not_applicable"
    assert report.by_scope["protected_private"].exact_match is None
    assert report.access.status == "not_applicable"
