"""Phase 5.5/T2 统一 evaluator 专项测试。"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    ByteTokenizer,
    GatedDecoderTransformer,
    PlainDecoderTransformer,
    PlainGenerationOutput,
    T2Evaluator,
    TransformerConfig,
    generate_t2_cap_corpus,
)


def _config() -> TransformerConfig:
    """返回 evaluator CPU 测试使用的最小模型配置。"""

    return TransformerConfig(
        max_seq_len=256,
        num_layers=2,
        cut_layer=1,
        d_model=16,
        num_heads=4,
        d_ff=32,
    )


def _rows():
    """返回一个完整 CAP dev 四元组。"""

    return generate_t2_cap_corpus(501, 1, 1, 1, 1)["dev"]


def _can():
    """构造 CAN evaluator 所需模型和 credential 生成器。"""

    params = LWEParams(n=32, m=64)
    matrix, secret, vector = generate_keypair(params, np.random.default_rng(31))
    model = GatedDecoderTransformer(matrix, vector, params, _config())
    generator = CredentialGenerator(matrix, secret, vector, params, seed=32)
    return model, generator


def test_plain_evaluator_reports_oracle_head_without_gate_metrics() -> None:
    """Plain 结果必须明确 oracle-head，且不能伪造 routing 证据。"""

    model = PlainDecoderTransformer(_config())
    result = T2Evaluator(
        model, ByteTokenizer(), torch.device("cpu"), max_new_tokens=1, batch_size=4
    ).evaluate(_rows())
    assert result["status"] == "ok"
    assert result["route_mode"] == "oracle_head"
    assert result["gate_or_credential"] is False
    assert result["routing"] is None
    assert len(result["diagnostics"]) == 4
    assert set(result["teacher_forced_by_scope"]) == {
        "public",
        "protected_public",
        "protected_private",
        "refusal",
    }


def test_can_evaluator_enforces_route_once_and_complete_coverage() -> None:
    """CAN 一批四元组必须得到 2/2 路由和逐序列一次提交。"""

    model, generator = _can()
    result = T2Evaluator(
        model,
        ByteTokenizer(),
        torch.device("cpu"),
        generator,
        max_new_tokens=1,
        batch_size=4,
    ).evaluate(_rows())
    routing = result["routing"]
    assert result["route_mode"] == "credential_gate"
    assert routing == {
        "route_calls": 4,
        "valid": 2,
        "invalid": 2,
        "protected_indices": 2,
        "public_indices": 2,
        "rejected_indices": 0,
        "invalid_protected_block_calls": 0,
    }


def test_can_evaluator_never_sends_invalid_rows_to_protected_block(monkeypatch) -> None:
    """protected block 的生成子批大小只能等于 valid 行数。"""

    model, generator = _can()
    original = model._forward_protected
    observed = []

    def wrapped(hidden, attention_mask):
        """记录 protected 路径实际接收的 batch 大小。"""

        observed.append(hidden.shape[0])
        return original(hidden, attention_mask)

    monkeypatch.setattr(model, "_forward_protected", wrapped)
    T2Evaluator(
        model,
        ByteTokenizer(),
        torch.device("cpu"),
        generator,
        max_new_tokens=1,
        batch_size=4,
    ).evaluate(_rows())
    # 一次 mixed 生成接收 2 个 valid；teacher-forced 只逐条接收 protected 行。
    assert observed[0] == 2
    assert all(size in {1, 2} for size in observed)


def test_evaluator_aligns_diagnostics_by_sample_id() -> None:
    """逐样本诊断必须与输入 sample ID 一一对应。"""

    rows = _rows()
    result = T2Evaluator(
        PlainDecoderTransformer(_config()),
        ByteTokenizer(),
        torch.device("cpu"),
        max_new_tokens=1,
        batch_size=4,
    ).evaluate(rows)
    assert [item["sample_id"] for item in result["diagnostics"]] == [
        row.sample_id for row in rows
    ]
    assert all(
        np.isfinite(item["teacher_forced_token_loss"])
        and 0.0 <= item["teacher_forced_token_accuracy"] <= 1.0
        for item in result["diagnostics"]
    )


@pytest.mark.parametrize("rows", [[], "rows"])
def test_evaluator_rejects_invalid_example_container(rows) -> None:
    """空输入和裸字符串不得被解释为样本列表。"""

    evaluator = T2Evaluator(
        PlainDecoderTransformer(_config()),
        ByteTokenizer(),
        torch.device("cpu"),
        max_new_tokens=1,
        batch_size=4,
    )
    with pytest.raises((TypeError, ValueError)):
        evaluator.evaluate(rows)


def test_evaluator_rejects_mixed_split() -> None:
    """一次评估混合 dev/validation 必须失败。"""

    corpus = generate_t2_cap_corpus(503, 1, 1, 1, 1)
    rows = list(corpus["dev"])
    rows[0] = corpus["validation"][0]
    evaluator = T2Evaluator(
        PlainDecoderTransformer(_config()),
        ByteTokenizer(),
        torch.device("cpu"),
        max_new_tokens=1,
        batch_size=4,
    )
    with pytest.raises(ValueError, match="一个 suite 和一个 split"):
        evaluator.evaluate(rows)


def test_evaluator_rejects_partial_quadruplet() -> None:
    """最后不足四元组的评估 batch 不得静默跳过。"""

    evaluator = T2Evaluator(
        PlainDecoderTransformer(_config()),
        ByteTokenizer(),
        torch.device("cpu"),
        max_new_tokens=1,
        batch_size=4,
    )
    with pytest.raises(ValueError, match="完整 T2 四元组"):
        evaluator.evaluate(_rows()[:-1])


def test_evaluator_enforces_credential_boundary() -> None:
    """CAN 缺 credential 或 Plain 接收 credential 都必须失败。"""

    can, generator = _can()
    with pytest.raises(ValueError, match="CAN 完整评估"):
        T2Evaluator(can, ByteTokenizer(), torch.device("cpu"))
    plain = PlainDecoderTransformer(_config())
    with pytest.raises(ValueError, match="Plain evaluator"):
        T2Evaluator(plain, ByteTokenizer(), torch.device("cpu"), generator)


@pytest.mark.parametrize("batch_size", [1, 6, True])
def test_evaluator_rejects_invalid_batch_size(batch_size) -> None:
    """evaluator batch 必须遵守完整四元组边界。"""

    with pytest.raises((TypeError, ValueError)):
        T2Evaluator(
            PlainDecoderTransformer(_config()),
            ByteTokenizer(),
            torch.device("cpu"),
            batch_size=batch_size,
        )


def test_evaluator_rejects_duplicate_sample_id() -> None:
    """重复 sample ID 必须在生成前失败。"""

    rows = _rows()
    rows[1] = replace(rows[1], sample_id=rows[0].sample_id)
    evaluator = T2Evaluator(
        PlainDecoderTransformer(_config()),
        ByteTokenizer(),
        torch.device("cpu"),
        max_new_tokens=1,
        batch_size=4,
    )
    with pytest.raises(ValueError, match="sample_id"):
        evaluator.evaluate(rows)


@pytest.mark.parametrize(
    "generated_suffix,expected_reason,expected_control_count",
    [
        ((0,), "invalid_control_token", 1),
        ((0xC2, 0x80), "invalid_control_token", 1),
        ((ByteTokenizer.pad_token_id,), "invalid_special_token", 0),
        ((0xFF,), "invalid_utf8", 0),
    ],
)
def test_evaluator_records_invalid_generated_tokens_without_crashing(
    monkeypatch,
    generated_suffix,
    expected_reason: str,
    expected_control_count: int,
) -> None:
    """控制、特殊及非法 UTF-8 token 必须稳定计为失败并留下诊断。"""

    model = PlainDecoderTransformer(_config())

    def fake_generate(
        input_ids,
        head,
        attention_mask=None,
        max_new_tokens=16,
        eos_token_id=ByteTokenizer.eos_token_id,
        pad_token_id=ByteTokenizer.pad_token_id,
        cache_mode="none",
    ):
        """返回带指定异常后缀的确定性生成结果。"""

        del max_new_tokens, eos_token_id, pad_token_id, cache_mode
        assert attention_mask is not None
        lengths = attention_mask.sum(dim=1).tolist()
        sequences = tuple(
            tuple(input_ids[index, : int(length)].tolist()) + generated_suffix
            for index, length in enumerate(lengths)
        )
        return PlainGenerationOutput(
            token_ids=sequences,
            head=head,
            stop_reasons=("max_new_tokens",) * len(sequences),
            cache_lengths=(),
        )

    monkeypatch.setattr(model, "generate", fake_generate)
    result = T2Evaluator(
        model,
        ByteTokenizer(),
        torch.device("cpu"),
        max_new_tokens=max(1, len(generated_suffix)),
        batch_size=4,
    ).evaluate(_rows())

    assert result["status"] == "ok"
    assert result["generation_safety"]["status"] == "invalid_generation_observed"
    assert result["generation_safety"]["invalid_sequences"] == 4
    assert result["generation_safety"][f"{expected_reason}_sequences"] == 4
    for diagnostic in result["diagnostics"]:
        assert diagnostic["generation_valid"] is False
        assert diagnostic["generated_text"] == "[INVALID-GENERATION]"
        assert "\x00" not in diagnostic["generated_text"]
        assert diagnostic["generated_token_ids"] == list(generated_suffix)
        assert diagnostic["invalid_generation_reason"] == expected_reason
        assert diagnostic["invalid_control_token_count"] == expected_control_count
        assert diagnostic["stop_reason"] == expected_reason
        assert diagnostic["model_stop_reason"] == "max_new_tokens"
