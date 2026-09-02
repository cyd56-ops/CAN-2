"""Phase 5 T1 reference、规范化和恢复率专项测试。"""

import numpy as np
import pytest
import torch

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.transformer import (
    GatedDecoderTransformer,
    ProbeConfig,
    RecoveryConfig,
    TransformerConfig,
    classify_refusal,
    compute_recovery_rate,
    freeze_record_sha256,
    load_freeze_record,
    normalize_answer,
    run_probe,
    validate_direct_reference,
    validate_generation_reference,
    validate_runtime_against_freeze,
)


def _fixture():
    """构造确定性小模型及 valid/invalid credential。"""
    torch.manual_seed(9)
    params = LWEParams(n=16, m=32)
    A, secret, b = generate_keypair(params, np.random.default_rng(9))
    model = GatedDecoderTransformer(
        A,
        b,
        params,
        TransformerConfig(num_layers=3, cut_layer=1, d_model=32, num_heads=4, d_ff=64),
    ).eval()
    invalid = np.full(params.n, 10.0, dtype=np.float32)
    return (
        model,
        torch.tensor([[256, 65], [256, 66]]),
        torch.tensor(np.stack([secret, invalid]), dtype=torch.float32),
    )


def test_reference_matches_per_sample_generation() -> None:
    """mixed batch 应与逐样本 reference 完全一致。"""
    model, tokens, credentials = _fixture()
    result = validate_generation_reference(model, tokens, credentials, max_new_tokens=2)
    assert result.trace.status == "ok"
    assert result.trace.matched_sequences == 2
    assert result.trace.max_token_mismatch == 0
    assert result.trace.max_abs_difference is not None
    assert result.empty_protected == 0
    assert result.empty_public == 0


def test_reference_kv_is_explicitly_blocked() -> None:
    """未交付 KV cache 时必须显式 blocked。"""
    model, tokens, credentials = _fixture()
    result = validate_generation_reference(model, tokens, credentials, cache_mode="kv")
    assert result.trace.status == "blocked"


def test_reference_counts_empty_public_subbatch() -> None:
    """全 valid batch 应记录 public 空子批。"""
    model, tokens, credentials = _fixture()
    result = validate_generation_reference(
        model, tokens[:1], credentials[:1], max_new_tokens=1
    )
    assert result.empty_public == 1


def test_reference_rejects_bad_shapes() -> None:
    """reference 输入 shape 错误必须 fail closed。"""
    model, tokens, credentials = _fixture()
    with pytest.raises(ValueError):
        validate_generation_reference(model, tokens[:1], credentials)


def test_direct_reference_reports_equivalence() -> None:
    """protected routed logits 应与 direct reference 一致。"""
    model, tokens, credentials = _fixture()
    result = validate_direct_reference(model, tokens[:1], credentials[:1])
    assert result.logits_allclose
    assert result.greedy_token_match
    assert result.max_abs_difference <= 1e-5


def test_recovery_rate_preserves_negative_values() -> None:
    """恢复率不得将负值截断。"""
    assert compute_recovery_rate(0.1, 0.8) < 0


def test_phase5_configs_validate() -> None:
    """probe/recovery 配置拒绝非法预算。"""
    with pytest.raises(ValueError):
        ProbeConfig(target_layer=-1)
    with pytest.raises(ValueError):
        RecoveryConfig("a" * 64, "x", 0, 1, "b" * 64, 1)


@pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_recovery_rate_bounds(value: float) -> None:
    """恢复率输入边界可计算。"""
    assert np.isfinite(compute_recovery_rate(value, 0.2))


@pytest.mark.parametrize("mode", ["none", "kv"])
def test_reference_modes_are_explicit(mode: str) -> None:
    """reference 模式必须有明确状态。"""
    model, tokens, credentials = _fixture()
    result = validate_generation_reference(
        model, tokens, credentials, max_new_tokens=1, cache_mode=mode
    )
    assert result.trace.status in {"ok", "blocked"}


def test_kv_cache_generation_matches_recompute() -> None:
    """真实 KV 增量路径应与 none 模式生成完全一致。"""
    model, tokens, credentials = _fixture()
    baseline = model.generate(tokens, credentials, max_new_tokens=2)
    cached = model.generate(tokens, credentials, max_new_tokens=2, cache_mode="kv")
    assert cached.token_ids == baseline.token_ids
    assert cached.cache_lengths == (4, 4)


@pytest.mark.parametrize(
    "raw,expected", [(" A  B ", "a b"), ("Ａ", "a"), ("X\nY", "x y"), ("  ", "")]
)
def test_normalization_contract(raw: str, expected: str) -> None:
    """答案规范化遵循冻结规则。"""
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(
    "generated,expected",
    [
        ("DENIED", "refusal"),
        ("PRIVATE-1", "leaked_private"),
        ("PUBLIC-1", "public_scope_compliant"),
        ("noise", "other"),
    ],
)
def test_refusal_classification_contract(generated: str, expected: str) -> None:
    """拒答分类四类互斥。"""
    assert classify_refusal(generated, "DENIED", "PRIVATE-1", "PUBLIC-1") == expected


@pytest.mark.parametrize("baseline", [0.0, 0.2, 0.5, 0.8])
def test_recovery_rate_is_finite(baseline: float) -> None:
    """合法 baseline 下 recovery rate 必须有限。"""
    assert np.isfinite(compute_recovery_rate(0.4, baseline))


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_recovery_rate_rejects_out_of_range(bad: float) -> None:
    """exact match 越界必须拒绝。"""
    with pytest.raises(ValueError):
        compute_recovery_rate(bad, 0.2)


def test_probe_returns_true_auc_and_separability() -> None:
    """probe 输出 ROC-AUC 和方向无关 separability。"""
    x = np.array([[0.0], [0.1], [1.0], [1.1]])
    result = run_probe(x, [0, 1, 0, 1], x, [0, 1, 0, 1], ProbeConfig(target_layer=1))
    assert 0.0 <= result.auc <= 1.0
    assert result.separability >= 0.5


def test_probe_rejects_single_class_training() -> None:
    """probe 训练集单类别时必须失败。"""
    with pytest.raises(ValueError):
        run_probe(
            np.ones((2, 1)),
            [0, 0],
            np.ones((2, 1)),
            [0, 1],
            ProbeConfig(target_layer=1),
        )


def test_freeze_record_load_and_runtime_validation(tmp_path) -> None:
    """freeze record 可读取并拒绝运行时不一致配置。"""
    path = tmp_path / "freeze.json"
    path.write_text(
        '{"freeze_version":"v1","generator_version":"phase5-t1-private-query-v2","batch_size":144,"cache_mode":"kv"}',
        encoding="utf-8",
    )
    record = load_freeze_record(path)
    validate_runtime_against_freeze(record, batch_size=144, cache_mode="kv")
    assert len(freeze_record_sha256(path)) == 64
    with pytest.raises(ValueError):
        validate_runtime_against_freeze(record, batch_size=192)


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"freeze_version":"v1","generator_version":"x","batch_size":5,"cache_mode":"kv"}',
        '{"freeze_version":"v1","generator_version":"x","batch_size":6,"cache_mode":"bad"}',
    ],
)
def test_freeze_record_rejects_invalid_payloads(tmp_path, payload: str) -> None:
    """freeze record 缺字段或非法值必须 fail-fast。"""
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_freeze_record(path)
