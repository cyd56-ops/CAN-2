"""PlainDecoderTransformer 对照模型的契约测试。"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.train_phase5_plain_exploratory import (
    _protocol_from_freeze,
)
from scripts.train_phase5_plain_exploratory import main as plain_main
from src.can.v2.crypto.lwe import LWEParams
from src.can.v2.transformer import (
    ByteTokenizer,
    EntityTripletBatchSampler,
    GatedDecoderTransformer,
    PlainDecoderTrainer,
    PlainDecoderTransformer,
    PlainGenerationOutput,
    SyntheticKnowledgeDataset,
    TransformerConfig,
    build_memorization_validation,
    collate_causal_lm_batch,
    generate_synthetic_corpus,
)


def _config() -> TransformerConfig:
    """构造与 T0 拓扑一致的小型测试配置。"""
    return TransformerConfig(
        max_seq_len=256,
        num_layers=3,
        cut_layer=1,
        d_model=32,
        num_heads=4,
        d_ff=64,
    )


def _model() -> PlainDecoderTransformer:
    """构造固定随机状态的 Plain 模型。"""
    torch.manual_seed(20260903)
    return PlainDecoderTransformer(_config())


def test_plain_model_has_no_gate_or_credential_and_returns_both_heads() -> None:
    """Plain 模型不应挂载 Gate，forward 应返回两个完整 logits。"""
    model = _model()
    assert not hasattr(model, "gate_layer")
    output = model(torch.tensor([[256, 65, 66]], dtype=torch.long))
    assert output.protected_logits.shape == (1, 3, 260)
    assert output.public_logits.shape == (1, 3, 260)


def test_plain_and_gated_models_share_identical_initial_language_weights() -> None:
    """相同 torch seed 下两模型的全部语言建模参数应逐项相同。"""
    config = _config()
    params = LWEParams(n=16, m=32)
    A = np.zeros((params.m, params.n), dtype=np.float32)
    b = np.zeros(params.m, dtype=np.float32)
    torch.manual_seed(20260903)
    gated = GatedDecoderTransformer(A, b, params, config)
    torch.manual_seed(20260903)
    plain = PlainDecoderTransformer(config)
    gated_state = gated.state_dict()
    plain_state = plain.state_dict()
    assert set(plain_state).issubset(gated_state)
    for name, value in plain_state.items():
        assert torch.equal(value, gated_state[name]), name


def test_plain_model_reuses_config_and_rejects_bad_head() -> None:
    """Plain 模型应接受 canonical TransformerConfig 并拒绝未知 head。"""
    model = _model().eval()
    ids = torch.tensor([[256, 65, 66]], dtype=torch.long)
    assert model.config == _config()
    with pytest.raises(ValueError, match="head"):
        model.logits(ids, "invalid")
    with pytest.raises(ValueError, match="head"):
        model.generate(ids, "invalid")


def test_plain_generation_none_and_kv_are_identical_for_both_heads() -> None:
    """两种缓存模式在 greedy 生成中应保持 token 和停止原因一致。"""
    model = _model().eval()
    ids = torch.tensor([[256, 65, 66], [256, 67, 258]], dtype=torch.long)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    for head in ("public", "protected"):
        none = model.generate(ids, head, mask, max_new_tokens=3, cache_mode="none")
        kv = model.generate(ids, head, mask, max_new_tokens=3, cache_mode="kv")
        assert isinstance(none, PlainGenerationOutput)
        assert none.token_ids == kv.token_ids
        assert none.stop_reasons == kv.stop_reasons
        assert none.cache_lengths == kv.cache_lengths


def test_plain_generation_requires_eval_and_valid_inputs() -> None:
    """生成接口应拒绝训练态、空 batch 和错误 mask。"""
    model = _model()
    ids = torch.tensor([[256, 65]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="eval"):
        model.generate(ids, "public")
    model.eval()
    with pytest.raises(ValueError):
        model(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(TypeError):
        model(ids, torch.ones_like(ids))  # type: ignore[arg-type]


def test_plain_trainer_runs_same_triplet_batch_contract() -> None:
    """Plain 训练器应在相同 entity-triplet 数据上产生有限 loss 和 token 计数。"""
    tokenizer = ByteTokenizer()
    examples = generate_synthetic_corpus(20260903, 2, 1, 1)["train"]
    sampler = EntityTripletBatchSampler(examples, batch_size=6, seed=20260903)
    loader = DataLoader(
        SyntheticKnowledgeDataset(examples, tokenizer, max_length=256),
        batch_sampler=sampler,
        collate_fn=collate_causal_lm_batch,
    )
    model = _model()
    trainer = PlainDecoderTrainer(
        model,
        loader,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        torch.device("cpu"),
    )
    metrics = trainer.train_epoch()
    assert metrics["tokens"] > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_plain_generation_uses_explicit_oracle_head_on_same_private_prompt() -> None:
    """private/refusal 相同 prompt 时必须由评估器显式选择 head。"""
    model = _model().eval()
    examples = build_memorization_validation(
        generate_synthetic_corpus(20260903, 2, 1, 1)["train"], 1
    )
    private = next(item for item in examples if item.scope == "private")
    refusal = next(item for item in examples if item.scope == "refusal")
    assert private.prompt == refusal.prompt
    ids = torch.tensor(
        [ByteTokenizer().encode(private.prompt, add_eos=False)], dtype=torch.long
    )
    protected = model.generate(ids, "protected", max_new_tokens=1)
    public = model.generate(ids, "public", max_new_tokens=1)
    assert protected.head == "protected"
    assert public.head == "public"


def test_plain_protocol_reads_v3_and_rejects_runtime_drift(tmp_path: Path) -> None:
    """Plain 入口必须从 freeze v3 取值，并拒绝 batch 等运行时漂移。"""
    source = Path("experiments/phase5_freeze_v3/freeze_record.json")
    record = tmp_path / "freeze_record.json"
    record.write_bytes(source.read_bytes())
    args = argparse.Namespace(
        batch_size=None,
        validation_interval=None,
        max_new_tokens=None,
        cache_mode=None,
    )
    protocol = _protocol_from_freeze(record, 20260903, args)
    assert protocol["batch_size"] == 144
    assert protocol["config"] == TransformerConfig()
    args.batch_size = 6
    with pytest.raises(ValueError, match="batch size"):
        _protocol_from_freeze(record, 20260903, args)


def test_plain_protocol_rejects_unknown_seed(tmp_path: Path) -> None:
    """未列入 freeze v3 的 seed 不得伪装成严格对照。"""
    source = Path("experiments/phase5_freeze_v3/freeze_record.json")
    record = tmp_path / "freeze_record.json"
    record.write_bytes(source.read_bytes())
    args = argparse.Namespace(
        batch_size=None,
        validation_interval=None,
        max_new_tokens=None,
        cache_mode=None,
    )
    with pytest.raises(ValueError, match="seed"):
        _protocol_from_freeze(record, 7, args)


@pytest.mark.parametrize("value", [True, "144", 0])
def test_plain_protocol_rejects_noncanonical_batch(
    tmp_path: Path, value: object
) -> None:
    """外部 freeze record 的类型混淆和非正 batch 必须 fail closed。"""
    source = Path("experiments/phase5_freeze_v3/freeze_record.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["batch_size"] = value
    record = tmp_path / "freeze_record.json"
    record.write_text(json.dumps(payload), encoding="utf-8")
    args = argparse.Namespace(
        batch_size=None,
        validation_interval=None,
        max_new_tokens=None,
        cache_mode=None,
    )
    with pytest.raises(ValueError, match="batch_size"):
        _protocol_from_freeze(record, 20260903, args)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--seed", "20260904", "--budget", "5000000"],
        ["--seed", "20260903", "--budget", "6000000"],
    ],
)
def test_plain_e1_rejects_seed_or_budget_drift(
    tmp_path: Path, arguments: List[str]
) -> None:
    """一对一 E1 对照不得改变 CAN E1 的 seed 或 token 预算。"""
    with pytest.raises(ValueError, match="Plain E1"):
        plain_main(["--output", str(tmp_path / "result"), *arguments])
