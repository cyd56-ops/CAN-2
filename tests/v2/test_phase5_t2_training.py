"""Phase 5.5/T2 四路 T-pretrain trainer 专项测试。"""

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
    T2CanPretrainer,
    T2CausalLMDataset,
    T2PlainPretrainer,
    TransformerConfig,
    build_t2_scope_masks,
    collate_t2_causal_lm_batch,
    generate_t2_cap_corpus,
)


def _config() -> TransformerConfig:
    """返回适合 CPU 单测的最小合法 Transformer 配置。"""

    return TransformerConfig(
        max_seq_len=256,
        num_layers=2,
        cut_layer=1,
        d_model=16,
        num_heads=4,
        d_ff=32,
    )


def _batch() -> dict:
    """构造一个完整且 metadata 对齐的 CAP 四元组 batch。"""

    rows = generate_t2_cap_corpus(401, 1, 1, 1, 1)["train"]
    dataset = T2CausalLMDataset(rows, ByteTokenizer())
    return collate_t2_causal_lm_batch([dataset[index] for index in range(4)])


def _can_fixture():
    """构造安全性足够用于测试的 CAN 模型和 credential 生成器。"""

    params = LWEParams(n=32, m=64)
    matrix, secret, vector = generate_keypair(params, np.random.default_rng(17))
    model = GatedDecoderTransformer(matrix, vector, params, _config())
    generator = CredentialGenerator(matrix, secret, vector, params, seed=18)
    return model, generator


def test_scope_masks_match_frozen_t2_contract() -> None:
    """四类 scope 必须映射到固定的 head 与 credential mask。"""

    scopes = ["public", "protected_public", "protected_private", "refusal"]
    masks = build_t2_scope_masks(scopes, torch.device("cpu"))
    assert masks.protected.tolist() == [False, True, True, False]
    assert masks.public.tolist() == [True, False, False, True]
    assert torch.equal(masks.valid, masks.protected)
    assert torch.equal(masks.invalid, masks.public)


@pytest.mark.parametrize(
    "scopes,error",
    [
        ([], ValueError),
        ("public", TypeError),
        (["unknown"], ValueError),
        ([1], ValueError),
    ],
)
def test_scope_masks_reject_invalid_values(scopes, error) -> None:
    """空、裸字符串、未知值和类型混淆必须 fail closed。"""

    with pytest.raises(error):
        build_t2_scope_masks(scopes, torch.device("cpu"))


def test_scope_masks_reject_non_device() -> None:
    """mask device 必须显式使用 torch.device。"""

    with pytest.raises(TypeError, match="torch.device"):
        build_t2_scope_masks(["public"], "cpu")


def test_plain_train_batch_updates_model_and_counts_tokens() -> None:
    """Plain trainer 应执行一次有限 loss 更新并记录 token。"""

    torch.manual_seed(23)
    model = PlainDecoderTransformer(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = T2PlainPretrainer(model, optimizer, torch.device("cpu"))
    before = model.public_head.weight.detach().clone()
    metrics = trainer.train_batch(_batch())
    assert metrics["loss"] > 0.0
    assert metrics["samples"] == 4.0
    assert metrics["tokens"] > 0.0
    assert metrics["global_step"] == 1.0
    assert not torch.equal(before, model.public_head.weight)


def test_can_train_batch_verifies_mixed_route_and_updates_model() -> None:
    """CAN trainer 应用真实 credential 形成 2 valid/2 invalid 路由。"""

    torch.manual_seed(29)
    model, generator = _can_fixture()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = T2CanPretrainer(model, optimizer, torch.device("cpu"), generator)
    metrics = trainer.train_batch(_batch())
    assert metrics["loss"] > 0.0
    assert metrics["global_step"] == 1.0
    assert trainer.global_step == 1


@pytest.mark.parametrize("field", ["scopes", "credential_classes"])
def test_train_batch_rejects_metadata_length_mismatch(field: str) -> None:
    """scope 或 credential metadata 长度漂移必须在 forward 前失败。"""

    model = PlainDecoderTransformer(_config())
    trainer = T2PlainPretrainer(
        model, torch.optim.AdamW(model.parameters()), torch.device("cpu")
    )
    batch = _batch()
    batch[field] = batch[field][:-1]
    with pytest.raises(ValueError, match="metadata"):
        trainer.train_batch(batch)


def test_train_batch_rejects_credential_scope_mismatch() -> None:
    """调用方不得用 metadata 把 invalid scope 标成 valid。"""

    model = PlainDecoderTransformer(_config())
    trainer = T2PlainPretrainer(
        model, torch.optim.AdamW(model.parameters()), torch.device("cpu")
    )
    batch = _batch()
    batch["credential_classes"][0] = "valid"
    with pytest.raises(ValueError, match="credential_classes"):
        trainer.train_batch(batch)


def test_train_batch_rejects_incomplete_quadruplet() -> None:
    """缺少任一 scope 的 batch 不得进入优化器。"""

    model = PlainDecoderTransformer(_config())
    trainer = T2PlainPretrainer(
        model, torch.optim.AdamW(model.parameters()), torch.device("cpu")
    )
    batch = _batch()
    for key in ("input_ids", "labels", "attention_mask"):
        batch[key] = batch[key][:-1]
    for key in ("scopes", "credential_classes"):
        batch[key] = batch[key][:-1]
    with pytest.raises(ValueError, match="完整四元组"):
        trainer.train_batch(batch)


def test_trainer_rejects_optimizer_for_another_model() -> None:
    """optimizer 与模型错配时必须在初始化阶段失败。"""

    model = PlainDecoderTransformer(_config())
    other = PlainDecoderTransformer(_config())
    with pytest.raises(ValueError, match="optimizer"):
        T2PlainPretrainer(
            model, torch.optim.AdamW(other.parameters()), torch.device("cpu")
        )


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), True])
def test_trainer_rejects_invalid_loss_weight(weight) -> None:
    """监督权重必须是有限正数且不能是 bool。"""

    model = PlainDecoderTransformer(_config())
    with pytest.raises((TypeError, ValueError)):
        T2PlainPretrainer(
            model,
            torch.optim.AdamW(model.parameters()),
            torch.device("cpu"),
            protected_weight=weight,
        )


def test_train_batch_rejects_non_finite_loss() -> None:
    """模型参数污染导致的非有限 loss 必须在 optimizer step 前中止。"""

    model = PlainDecoderTransformer(_config())
    with torch.no_grad():
        model.public_head.weight.fill_(float("nan"))
    trainer = T2PlainPretrainer(
        model, torch.optim.AdamW(model.parameters()), torch.device("cpu")
    )
    with pytest.raises(FloatingPointError, match="非有限 loss"):
        trainer.train_batch(_batch())


def test_train_batch_rejects_non_tensor_inputs() -> None:
    """collate 张量被外部类型替换后必须 fail closed。"""

    model = PlainDecoderTransformer(_config())
    trainer = T2PlainPretrainer(
        model, torch.optim.AdamW(model.parameters()), torch.device("cpu")
    )
    batch = _batch()
    batch["input_ids"] = []
    with pytest.raises(TypeError, match="必须是 Tensor"):
        trainer.train_batch(batch)
