"""Phase 5.5/T2 四路 T-pretrain trainer 专项测试。"""

import copy
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
    t2_parameter_groups,
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


def test_plain_train_batch_reports_scope_and_gradient_diagnostics() -> None:
    """Plain 一步训练应报告四 scope、双 head 和三参数组观测。"""

    torch.manual_seed(24)
    model = PlainDecoderTransformer(_config())
    trainer = T2PlainPretrainer(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3), torch.device("cpu")
    )
    metrics = trainer.train_batch(_batch())
    assert metrics["total_loss"] == pytest.approx(metrics["loss"])
    assert set(metrics["scope_losses"]) == {
        "public",
        "protected_public",
        "protected_private",
        "refusal",
    }
    assert all(value > 0 for value in metrics["scope_answer_tokens"].values())
    assert set(metrics["gradient_norms"]) == {
        "shared_prefix",
        "protected_path",
        "public_path",
    }
    assert metrics["gate"] is None


def test_scope_losses_reconstruct_token_weighted_head_losses() -> None:
    """四 scope loss 应按答案 token 数重建原双 head 训练 loss。"""

    torch.manual_seed(26)
    model = PlainDecoderTransformer(_config())
    trainer = T2PlainPretrainer(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3), torch.device("cpu")
    )
    metrics = trainer.train_batch(_batch())
    losses = metrics["scope_losses"]
    tokens = metrics["scope_answer_tokens"]

    def weighted(scopes) -> float:
        """按各 scope 的答案 token 数计算手工加权均值。"""

        return sum(losses[scope] * tokens[scope] for scope in scopes) / sum(
            tokens[scope] for scope in scopes
        )

    assert metrics["public_head_loss"] == pytest.approx(
        weighted(("public", "refusal")), rel=1e-6
    )
    assert metrics["protected_head_loss"] == pytest.approx(
        weighted(("protected_public", "protected_private")), rel=1e-6
    )


def test_diagnostics_do_not_change_plain_parameter_update() -> None:
    """启用观测不得改变 Plain 一步优化后的任何参数。"""

    torch.manual_seed(25)
    observed = PlainDecoderTransformer(_config())
    silent = PlainDecoderTransformer(_config())
    silent.load_state_dict(observed.state_dict())
    observed_trainer = T2PlainPretrainer(
        observed,
        torch.optim.SGD(observed.parameters(), lr=1e-3),
        torch.device("cpu"),
        collect_diagnostics=True,
    )
    silent_trainer = T2PlainPretrainer(
        silent,
        torch.optim.SGD(silent.parameters(), lr=1e-3),
        torch.device("cpu"),
        collect_diagnostics=False,
    )
    observed_trainer.train_batch(_batch())
    silent_trainer.train_batch(_batch())
    for left, right in zip(observed.parameters(), silent.parameters()):
        assert torch.equal(left, right)


def test_parameter_groups_are_mutually_exclusive_and_complete() -> None:
    """三组梯度参数必须互斥并完整覆盖全部可训练参数。"""

    model = PlainDecoderTransformer(_config())
    groups = t2_parameter_groups(model)
    identifiers = [
        {id(parameter) for parameter in values} for values in groups.values()
    ]
    assert not identifiers[0] & identifiers[1]
    assert not identifiers[0] & identifiers[2]
    assert not identifiers[1] & identifiers[2]
    assert set().union(*identifiers) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_can_diagnostics_leave_parameter_update_identical() -> None:
    """CAN 真实 Gate 与相同凭证下，开启观测不得改变一步更新。"""
    torch.manual_seed(27)
    model, generator = _can_fixture()
    other, other_generator = copy.deepcopy(model), copy.deepcopy(generator)
    left = T2CanPretrainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=0.001),
        torch.device("cpu"),
        generator,
    )
    right = T2CanPretrainer(
        other,
        torch.optim.AdamW(other.parameters(), lr=0.001),
        torch.device("cpu"),
        other_generator,
        collect_diagnostics=False,
    )
    left.train_batch(_batch())
    right.train_batch(_batch())
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), other.parameters())
    )


def test_can_train_batch_reports_gate_subsets() -> None:
    """CAN Gate 摘要应分别覆盖两个 valid 与两个 invalid 样本。"""

    torch.manual_seed(28)
    model, generator = _can_fixture()
    trainer = T2CanPretrainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        torch.device("cpu"),
        generator,
    )
    gate = trainer.train_batch(_batch())["gate"]
    assert gate["valid"]["count"] == 2
    assert gate["invalid"]["count"] == 2
    for credential_class in ("valid", "invalid"):
        for metric in ("signal", "error_norm"):
            values = gate[credential_class][metric]
            assert values["min"] <= values["mean"] <= values["max"]


def test_parameter_groups_reject_unrelated_model() -> None:
    """参数分组 API 不接受任意 nn.Module。"""

    with pytest.raises(TypeError, match="T2 Plain/CAN"):
        t2_parameter_groups(torch.nn.Linear(2, 2))


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
