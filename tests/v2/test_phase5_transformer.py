"""Phase 5 T0 tokenizer、数据、模型与训练契约测试。"""

import copy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    ByteTokenizer,
    EntityTripletBatchSampler,
    GatedDecoderTransformer,
    KnowledgeExample,
    Phase5Trainer,
    PretrainMetrics,
    SyntheticKnowledgeDataset,
    TransformerConfig,
    build_memorization_validation,
    causal_distillation_loss,
    collate_causal_lm_batch,
    configure_stage,
    freeze_teacher,
    generate_synthetic_corpus,
    masked_causal_lm_loss,
    pretrain_go_no_go,
    validate_mixed_routing,
)


def _model_fixture() -> (
    Tuple[GatedDecoderTransformer, LWEParams, np.ndarray, np.ndarray]
):
    """构造确定性的小尺寸 Transformer 和 valid/invalid credential。"""

    torch.manual_seed(20260901)
    params = LWEParams(n=16, m=32)
    A, secret, b = generate_keypair(params, np.random.default_rng(20260901))
    config = TransformerConfig(
        max_seq_len=64,
        num_layers=3,
        cut_layer=1,
        d_model=32,
        num_heads=4,
        d_ff=64,
    )
    model = GatedDecoderTransformer(A, b, params, config)
    invalid = np.full(params.n, 10.0, dtype=np.float32)
    return model, params, secret, invalid


def _short_examples() -> List[KnowledgeExample]:
    """构造两个实体的短 triplet，供训练循环测试使用。"""

    examples: List[KnowledgeExample] = []
    for entity in ("e0", "e1"):
        examples.extend(
            [
                KnowledgeExample(
                    f"{entity}-public",
                    "public",
                    entity,
                    "lookup",
                    f"Public {entity}:",
                    " OK",
                    "train",
                    3,
                ),
                KnowledgeExample(
                    f"{entity}-private",
                    "private",
                    entity,
                    "lookup",
                    f"Private {entity}:",
                    " SECRET",
                    "train",
                    3,
                ),
                KnowledgeExample(
                    f"{entity}-refusal",
                    "refusal",
                    entity,
                    "lookup",
                    f"Private {entity}:",
                    " DENIED",
                    "train",
                    3,
                ),
            ]
        )
    return examples


def test_byte_tokenizer_round_trip_and_fixed_vocabulary() -> None:
    """byte tokenizer 应保持 UTF-8 文本并固定为 260 项词表。"""

    tokenizer = ByteTokenizer()
    token_ids = tokenizer.encode("CAN 测试")
    assert tokenizer.vocab_size == 260
    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(token_ids) == "CAN 测试"


@pytest.mark.parametrize("value", [-1, 260, True, "1"])
def test_byte_tokenizer_rejects_invalid_token_ids(value: object) -> None:
    """tokenizer 必须拒绝越界或类型混淆 token。"""

    tokenizer = ByteTokenizer()
    with pytest.raises((TypeError, ValueError)):
        tokenizer.decode([value])  # type: ignore[list-item]


def test_byte_tokenizer_rejects_invalid_api_inputs() -> None:
    """tokenizer 的文本、标志和长度参数必须严格校验。"""

    tokenizer = ByteTokenizer()
    with pytest.raises(TypeError):
        tokenizer.encode(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tokenizer.encode("x", add_bos=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tokenizer.encode("x", max_length=True)
    with pytest.raises(ValueError):
        tokenizer.encode("x", max_length=0)
    with pytest.raises(ValueError):
        tokenizer.encode("too-long", max_length=1)
    with pytest.raises(TypeError):
        tokenizer.decode([], skip_special=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tokenizer.decode("bad")
    with pytest.raises(TypeError):
        tokenizer.decode(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tokenizer.decode([tokenizer.bos_token_id], skip_special=False)


def test_synthetic_corpus_is_deterministic_and_entity_disjoint() -> None:
    """固定 seed 应产生相同语料且 split 实体严格隔离。"""

    first = generate_synthetic_corpus(7, 4, 2, 2)
    second = generate_synthetic_corpus(7, 4, 2, 2)
    assert first == second
    entities = {
        split: {item.entity_id for item in examples}
        for split, examples in first.items()
    }
    assert entities["train"].isdisjoint(entities["validation"])
    assert entities["train"].isdisjoint(entities["test"])
    assert entities["validation"].isdisjoint(entities["test"])
    assert {item.scope for item in first["train"]} == {
        "public",
        "private",
        "refusal",
    }
    # private prompt 不得泄露其对应答案，避免复制上下文造成假阳性。
    for item in first["train"]:
        if item.scope == "private":
            assert "PRIVATE-" not in item.prompt
            assert "PRIVATE-" in item.answer


def test_memorization_validation_reuses_facts_with_unseen_prompts() -> None:
    """记忆 validation 应保留训练映射，但不得复用训练 prompt。"""

    train = generate_synthetic_corpus(7, 4, 2, 2)["train"]
    validation = build_memorization_validation(train, 2)
    train_by_key = {(item.entity_id, item.scope): item for item in train}
    assert len(validation) == 6
    for item in validation:
        source = train_by_key[(item.entity_id, item.scope)]
        assert item.answer == source.answer
        assert item.prompt != source.prompt
        assert item.split == "validation"
        if item.scope == "private":
            assert item.answer.strip() not in item.prompt


def test_synthetic_data_rejects_invalid_configuration() -> None:
    """语料和 Dataset 必须拒绝非法计数、重复 ID 与超长样本。"""

    with pytest.raises(TypeError):
        generate_synthetic_corpus(True)
    with pytest.raises(TypeError):
        generate_synthetic_corpus(1, train_entities="2")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        generate_synthetic_corpus(1, train_entities=0)
    example = _short_examples()[0]
    with pytest.raises(TypeError):
        SyntheticKnowledgeDataset([example], object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SyntheticKnowledgeDataset([example], ByteTokenizer(), max_length=True)
    with pytest.raises(ValueError):
        SyntheticKnowledgeDataset([example], ByteTokenizer(), max_length=1)
    with pytest.raises(ValueError):
        SyntheticKnowledgeDataset([], ByteTokenizer())
    with pytest.raises(TypeError):
        SyntheticKnowledgeDataset([object()], ByteTokenizer())  # type: ignore[list-item]
    with pytest.raises(ValueError):
        SyntheticKnowledgeDataset([example, example], ByteTokenizer())
    combined_too_long = KnowledgeExample(
        "long", "public", "e", "lookup", "abcde", " vwxyz", "train", 1
    )
    dataset = SyntheticKnowledgeDataset(
        [combined_too_long], ByteTokenizer(), max_length=8
    )
    with pytest.raises(ValueError, match="超过 max_length"):
        dataset[0]


def test_dataset_and_collate_preserve_prompt_mask() -> None:
    """collate 应右填充输入并保留 prompt 的 -100 loss mask。"""

    corpus = generate_synthetic_corpus(8, 2, 1, 1)
    dataset = SyntheticKnowledgeDataset(
        corpus["train"][:2], ByteTokenizer(), max_length=256
    )
    batch = collate_causal_lm_batch([dataset[0], dataset[1]])
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert input_ids.shape == labels.shape == attention_mask.shape
    assert bool((labels == -100).any().item())
    assert bool(attention_mask[:, 0].all().item())


def test_entity_triplet_sampler_guarantees_mixed_batch() -> None:
    """实体 sampler 的每个 batch 应恰含至少两个 private 样本。"""

    examples = _short_examples()
    sampler = EntityTripletBatchSampler(examples, batch_size=6, seed=11)
    rows = next(iter(sampler))
    scopes = [examples[index].scope for index in rows]
    assert scopes.count("private") == 2
    assert scopes.count("public") == 2
    assert scopes.count("refusal") == 2
    sampler.set_epoch(1)
    assert len(sampler) == 1


def test_sampler_and_collate_reject_invalid_inputs() -> None:
    """mixed sampler 与 collate 必须 fail-fast 拒绝不完整结构。"""

    examples = _short_examples()
    with pytest.raises(TypeError):
        EntityTripletBatchSampler(examples, batch_size=True, seed=1)
    with pytest.raises(ValueError):
        EntityTripletBatchSampler(examples, batch_size=5, seed=1)
    with pytest.raises(TypeError):
        EntityTripletBatchSampler(examples, batch_size=6, seed=True)
    with pytest.raises(TypeError):
        EntityTripletBatchSampler([object()], batch_size=6, seed=1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="恰含"):
        EntityTripletBatchSampler(examples[:2], batch_size=6, seed=1)
    with pytest.raises(ValueError, match="实体数量不足"):
        EntityTripletBatchSampler(examples[:3], batch_size=6, seed=1)
    sampler = EntityTripletBatchSampler(examples, batch_size=6, seed=1)
    with pytest.raises(ValueError):
        sampler.set_epoch(-1)
    with pytest.raises(ValueError):
        collate_causal_lm_batch([])
    with pytest.raises(TypeError):
        collate_causal_lm_batch([{"input_ids": [], "labels": []}])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 259},
        {"num_layers": 2, "cut_layer": 2},
        {"d_model": 30, "num_heads": 8},
        {"dropout": 1.0},
    ],
)
def test_transformer_config_rejects_invalid_values(kwargs: dict) -> None:
    """模型配置必须拒绝不符合 T0 契约的结构。"""

    with pytest.raises((TypeError, ValueError)):
        TransformerConfig(**kwargs)


def test_model_rejects_invalid_token_and_attention_inputs() -> None:
    """模型必须拒绝 dtype、词表、shape 和非右填充 mask 错误。"""

    model, _, secret, _ = _model_fixture()
    model.eval()
    credential = torch.tensor(secret, dtype=torch.float32)
    with pytest.raises(TypeError):
        model(torch.tensor([[1.0]]), credential)
    with pytest.raises(ValueError):
        model(torch.empty((1, 0), dtype=torch.long), credential)
    with pytest.raises(ValueError, match="词表外"):
        model(torch.tensor([[260]], dtype=torch.long), credential)
    tokens = torch.tensor([[256, 65]], dtype=torch.long)
    with pytest.raises(TypeError):
        model(tokens, credential, attention_mask=[[True, True]])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        model(tokens, credential, torch.ones((1, 1), dtype=torch.bool))
    with pytest.raises(TypeError):
        model(tokens, credential, torch.ones_like(tokens))
    with pytest.raises(ValueError, match="至少包含"):
        model(tokens, credential, torch.zeros_like(tokens, dtype=torch.bool))
    bad_tokens = torch.tensor([[256, 65, 66]], dtype=torch.long)
    bad_padding = torch.tensor([[True, False, True]], dtype=torch.bool)
    with pytest.raises(ValueError, match="右侧"):
        model(bad_tokens, credential, bad_padding)


def test_generate_rejects_training_mode_and_invalid_budget() -> None:
    """生成接口必须要求 eval 模式和有限正 token budget。"""

    model, _, secret, _ = _model_fixture()
    tokens = torch.tensor([[256, 65]], dtype=torch.long)
    credential = torch.tensor(secret, dtype=torch.float32)
    model.train()
    with pytest.raises(RuntimeError):
        model.generate(tokens, credential)
    model.eval()
    with pytest.raises(TypeError):
        model.generate(tokens, credential, max_new_tokens=True)
    with pytest.raises(ValueError):
        model.generate(tokens, credential, max_new_tokens=0)


def test_eval_routes_valid_and_invalid_to_sparse_outputs() -> None:
    """valid/invalid credential 应分别只产生 protected/public logits。"""

    model, _, secret, invalid = _model_fixture()
    model.eval()
    input_ids = torch.tensor([[256, 65, 66], [256, 67, 68]], dtype=torch.long)
    credentials = torch.tensor(np.stack([secret, invalid]), dtype=torch.float32)
    output = model(input_ids, credentials)
    assert output.protected_indices.tolist() == [0]
    assert output.public_indices.tolist() == [1]
    assert output.rejected_indices.numel() == 0
    assert output.protected_logits.shape == (1, 3, 260)
    assert output.public_logits.shape == (1, 3, 260)


def test_gate_decision_is_independent_of_tokens() -> None:
    """相同 credential 的判决和误差不得随 token/hidden state 改变。"""

    model, _, secret, _ = _model_fixture()
    model.eval()
    credential = torch.tensor(secret, dtype=torch.float32)
    first = model(torch.tensor([[256, 65]], dtype=torch.long), credential)
    second = model(torch.tensor([[256, 90, 91]], dtype=torch.long), credential)
    assert torch.equal(first.decision.allow, second.decision.allow)
    assert torch.equal(
        first.decision.evidence.error_norm, second.decision.evidence.error_norm
    )


def test_malformed_credential_is_rejected_without_route_logits() -> None:
    """非有限 credential 必须进入 rejected，而不是静默降级 public。"""

    model, params, _, _ = _model_fixture()
    model.eval()
    credential = torch.full((1, params.n), float("nan"), dtype=torch.float32)
    output = model(torch.tensor([[256, 65]], dtype=torch.long), credential)
    assert output.rejected_indices.tolist() == [0]
    assert output.protected_logits.shape[0] == 0
    assert output.public_logits.shape[0] == 0


def test_all_invalid_batch_never_calls_protected_blocks() -> None:
    """全 invalid 推理不得执行 cut 后 protected blocks。"""

    model, _, _, invalid = _model_fixture()
    model.eval()
    calls = {"count": 0}

    def count_call(
        module: torch.nn.Module, inputs: tuple, output: torch.Tensor
    ) -> None:
        """统计 protected block 的实际调用次数。"""

        del module, inputs, output
        calls["count"] += 1

    handle = model.blocks[model.config.cut_layer].register_forward_hook(count_call)
    try:
        credentials = torch.tensor(np.stack([invalid, invalid]), dtype=torch.float32)
        output = model(
            torch.tensor([[256, 65], [256, 66]], dtype=torch.long), credentials
        )
    finally:
        handle.remove()
    assert calls["count"] == 0
    assert output.public_indices.tolist() == [0, 1]


def test_valid_routed_logits_match_direct_full_path() -> None:
    """eval 态 valid routed logits 应匹配同 checkpoint direct full path。"""

    model, _, secret, _ = _model_fixture()
    model.eval()
    input_ids = torch.tensor([[256, 65, 66]], dtype=torch.long)
    credential = torch.tensor(secret, dtype=torch.float32)
    routed = model(input_ids, credential)
    direct = model.direct_protected_logits(input_ids)
    torch.testing.assert_close(routed.protected_logits, direct)


def test_training_output_keeps_full_batch_and_backpropagates() -> None:
    """训练态应保留完整 batch 占位并允许 public loss 反向传播。"""

    model, _, secret, invalid = _model_fixture()
    model.train()
    input_ids = torch.tensor([[256, 65, 66], [256, 67, 68]], dtype=torch.long)
    credentials = torch.tensor(np.stack([secret, invalid]), dtype=torch.float32)
    output = model(input_ids, credentials)
    assert output.protected_logits.shape == (2, 3, 260)
    assert output.public_logits.shape == (2, 3, 260)
    assert torch.count_nonzero(output.protected_logits[1]).item() == 0
    output.public_logits.sum().backward()
    assert model.public_head.weight.grad is not None


def test_generate_commits_one_route_and_matches_single_references() -> None:
    """mixed generation 应只路由一次并匹配逐样本 reference generation。"""

    model, _, secret, invalid = _model_fixture()
    model.eval()
    input_ids = torch.tensor([[256, 65], [256, 66]], dtype=torch.long)
    credentials = torch.tensor(np.stack([secret, invalid]), dtype=torch.float32)
    mixed = model.generate(input_ids, credentials, max_new_tokens=3)
    valid = model.generate(input_ids[:1], credentials[:1], max_new_tokens=3)
    public = model.generate(input_ids[1:], credentials[1:], max_new_tokens=3)
    assert mixed.route_call_count.tolist() == [1, 1]
    assert mixed.capability_levels == ("protected", "public")
    assert mixed.token_ids[0] == valid.token_ids[0]
    assert mixed.token_ids[1] == public.token_ids[0]


def test_generate_malformed_credential_has_zero_committed_routes() -> None:
    """格式错误 credential 不得提交 public/protected route。"""

    model, params, _, _ = _model_fixture()
    model.eval()
    credential = torch.full((1, params.n), float("inf"), dtype=torch.float32)
    output = model.generate(
        torch.tensor([[256, 65]], dtype=torch.long), credential, max_new_tokens=2
    )
    assert output.capability_levels == ("rejected",)
    assert output.route_call_count.tolist() == [0]
    assert output.stop_reasons == ("invalid_credential_format",)


def test_generate_does_not_exceed_context_limit() -> None:
    """已达到 max_seq_len 的 prompt 不得再追加 token。"""

    model, _, secret, _ = _model_fixture()
    model.eval()
    input_ids = torch.full((1, model.config.max_seq_len), 65, dtype=torch.long)
    input_ids[0, 0] = 256
    output = model.generate(
        input_ids, torch.tensor(secret, dtype=torch.float32), max_new_tokens=2
    )
    assert len(output.token_ids[0]) == model.config.max_seq_len
    assert output.stop_reasons == ("max_seq_len",)


def test_masked_causal_lm_loss_handles_empty_route() -> None:
    """空路由 loss 应为图连接零值。"""

    logits = torch.randn(2, 4, 260, requires_grad=True)
    labels = torch.tensor(
        [[-100, -100, 65, 257], [-100, -100, 66, 257]], dtype=torch.long
    )
    loss = masked_causal_lm_loss(logits, labels, torch.zeros(2, dtype=torch.bool))
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_distillation_loss_is_finite_and_has_student_gradient() -> None:
    """KD loss 应有限且只需向 student 回传梯度。"""

    student = torch.randn(2, 4, 260, requires_grad=True)
    teacher = torch.randn(2, 4, 260)
    labels = torch.tensor(
        [[-100, -100, 65, 257], [-100, -100, 66, 257]], dtype=torch.long
    )
    loss = causal_distillation_loss(student, teacher, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_stage_configuration_and_teacher_freeze() -> None:
    """Stage A/B 只训练 public head，teacher 必须完全冻结。"""

    model, _, _, _ = _model_fixture()
    summary = configure_stage(model, "A")
    assert 0 < summary["trainable"] < summary["total"]
    assert model.public_head.weight.requires_grad
    assert not model.token_embedding.weight.requires_grad
    freeze_teacher(model)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_pretrain_go_no_go_uses_absolute_thresholds() -> None:
    """T-pretrain 必须同时满足 public/private/refusal 三项门槛。"""

    assert pretrain_go_no_go(PretrainMetrics(0.80, 0.81, 0.90))
    assert not pretrain_go_no_go(PretrainMetrics(0.79, 0.99, 0.99))
    with pytest.raises(ValueError, match="有限"):
        pretrain_go_no_go(PretrainMetrics(float("nan"), 0.9, 0.9))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pretrain_go_no_go(PretrainMetrics(1.1, 0.9, 0.9))


def test_mixed_routing_requires_two_valid_and_one_invalid() -> None:
    """Stage C mixed batch 必须满足 2 valid + 1 invalid。"""

    model, _, secret, invalid = _model_fixture()
    model.eval()
    input_ids = torch.tensor([[256, 65], [256, 66], [256, 67]])
    credentials = torch.tensor(np.stack([secret, secret, invalid]), dtype=torch.float32)
    decision = model(input_ids, credentials).decision
    validate_mixed_routing(decision)
    bad_decision = model(input_ids[:2], credentials[[0, 2]]).decision
    with pytest.raises(ValueError, match="2 valid"):
        validate_mixed_routing(bad_decision)


def test_phase5_trainer_runs_tpretrain_and_stage_c(tmp_path: Path) -> None:
    """最小训练器应能执行 T-pretrain、Stage C 并保存/恢复状态。"""

    model, params, secret, _ = _model_fixture()
    examples = _short_examples()
    dataset = SyntheticKnowledgeDataset(examples, ByteTokenizer(), max_length=64)
    sampler = EntityTripletBatchSampler(examples, batch_size=6, seed=9)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_causal_lm_batch
    )

    pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pretrainer = Phase5Trainer(
        model,
        loader,
        pretrain_optimizer,
        torch.device("cpu"),
        stage="T-pretrain",
    )
    pretrain_metrics = pretrainer.train_epoch()
    assert pretrain_metrics["global_step"] == 1.0
    assert np.isfinite(pretrain_metrics["loss"])
    assert model.public_head.weight.grad is not None
    with pytest.raises(TypeError, match="progress"):
        pretrainer.train_epoch(progress=1)  # type: ignore[arg-type]

    A = model.gate_layer.verifier.A.detach().cpu().numpy()
    b = model.gate_layer.verifier.b.detach().cpu().numpy()
    generator = CredentialGenerator(A, secret, b, params, seed=19)
    teacher = copy.deepcopy(model)
    stage_c_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Phase5Trainer(
        model,
        loader,
        stage_c_optimizer,
        torch.device("cpu"),
        stage="C",
        credential_generator=generator,
        teacher=teacher,
        teacher_identity={
            "checkpoint_sha256": "0" * 64,
            "manifest_sha256": "1" * 64,
        },
    )
    metrics = trainer.train_epoch()
    assert metrics["global_step"] == 1.0
    checkpoint = tmp_path / "phase5.ckpt"
    trainer.save_checkpoint(checkpoint)
    trainer.global_step = 99
    trainer.load_checkpoint(checkpoint)
    assert trainer.global_step == 1
    assert trainer.current_epoch == 1


def test_phase5_trainer_requires_teacher_identity() -> None:
    """Stage B/C 缺少可信 teacher identity 时必须 fail fast。"""

    model, params, secret, _ = _model_fixture()
    examples = _short_examples()
    dataset = SyntheticKnowledgeDataset(examples, ByteTokenizer(), max_length=64)
    sampler = EntityTripletBatchSampler(examples, batch_size=6, seed=9)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_causal_lm_batch
    )
    A = model.gate_layer.verifier.A.detach().cpu().numpy()
    b = model.gate_layer.verifier.b.detach().cpu().numpy()
    generator = CredentialGenerator(A, secret, b, params, seed=19)
    with pytest.raises(ValueError, match="teacher_identity"):
        Phase5Trainer(
            model,
            loader,
            torch.optim.AdamW(model.parameters()),
            torch.device("cpu"),
            stage="B",
            credential_generator=generator,
            teacher=copy.deepcopy(model),
        )
