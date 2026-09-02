"""运行 Phase 5 T1 CPU smoke 的工程接线检查。"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    ByteTokenizer,
    EntityTripletBatchSampler,
    GatedDecoderTransformer,
    Phase5Trainer,
    Phase5Evaluator,
    PretrainMetrics,
    SyntheticKnowledgeDataset,
    TransformerConfig,
    collate_causal_lm_batch,
    generate_synthetic_corpus,
    pretrain_go_no_go,
    freeze_teacher,
)


def main() -> int:
    """执行单步 T-pretrain，并在未通过 smoke 门时阻断后续阶段。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/phase5_smoke"))
    parser.add_argument(
        "--pipeline-fixture",
        action="store_true",
        help="使用显式合成 go/no-go 仅验证 A/B/C 接线",
    )
    args = parser.parse_args()
    torch.manual_seed(20260901)
    params = LWEParams(n=16, m=32)
    A, secret, b = generate_keypair(params, np.random.default_rng(20260901))
    model = GatedDecoderTransformer(
        A,
        b,
        params,
        TransformerConfig(
            num_layers=3, cut_layer=1, d_model=32, num_heads=4, d_ff=64, max_seq_len=256
        ),
    )
    corpus = generate_synthetic_corpus(20260901, 2, 2, 2)
    dataset = SyntheticKnowledgeDataset(
        corpus["train"], ByteTokenizer(), max_length=256
    )
    sampler = EntityTripletBatchSampler(corpus["train"], batch_size=6, seed=20260901)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_causal_lm_batch
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Phase5Trainer(model, loader, optimizer, torch.device("cpu"), "T-pretrain")
    metrics = trainer.train_epoch()
    # 使用真实 validation 指标执行 go/no-go；短 smoke 未达门槛时阻断后续阶段。
    generator = CredentialGenerator(A, secret, b, params, seed=20260901 + 500)
    validation = SyntheticKnowledgeDataset(
        corpus["validation"], ByteTokenizer(), max_length=256
    )
    evaluated = Phase5Evaluator(
        model, ByteTokenizer(), torch.device("cpu"), generator, max_new_tokens=8
    ).evaluate(validation)
    pretrain_metrics = PretrainMetrics(
        evaluated["public"].exact_match or 0.0,
        evaluated["protected"].exact_match or 0.0,
        evaluated["refusal"].refusal_rate or 0.0,
    )
    go = pretrain_go_no_go(pretrain_metrics)
    # smoke 级硬检查：指标有限、索引路由互斥且拒答四分类和为 1。
    if not all(
        np.isfinite(v)
        for v in (
            pretrain_metrics.public_exact_match,
            pretrain_metrics.private_exact_match,
            pretrain_metrics.refusal_rate,
        )
    ):
        raise RuntimeError("smoke 指标出现 NaN/Inf")
    refusal_sum = sum(
        getattr(evaluated["refusal"], name)
        for name in (
            "refusal_rate",
            "leaked_private_rate",
            "public_scope_compliance",
            "other_rate",
        )
    )
    if not np.isclose(refusal_sum, 1.0):
        raise RuntimeError("拒答四分类未覆盖全部样本")
    if go and not args.pipeline_fixture:
        raise RuntimeError("smoke fixture 不应通过正式 go/no-go")
    stage_results = {}
    if args.pipeline_fixture:
        teacher = copy.deepcopy(model)
        freeze_teacher(teacher)
        for stage in ("A", "B", "C"):
            stage_model = model
            stage_optimizer = torch.optim.AdamW(stage_model.parameters(), lr=1e-3)
            stage_trainer = Phase5Trainer(
                stage_model,
                loader,
                stage_optimizer,
                torch.device("cpu"),
                stage,
                credential_generator=generator,
                teacher=teacher,
                teacher_identity={
                    "checkpoint_sha256": "0" * 64,
                    "manifest_sha256": "1" * 64,
                },
            )
            stage_results[stage] = stage_trainer.train_epoch()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.output / "t_pretrain_smoke.ckpt")
    print(
        {
            "smoke_mode": "pipeline_fixture" if args.pipeline_fixture else "strict",
            "train": metrics,
            "validation": evaluated,
            "go_no_go": go,
            "stages": stage_results,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
