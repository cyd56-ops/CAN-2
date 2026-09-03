"""测量 Phase 5 T-pretrain 的 GPU 显存、吞吐与 step 时延。"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

# 允许从仓库根目录以外直接执行本脚本，无需预先设置 PYTHONPATH。
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.transformer import (
    ByteTokenizer,
    EntityTripletBatchSampler,
    GatedDecoderTransformer,
    Phase5Trainer,
    SyntheticKnowledgeDataset,
    TransformerConfig,
    build_memorization_validation,
    collate_causal_lm_batch,
    count_non_padding_input_tokens,
    evaluate_pretrain_validation,
    generate_synthetic_corpus,
)


def _parse_args() -> argparse.Namespace:
    """解析 GPU benchmark 参数并保留安全默认值。"""

    parser = argparse.ArgumentParser(description="Phase 5 GPU smoke benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--force-overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 6 or args.batch_size % 3:
        parser.error("--batch-size 必须至少为 6 且能被 3 整除")
    if args.warmup_steps < 0 or args.measure_steps <= 0:
        parser.error("warmup 必须非负，measure-steps 必须为正数")
    if args.output.exists() and not args.force_overwrite:
        parser.error(f"输出已存在，拒绝覆盖: {args.output}")
    if not torch.cuda.is_available():
        parser.error("GPU benchmark 要求 CUDA 可用")
    return args


def _build_loader(seed: int, batch_size: int) -> DataLoader:
    """构造固定新数据协议的训练 DataLoader。"""

    # 至少准备一个完整 batch 的 entity triplet；高 batch benchmark 不能因数据不足失败。
    train_entities = max(24, (batch_size + 2) // 3)
    corpus = generate_synthetic_corpus(
        seed,
        train_entities=train_entities,
        validation_entities=2,
        test_entities=2,
    )
    dataset = SyntheticKnowledgeDataset(
        corpus["train"], ByteTokenizer(), max_length=256
    )
    sampler = EntityTripletBatchSampler(
        corpus["train"], batch_size=batch_size, seed=seed
    )
    return DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_causal_lm_batch
    )


def main() -> int:
    """运行有限 T-pretrain step 并输出可冻结预算的测量记录。"""

    args = _parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    params = LWEParams()
    A, secret, b = generate_keypair(params, np.random.default_rng(args.seed + 100))
    config = TransformerConfig()
    model = GatedDecoderTransformer(A, b, params, config)
    trainer = Phase5Trainer(
        model,
        _build_loader(args.seed, args.batch_size),
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        device,
        stage="T-pretrain",
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(trainer.train_loader)

    def step() -> Dict[str, float]:
        """执行一个训练 batch，并返回 trainer 统计。"""

        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(trainer.train_loader)
            batch = next(iterator)
        input_ids, labels, attention_mask, scopes = trainer._prepare_batch(batch)
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = trainer._batch_loss(input_ids, labels, attention_mask, scopes)
        if not torch.isfinite(loss):
            raise RuntimeError("GPU benchmark 出现非有限 loss")
        loss.backward()
        trainer.optimizer.step()
        return {
            # 正式预算口径：输入序列中所有非 padding token（prompt + target）。
            "non_padding_input_tokens": float(
                count_non_padding_input_tokens(attention_mask)
            ),
            "supervised_target_tokens": float((labels[:, 1:] != -100).sum().item()),
            "loss": float(loss.item()),
        }

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    tokens = 0.0
    target_tokens = 0.0
    losses = []
    for _ in range(args.measure_steps):
        result = step()
        tokens += result["non_padding_input_tokens"]
        target_tokens += result["supervised_target_tokens"]
        losses.append(result["loss"])
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    # 使用正式训练相同的 20-entity memorization validation 路径测量完整生成开销。
    validation_corpus = generate_synthetic_corpus(
        args.seed,
        train_entities=48,
        validation_entities=20,
        test_entities=20,
    )
    validation_examples = build_memorization_validation(validation_corpus["train"], 20)
    torch.cuda.synchronize(device)
    validation_started = time.perf_counter()
    validation_metrics = evaluate_pretrain_validation(
        model,
        validation_examples,
        ByteTokenizer(),
        A,
        secret,
        b,
        params,
        args.seed,
        device,
        max_new_tokens=16,
        cache_mode="kv",
    )
    torch.cuda.synchronize(device)
    validation_wall_seconds = time.perf_counter() - validation_started
    report = {
        "schema_version": 1,
        "kind": "phase5_gpu_smoke_benchmark",
        "research_result": False,
        "generator_version": "phase5-t1-private-query-v2",
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "budget_token_unit": "non_padding_input_tokens",
        "non_padding_input_tokens_measured": int(tokens),
        "supervised_target_tokens_measured": int(target_tokens),
        "tokens_per_second": tokens / elapsed,
        "seconds_per_step": elapsed / args.measure_steps,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "validation_entities": 20,
        "validation_max_new_tokens": 16,
        "validation_cache_mode": "kv",
        "validation_wall_seconds": validation_wall_seconds,
        "validation_metrics": validation_metrics,
        "loss_mean": float(np.mean(losses)),
        "transformer_config": config.__dict__,
        "lwe": {
            "n": params.n,
            "m": params.m,
            "error_threshold": params.error_threshold,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
