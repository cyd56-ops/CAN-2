"""运行 Phase 5 E2 exploratory 的 Plain/CAN 成对训练实验。"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_phase5 import _loader
from scripts.train_phase5_exploratory import _validation_metrics
from scripts.train_phase5_plain_exploratory import _evaluate as _plain_evaluate
from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    ByteTokenizer,
    GatedDecoderTransformer,
    Phase5Trainer,
    PlainDecoderTrainer,
    PlainDecoderTransformer,
    TransformerConfig,
    build_checkpoint_manifest,
    build_same_template_validation,
    build_sample_diagnostics,
    generate_e2_corpus,
    load_freeze_record,
    write_manifest,
)

FREEZE_V3_SHA256 = "9ce8876343c96c2c11cb9b9993152f1631937cadc6877691a55c4cf252598869"


def _seed_everything(seed: int) -> None:
    """固定 E2 运行使用的 Python、NumPy 和 PyTorch 随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _trainable_epoch(
    trainer: Any, loader: DataLoader, budget: int, total_tokens: int
) -> Dict[str, float]:
    """执行一个完整 epoch，并在预算不足时拒绝部分 epoch。"""
    epoch_tokens = sum(int(batch["attention_mask"].sum().item()) for batch in loader)
    if total_tokens + epoch_tokens > budget:
        return {}
    return trainer.train_epoch(progress=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """执行一个独立 E2 exploratory 训练并保存可审计 JSON。"""
    parser = argparse.ArgumentParser(description="Phase 5 E2 exploratory")
    parser.add_argument("--model", choices=("plain", "can"), required=True)
    parser.add_argument(
        "--protocol", choices=("structured", "random-short"), default="structured"
    )
    parser.add_argument("--prompt-mode", choices=("same", "paraphrase"), default="same")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=500_000)
    # E2-A 使用 12 个实体；每个实体含 3 条样本，因此 batch=36 才能组成完整 triplet。
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {args.output}")
    freeze_path = ROOT / "experiments/phase5_freeze_v3/freeze_record.json"
    record = load_freeze_record(freeze_path)
    digest = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    if digest != FREEZE_V3_SHA256:
        raise ValueError("freeze v3 SHA-256 与可信登记值不一致")
    if args.batch_size < 6 or args.batch_size % 3:
        raise ValueError("E2 batch-size 必须至少为 6 且能被 3 整除")
    if args.budget <= 0:
        raise ValueError("budget 必须为正数")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用")
    _seed_everything(args.seed)
    corpus = generate_e2_corpus(
        args.seed, 12, 4, 4, protocol=args.protocol, prompt_mode=args.prompt_mode
    )
    tokenizer = ByteTokenizer()
    loader = _loader(corpus["train"], tokenizer, args.batch_size, args.seed, 256)
    validation = build_same_template_validation(
        corpus["train"], min(4, 12), prompt_mode=args.prompt_mode
    )
    config = TransformerConfig(**record["model_config"])
    params = LWEParams(n=128, m=256)
    if args.model == "plain":
        model = PlainDecoderTransformer(config).to(device)
        trainer = PlainDecoderTrainer(
            model,
            loader,
            torch.optim.AdamW(model.parameters(), lr=record["learning_rate"]),
            device,
            protected_weight=2.0,
            public_weight=1.0,
        )
    else:
        A, secret, b = generate_keypair(params, np.random.default_rng(args.seed))
        model = GatedDecoderTransformer(A, b, params, config=config).to(device)
        generator = CredentialGenerator(A, secret, b, params, seed=args.seed + 500)
        trainer = Phase5Trainer(
            model,
            loader,
            torch.optim.AdamW(model.parameters(), lr=record["learning_rate"]),
            device,
            "T-pretrain",
            pretrain_protected_weight=2.0,
            pretrain_public_weight=1.0,
        )
    args.output.mkdir(parents=True, exist_ok=True)

    def evaluate_validation() -> Dict[str, Any]:
        """使用当前模型计算紧凑 validation 摘要，不生成逐样本诊断。"""
        if args.model == "plain":
            return _plain_evaluate(
                model,
                validation,
                tokenizer,
                device,
                int(record["max_new_tokens"]),
                record["cache_mode"],
            )
        return _validation_metrics(
            model,
            validation,
            tokenizer,
            A,
            secret,
            b,
            params,
            args.seed,
            device,
            int(record["max_new_tokens"]),
            record["cache_mode"],
        )

    total_tokens = 0
    history: List[Dict[str, Any]] = []
    validation_interval = int(record["validation_interval_tokens"])
    next_validation = validation_interval
    while total_tokens < args.budget:
        metrics = _trainable_epoch(trainer, loader, args.budget, total_tokens)
        if not metrics:
            break
        total_tokens += int(metrics["tokens"])
        entry: Dict[str, Any] = {"train": metrics, "tokens": total_tokens}
        if total_tokens >= next_validation:
            entry["validation"] = evaluate_validation()
            while next_validation <= total_tokens:
                next_validation += validation_interval
        history.append(entry)
    checkpoint = {
        "schema_version": 1,
        "model_kind": args.model,
        "model": model.state_dict(),
        "config": config.__dict__,
        "seed": args.seed,
        "actual_tokens": total_tokens,
        "budget": args.budget,
        "batch_size": args.batch_size,
        "freeze_record_sha256": digest,
    }
    torch.save(checkpoint, args.output / "final.ckpt")
    if args.model == "plain":
        validation_metrics = evaluate_validation()
        diagnostic_generator = None
    else:
        validation_metrics = evaluate_validation()
        diagnostic_generator = generator
    diagnostic = {
        "schema_version": 1,
        "experiment_kind": "exploratory_e2",
        "model_kind": args.model,
        "protocol": args.protocol,
        "prompt_mode": args.prompt_mode,
        "seed": args.seed,
        "train": build_sample_diagnostics(
            model,
            corpus["train"],
            tokenizer,
            device,
            int(record["max_new_tokens"]),
            record["cache_mode"],
            diagnostic_generator,
        ),
        "validation": build_sample_diagnostics(
            model,
            validation,
            tokenizer,
            device,
            int(record["max_new_tokens"]),
            record["cache_mode"],
            diagnostic_generator,
        ),
    }
    (args.output / "diagnostic.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "experiment_kind": "exploratory_e2",
        "research_result": False,
        "model_kind": args.model,
        "protocol": args.protocol,
        "prompt_mode": args.prompt_mode,
        "seed": args.seed,
        "budget": args.budget,
        "actual_tokens": total_tokens,
        "batch_size": args.batch_size,
        "freeze_record_sha256": digest,
        "validation": validation_metrics,
        "history": history,
        "final_checkpoint": "final.ckpt",
        "diagnostic": "diagnostic.json",
    }
    (args.output / "resolved_config.json").write_text(
        json.dumps(
            {**summary, "model_config": config.__dict__}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output / "exploratory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_checkpoint_manifest(
        args.output,
        {
            "final.ckpt": {
                "seed": args.seed,
                "protocol": args.protocol,
                "model": args.model,
            }
        },
    )
    manifest["experiment"] = {
        "protocol": args.protocol,
        "prompt_mode": args.prompt_mode,
        "freeze_record_sha256": digest,
    }
    write_manifest(args.output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "model": args.model,
                "protocol": args.protocol,
                "actual_tokens": total_tokens,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
