"""Phase 5 exploratory T-pretrain：允许预算/监督权重探索，不产生正式结果。"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_phase5 import _loader, _validation_metrics
from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.transformer import (
    ByteTokenizer,
    GatedDecoderTransformer,
    Phase5Trainer,
    TransformerConfig,
    build_memorization_validation,
    count_non_padding_input_tokens,
    generate_synthetic_corpus,
)
from src.can.v2.training.data import CredentialGenerator


def _seed(seed: int) -> None:
    """固定 exploratory 运行的 Python、NumPy 和 PyTorch 随机源。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """运行仅限 validation 的 exploratory T-pretrain。"""

    parser = argparse.ArgumentParser(description="Phase 5 exploratory T-pretrain")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--validation-interval", type=int, default=50_000)
    parser.add_argument("--protected-weight", type=float, default=1.0)
    parser.add_argument("--public-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {args.output}")
    if args.budget <= 0 or args.validation_interval <= 0 or args.batch_size < 6:
        raise ValueError("budget、validation-interval 必须为正，batch-size 至少为 6")
    if args.batch_size % 3:
        raise ValueError("batch-size 必须能被 3 整除")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用")
    _seed(args.seed)
    params = LWEParams(n=128, m=256)
    A, secret, b = generate_keypair(params, np.random.default_rng(args.seed))
    config = TransformerConfig()
    model = GatedDecoderTransformer(A, b, params, config=config).to(device)
    tokenizer = ByteTokenizer()
    corpus = generate_synthetic_corpus(args.seed, 48, 20, 20)
    train_loader = _loader(corpus["train"], tokenizer, args.batch_size, args.seed, 256)
    validation = build_memorization_validation(corpus["train"], 20)
    generator = CredentialGenerator(A, secret, b, params, seed=args.seed + 500)
    trainer = Phase5Trainer(
        model,
        train_loader,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        device,
        "T-pretrain",
        pretrain_protected_weight=args.protected_weight,
        pretrain_public_weight=args.public_weight,
    )
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    history = []
    total_tokens = 0
    next_validation = args.validation_interval
    try:
        from tqdm.auto import tqdm

        progress = tqdm(total=args.budget, desc="exploratory T-pretrain tokens", unit="tok")
    except ImportError:
        progress = None
    while total_tokens < args.budget:
        epoch_tokens = sum(
            count_non_padding_input_tokens(batch["attention_mask"])
            for batch in train_loader
        )
        if total_tokens + epoch_tokens > args.budget:
            break
        metrics = trainer.train_epoch(progress=False)
        total_tokens += int(metrics["tokens"])
        if progress is not None:
            progress.update(int(metrics["tokens"]))
        entry: Dict[str, Any] = {"train": metrics}
        if total_tokens >= next_validation:
            print(f"[exploratory] validation start tokens={total_tokens}", flush=True)
            result = _validation_metrics(
                model, validation, tokenizer, A, secret, b, params, args.seed,
                device, 16, "kv"
            )
            print(f"[exploratory] validation end tokens={total_tokens}", flush=True)
            entry["validation"] = result
            while next_validation <= total_tokens:
                next_validation += args.validation_interval
        history.append(entry)
    if progress is not None:
        progress.close()
    summary = {
        "experiment_kind": "exploratory",
        "research_result": False,
        "baseline_freeze": "phase5-freeze-v3",
        "seed": args.seed,
        "budget": args.budget,
        "actual_tokens": total_tokens,
        "batch_size": args.batch_size,
        "protected_weight": args.protected_weight,
        "public_weight": args.public_weight,
        "history": history,
    }
    (out / "exploratory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("experiment_kind", "actual_tokens")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
