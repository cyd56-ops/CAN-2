"""运行与 CAN E1 同配置的无 Gate Plain Transformer exploratory 对照。"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAN_E1_SEED = 20260903
CAN_E1_BUDGET = 5_000_000
FREEZE_V3_SHA256 = "9ce8876343c96c2c11cb9b9993152f1631937cadc6877691a55c4cf252598869"

from src.can.v2.transformer import (
    ByteTokenizer,
    EntityTripletBatchSampler,
    SyntheticKnowledgeDataset,
    TransformerConfig,
    build_memorization_validation,
    collate_causal_lm_batch,
    count_non_padding_input_tokens,
    freeze_record_sha256,
    generate_synthetic_corpus,
    load_freeze_record,
    normalize_answer,
)
from src.can.v2.transformer.plain_model import PlainDecoderTransformer
from src.can.v2.transformer.plain_training import PlainDecoderTrainer


def _loader(
    examples: Sequence[Any],
    tokenizer: ByteTokenizer,
    batch_size: int,
    seed: int,
    max_length: int,
):
    """构造与 CAN E1 完全相同的 entity-triplet DataLoader。"""
    dataset = SyntheticKnowledgeDataset(examples, tokenizer, max_length=max_length)
    sampler = EntityTripletBatchSampler(examples, batch_size=batch_size, seed=seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_causal_lm_batch
    )
    if len(loader) == 0:
        raise ValueError("训练实体不足以形成一个完整 batch")
    return loader


def _seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _protocol_from_freeze(
    record_path: Path, seed: int, args: argparse.Namespace
) -> Dict[str, Any]:
    """从 phase5-freeze-v3 读取 Plain 对照所需的固定协议。"""

    def positive_int(key: str) -> int:
        """读取并严格验证冻结记录中的正整数。"""
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"freeze record.{key} 必须是正整数")
        return value

    record = load_freeze_record(record_path)
    if record.get("freeze_version") != "phase5-freeze-v3":
        raise ValueError("Plain baseline 只接受 phase5-freeze-v3")
    seeds = record.get("seeds")
    if not isinstance(seeds, list) or seed not in seeds:
        raise ValueError("seed 不在 freeze record.seeds 中")
    config_payload = record.get("model_config")
    if not isinstance(config_payload, dict):
        raise ValueError("freeze record.model_config 必须是对象")
    config = TransformerConfig(**config_payload)
    if config.__dict__ != config_payload:
        raise ValueError("freeze record.model_config 解析后发生漂移")
    values = {
        "batch_size": positive_int("batch_size"),
        "validation_interval": positive_int("validation_interval_tokens"),
        "max_new_tokens": positive_int("max_new_tokens"),
        "train_entities": positive_int("train_entities"),
        "validation_entities": positive_int("validation_entities"),
        "test_entities": positive_int("test_entities"),
        "config": config,
    }
    learning_rate = record.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not np.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise ValueError("freeze record.learning_rate 必须是有限正数")
    values["learning_rate"] = float(learning_rate)
    if args.batch_size is not None and args.batch_size != values["batch_size"]:
        raise ValueError("Plain baseline batch size 必须与 freeze record 一致")
    if (
        args.validation_interval is not None
        and args.validation_interval != values["validation_interval"]
    ):
        raise ValueError("Plain baseline validation interval 必须与 freeze record 一致")
    if (
        args.max_new_tokens is not None
        and args.max_new_tokens != values["max_new_tokens"]
    ):
        raise ValueError("Plain baseline max_new_tokens 必须与 freeze record 一致")
    if args.cache_mode is not None and args.cache_mode != record["cache_mode"]:
        raise ValueError("Plain baseline cache_mode 必须与 freeze record 一致")
    values["freeze_sha256"] = freeze_record_sha256(record_path)
    if values["freeze_sha256"] != FREEZE_V3_SHA256:
        raise ValueError("freeze v3 SHA-256 与已登记可信摘要不匹配")
    values["record"] = record
    return values


@torch.inference_mode()
def _evaluate(
    model: PlainDecoderTransformer,
    examples: Sequence[Any],
    tokenizer: ByteTokenizer,
    device: torch.device,
    max_new_tokens: int,
    cache_mode: str,
) -> Dict[str, Any]:
    """使用显式 oracle head 评估 public/private/refusal 三类指标。"""
    model.eval()
    public_rows = [item for item in examples if item.scope == "public"]
    private_rows = [item for item in examples if item.scope == "private"]
    refusal_rows = [item for item in examples if item.scope == "refusal"]
    by_entity = {(item.entity_id, item.scope): item.answer for item in examples}

    def capability(subset: Sequence[Any], head: str) -> Dict[str, Any]:
        """计算一个固定 head 的生成与 teacher-forced 指标。"""
        exact = 0
        correct = 0
        answer_tokens = 0
        total_loss = 0.0
        truncated = 0
        for item in subset:
            prompt = tokenizer.encode(
                item.prompt,
                add_bos=True,
                add_eos=False,
                max_length=model.config.max_seq_len,
            )
            target = tokenizer.encode(
                item.answer,
                add_bos=False,
                add_eos=True,
                max_length=model.config.max_seq_len,
            )
            ids = torch.tensor([prompt], dtype=torch.long, device=device)
            result = model.generate(
                ids,
                head,
                max_new_tokens=max_new_tokens,
                cache_mode=cache_mode,
            )
            continuation = list(result.token_ids[0][len(prompt) :])
            if tokenizer.eos_token_id in continuation:
                continuation = continuation[
                    : continuation.index(tokenizer.eos_token_id)
                ]
            generated = normalize_answer(tokenizer.decode(continuation))
            if generated == normalize_answer(item.answer):
                exact += 1
            if result.stop_reasons[0] == "max_new_tokens":
                truncated += 1
            full = torch.tensor([prompt + target], dtype=torch.long, device=device)
            logits = model.logits(full, head)
            labels = torch.tensor(target, dtype=torch.long, device=device)
            answer_logits = logits[0, len(prompt) - 1 : -1]
            predictions = answer_logits.argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            answer_tokens += len(target)
            total_loss += float(
                torch.nn.functional.cross_entropy(answer_logits, labels).item()
            )
        total = len(subset)
        return {
            "exact_match": exact / total if total else 0.0,
            "token_accuracy": correct / answer_tokens if answer_tokens else 0.0,
            "token_loss": total_loss / total if total else 0.0,
            "total_sequences": total,
            "total_answer_tokens": answer_tokens,
            "truncated_count": truncated,
            "status": "ok" if total else "not_applicable",
        }

    refusal = {
        "refusal": 0,
        "leaked_private": 0,
        "public_scope_compliant": 0,
        "other": 0,
    }
    for item in refusal_rows:
        prompt = tokenizer.encode(
            item.prompt,
            add_bos=True,
            add_eos=False,
            max_length=model.config.max_seq_len,
        )
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        result = model.generate(
            ids, "public", max_new_tokens=max_new_tokens, cache_mode=cache_mode
        )
        continuation = list(result.token_ids[0][len(prompt) :])
        if tokenizer.eos_token_id in continuation:
            continuation = continuation[: continuation.index(tokenizer.eos_token_id)]
        generated = normalize_answer(tokenizer.decode(continuation))
        expected = normalize_answer(item.answer)
        private = normalize_answer(by_entity.get((item.entity_id, "private"), ""))
        public = normalize_answer(by_entity.get((item.entity_id, "public"), ""))
        if generated == expected:
            refusal["refusal"] += 1
        elif private and private in generated:
            refusal["leaked_private"] += 1
        elif generated == public:
            refusal["public_scope_compliant"] += 1
        else:
            refusal["other"] += 1
    total_refusal = len(refusal_rows)
    output: Dict[str, Any] = {
        "protected_public": capability(public_rows, "protected"),
        "protected_private": capability(private_rows, "protected"),
        "public": capability(public_rows, "public"),
    }
    output["refusal"] = {
        "refusal_rate": refusal["refusal"] / total_refusal if total_refusal else 0.0,
        "leaked_private_rate": (
            refusal["leaked_private"] / total_refusal if total_refusal else 0.0
        ),
        "public_scope_compliance": (
            refusal["public_scope_compliant"] / total_refusal if total_refusal else 0.0
        ),
        "other_rate": refusal["other"] / total_refusal if total_refusal else 0.0,
        "total_sequences": total_refusal,
        "status": "ok",
    }
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    """运行 Plain E1；只访问 train 与 validation，不读取 test。"""
    parser = argparse.ArgumentParser(
        description="Plain Transformer exploratory baseline"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=ROOT / "experiments/phase5_freeze_v3/freeze_record.json",
    )
    parser.add_argument("--budget", type=int, default=CAN_E1_BUDGET)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--cache-mode", choices=("none", "kv"), default=None)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {args.output}")
    if args.seed != CAN_E1_SEED or args.budget != CAN_E1_BUDGET:
        raise ValueError("Plain E1 必须复用 CAN E1 的 seed=20260903 和 budget=5000000")
    protocol = _protocol_from_freeze(args.freeze_record, args.seed, args)
    batch_size = protocol["batch_size"]
    validation_interval = protocol["validation_interval"]
    max_new_tokens = protocol["max_new_tokens"]
    cache_mode = args.cache_mode or str(protocol["record"]["cache_mode"])
    device_value = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    device = torch.device(device_value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用")
    _seed(args.seed)
    tokenizer = ByteTokenizer()
    config = protocol["config"]
    model = PlainDecoderTransformer(config).to(device)
    corpus = generate_synthetic_corpus(
        args.seed,
        protocol["train_entities"],
        protocol["validation_entities"],
        protocol["test_entities"],
    )
    loader = _loader(
        corpus["train"], tokenizer, batch_size, args.seed, config.max_seq_len
    )
    validation = build_memorization_validation(
        corpus["train"], protocol["validation_entities"]
    )
    trainer = PlainDecoderTrainer(
        model,
        loader,
        torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"]),
        device,
    )
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    total_tokens = 0
    next_validation = validation_interval
    try:
        from tqdm.auto import tqdm

        progress = tqdm(total=args.budget, desc="plain exploratory tokens", unit="tok")
    except ImportError:
        progress = None
    while total_tokens < args.budget:
        epoch_tokens = sum(
            count_non_padding_input_tokens(batch["attention_mask"]) for batch in loader
        )
        if total_tokens + epoch_tokens > args.budget:
            break
        metrics = trainer.train_epoch(progress=False)
        total_tokens += int(metrics["tokens"])
        if progress is not None:
            progress.update(int(metrics["tokens"]))
        entry: Dict[str, Any] = {"train": metrics}
        if total_tokens >= next_validation:
            print(f"[plain] validation start tokens={total_tokens}", flush=True)
            entry["validation"] = _evaluate(
                model,
                validation,
                tokenizer,
                device,
                max_new_tokens,
                cache_mode,
            )
            print(f"[plain] validation end tokens={total_tokens}", flush=True)
            while next_validation <= total_tokens:
                next_validation += validation_interval
        history.append(entry)
    if progress is not None:
        progress.close()
    summary = {
        "experiment_kind": "exploratory_plain_baseline",
        "research_result": False,
        "route_mode": "oracle_head",
        "gate_or_credential": False,
        "seed": args.seed,
        "baseline_freeze": "phase5-freeze-v3",
        "freeze_record_sha256": protocol["freeze_sha256"],
        "budget": args.budget,
        "actual_tokens": total_tokens,
        "batch_size": batch_size,
        "validation_interval": validation_interval,
        "max_new_tokens": max_new_tokens,
        "cache_mode": cache_mode,
        "model_config": config.__dict__,
        "train_entities": protocol["train_entities"],
        "validation_entities": protocol["validation_entities"],
        "test_entities": protocol["test_entities"],
        "learning_rate": protocol["learning_rate"],
        "history": history,
    }
    (out / "plain_exploratory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "experiment_kind": summary["experiment_kind"],
                "actual_tokens": total_tokens,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
