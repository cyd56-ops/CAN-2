"""运行 Phase 5.5/T2 Plain/CAN 成对 T-pretrain。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.can.v2.crypto.lwe import LWEParams
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    ByteTokenizer,
    GatedDecoderTransformer,
    PlainDecoderTransformer,
    T2CanPretrainer,
    T2CausalLMDataset,
    T2Evaluator,
    T2PlainPretrainer,
    T2QuadrupletBatchSampler,
    T2RuntimeConfig,
    TransformerConfig,
    atomic_write_json,
    build_checkpoint_manifest,
    collate_t2_causal_lm_batch,
    derive_lwe_keypair,
    file_sha256,
    generate_t2_split,
    initialize_run_state,
    load_frozen_runtime,
    load_run_state,
    load_t2_checkpoint,
    model_tensor_sha256,
    save_t2_checkpoint,
    t2_split_sha256,
    write_manifest,
)

DEFAULT_COUNTS = {"train": 12, "dev": 4, "validation": 4, "test": 4}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析 T2 训练参数；正式模式参数最终以 freeze 为准。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("dev-pilot", "frozen-validation"), required=True
    )
    parser.add_argument("--models", choices=("both", "plain", "can"))
    parser.add_argument("--suite", choices=("cap", "mem"), required=True)
    parser.add_argument("--prompt-group", choices=("C0", "C1", "C2"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--validation-interval-tokens", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--cache-mode", choices=("none", "kv"))
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--dev-count", type=int)
    parser.add_argument("--validation-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--freeze-record", type=Path)
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--smoke-model", action="store_true", help="使用最小 CPU 模型，仅供工程 smoke"
    )
    return parser.parse_args(argv)


def _seed_everything(seed: int) -> None:
    """重置 Plain/CAN 共用的确定性初始化随机状态。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    """解析设备，并在显式 CUDA 不可用时失败。"""

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但 CUDA 不可用")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _config_from_args(args: argparse.Namespace) -> T2RuntimeConfig:
    """构造 dev 配置，或加载并交叉校验正式 freeze。"""

    suite_id = "t2_nl_cap" if args.suite == "cap" else "t2_nl_mem"
    if args.mode == "frozen-validation":
        if args.freeze_record is None or args.expected_freeze_sha256 is None:
            raise ValueError("frozen-validation 必须提供 freeze record 与可信摘要")
        config = load_frozen_runtime(args.freeze_record, args.expected_freeze_sha256)
        expected = (suite_id, args.prompt_group, args.seed)
        actual = (config.suite_id, config.prompt_group, config.seed)
        if expected != actual:
            raise ValueError("CLI suite/prompt-group/seed/models 与 freeze 不一致")
        if args.models is not None and args.models != config.models:
            raise ValueError("CLI models 与 freeze 不一致")
        overrides = {
            "token_budget": args.token_budget,
            "validation_interval_tokens": args.validation_interval_tokens,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_new_tokens": args.max_new_tokens,
            "cache_mode": args.cache_mode,
            "train_count": args.train_count,
            "dev_count": args.dev_count,
            "validation_count": args.validation_count,
            "test_count": args.test_count,
        }
        frozen_values = {
            "token_budget": config.token_budget,
            "validation_interval_tokens": config.validation_interval_tokens,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_new_tokens": config.max_new_tokens,
            "cache_mode": config.cache_mode,
            "train_count": config.split_counts["train"],
            "dev_count": config.split_counts["dev"],
            "validation_count": config.split_counts["validation"],
            "test_count": config.split_counts["test"],
        }
        mismatches = [
            name
            for name, value in overrides.items()
            if value is not None and value != frozen_values[name]
        ]
        if mismatches:
            raise ValueError(
                "CLI 显式参数与 freeze 不一致: " + ", ".join(sorted(mismatches))
            )
        if args.smoke_model:
            raise ValueError("frozen-validation 禁止 --smoke-model")
        return config
    if args.freeze_record is not None or args.expected_freeze_sha256 is not None:
        raise ValueError("dev-pilot 禁止伪装为 frozen run")
    model_config = (
        TransformerConfig(num_layers=2, cut_layer=1, d_model=16, num_heads=4, d_ff=32)
        if args.smoke_model
        else TransformerConfig()
    )
    counts = {
        "train": (
            args.train_count
            if args.train_count is not None
            else DEFAULT_COUNTS["train"]
        ),
        "dev": args.dev_count if args.dev_count is not None else DEFAULT_COUNTS["dev"],
        "validation": (
            args.validation_count
            if args.validation_count is not None
            else DEFAULT_COUNTS["validation"]
        ),
        "test": (
            args.test_count if args.test_count is not None else DEFAULT_COUNTS["test"]
        ),
    }
    if suite_id == "t2_nl_mem" and len(set(counts.values())) != 1:
        raise ValueError("MEM 四路 count 必须相同")
    return T2RuntimeConfig(
        mode=args.mode,
        suite_id=suite_id,
        prompt_group=args.prompt_group,
        seed=args.seed,
        models=args.models or "both",
        split_counts=counts,
        batch_size=args.batch_size if args.batch_size is not None else 4,
        token_budget=args.token_budget if args.token_budget is not None else 4096,
        validation_interval_tokens=(
            args.validation_interval_tokens
            if args.validation_interval_tokens is not None
            else 2048
        ),
        learning_rate=args.learning_rate if args.learning_rate is not None else 1e-3,
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else 16,
        cache_mode=args.cache_mode or "none",
        model_config=model_config,
    )


def _loader(
    examples: Sequence[Any], config: T2RuntimeConfig, epoch: int
) -> Tuple[DataLoader, T2QuadrupletBatchSampler]:
    """构造完整四元组 loader，并将 sampler 定位到指定 epoch。"""

    dataset = T2CausalLMDataset(
        examples, ByteTokenizer(), config.model_config.max_seq_len
    )
    sampler = T2QuadrupletBatchSampler(examples, config.batch_size, config.seed)
    sampler.set_epoch(epoch)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_t2_causal_lm_batch
    )
    if len(loader) == 0:
        raise ValueError("T2 train split 不足以形成一个完整 batch")
    return loader, sampler


def _batch_ids(batch: Mapping[str, object]) -> Iterable[str]:
    """从 collate batch 中提取规范 sample ID 序列。"""

    values = batch.get("sample_ids")
    if not isinstance(values, list) or any(
        not isinstance(value, str) for value in values
    ):
        raise ValueError("batch 缺少规范 sample_ids")
    return values


def _build_model(
    kind: str, config: T2RuntimeConfig, device: torch.device
) -> Tuple[torch.nn.Module, Optional[CredentialGenerator]]:
    """按模型种类重置初始化，并构造独立 CAN credential RNG。"""

    _seed_everything(config.seed)
    if kind == "plain":
        return PlainDecoderTransformer(config.model_config).to(device), None
    params = LWEParams(n=config.lwe_n, m=config.lwe_m, sigma=config.lwe_sigma)
    matrix, secret, vector = derive_lwe_keypair(config.seed, params)
    model = GatedDecoderTransformer(matrix, vector, params, config.model_config).to(
        device
    )
    generator = CredentialGenerator(
        matrix, secret, vector, params, seed=config.seed + 20_000
    )
    return model, generator


def _shared_initial_hash(model: torch.nn.Module) -> str:
    """计算排除 Gate buffer 的共享可训练初始化摘要。"""

    return model_tensor_sha256(model, trainable_only=True)


def _checkpoint_metadata(
    config: T2RuntimeConfig,
    kind: str,
    train_hash: str,
    eval_hash: str,
    eval_split: str,
) -> Dict[str, Any]:
    """构造 checkpoint 与恢复使用的稳定身份元数据。"""

    return {
        "suite_id": config.suite_id,
        "prompt_group": config.prompt_group,
        "seed": config.seed,
        "model_kind": kind,
        "mode": config.mode,
        "generator_version": "phase5-t2-natural-language-pilot-v1",
        "normalization_version": "t2-nfkc-casefold-punct-articles-v1",
        "tokenizer_version": "byte-tokenizer-v1",
        "train_split_sha256": train_hash,
        "evaluation_split": eval_split,
        "evaluation_split_sha256": eval_hash,
        "split_counts": dict(config.split_counts),
    }


def _selection_score(report: Mapping[str, Any]) -> float:
    """用四个 scope 的 macro token-F1 均值选择 T-pretrain best。"""

    by_scope = report["text_metrics"]["by_scope"]
    values = [
        by_scope[scope]["token_f1"]
        for scope in ("public", "protected_public", "protected_private", "refusal")
    ]
    if any(
        not isinstance(value, (int, float)) or not np.isfinite(value)
        for value in values
    ):
        raise FloatingPointError("T2 checkpoint selection score 非有限或缺失")
    return float(np.mean(values))


def _strictly_improves(candidate: float, best: Optional[float]) -> bool:
    """仅在有限候选分数严格更高时允许替换历史 best。"""

    if not np.isfinite(candidate) or (best is not None and not np.isfinite(best)):
        raise FloatingPointError("T2 best 比较不得包含非有限值")
    return best is None or candidate > best


def _compact_evaluation(report: Mapping[str, Any]) -> Dict[str, Any]:
    """复制 evaluator 摘要并剥离单独保存的逐样本诊断。"""

    required = {
        "schema_version",
        "status",
        "model_kind",
        "suite_id",
        "split",
        "route_mode",
        "gate_or_credential",
        "text_metrics",
        "teacher_forced_by_scope",
        "routing",
        "generation_safety",
        "diagnostics",
    }
    if set(report) != required:
        raise ValueError("T2 evaluator report schema 不匹配")
    return {key: value for key, value in report.items() if key != "diagnostics"}


def _best_progress(
    score: Optional[float],
    at_tokens: Optional[int],
    at_global_step: Optional[int],
    evaluation: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """构造 checkpoint 中绑定最佳时点和摘要的稳定字段。"""

    return {
        "best_score": score,
        "best_at_tokens": at_tokens,
        "best_at_global_step": at_global_step,
        "best_evaluation": None if evaluation is None else dict(evaluation),
    }


def _train_one(
    kind: str,
    config: T2RuntimeConfig,
    train_rows: Sequence[Any],
    eval_rows: Sequence[Any],
    output: Path,
    device: torch.device,
    resume: bool,
    progress_enabled: bool,
) -> Dict[str, Any]:
    """训练单个 Plain/CAN 模型，并保存可恢复 checkpoint 与 dev 指标。"""

    model, generator = _build_model(kind, config, device)
    initial_hash = _shared_initial_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    trainer = (
        T2PlainPretrainer(model, optimizer, device)
        if kind == "plain"
        else T2CanPretrainer(model, optimizer, device, generator)  # type: ignore[arg-type]
    )
    train_hash = t2_split_sha256(train_rows)
    eval_hash = t2_split_sha256(eval_rows)
    eval_split = eval_rows[0].split
    metadata = _checkpoint_metadata(config, kind, train_hash, eval_hash, eval_split)
    model_dir = output / kind
    last_path = model_dir / "last.ckpt"
    best_path = model_dir / "best.ckpt"
    epoch = 0
    batch_offset = 0
    total_tokens = 0
    next_validation = config.validation_interval_tokens
    batch_digest = hashlib.sha256()
    consumed_sample_ids: List[str] = []
    history: List[Dict[str, Any]] = []
    best_score: Optional[float] = None
    best_at_tokens: Optional[int] = None
    best_at_global_step: Optional[int] = None
    best_evaluation: Optional[Dict[str, Any]] = None
    resume_count = 0
    credential_rng_state: Optional[Mapping[str, Any]] = None
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"{kind} resume checkpoint 不存在")
        payload = load_t2_checkpoint(last_path, model, optimizer, metadata)
        saved = payload["progress"]
        epoch = int(saved["epoch"])
        batch_offset = int(saved["batch_offset"])
        total_tokens = int(saved["total_tokens"])
        next_validation = int(saved["next_validation_tokens"])
        best_score = (
            None if saved.get("best_score") is None else float(saved["best_score"])
        )
        resume_count = int(saved.get("resume_count", 0)) + 1
        if best_score is not None:
            required_best = (
                saved.get("best_at_tokens"),
                saved.get("best_at_global_step"),
                saved.get("best_evaluation"),
            )
            if any(value is None for value in required_best):
                raise ValueError("旧版 checkpoint 缺少 schema v2 best 恢复字段")
            best_at_tokens = int(saved["best_at_tokens"])
            best_at_global_step = int(saved["best_at_global_step"])
            if not isinstance(saved["best_evaluation"], Mapping):
                raise ValueError("checkpoint best_evaluation 必须是 Mapping")
            best_evaluation = dict(saved["best_evaluation"])
        consumed_sample_ids = list(saved.get("consumed_sample_ids", []))
        saved_history = saved.get("history", [])
        if not isinstance(saved_history, list):
            raise ValueError("checkpoint history 必须是 list")
        history = list(saved_history)
        for sample_id in consumed_sample_ids:
            batch_digest.update(sample_id.encode("utf-8") + b"\0")
        trainer.global_step = int(saved["global_step"])
        credential_rng_state = payload["credential_rng_state"]
        if generator is not None and credential_rng_state is not None:
            generator.rng.bit_generator.state = dict(credential_rng_state)
        if best_score is not None:
            if (
                not best_path.is_file()
                or not (model_dir / "best_diagnostic.json").is_file()
            ):
                raise ValueError("resume 缺少与 best 进度绑定的 checkpoint/diagnostic")
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            best_saved = best_payload["progress"]
            if any(
                best_saved.get(key) != saved.get(key)
                for key in (
                    "best_score",
                    "best_at_tokens",
                    "best_at_global_step",
                    "best_evaluation",
                )
            ):
                raise ValueError("best 与 last 恢复状态不一致；保留目录排查中断写入")
            if not (
                0 < best_at_global_step <= trainer.global_step
                and 0 < best_at_tokens <= total_tokens
            ):
                raise ValueError("best 绝对时点不在已完成训练范围内")
    bar = tqdm(
        total=config.token_budget,
        unit="tok",
        desc=f"T2 {kind}",
        disable=not progress_enabled,
        dynamic_ncols=True,
    )
    if total_tokens:
        bar.update(total_tokens)
    try:
        stop = False
        while not stop:
            loader, _ = _loader(train_rows, config, epoch)
            advanced = False
            for index, batch in enumerate(loader):
                if index < batch_offset:
                    continue
                token_count = int(batch["attention_mask"].sum().item())
                if total_tokens + token_count > config.token_budget:
                    stop = True
                    break
                advanced = True
                metrics = trainer.train_batch(batch)
                total_tokens += int(metrics["tokens"])
                for sample_id in _batch_ids(batch):
                    batch_digest.update(sample_id.encode("utf-8") + b"\0")
                    consumed_sample_ids.append(sample_id)
                batch_offset = index + 1
                history.append({**metrics, "total_tokens": total_tokens})
                bar.update(int(metrics["tokens"]))
                candidate_best = False
                if total_tokens >= next_validation:
                    print(
                        f"[{kind}] {eval_split} evaluation start @ {total_tokens} tokens",
                        flush=True,
                    )
                    evaluator = T2Evaluator(
                        model,
                        ByteTokenizer(),
                        device,
                        generator,
                        max_new_tokens=config.max_new_tokens,
                        cache_mode=config.cache_mode,
                        batch_size=config.batch_size,
                    )
                    report = evaluator.evaluate(eval_rows)
                    history[-1]["evaluation"] = report["text_metrics"]
                    score = _selection_score(report)
                    compact_report = _compact_evaluation(report)
                    if _strictly_improves(score, best_score):
                        best_score = score
                        best_at_tokens = total_tokens
                        best_at_global_step = trainer.global_step
                        best_evaluation = compact_report
                        candidate_best = True
                    print(f"[{kind}] {eval_split} evaluation end", flush=True)
                    while next_validation <= total_tokens:
                        next_validation += config.validation_interval_tokens
                state = {
                    "epoch": epoch,
                    "batch_offset": batch_offset,
                    "total_tokens": total_tokens,
                    "next_validation_tokens": next_validation,
                    "global_step": trainer.global_step,
                    "batch_order_sha256_prefix": batch_digest.hexdigest(),
                    "consumed_sample_ids": list(consumed_sample_ids),
                    "resume_count": resume_count,
                    **_best_progress(
                        best_score,
                        best_at_tokens,
                        best_at_global_step,
                        best_evaluation,
                    ),
                    "history": list(history),
                }
                credential_state = (
                    generator.rng.bit_generator.state if generator is not None else None
                )
                save_t2_checkpoint(
                    last_path, model, optimizer, metadata, state, credential_state
                )
                if candidate_best:
                    save_t2_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        metadata,
                        state,
                        credential_state,
                    )
                    atomic_write_json(
                        model_dir / "best_diagnostic.json",
                        {"diagnostics": report["diagnostics"]},
                    )
            if stop:
                break
            if not advanced and batch_offset == 0:
                raise RuntimeError("T2 loader 未产生可训练 batch")
            epoch += 1
            batch_offset = 0
    finally:
        bar.close()
    if total_tokens <= 0:
        raise ValueError("token budget 不足以容纳一个完整 batch")
    evaluator = T2Evaluator(
        model,
        ByteTokenizer(),
        device,
        generator,
        max_new_tokens=config.max_new_tokens,
        cache_mode=config.cache_mode,
        batch_size=config.batch_size,
    )
    final_report = evaluator.evaluate(eval_rows)
    final_score = _selection_score(final_report)
    final_compact = _compact_evaluation(final_report)
    final_is_best = _strictly_improves(final_score, best_score)
    if final_is_best:
        best_score = final_score
        best_at_tokens = total_tokens
        best_at_global_step = trainer.global_step
        best_evaluation = final_compact
    final_state = {
        "epoch": epoch,
        "batch_offset": batch_offset,
        "total_tokens": total_tokens,
        "next_validation_tokens": next_validation,
        "global_step": trainer.global_step,
        "batch_order_sha256_prefix": batch_digest.hexdigest(),
        "consumed_sample_ids": list(consumed_sample_ids),
        "resume_count": resume_count,
        **_best_progress(
            best_score,
            best_at_tokens,
            best_at_global_step,
            best_evaluation,
        ),
        "history": list(history),
    }
    credential_state = (
        generator.rng.bit_generator.state if generator is not None else None
    )
    if final_is_best:
        save_t2_checkpoint(
            best_path, model, optimizer, metadata, final_state, credential_state
        )
        atomic_write_json(
            model_dir / "best_diagnostic.json",
            {"diagnostics": final_report["diagnostics"]},
        )
    save_t2_checkpoint(
        last_path, model, optimizer, metadata, final_state, credential_state
    )
    atomic_write_json(
        model_dir / "final_diagnostic.json",
        {"diagnostics": final_report["diagnostics"]},
    )
    if (
        best_score is None
        or best_at_tokens is None
        or best_at_global_step is None
        or best_evaluation is None
        or not best_path.is_file()
        or not (model_dir / "best_diagnostic.json").is_file()
    ):
        raise RuntimeError("T2 best checkpoint/evaluation 未形成完整绑定")
    best_is_final = (
        best_at_tokens == total_tokens and best_at_global_step == trainer.global_step
    )
    summary = {
        "schema_version": 2,
        "status": "completed_budget",
        "research_result": config.mode == "frozen-validation",
        "lifecycle_stage": "T-pretrain",
        "model_kind": kind,
        "total_tokens": total_tokens,
        "global_step": trainer.global_step,
        "batch_order_sha256": batch_digest.hexdigest(),
        "shared_initial_tensor_sha256": initial_hash,
        "final_model_tensor_sha256": model_tensor_sha256(model),
        "resume_count": resume_count,
        "best_selection_score": best_score,
        "best_at_tokens": best_at_tokens,
        "best_at_global_step": best_at_global_step,
        "best_evaluation": best_evaluation,
        "final_selection_score": final_score,
        "final_at_tokens": total_tokens,
        "final_at_global_step": trainer.global_step,
        "final_evaluation": final_compact,
        "best_is_final": best_is_final,
        "best_checkpoint": {
            "path": "best.ckpt",
            "sha256": file_sha256(best_path),
        },
        "final_checkpoint": {
            "path": "last.ckpt",
            "sha256": file_sha256(last_path),
        },
        "history": history,
        "stages": {
            "T-pretrain": "completed",
            "A": "not_run",
            "B": "not_run",
            "C": "not_run",
        },
    }
    manifest = build_checkpoint_manifest(
        model_dir,
        {
            "last.ckpt": {
                "seed": config.seed,
                "stage": "T-pretrain",
                "model_kind": kind,
            },
            "best.ckpt": {
                "seed": config.seed,
                "stage": "T-pretrain",
                "model_kind": kind,
            },
        },
    )
    manifest_sha = write_manifest(model_dir / "checkpoint_manifest.json", manifest)
    (model_dir / "checkpoint_manifest.sha256").write_text(
        manifest_sha + "\n", encoding="ascii"
    )
    summary["checkpoint_manifest_sha256"] = manifest_sha
    atomic_write_json(model_dir / "summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """执行 T2 成对训练并形成受管、可恢复的实验目录。"""

    args = _parse_args(argv)
    config = _config_from_args(args)
    if config.mode == "frozen-validation" and config.models != "both":
        raise ValueError("正式 frozen-validation 必须运行 both")
    output = args.output.resolve()
    state_path = output / "pair_run_state.json"
    identity = config.to_dict()
    if args.resume:
        state = load_run_state(state_path, identity)
        # 完成的旧结果保持原样，禁止借 resume 将 schema v1 历史重写为 v2。
        if state.get("status") == "completed":
            print(json.dumps({"status": "already_completed"}))
            return 0
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("output 非空；恢复必须显式使用 --resume")
        output.mkdir(parents=True, exist_ok=True)
        state = initialize_run_state(state_path, identity)
        atomic_write_json(output / "resolved_config.json", identity)
    # 延迟生成只允许当前模式所需的两个 split；此处绝不生成 test。
    eval_split = "dev" if config.mode == "dev-pilot" else "validation"
    train_rows = generate_t2_split(
        config.suite_id,
        "train",
        config.seed,
        config.split_counts,
        prompt_group=config.prompt_group,
    )
    eval_rows = generate_t2_split(
        config.suite_id,
        eval_split,
        config.seed,
        config.split_counts,
        prompt_group=config.prompt_group,
    )
    corpus_manifest = {
        "schema_version": 1,
        "suite_id": config.suite_id,
        "prompt_group": config.prompt_group,
        "seed": config.seed,
        "splits": {
            "train": t2_split_sha256(train_rows),
            eval_split: t2_split_sha256(eval_rows),
        },
        "test_materialized": False,
    }
    atomic_write_json(output / "corpus_manifest.json", corpus_manifest)
    device = _device(args.device)
    selected = ["plain", "can"] if config.models == "both" else [config.models]
    summaries: Dict[str, Any] = dict(state["models"])
    for kind in selected:
        if kind in state["completed_models"]:
            continue
        summary = _train_one(
            kind,
            config,
            train_rows,
            eval_rows,
            output,
            device,
            args.resume,
            progress_enabled=not args.no_progress,
        )
        summaries[kind] = summary
        state["models"] = summaries
        state["completed_models"].append(kind)
        atomic_write_json(state_path, state)
    if config.models == "both":
        if (
            summaries["plain"]["batch_order_sha256"]
            != summaries["can"]["batch_order_sha256"]
        ):
            raise RuntimeError("Plain/CAN batch 顺序不一致")
        if (
            summaries["plain"]["shared_initial_tensor_sha256"]
            != summaries["can"]["shared_initial_tensor_sha256"]
        ):
            raise RuntimeError("Plain/CAN 共享初始权重不一致")
    state["status"] = "completed"
    state["stages"]["T-pretrain"] = "completed"
    atomic_write_json(state_path, state)
    pair_summary = {
        "schema_version": 2,
        "status": "completed",
        "research_result": config.mode == "frozen-validation",
        "identity": identity,
        "models": summaries,
        "test_materialized": False,
    }
    atomic_write_json(output / "pair_summary.json", pair_summary)
    print(json.dumps({"status": "completed", "models": selected}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
