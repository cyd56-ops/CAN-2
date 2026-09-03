"""Phase 5 正式训练入口：按冻结配置执行可恢复的 T-pretrain/A/B/C。"""

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

PRETRAIN_PUBLIC_THRESHOLD = 0.80
PRETRAIN_PRIVATE_THRESHOLD = 0.80
PRETRAIN_REFUSAL_THRESHOLD = 0.90

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    ByteTokenizer,
    EntityTripletBatchSampler,
    GatedDecoderTransformer,
    Phase5Trainer,
    PretrainMetrics,
    SyntheticKnowledgeDataset,
    TransformerConfig,
    build_checkpoint_manifest,
    build_memorization_validation,
    collate_causal_lm_batch,
    configure_stage,
    count_non_padding_input_tokens,
    evaluate_pretrain_validation,
    freeze_record_sha256,
    freeze_teacher,
    generate_synthetic_corpus,
    load_freeze_record,
    pretrain_go_no_go,
    sha256_file,
    validate_runtime_against_freeze,
    verify_manifest_entry,
    write_manifest,
)

GENERATOR_VERSION = "phase5-t1-private-query-v2"
STAGES = ("T-pretrain", "A", "B", "C")
STAGE_DIRS = {
    "T-pretrain": "t_pretrain",
    "A": "stage_a",
    "B": "stage_b",
    "C": "stage_c",
}
BUDGET_KEYS = {
    "T-pretrain": "t_pretrain_token_budget",
    "A": "stage_a_token_budget",
    "B": "stage_b_token_budget",
    "C": "stage_c_token_budget",
}


def _device(value: str) -> torch.device:
    """解析设备参数；auto 在 CUDA 可用时选择 CUDA。"""

    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用")
    return result


def _seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机源。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """以原子替换方式写入规范 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reset_managed_output(path: Path) -> None:
    """仅删除含本脚本 run state 的受管输出目录。"""

    resolved = path.resolve()
    if resolved in {Path.cwd().resolve(), Path.home().resolve(), Path(resolved.anchor)}:
        raise ValueError("拒绝覆盖工作区、用户目录或文件系统根目录")
    marker = resolved / "run_state.json"
    if not marker.is_file():
        raise ValueError("--force-overwrite 只允许清理含 run_state.json 的受管目录")
    shutil.rmtree(resolved)


def _frozen_int(record: Mapping[str, Any], key: str, minimum: int = 1) -> int:
    """读取正式训练必需的冻结整数，缺失或类型错误时拒绝启动。"""

    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"freeze record.{key} 必须是 >= {minimum} 的整数")
    return value


def _validate_formal_freeze(record: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    """验证正式训练的数据规模、预算、seed 和模型结构均已冻结。"""

    expected_v3 = {
        "freeze_version": "phase5-freeze-v3",
        "budget_token_unit": "non_padding_input_tokens",
        "t_pretrain_stop_policy": "go-no-go-or-full-budget-v3",
        "t_pretrain_checkpoint_policy": "threshold-ratio-loss-tiebreak-v3",
    }
    for key, expected in expected_v3.items():
        if record.get(key) != expected:
            raise ValueError(f"freeze record.{key} 必须为 {expected!r}")
    if record.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("freeze record.generator_version 与当前数据协议不一致")
    seeds = record.get("seeds")
    if (
        not isinstance(seeds, list)
        or seed not in seeds
        or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds)
    ):
        raise ValueError("正式 seed 必须出现在 freeze record.seeds 整数列表中")
    if record.get("model_config") != TransformerConfig().__dict__:
        raise ValueError(
            "freeze record.model_config 必须与正式 TransformerConfig 完全一致"
        )
    values = {
        "train_entities": _frozen_int(record, "train_entities", 2),
        "validation_entities": _frozen_int(record, "validation_entities", 20),
        "test_entities": _frozen_int(record, "test_entities"),
        "max_new_tokens": _frozen_int(record, "max_new_tokens"),
        "validation_interval_tokens": _frozen_int(record, "validation_interval_tokens"),
    }
    for key in BUDGET_KEYS.values():
        values[key] = _frozen_int(record, key)
    if values["t_pretrain_token_budget"] > 2_000_000:
        raise ValueError("T-pretrain token budget 不得超过 2,000,000")
    if values["validation_interval_tokens"] != 50_000:
        raise ValueError("正式 validation interval 必须冻结为 50,000 tokens")
    if values["validation_entities"] > values["train_entities"]:
        raise ValueError("记忆 validation 实体数不能超过训练实体数")
    learning_rate = record.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not np.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise ValueError("freeze record.learning_rate 必须是有限正数")
    values["learning_rate"] = float(learning_rate)
    benchmark = record.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise ValueError("freeze record.benchmark 必须是对象")
    if benchmark.get("budget_token_unit") != "non_padding_input_tokens":
        raise ValueError("benchmark token unit 与 v3 正式口径不一致")
    for key in ("tokens_per_second", "seconds_per_step", "validation_wall_seconds"):
        value = benchmark.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"freeze record.benchmark.{key} 必须是有限正数")
    benchmark_sha = benchmark.get("sha256")
    if (
        not isinstance(benchmark_sha, str)
        or len(benchmark_sha) != 64
        or any(character not in "0123456789abcdef" for character in benchmark_sha)
    ):
        raise ValueError("freeze record.benchmark.sha256 必须是小写 SHA-256")
    return values


def _loader(
    examples: Sequence[Any],
    tokenizer: ByteTokenizer,
    batch_size: int,
    seed: int,
    max_length: int,
) -> DataLoader:
    """构造确定性 entity-triplet DataLoader 并立即验证非空。"""

    dataset = SyntheticKnowledgeDataset(examples, tokenizer, max_length=max_length)
    sampler = EntityTripletBatchSampler(examples, batch_size=batch_size, seed=seed)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_causal_lm_batch
    )
    if len(loader) == 0:
        raise ValueError("训练实体不足以形成一个完整 batch")
    return loader


def _epoch_token_count(loader: DataLoader, epoch: int) -> int:
    """计算指定完整 epoch 的有效训练 token 数。"""

    sampler = getattr(loader, "batch_sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    return sum(
        count_non_padding_input_tokens(batch["attention_mask"]) for batch in loader
    )


def _validation_metrics(
    model: GatedDecoderTransformer,
    examples: Sequence[Any],
    tokenizer: ByteTokenizer,
    A: np.ndarray,
    secret: np.ndarray,
    b: np.ndarray,
    params: LWEParams,
    seed: int,
    device: torch.device,
    max_new_tokens: int,
    cache_mode: str,
) -> Dict[str, Any]:
    """分别计算 public/private/refusal，禁止用混合 protected EM 代替 private EM。"""

    return evaluate_pretrain_validation(
        model,
        examples,
        tokenizer,
        A,
        secret,
        b,
        params,
        seed,
        device,
        max_new_tokens,
        cache_mode,
    )


def _score(stage: str, metrics: Mapping[str, Any]) -> float:
    """返回固定的 validation checkpoint 选择分数。"""

    if stage == "T-pretrain":
        values = (
            metrics["protected_public"]["exact_match"],
            metrics["protected_private"]["exact_match"],
        )
    elif stage in {"A", "B"}:
        values = (
            metrics["public"]["exact_match"],
            metrics["refusal"]["refusal_rate"],
        )
    else:
        values = (
            metrics["protected_private"]["exact_match"],
            metrics["public"]["exact_match"],
            metrics["refusal"]["refusal_rate"],
        )
    return min(float(value or 0.0) for value in values)


def _pretrain_metrics(metrics: Mapping[str, Any]) -> PretrainMetrics:
    """从 validation JSON 提取 T-pretrain 三项门槛指标。"""

    return PretrainMetrics(
        float(metrics["protected_public"]["exact_match"] or 0.0),
        float(metrics["protected_private"]["exact_match"] or 0.0),
        float(metrics["refusal"]["refusal_rate"] or 0.0),
    )


def _diagnostic_score(metrics: Mapping[str, Any]) -> tuple[float, float, int]:
    """返回未过门槛 checkpoint 的确定性诊断排序键。"""

    go = _pretrain_metrics(metrics)
    ratio = min(
        go.public_exact_match / PRETRAIN_PUBLIC_THRESHOLD,
        go.private_exact_match / PRETRAIN_PRIVATE_THRESHOLD,
        go.refusal_rate / PRETRAIN_REFUSAL_THRESHOLD,
    )
    public_loss = float(metrics["protected_public"].get("token_loss") or float("inf"))
    private_loss = float(metrics["protected_private"].get("token_loss") or float("inf"))
    loss_key = -(public_loss + private_loss) / 2.0
    return ratio, loss_key, 0


def _is_better_diagnostic(
    candidate: Mapping[str, Any],
    candidate_tokens: int,
    incumbent: Optional[Mapping[str, Any]],
    incumbent_tokens: Optional[int],
) -> bool:
    """按 v3 ratio、protected loss、累计 token 规则比较诊断 checkpoint。"""

    if incumbent is None or incumbent_tokens is None:
        return True
    candidate_key = _diagnostic_score(candidate)
    incumbent_key = _diagnostic_score(incumbent)
    if candidate_key[:2] != incumbent_key[:2]:
        return candidate_key[:2] > incumbent_key[:2]
    return candidate_tokens < incumbent_tokens


def _load_model_checkpoint(
    model: GatedDecoderTransformer,
    path: Path,
    expected_stage: str,
    expected_sha256: str,
) -> None:
    """校验摘要与元数据后恢复跨阶段初始化所需的模型权重。"""

    if sha256_file(path) != expected_sha256:
        raise ValueError("run state 中的 checkpoint SHA-256 不匹配")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("stage") != expected_stage:
        raise ValueError("跨阶段 checkpoint schema/stage 不匹配")
    if payload.get("config") != model.config.__dict__:
        raise ValueError("跨阶段 checkpoint 模型配置不匹配")
    model.load_state_dict(payload["model"])


def _new_state(
    seed: int, freeze_sha256: str, batch_size: int, frozen: Mapping[str, Any]
) -> Dict[str, Any]:
    """建立可恢复的正式运行状态。"""

    interval = frozen["validation_interval_tokens"]
    return {
        "schema_version": 1,
        "status": "running",
        "seed": seed,
        "freeze_record_sha256": freeze_sha256,
        "batch_size": batch_size,
        "frozen": dict(frozen),
        "current_stage": "T-pretrain",
        "completed_stages": [],
        "stage_tokens": {stage: 0 for stage in STAGES},
        "histories": {stage: [] for stage in STAGES},
        "best_scores": {stage: None for stage in STAGES},
        "checkpoint_sha256": {},
        "diagnostic_best_validation": {stage: None for stage in STAGES},
        "diagnostic_best_tokens": {stage: None for stage in STAGES},
        "next_validation_tokens": {stage: interval for stage in STAGES},
        "last_validation_score": {stage: None for stage in STAGES},
        "small_improvement_count": {stage: 0 for stage in STAGES},
    }


def _load_resume_state(
    path: Path,
    seed: int,
    freeze_sha256: str,
    batch_size: int,
    frozen: Mapping[str, Any],
) -> Dict[str, Any]:
    """解析并校验 run state 的身份、阶段顺序和核心映射。"""

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("resume state 不是有效 JSON") from exc
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("resume state schema 非法")
    expected = (seed, freeze_sha256, batch_size, dict(frozen))
    actual = (
        state.get("seed"),
        state.get("freeze_record_sha256"),
        state.get("batch_size"),
        state.get("frozen"),
    )
    if actual != expected:
        raise ValueError("resume state 与 seed/freeze/config 不一致")
    completed = state.get("completed_stages")
    if not isinstance(completed, list) or completed != list(STAGES[: len(completed)]):
        raise ValueError("resume state 的已完成阶段不是合法前缀")
    for key in (
        "stage_tokens",
        "histories",
        "best_scores",
        "checkpoint_sha256",
        "next_validation_tokens",
    ):
        if not isinstance(state.get(key), dict):
            raise ValueError(f"resume state.{key} 必须是对象")
    return state


def main(argv: Optional[Sequence[str]] = None) -> int:
    """执行正式 Phase 5 训练；T-pretrain 未通过 go/no-go 时停止。"""

    parser = argparse.ArgumentParser(description="Phase 5 formal training")
    parser.add_argument("--freeze-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cache-mode", choices=("none", "kv"), default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    record = load_freeze_record(args.freeze_record)
    frozen = _validate_formal_freeze(record, args.seed)
    batch_size = args.batch_size or int(record["batch_size"])
    cache_mode = args.cache_mode or str(record["cache_mode"])
    validate_runtime_against_freeze(
        record,
        batch_size=batch_size,
        max_new_tokens=frozen["max_new_tokens"],
        cache_mode=cache_mode,
        generator_version=GENERATOR_VERSION,
    )
    freeze_sha = freeze_record_sha256(args.freeze_record)
    state_path = args.output / "run_state.json"
    if args.resume is not None:
        if args.force_overwrite:
            raise ValueError("--resume 与 --force-overwrite 不能同时使用")
        if args.resume.resolve() != state_path.resolve() or not state_path.is_file():
            raise ValueError("--resume 必须指向当前 output/run_state.json")
        state = _load_resume_state(
            state_path, args.seed, freeze_sha, batch_size, frozen
        )
    else:
        if args.output.exists() and any(args.output.iterdir()):
            if not args.force_overwrite:
                raise FileExistsError("输出目录非空；正式恢复必须使用 --resume")
            _reset_managed_output(args.output)
        state = _new_state(args.seed, freeze_sha, batch_size, frozen)

    _seed_everything(args.seed)
    device = _device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(state_path, state)

    params = LWEParams(n=128, m=256)
    A, secret, b = generate_keypair(params, np.random.default_rng(args.seed))
    config = TransformerConfig()
    model = GatedDecoderTransformer(A, b, params, config=config).to(device)
    tokenizer = ByteTokenizer()
    corpus = generate_synthetic_corpus(
        args.seed,
        frozen["train_entities"],
        frozen["validation_entities"],
        frozen["test_entities"],
    )
    train_loader = _loader(
        corpus["train"], tokenizer, batch_size, args.seed, config.max_seq_len
    )
    # go/no-go 使用训练实体的未见模板，衡量随机私有映射的记忆能力；
    # corpus 自带的实体互斥 validation/test 留给独立泛化报告。
    validation_examples = build_memorization_validation(
        corpus["train"], frozen["validation_entities"]
    )
    credential_generator = CredentialGenerator(
        A, secret, b, params, seed=args.seed + 500
    )

    completed = list(state["completed_stages"])
    if completed:
        previous = completed[-1]
        previous_path = args.output / STAGE_DIRS[previous] / "best.ckpt"
        _load_model_checkpoint(
            model,
            previous_path,
            previous,
            state["checkpoint_sha256"][f"{previous}:best"],
        )

    teacher: Optional[GatedDecoderTransformer] = None
    teacher_identity: Optional[Dict[str, str]] = None

    def prepare_teacher() -> None:
        """从不可变 T-pretrain best 构造独立只读 teacher。"""

        nonlocal teacher, teacher_identity
        teacher_path = args.output / STAGE_DIRS["T-pretrain"] / "best.ckpt"
        teacher = GatedDecoderTransformer(A, b, params, config=config)
        teacher_sha = state["checkpoint_sha256"]["T-pretrain:best"]
        _load_model_checkpoint(teacher, teacher_path, "T-pretrain", teacher_sha)
        freeze_teacher(teacher)
        manifest_path = args.output / "teacher_manifest.json"
        if not manifest_path.is_file():
            manifest = build_checkpoint_manifest(
                args.output,
                {
                    "t_pretrain/best.ckpt": {
                        "seed": args.seed,
                        "stage": "T-pretrain",
                    }
                },
            )
            manifest_sha = write_manifest(manifest_path, manifest)
        else:
            manifest_sha = sha256_file(manifest_path)
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_manifest_entry(teacher_path, manifest_payload, "t_pretrain/best.ckpt")
        teacher_identity = {
            "checkpoint_sha256": teacher_sha,
            "manifest_sha256": manifest_sha,
        }
        recorded_identity = state.get("teacher_identity")
        if recorded_identity is not None and recorded_identity != teacher_identity:
            raise ValueError("teacher manifest identity 与 resume state 不一致")
        state["teacher_identity"] = teacher_identity

    def evaluate() -> Dict[str, Any]:
        """以独立 credential RNG 执行确定性 validation。"""

        return _validation_metrics(
            model,
            validation_examples,
            tokenizer,
            A,
            secret,
            b,
            params,
            args.seed,
            device,
            frozen["max_new_tokens"],
            cache_mode,
        )

    for stage in STAGES:
        if stage in state["completed_stages"]:
            continue
        state["current_stage"] = stage
        if stage in {"B", "C"} and teacher is None:
            prepare_teacher()
        configure_stage(model, stage)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=frozen["learning_rate"],
        )
        trainer = Phase5Trainer(
            model,
            train_loader,
            optimizer,
            device,
            stage,
            credential_generator if stage != "T-pretrain" else None,
            teacher if stage in {"B", "C"} else None,
            teacher_identity if stage in {"B", "C"} else None,
        )
        stage_dir = args.output / STAGE_DIRS[stage]
        stage_dir.mkdir(parents=True, exist_ok=True)
        last_path = stage_dir / "last.ckpt"
        if state["stage_tokens"][stage] > 0:
            if sha256_file(last_path) != state["checkpoint_sha256"][f"{stage}:last"]:
                raise ValueError("resume last.ckpt 摘要不匹配")
            trainer.load_checkpoint(last_path)

        budget = frozen[BUDGET_KEYS[stage]]
        last_validation: Optional[Dict[str, Any]] = None
        stop_stage = False
        progress_bar = None
        if not args.no_progress:
            try:
                from tqdm.auto import tqdm

                progress_bar = tqdm(
                    total=budget,
                    initial=state["stage_tokens"][stage],
                    desc=f"{stage} tokens",
                    unit="tok",
                    dynamic_ncols=True,
                )
            except ImportError:
                print(f"[{stage}] token progress unavailable (tqdm not installed)")
        while state["stage_tokens"][stage] < budget:
            epoch_tokens = _epoch_token_count(train_loader, trainer.current_epoch)
            if epoch_tokens <= 0:
                raise RuntimeError("训练 epoch 不包含有效 token")
            if state["stage_tokens"][stage] + epoch_tokens > budget:
                if state["stage_tokens"][stage] == 0:
                    raise ValueError(
                        f"{stage} token budget 小于一个完整 epoch，无法产生训练更新"
                    )
                break
            try:
                train_metrics = trainer.train_epoch(progress=False)
            except FloatingPointError as exc:
                state["status"] = "failed_non_finite_loss"
                state["failure"] = {"stage": stage, "message": str(exc)}
                _atomic_json(state_path, state)
                _atomic_json(args.output / "failure.json", state["failure"])
                raise
            state["stage_tokens"][stage] += int(train_metrics["tokens"])
            if progress_bar is not None:
                progress_bar.update(int(train_metrics["tokens"]))
            trainer.save_checkpoint(last_path)
            state["checkpoint_sha256"][f"{stage}:last"] = sha256_file(last_path)
            entry: Dict[str, Any] = {"train": train_metrics}
            validation_due = (
                state["stage_tokens"][stage] >= state["next_validation_tokens"][stage]
                or state["stage_tokens"][stage] == budget
            )
            if validation_due:
                print(
                    f"[{stage}] validation start tokens={state['stage_tokens'][stage]}",
                    flush=True,
                )
                last_validation = evaluate()
                print(
                    f"[{stage}] validation end tokens={state['stage_tokens'][stage]}",
                    flush=True,
                )
                entry["validation"] = last_validation
                current_score = _score(stage, last_validation)
                best_score = state["best_scores"][stage]
                em_better = best_score is None or current_score > best_score
                previous_score = state["last_validation_score"][stage]
                if previous_score is not None and current_score - previous_score < 0.01:
                    state["small_improvement_count"][stage] += 1
                else:
                    state["small_improvement_count"][stage] = 0
                state["last_validation_score"][stage] = current_score
                if stage == "T-pretrain":
                    current_go = _pretrain_metrics(last_validation)
                    if pretrain_go_no_go(current_go):
                        # 只有实际通过三项硬门槛的 checkpoint 才能成为 teacher。
                        trainer.save_checkpoint(stage_dir / "best.ckpt")
                        state["checkpoint_sha256"][f"{stage}:best"] = sha256_file(
                            stage_dir / "best.ckpt"
                        )
                        state["best_scores"][stage] = current_score
                        state["stage_stop_reason"] = "go_no_go_passed"
                        stop_stage = True
                    elif _is_better_diagnostic(
                        last_validation,
                        state["stage_tokens"][stage],
                        state.get("diagnostic_best_validation", {}).get(stage),
                        state.get("diagnostic_best_tokens", {}).get(stage),
                    ):
                        # 未通过门槛的 checkpoint 只能写入独立诊断文件，不能污染正式 best。
                        diagnostic_path = stage_dir / "diagnostic_best.ckpt"
                        trainer.save_checkpoint(diagnostic_path)
                        state["checkpoint_sha256"][f"{stage}:diagnostic_best"] = (
                            sha256_file(diagnostic_path)
                        )
                        state["diagnostic_best_validation"][stage] = last_validation
                        state["diagnostic_best_tokens"][stage] = state["stage_tokens"][
                            stage
                        ]
                elif em_better:
                    trainer.save_checkpoint(stage_dir / "best.ckpt")
                    state["checkpoint_sha256"][f"{stage}:best"] = sha256_file(
                        stage_dir / "best.ckpt"
                    )
                    state["best_scores"][stage] = current_score
                while (
                    state["next_validation_tokens"][stage]
                    <= state["stage_tokens"][stage]
                ):
                    state["next_validation_tokens"][stage] += frozen[
                        "validation_interval_tokens"
                    ]
            state["histories"][stage].append(entry)
            _atomic_json(state_path, state)
            if stop_stage:
                break
        if progress_bar is not None:
            progress_bar.close()

        # 阶段结束时评估当前 last.ckpt；T-pretrain 只有通过门槛才建立正式 best。
        last_validation = evaluate()
        current_score = _score(stage, last_validation)
        if stage == "T-pretrain":
            final_go = _pretrain_metrics(last_validation)
            if pretrain_go_no_go(final_go):
                trainer.save_checkpoint(stage_dir / "best.ckpt")
                state["checkpoint_sha256"][f"{stage}:best"] = sha256_file(
                    stage_dir / "best.ckpt"
                )
                state["best_scores"][stage] = current_score
            elif _is_better_diagnostic(
                last_validation,
                state["stage_tokens"][stage],
                state.get("diagnostic_best_validation", {}).get(stage),
                state.get("diagnostic_best_tokens", {}).get(stage),
            ):
                diagnostic_path = stage_dir / "diagnostic_best.ckpt"
                trainer.save_checkpoint(diagnostic_path)
                state["checkpoint_sha256"][f"{stage}:diagnostic_best"] = sha256_file(
                    diagnostic_path
                )
                state["diagnostic_best_validation"][stage] = last_validation
                state["diagnostic_best_tokens"][stage] = state["stage_tokens"][stage]
        elif (
            state["best_scores"][stage] is None
            or current_score > state["best_scores"][stage]
        ):
            trainer.save_checkpoint(stage_dir / "best.ckpt")
            state["checkpoint_sha256"][f"{stage}:best"] = sha256_file(
                stage_dir / "best.ckpt"
            )
            state["best_scores"][stage] = current_score
        state["histories"][stage].append({"final_validation": last_validation})

        if stage == "T-pretrain":
            go_metrics = final_go
            state["go_no_go"] = asdict(go_metrics)
            state["go_no_go"]["passed"] = pretrain_go_no_go(go_metrics)
            if not state["go_no_go"]["passed"]:
                state["status"] = "blocked_go_no_go"
                _atomic_json(state_path, state)
                _atomic_json(args.output / "training_summary.json", state)
                return 2
        # 后续阶段和最终 go/no-go 都以正式 best checkpoint 为准。
        _load_model_checkpoint(
            model,
            stage_dir / "best.ckpt",
            stage,
            state["checkpoint_sha256"][f"{stage}:best"],
        )

        state["completed_stages"].append(stage)
        state["current_stage"] = None
        _atomic_json(stage_dir / "metrics.json", {"history": state["histories"][stage]})
        _atomic_json(state_path, state)

    checkpoints = {
        f"{STAGE_DIRS[stage]}/best.ckpt": {"seed": args.seed, "stage": stage}
        for stage in STAGES
    }
    manifest = build_checkpoint_manifest(args.output, checkpoints)
    state["checkpoint_manifest_sha256"] = write_manifest(
        args.output / "checkpoint_manifest.json", manifest
    )
    state["status"] = "completed"
    _atomic_json(state_path, state)
    _atomic_json(args.output / "training_summary.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
