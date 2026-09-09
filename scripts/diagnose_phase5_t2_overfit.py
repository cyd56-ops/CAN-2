"""运行 Phase 5.5/T2 固定单四元组 train-only 过拟合诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.can.v2.crypto.lwe import LWEParams
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    T2_DIAGNOSTIC_PROTOCOL,
    ByteTokenizer,
    GatedDecoderTransformer,
    PlainDecoderTransformer,
    T2CanDirectPretrainer,
    T2CanPretrainer,
    T2CausalLMDataset,
    T2Evaluator,
    T2OverfitTracker,
    T2PlainPretrainer,
    TransformerConfig,
    array_sha256,
    atomic_write_json,
    classify_t2_overfit_outcome,
    collate_t2_causal_lm_batch,
    compact_t2_evaluation,
    derive_lwe_keypair,
    materialize_t2_diagnostic_quartet,
    model_tensor_sha256,
    overfit_snapshot,
    save_t2_diagnostic_checkpoint,
    t2_split_sha256,
)


class _ScheduleGenerator(CredentialGenerator):
    """仅在内存比对两个 CAN 的共同凭证序列，不写出内容或内容摘要。"""

    def __init__(
        self, source: CredentialGenerator, reference: list, record: bool
    ) -> None:
        """复用既有生成器状态；soft 记录，direct 对共同前缀逐行比对。"""

        self.__dict__.update(source.__dict__)
        self.reference = reference
        self.record = record
        self.calls = 0
        self.compared = 0
        self.matches = True

    def generate(self, is_valid: bool) -> np.ndarray:
        """生成一行真实凭证，并比对其类别、顺序及实际数组内容。"""

        value = super().generate(is_valid)
        if self.record:
            self.reference.append((is_valid, value.copy()))
        elif self.calls < len(self.reference):
            expected_class, expected_value = self.reference[self.calls]
            self.compared += 1
            self.matches = (
                self.matches
                and expected_class == is_valid
                and np.array_equal(value, expected_value)
            )
            if not self.matches:
                raise RuntimeError("CAN 凭证共同序列不一致")
        self.calls += 1
        return value


def _seed_everything(seed: int) -> None:
    """重置诊断所需的 Python、NumPy 和 PyTorch 随机状态。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    """解析诊断设备，并拒绝不可用的显式 CUDA。"""

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但 CUDA 不可用")
    return torch.device(name)


def _batch(quartet: Sequence[Any], config: TransformerConfig) -> Mapping[str, object]:
    """把固定四元组编码为单个 T2 causal-LM batch。"""

    dataset = T2CausalLMDataset(quartet, ByteTokenizer(), config.max_seq_len)
    return collate_t2_causal_lm_batch([dataset[index] for index in range(4)])


def _build_variant(
    variant: str, config: TransformerConfig, device: torch.device, seed: int
) -> Tuple[torch.nn.Module, Any, str, Optional[CredentialGenerator]]:
    """以相同 seed 构造 Plain、CAN soft 或 CAN direct 诊断变体。"""

    _seed_everything(seed)
    if variant == "plain":
        model = PlainDecoderTransformer(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = T2PlainPretrainer(model, optimizer, device)
        return model, trainer, model_tensor_sha256(model, trainable_only=True), None
    params = LWEParams(n=32, m=64)
    matrix, secret, vector = derive_lwe_keypair(seed, params)
    model = GatedDecoderTransformer(matrix, vector, params, config).to(device)
    generator = CredentialGenerator(matrix, secret, vector, params, seed=seed + 20_000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if variant == "can_soft":
        trainer = T2CanPretrainer(model, optimizer, device, generator)
    elif variant == "can_direct":
        trainer = T2CanDirectPretrainer(model, optimizer, device, generator)
    else:
        raise ValueError("未知诊断 variant")
    return model, trainer, model_tensor_sha256(model, trainable_only=True), generator


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析固定协议 CLI；研究身份字段不开放覆盖。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """执行 train-only 单四元组诊断并写入受管 JSON 结果。"""

    args = _parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("诊断 output 非空；本协议不支持覆盖或 resume")
    output.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    seed = 20260903
    config = TransformerConfig()
    # helper 的 spy 测试锁定只允许调用一次 train split 生成器。
    quartet = materialize_t2_diagnostic_quartet()
    quartet_hash = t2_split_sha256(quartet)
    batch = _batch(quartet, config)
    atomic_write_json(
        output / "diagnostic_config.json",
        {
            "schema_version": 1,
            "protocol": T2_DIAGNOSTIC_PROTOCOL,
            "diagnostic_only": True,
            "suite_id": "t2_nl_cap",
            "prompt_group": "C0",
            "seed": seed,
            "batch_size": 4,
            "max_updates": 512,
            "evaluation_interval_updates": 16,
            "model_config": asdict(config),
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "max_new_tokens": 16,
            "cache_mode": "none",
            "tokenizer_version": "byte-tokenizer-v1",
            "generator_version": quartet[0].generator_version,
            "normalization_version": "t2-nfkc-casefold-punct-articles-v1",
            "credential_seed_offset": 20000,
            "keypair_seed_offset": 10000,
            "lwe_params": asdict(LWEParams(n=32, m=64)),
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
            },
            "splits_materialized": {
                "train": True,
                "dev": False,
                "validation": False,
                "test": False,
            },
        },
    )
    atomic_write_json(
        output / "quartet_manifest.json",
        {
            "schema_version": 1,
            "protocol": T2_DIAGNOSTIC_PROTOCOL,
            "split": "train",
            "quartet_sha256": quartet_hash,
            "sample_ids": [row.sample_id for row in quartet],
            "source_ids": [row.source_id for row in quartet],
            "credential_classes": [row.credential_class for row in quartet],
        },
    )
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": T2_DIAGNOSTIC_PROTOCOL,
        "diagnostic_only": True,
        "identity": {
            "suite_id": "t2_nl_cap",
            "prompt_group": "C0",
            "seed": seed,
            "batch_size": 4,
            "quartet_sha256": quartet_hash,
        },
        "splits_materialized": {
            "train": True,
            "dev": False,
            "validation": False,
            "test": False,
        },
        "variants": {},
    }
    invalid_runs = 0
    credential_reference: list = []
    compared_credentials = 0
    matching_credentials = True
    expected_initial_hash = None
    for variant in ("plain", "can_soft", "can_direct"):
        model, trainer, initial_hash, generator = _build_variant(
            variant, config, device, seed
        )
        if generator is not None:
            generator = _ScheduleGenerator(
                generator, credential_reference, variant == "can_soft"
            )
            trainer.credential_generator = generator
        variant_dir = output / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        tracker = T2OverfitTracker()
        history: List[Dict[str, Any]] = []
        total_tokens = 0
        last_report: Optional[Dict[str, Any]] = None
        metrics: Optional[Mapping[str, Any]] = None
        update = 0
        invalid_error: Optional[Dict[str, str]] = None
        batch_digest = hashlib.sha256()
        bar = tqdm(
            total=tracker.max_updates,
            desc=f"T2 diagnostic {variant}",
            unit="upd",
            disable=args.no_progress,
            dynamic_ncols=True,
        )
        try:
            if expected_initial_hash is None:
                expected_initial_hash = initial_hash
            elif initial_hash != expected_initial_hash:
                raise RuntimeError("诊断变体初始化不一致")
            for update in range(1, tracker.max_updates + 1):
                metrics = trainer.train_batch(batch)
                total_tokens += int(metrics["tokens"])
                for row in quartet:
                    batch_digest.update(row.sample_id.encode("utf-8") + b"\0")
                bar.update(1)
                history.append(
                    {"update": update, "tokens": total_tokens, "train": metrics}
                )
                if update % 16 == 0:
                    evaluator = T2Evaluator(
                        model,
                        ByteTokenizer(),
                        device,
                        generator,
                        max_new_tokens=16,
                        cache_mode="none",
                        batch_size=4,
                    )
                    candidate_report = evaluator.evaluate(quartet)
                    snapshot = overfit_snapshot(candidate_report, metrics)
                    tracker.observe(update, total_tokens, snapshot)
                    last_report = candidate_report
                    history[-1]["evaluation"] = snapshot
                    if tracker.passed_at_update is not None:
                        break
        except (
            Exception
        ) as exc:  # noqa: BLE001 - invalid_run 必须留痕后由 CLI 返回失败。
            invalid_error = {
                "type": type(exc).__name__,
                "message": "训练、数值或诊断协议检查失败；本结果不可用于性能结论",
            }
            invalid_runs += 1
        finally:
            bar.close()
        snapshot = tracker.snapshots[-1] if tracker.snapshots else None
        completed_updates = trainer.global_step
        status = tracker.status(completed_updates, invalid=invalid_error is not None)
        atomic_write_json(variant_dir / "history.json", {"history": history})
        if invalid_error is None:
            save_t2_diagnostic_checkpoint(
                variant_dir / "final.ckpt",
                model,
                variant,
                completed_updates,
                total_tokens,
            )
        public_identity = None
        if isinstance(model, GatedDecoderTransformer):
            public_identity = {
                "A_sha256": array_sha256(
                    model.gate_layer.verifier.A.detach().cpu().numpy()
                ),
                "b_sha256": array_sha256(
                    model.gate_layer.verifier.b.detach().cpu().numpy()
                ),
            }
        summary["variants"][variant] = {
            "status": status,
            "completed_updates": completed_updates,
            "total_tokens": total_tokens,
            "initial_trainable_tensor_sha256": initial_hash,
            "final_trainable_tensor_sha256": model_tensor_sha256(
                model, trainable_only=True
            ),
            "first_scope_em_update": tracker.first_scope_em_update,
            "first_pass_update": tracker.first_pass_update,
            "first_pass_tokens": tracker.first_pass_tokens,
            "batch_order_sha256": batch_digest.hexdigest(),
            "sample_ids": [row.sample_id for row in quartet],
            "passed_at_update": tracker.passed_at_update,
            "passed_at_tokens": tracker.passed_at_tokens,
            "credential_schedule_id": (
                None if variant == "plain" else f"derived-seed-{seed + 20_000}"
            ),
            "lwe_public_identity": public_identity,
            "final_train_metrics": metrics if invalid_error is None else None,
            "final_snapshot": snapshot,
            "final_evaluation": (
                None if last_report is None else compact_t2_evaluation(last_report)
            ),
            "diagnostics": [] if last_report is None else last_report["diagnostics"],
            "invalid_error": invalid_error,
            "evaluation_at_update": None if snapshot is None else snapshot["update"],
        }
        if variant == "can_direct" and isinstance(generator, _ScheduleGenerator):
            compared_credentials = generator.compared
            matching_credentials = generator.matches
        atomic_write_json(output / "overfit_summary.json", summary)
    credential_reference.clear()
    summary["same_credential_schedule"] = (
        matching_credentials and compared_credentials > 0
    )
    summary["credential_rows_compared"] = compared_credentials
    summary["credential_comparison_scope"] = (
        "common_executed_prefix_including_evaluation"
    )
    initial_hashes = {
        value["initial_trainable_tensor_sha256"]
        for value in summary["variants"].values()
    }
    summary["shared_initialization"] = len(initial_hashes) == 1
    statuses = {
        variant: value["status"] for variant, value in summary["variants"].items()
    }
    summary["decision"] = classify_t2_overfit_outcome(statuses)
    if not summary["same_credential_schedule"] or not summary["shared_initialization"]:
        invalid_runs += 1
        summary["decision"] = "invalid_protocol"
    summary["status"] = "invalid_run" if invalid_runs else "completed"
    atomic_write_json(output / "overfit_summary.json", summary)
    terminal = "invalid_run" if invalid_runs else "completed"
    print(json.dumps({"status": terminal, "output": str(output)}, ensure_ascii=False))
    return 2 if invalid_runs else 0


if __name__ == "__main__":
    raise SystemExit(main())
