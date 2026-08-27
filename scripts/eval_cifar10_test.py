"""Phase 3 CIFAR-10 test split evaluator 命令行入口。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.experiments.test_evaluator import TestSplitEvaluator
from src.can.v2.models.gated_resnet import GatedResNet18
from src.can.v2.training.data import (
    CIFAR10WithCoarse,
    CredentialGenerator,
    DATA_MAPPING_VERSION,
    get_cifar_transforms,
)


def _sha256(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_integrity(
    args: argparse.Namespace, checkpoint: Path
) -> tuple[str, str, dict]:
    """按直接摘要或 manifest 校验 checkpoint 完整性。"""
    direct = args.expected_checkpoint_sha256
    manifest = args.checkpoint_manifest
    if direct and manifest:
        raise ValueError("--expected-checkpoint-sha256 与 --checkpoint-manifest 互斥")
    if bool(manifest) != bool(args.expected_manifest_sha256):
        raise ValueError("Manifest 模式必须同时提供 manifest 路径和其期望 SHA-256")
    actual = _sha256(checkpoint)
    if direct:
        if actual != direct:
            raise ValueError("checkpoint SHA-256 不匹配")
        return actual, "verified", {}
    if manifest:
        manifest_path = Path(manifest)
        if _sha256(manifest_path) != args.expected_manifest_sha256:
            raise ValueError("manifest SHA-256 不匹配")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = (Path(__file__).resolve().parent.parent / "checkpoints" / "v2").resolve()
        try:
            key = checkpoint.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("checkpoint 必须位于 checkpoints/v2 目录树下") from exc
        entry = data.get("checkpoints", {}).get(key)
        if not isinstance(entry, dict) or entry.get("sha256") != actual:
            raise ValueError("checkpoint 在 manifest 中缺失或 SHA-256 不匹配")
        if entry.get("size_bytes") != checkpoint.stat().st_size:
            raise ValueError("checkpoint 文件大小与 manifest 不一致")
        return actual, "verified", entry
    return actual, "not_performed", {}


def _aggregate(paths: list[str], output: Path, force: bool) -> None:
    """聚合恰好三个不同 seed 的 Stage C 结果。"""
    if output.exists() and not force:
        raise FileExistsError(f"输出已存在，使用 --force-overwrite 覆盖: {output}")
    records = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if len(records) != 3 or len({r.get("seed") for r in records}) != 3:
        raise ValueError("aggregate 必须接收恰好三个不同 seed 的结果")
    if any(r.get("checkpoint", {}).get("stage") != "C" for r in records):
        raise ValueError("aggregate 只接受 Stage C 结果")
    if len({r.get("eval_batch_size") for r in records}) != 1:
        raise ValueError("各结果 eval_batch_size 必须一致")
    if len({r.get("mapping_version") for r in records}) != 1:
        raise ValueError("各结果 mapping_version 必须一致")
    # 仅聚合可跨 seed 比较的标量；混淆矩阵、直方图和 per-class 数组保留在单 seed 文件中。
    metric_paths = [
        ("authorized", "protected_accuracy"),
        ("authorized", "protected_macro_f1"),
        ("authorized", "protected_total"),
        ("unauthorized", "public_accuracy"),
        ("unauthorized", "public_macro_f1"),
        ("unauthorized", "public_balanced_accuracy"),
        ("unauthorized", "public_total"),
        ("capability", "protected_coarse_accuracy"),
        ("capability", "capability_gap_fine"),
        ("gate", "far"),
        ("gate", "frr"),
        ("gate", "min_margin", "valid"),
        ("gate", "min_margin", "invalid"),
        ("gate", "min_margin", "all"),
        ("gate", "error_norm_stats", "valid", "mean"),
        ("gate", "error_norm_stats", "invalid", "mean"),
        ("mixed_batch", "routing_mismatches"),
        ("mixed_batch", "max_abs_difference"),
        ("mixed_batch", "max_relative_difference"),
        ("mixed_batch", "empty_subbatch_skips", "valid"),
        ("mixed_batch", "empty_subbatch_skips", "invalid"),
        ("latency", "all_valid", "mean_ms"),
        ("latency", "all_valid", "median_ms"),
        ("latency", "all_valid", "p95_ms"),
        ("latency", "all_invalid", "mean_ms"),
        ("latency", "all_invalid", "median_ms"),
        ("latency", "all_invalid", "p95_ms"),
        ("latency", "mixed", "mean_ms"),
        ("latency", "mixed", "median_ms"),
        ("latency", "mixed", "p95_ms"),
    ]
    metrics = {}
    for path in metric_paths:
        values = []
        for record in records:
            value = record
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            values.append(value)
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            metrics[".".join(path)] = {
                "values": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    quality_gate_paths = [
        ("checkpoint", "integrity_check", "verified"),
        ("provenance_check", None, "complete"),
        ("mixed_batch", "index_coverage_complete", True),
        ("mixed_batch", "reference_routing_logits_allclose", True),
        ("mixed_batch", "prediction_indices_exact", True),
    ]
    quality_gates = {}
    for section, key, expected in quality_gate_paths:
        values = [
            record.get(section) if key is None else record.get(section, {}).get(key)
            for record in records
        ]
        quality_gates[f"{section}.{key}" if key else section] = {
            "values": values,
            "expected": expected,
            "all_passed": all(value == expected for value in values),
        }
    metadata = {
        "eval_batch_size": records[0].get("eval_batch_size"),
        "mapping_version": records[0].get("mapping_version"),
        "latency_batch_size": records[0].get("latency", {}).get("batch_size"),
        "checkpoint_sha256": [r.get("checkpoint", {}).get("sha256") for r in records],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "seeds": [r["seed"] for r in records],
                "metadata": metadata,
                "quality_gates": quality_gates,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """解析参数并执行单 checkpoint test 评估。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", nargs="*")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root", default="data/cifar10")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mixed-ratio", type=float, default=0.5)
    parser.add_argument("--summary")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--measure-latency", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if args.aggregate is not None:
        _aggregate(args.aggregate, output, args.force_overwrite)
        return
    if not args.checkpoint:
        raise ValueError("非 aggregate 模式必须提供 --checkpoint")
    if output.exists() and not args.force_overwrite:
        raise FileExistsError(f"输出已存在，使用 --force-overwrite 覆盖: {output}")
    checkpoint = Path(args.checkpoint)
    actual_hash, integrity, manifest_entry = _verify_integrity(args, checkpoint)
    selected_device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    device = torch.device(selected_device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = payload["metadata"]
    rng_version = metadata.get("keypair_rng_version", 1)
    rng_source = "declared" if "keypair_rng_version" in metadata else "assumed_v1"
    if rng_version != 1:
        raise ValueError(f"不支持的 keypair_rng_version: {rng_version}")
    if manifest_entry:
        config_seed = int(metadata.get("config", {}).get("seed", -1))
        stage = str(payload.get("stage", "")).upper()
        epoch = int(payload.get("epoch", -1))
        if (
            manifest_entry.get("seed") != config_seed
            or manifest_entry.get("stage", "").upper() != stage
            or manifest_entry.get("epoch") != epoch
        ):
            raise ValueError(
                "Manifest 的 seed/stage/epoch 与 checkpoint metadata 不一致"
            )
    provenance = "partial"
    summary = None
    if args.summary:
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        required = (
            summary.get("split_hash"),
            summary.get("keypair", {}).get("A_sha256"),
            summary.get("keypair", {}).get("b_sha256"),
        )
        if any(value is None for value in required):
            raise ValueError("summary 缺少 split_hash 或 keypair 哈希")
        provenance = "complete"
    params = LWEParams(**metadata["lwe"])
    seed = int(metadata["config"]["seed"])
    A, secret, b = generate_keypair(params, rng=np.random.default_rng(seed + 100))
    if not np.array_equal(A, metadata["A"]) or not np.array_equal(b, metadata["b"]):
        raise ValueError("checkpoint LWE 公共参数不匹配")
    if summary is not None:
        if summary["keypair"]["A_sha256"] != hashlib.sha256(A.tobytes()).hexdigest():
            raise ValueError("summary A_sha256 与 checkpoint 不一致")
        if summary["keypair"]["b_sha256"] != hashlib.sha256(b.tobytes()).hexdigest():
            raise ValueError("summary b_sha256 与 checkpoint 不一致")
        if summary["split_hash"] != metadata.get("split", {}).get("split_hash"):
            raise ValueError("summary split_hash 与 checkpoint 不一致")
    model = GatedResNet18(A, b, params).to(device)
    model.load_state_dict(payload["model"])
    if metadata.get("mapping_version") != DATA_MAPPING_VERSION:
        raise ValueError("mapping_version 与当前数据映射不一致")
    dataset = CIFAR10WithCoarse(
        args.data_root,
        train=False,
        transform=get_cifar_transforms(False),
        download=False,
    )
    if len(dataset) != 10000:
        raise ValueError("CIFAR-10 test split 必须包含 10000 个样本")
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须大于 0")
    tail = len(dataset) % args.batch_size or args.batch_size
    tail_valid = int(round(tail * args.mixed_ratio))
    if not 0.0 < args.mixed_ratio < 1.0 or tail_valid < 2 or tail - tail_valid < 1:
        raise ValueError(
            "mixed-ratio 在尾批下必须保证至少两个 valid 和一个 invalid 样本"
        )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    generator = CredentialGenerator(A, secret, b, params, seed=seed + 500)
    result = TestSplitEvaluator(
        model, generator, params, device, args.mixed_ratio
    ).evaluate(loader)
    if args.measure_latency:
        latency_parts = []
        count = 0
        for images, _, _ in loader:
            latency_parts.append(images)
            count += images.shape[0]
            if count >= 256:
                break
        latency_images = torch.cat(latency_parts, dim=0)[:256]
        result["latency"] = {
            "measured": True,
            "scope": "forward_and_routing",
            "batch_size": 256,
            **TestSplitEvaluator(
                model, generator, params, device, args.mixed_ratio
            ).measure_latency(latency_images),
        }
    else:
        result["latency"] = {"measured": False, "batch_size": None}
    split = metadata.get("split", {})
    result.update(
        {
            "schema_version": 1,
            "seed": seed,
            "mapping_version": DATA_MAPPING_VERSION,
            "device": str(device),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": actual_hash,
                "integrity_check": integrity,
                "stage": str(payload.get("stage", "")).upper(),
                "epoch": int(payload.get("epoch", -1)),
            },
            "keypair": {
                "A_sha256": hashlib.sha256(A.tobytes()).hexdigest(),
                "b_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                "rng_scheme": "numpy.default_rng(seed + 100)",
                "rng_version": 1,
                "rng_scheme_source": rng_source,
            },
            "credential_rng_seed": seed + 500,
            "credential_rng_note": "评估使用显式派生 seed + 500；不构成统计或密码学独立性证明",
            "training_credential_rng_seed": metadata.get(
                "training_credential_rng_seed"
            ),
            "lwe": metadata["lwe"],
            "lwe_provenance": (
                "complete"
                if {"q", "secret_bound"}.issubset(metadata["lwe"])
                else "partial_legacy_defaults"
            ),
            "dataset": {
                "dataset_name": "CIFAR10",
                "split": "test",
                "train_flag": False,
                "size": len(dataset),
            },
            "leakage_check": {
                "level": "split_identity_and_metadata_only",
                "split_identity_verified": True,
                "train_size": len(split.get("train_indices", []))
                or split.get("train_size"),
                "val_size": len(split.get("validation_indices", []))
                or split.get("val_size"),
                "split_hash": split.get("split_hash") if summary is not None else None,
                "split_hash_matches_summary": True if summary is not None else None,
            },
            "eval_batch_size": args.batch_size,
            "provenance_check": provenance,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
