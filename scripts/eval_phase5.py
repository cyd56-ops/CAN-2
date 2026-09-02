"""Phase 5 T1 evaluator CLI：先执行完整性校验并生成可审计结果骨架。"""

import argparse
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.can.v2.crypto.lwe import LWEParams
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    ByteTokenizer,
    GatedDecoderTransformer,
    Phase5Evaluator,
    SyntheticKnowledgeDataset,
    TransformerConfig,
    freeze_record_sha256,
    generate_synthetic_corpus,
    load_freeze_record,
    validate_runtime_against_freeze,
)
from src.can.v2.transformer.manifest import (
    sha256_file,
    verify_checkpoint_integrity,
)


def _digest_bytes(path: Path) -> str:
    """计算 manifest 文件摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """校验 checkpoint/manifest 并写出 partial provenance 结果。"""
    parser = argparse.ArgumentParser(description="Phase 5 T1 evaluator")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--credential-file", type=Path, help="受限 valid credential 文件"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--manifest-key")
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--freeze-record", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cache-mode", choices=("none", "kv"))
    args = parser.parse_args()
    freeze = None
    freeze_sha = None
    if args.freeze_record:
        try:
            freeze = load_freeze_record(args.freeze_record)
            freeze_sha = freeze_record_sha256(args.freeze_record)
            validate_runtime_against_freeze(
                freeze,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                cache_mode=args.cache_mode,
                generator_version="phase5-t1-private-query-v2",
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if args.split == "test" and not args.confirm_test:
        parser.error("--split test 必须同时提供 --confirm-test")
    if args.split != "test" and args.confirm_test:
        parser.error("--confirm-test 仅允许与 --split test 搭配")
    if args.summary is None and args.split == "test":
        parser.error("正式 test 评估必须提供 --summary")
    if args.expected_checkpoint_sha256 and args.checkpoint_manifest:
        parser.error("--expected-checkpoint-sha256 与 --checkpoint-manifest 互斥")
    if args.checkpoint_manifest and not args.expected_manifest_sha256:
        parser.error("manifest 模式必须提供 --expected-manifest-sha256")
    if args.output.exists() and not args.force_overwrite:
        parser.error(f"输出已存在，拒绝覆盖: {args.output}")
    manifest: Optional[Dict[str, Any]] = None
    manifest_sha = None
    if args.checkpoint_manifest:
        manifest_sha = _digest_bytes(args.checkpoint_manifest)
        if manifest_sha != args.expected_manifest_sha256:
            parser.error("manifest SHA-256 不匹配")
        manifest = json.loads(args.checkpoint_manifest.read_text(encoding="utf-8"))
    integrity = "not_performed"
    actual = sha256_file(args.checkpoint)
    try:
        verify_checkpoint_integrity(
            args.checkpoint,
            expected_sha256=args.expected_checkpoint_sha256,
            manifest=manifest,
            manifest_key=args.manifest_key,
        )
        integrity = (
            "verified"
            if (args.expected_checkpoint_sha256 or manifest)
            else "not_performed"
        )
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    # secret 只允许由受限文件显式提供，绝不从 checkpoint 推断。
    if args.credential_file is None:
        result_status = "incomplete"
        note = "缺少 --credential-file；secret 不写入 checkpoint，未执行模型指标评估。"
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        required = {"A", "b", "lwe_params", "config", "model"}
        if not required.issubset(payload):
            parser.error("checkpoint 缺少模型重建 metadata")
        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("请求 CUDA 但当前不可用")
        p = LWEParams(**payload["lwe_params"])
        model = GatedDecoderTransformer(
            np.asarray(payload["A"]),
            np.asarray(payload["b"]),
            p,
            TransformerConfig(**payload["config"]),
        )
        model.load_state_dict(payload["model"])
        secret = np.load(args.credential_file)
        generator = CredentialGenerator(
            np.asarray(payload["A"]),
            secret,
            np.asarray(payload["b"]),
            p,
            seed=args.seed + 500,
        )
        corpus = generate_synthetic_corpus(args.seed, 4, 2, 2)
        dataset = SyntheticKnowledgeDataset(
            corpus[args.split], ByteTokenizer(), max_length=model.config.max_seq_len
        )
        metrics = Phase5Evaluator(
            model,
            ByteTokenizer(),
            device,
            generator,
            max_new_tokens=args.max_new_tokens,
        ).evaluate(dataset)
        result_status = "ok"
        note = "evaluator 已从 checkpoint metadata 重建模型并完成指标计算。"
    result = {
        "schema_version": 1,
        "status": result_status,
        "integrity_check": integrity,
        "checkpoint": {"path": str(args.checkpoint), "sha256": actual},
        "manifest_sha256": manifest_sha,
        "freeze_record": str(args.freeze_record) if args.freeze_record else None,
        "freeze_record_sha256": freeze_sha,
        "summary_path": str(args.summary) if args.summary else None,
        "confirm_test": bool(args.confirm_test),
        "capability": asdict(metrics["protected"]) if args.credential_file else None,
        "public": asdict(metrics["public"]) if args.credential_file else None,
        "refusal": asdict(metrics["refusal"]) if args.credential_file else None,
        "teacher_comparison": (
            asdict(metrics["teacher_comparison"]) if args.credential_file else None
        ),
        "probe": None,
        "recovery": None,
        "note": note,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
