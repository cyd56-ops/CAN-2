"""执行 Phase 5.5/T2 dev/validation/test 受管评估。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.can.v2.crypto.lwe import LWEParams, V_ref
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    ByteTokenizer,
    GatedDecoderTransformer,
    PlainDecoderTransformer,
    T2Evaluator,
    TransformerConfig,
    atomic_write_json,
    begin_test_access,
    derive_lwe_keypair,
    file_sha256,
    finish_test_access,
    generate_t2_split,
    load_frozen_runtime,
    load_strict_json,
    load_t2_checkpoint,
    runtime_config_from_mapping,
    verify_checkpoint_integrity,
)


def _args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析 T2 evaluator 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--manifest-key")
    parser.add_argument("--split", choices=("dev", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--freeze-record", type=Path)
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--validation-summary", type=Path)
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--cache-mode", choices=("none", "kv"), default="none")
    return parser.parse_args(argv)


def _load_secret(
    path: Path, expected_a: np.ndarray, expected_b: np.ndarray, params: LWEParams
) -> CredentialGenerator:
    """从受限 npz 文件读取 secret，并验证其与 checkpoint 公共参数匹配。"""

    if not path.is_file() or path.suffix.lower() != ".npz":
        raise ValueError("--credential-file 必须是外部 .npz 文件")
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"secret"}:
                raise ValueError("credential npz 只能包含 secret 字段")
            secret = np.asarray(payload["secret"])
    except (OSError, ValueError) as exc:
        raise ValueError("credential 文件无法安全读取") from exc
    if secret.shape != (params.n,) or not np.issubdtype(secret.dtype, np.floating):
        raise ValueError("credential secret shape/dtype 非法")
    if (
        not np.isfinite(secret).all()
        or V_ref({"vector": secret}, expected_a, expected_b, params) != 1
    ):
        raise ValueError("credential secret 不是 checkpoint 对应的有效 credential")
    return CredentialGenerator(
        expected_a, secret.astype(np.float32), expected_b, params, seed=0
    )


def _device(name: str) -> torch.device:
    """解析 evaluator 设备。"""

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但 CUDA 不可用")
    return torch.device(name)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """读取 checkpoint metadata，构造授权 split 并写出评估结果。"""

    args = _args(argv)
    if args.split == "test" and not args.confirm_test:
        raise ValueError("--split test 必须同时提供 --confirm-test")
    if args.split != "test" and args.confirm_test:
        raise ValueError("--confirm-test 只允许用于 --split test")
    if args.output.exists():
        raise FileExistsError("评估输出已存在，默认拒绝覆盖")
    if args.split == "test" and args.checkpoint_manifest is None:
        raise ValueError(
            "test 禁止 partial integrity，必须提供可信 checkpoint manifest"
        )
    if args.checkpoint_manifest is not None:
        if not args.expected_manifest_sha256 or not args.manifest_key:
            raise ValueError("manifest 模式必须同时提供 expected SHA 和 manifest-key")
        if file_sha256(args.checkpoint_manifest) != args.expected_manifest_sha256:
            raise ValueError("checkpoint manifest SHA-256 不匹配")
        manifest = load_strict_json(args.checkpoint_manifest)
    else:
        if args.expected_manifest_sha256 or args.manifest_key:
            raise ValueError("未提供 manifest 时不得提供其校验参数")
        manifest = None
    checkpoint_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint_payload, dict) or not isinstance(
        checkpoint_payload.get("metadata"), dict
    ):
        raise ValueError("checkpoint 缺少 T2 metadata")
    metadata = checkpoint_payload["metadata"]
    model_config = TransformerConfig(**checkpoint_payload["model_config"])
    kind = checkpoint_payload.get("model_kind")
    device = _device(args.device)
    credential_generator = None
    if kind == "plain":
        if args.credential_file is not None:
            raise ValueError("Plain evaluator 禁止 credential-file")
        model = PlainDecoderTransformer(model_config).to(device)
    elif kind == "can":
        public = checkpoint_payload.get("lwe_public")
        if not isinstance(public, dict) or args.credential_file is None:
            raise ValueError(
                "CAN evaluator 必须提供 checkpoint 公共参数和 credential-file"
            )
        params = LWEParams(**public["params"])
        matrix = np.asarray(public["A"])
        vector = np.asarray(public["b"])
        model = GatedDecoderTransformer(matrix, vector, params, model_config).to(device)
        credential_generator = _load_secret(
            args.credential_file, matrix, vector, params
        )
    else:
        raise ValueError("checkpoint model_kind 非法")
    expected_metadata = dict(metadata)
    load_t2_checkpoint(
        args.checkpoint, model, None, expected_metadata, restore_rng=False
    )
    if manifest is not None:
        verify_checkpoint_integrity(
            args.checkpoint, manifest=manifest, manifest_key=args.manifest_key
        )
    runtime_payload = {
        "schema_version": 1,
        "mode": metadata["mode"],
        "suite_id": metadata["suite_id"],
        "prompt_group": metadata["prompt_group"],
        "seed": metadata["seed"],
        "models": "both",
        "split_counts": metadata["split_counts"],
        "batch_size": args.batch_size,
        "token_budget": 1,
        "validation_interval_tokens": 1,
        "learning_rate": 1e-3,
        "max_new_tokens": args.max_new_tokens,
        "cache_mode": args.cache_mode,
        "model_config": checkpoint_payload["model_config"],
        "lwe_n": (
            int(checkpoint_payload.get("lwe_public", {}).get("params", {}).get("n", 32))
            if kind == "can"
            else 32
        ),
        "lwe_m": (
            int(checkpoint_payload.get("lwe_public", {}).get("params", {}).get("m", 64))
            if kind == "can"
            else 64
        ),
        "lwe_sigma": (
            float(
                checkpoint_payload.get("lwe_public", {})
                .get("params", {})
                .get("sigma", 1.0)
            )
            if kind == "can"
            else 1.0
        ),
        "generator_version": metadata["generator_version"],
        "normalization_version": metadata["normalization_version"],
        "tokenizer_version": metadata["tokenizer_version"],
    }
    # runtime_config_from_mapping 仅用于身份校验；评估 batch/budget 不会修改 checkpoint 身份。
    runtime_config_from_mapping(runtime_payload)
    if args.split == "validation":
        if args.freeze_record is None or args.expected_freeze_sha256 is None:
            raise ValueError("validation 必须提供 freeze record 与外部可信摘要")
        frozen = load_frozen_runtime(args.freeze_record, args.expected_freeze_sha256)
        if (
            frozen.suite_id != metadata["suite_id"]
            or frozen.prompt_group != metadata["prompt_group"]
            or frozen.seed != metadata["seed"]
        ):
            raise ValueError("validation freeze 与 checkpoint identity 不一致")
        if (args.batch_size, args.max_new_tokens, args.cache_mode) != (
            frozen.batch_size,
            frozen.max_new_tokens,
            frozen.cache_mode,
        ):
            raise ValueError("validation evaluator 参数与 freeze 不一致")
    elif args.split == "test":
        if (
            args.freeze_record is None
            or args.expected_freeze_sha256 is None
            or args.validation_summary is None
        ):
            raise ValueError("test 必须提供 freeze、可信摘要和 validation-summary")
        frozen = load_frozen_runtime(args.freeze_record, args.expected_freeze_sha256)
        validation = load_strict_json(args.validation_summary)
        if (
            validation.get("status") not in {"completed", "ok"}
            or validation.get("research_result") is not True
        ):
            raise ValueError("validation summary 未通过正式状态检查")
        if (args.batch_size, args.max_new_tokens, args.cache_mode) != (
            frozen.batch_size,
            frozen.max_new_tokens,
            frozen.cache_mode,
        ):
            raise ValueError("test evaluator 参数与 freeze 不一致")
    identity = {
        "suite_id": metadata["suite_id"],
        "prompt_group": metadata["prompt_group"],
        "seed": metadata["seed"],
        "checkpoint": file_sha256(args.checkpoint),
    }
    ledger = None
    if args.split == "test":
        ledger = begin_test_access(args.run_directory, identity)
    try:
        counts = metadata["split_counts"]
        rows = generate_t2_split(
            metadata["suite_id"],
            args.split,
            int(metadata["seed"]),
            counts,
            prompt_group=metadata["prompt_group"],
        )
        evaluator = T2Evaluator(
            model,
            ByteTokenizer(),
            device,
            credential_generator,
            max_new_tokens=args.max_new_tokens,
            cache_mode=args.cache_mode,
            batch_size=args.batch_size,
        )
        result = evaluator.evaluate(rows)
        result["provenance"] = {
            "checkpoint_sha256": identity["checkpoint"],
            "checkpoint_manifest_sha256": (
                file_sha256(args.checkpoint_manifest)
                if args.checkpoint_manifest
                else None
            ),
            "freeze_record_sha256": (
                args.expected_freeze_sha256 if args.split != "dev" else None
            ),
            "integrity": "verified" if manifest is not None else "not_performed",
            "partial": args.split == "dev" and manifest is None,
        }
        atomic_write_json(args.output, result)
        if ledger is not None:
            finish_test_access(ledger, identity, "completed")
    except Exception as exc:
        if ledger is not None:
            finish_test_access(ledger, identity, "failed", str(exc))
        raise
    print(
        json.dumps(
            {"status": "ok", "split": args.split, "output": str(args.output)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
