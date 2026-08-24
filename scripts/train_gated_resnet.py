"""Phase 2 CIFAR-10 三阶段训练入口。

默认不下载数据；使用 ``--download-data`` 才允许 torchvision 下载 CIFAR-10。
``--smoke-test`` 提供完全离线的小型 synthetic 数据路径，用于验证训练架构。
"""

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.can.v2.crypto.lwe import LWEParams, generate_keypair  # noqa: E402
from src.can.v2.models import GatedResNet18  # noqa: E402
from src.can.v2.training.data import (  # noqa: E402
    DATA_MAPPING_VERSION,
    CIFAR10WithCoarse,
    CredentialGenerator,
    fine_to_coarse,
    get_cifar_transforms,
    make_worker_init_fn,
    split_indices,
)
from src.can.v2.training.trainer import (  # noqa: E402
    GatedResNetTrainer,
    checkpoint_sha256,
)

_ALLOWED_TOP_LEVEL = {
    "schema_version",
    "seed",
    "device",
    "lwe",
    "data",
    "stages",
    "optimizer",
    "kd",
    "training",
}
_ALLOWED_STAGE = {
    "epochs",
    "patience",
    "valid_ratio",
    "alpha",
    "beta_ce",
    "beta_kd",
    "min_delta",
    "max_protected_drop",
}
_ALLOWED_LWE = {"n", "m", "sigma", "error_threshold"}
_ALLOWED_DATA = {
    "root",
    "download",
    "validation_fraction",
    "split_seed",
    "num_workers",
}
_ALLOWED_OPTIMIZER = {"name", "lr", "joint_lr", "momentum", "weight_decay"}
_ALLOWED_KD = {"temperature"}
_ALLOWED_TRAINING = {"batch_size", "max_grad_norm", "checkpoint_dir"}


class _UniqueLoader(yaml.SafeLoader):
    """拒绝重复 YAML key 的安全 loader。"""


def _construct_mapping(
    loader: yaml.Loader, node: yaml.Node, deep: bool = False
) -> dict:
    """构造 mapping，并拒绝重复字段。"""

    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"配置包含重复字段: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _check_keys(section: Any, allowed: set, name: str) -> None:
    """拒绝非 mapping 或未知字段。"""

    if not isinstance(section, dict):
        raise TypeError(f"{name} 必须是 mapping")
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"{name} 包含未知字段: {sorted(unknown)}")


def _positive_int(value: Any, name: str) -> int:
    """校验正整数配置。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _finite_float(value: Any, name: str, minimum: float = 0.0) -> float:
    """校验有限浮点配置。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是有限实数")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的有限实数")
    return result


def load_config(path: Path) -> dict:
    """加载并严格校验 Phase 2 YAML 配置。"""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=_UniqueLoader)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("配置 schema_version 必须为 1")
    _check_keys(config, _ALLOWED_TOP_LEVEL, "配置")
    _positive_int(config.get("seed"), "seed")
    if config.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 必须是 auto、cpu 或 cuda")

    _check_keys(config.get("lwe"), _ALLOWED_LWE, "lwe")
    _positive_int(config["lwe"].get("n"), "lwe.n")
    _positive_int(config["lwe"].get("m"), "lwe.m")
    _finite_float(config["lwe"].get("sigma"), "lwe.sigma", minimum=0.0)
    _finite_float(
        config["lwe"].get("error_threshold"), "lwe.error_threshold", minimum=0.0
    )

    _check_keys(config.get("data"), _ALLOWED_DATA, "data")
    if not isinstance(config["data"].get("root"), str):
        raise TypeError("data.root 必须是字符串")
    if not isinstance(config["data"].get("download"), bool):
        raise TypeError("data.download 必须是 bool")
    fraction = _finite_float(
        config["data"].get("validation_fraction"), "data.validation_fraction"
    )
    if not 0.0 < fraction < 1.0:
        raise ValueError("data.validation_fraction 必须位于 (0, 1)")
    _positive_int(config["data"].get("split_seed"), "data.split_seed")
    if (
        isinstance(config["data"].get("num_workers"), bool)
        or not isinstance(config["data"].get("num_workers"), int)
        or config["data"].get("num_workers") < 0
    ):
        raise ValueError("data.num_workers 必须是非负整数")

    for stage in ("A", "B", "C"):
        section = config.get("stages", {}).get(stage)
        _check_keys(section, _ALLOWED_STAGE, f"stages.{stage}")
        _positive_int(section.get("epochs"), f"stages.{stage}.epochs")
        if (
            isinstance(section.get("patience"), bool)
            or not isinstance(section.get("patience"), int)
            or section.get("patience") < 0
        ):
            raise ValueError(f"stages.{stage}.patience 必须是非负整数")
        ratio = _finite_float(section.get("valid_ratio"), f"stages.{stage}.valid_ratio")
        if ratio > 1.0:
            raise ValueError(f"stages.{stage}.valid_ratio 必须位于 [0, 1]")
        for name in ("alpha", "beta_ce", "beta_kd", "min_delta"):
            _finite_float(section.get(name, 0.0), f"stages.{stage}.{name}")
        if "max_protected_drop" in section:
            if stage != "C":
                raise ValueError("max_protected_drop 只允许配置在 Stage C")
            _finite_float(section["max_protected_drop"], "stages.C.max_protected_drop")
    _check_keys(config.get("optimizer"), _ALLOWED_OPTIMIZER, "optimizer")
    if config["optimizer"].get("name") != "sgd":
        raise ValueError("当前实现仅支持 optimizer.name=sgd")
    for name in ("lr", "joint_lr", "momentum", "weight_decay"):
        _finite_float(config["optimizer"].get(name), f"optimizer.{name}")
    _check_keys(config.get("kd"), _ALLOWED_KD, "kd")
    _finite_float(
        config["kd"].get("temperature"), "kd.temperature", minimum=np.finfo(float).eps
    )
    _check_keys(config.get("training"), _ALLOWED_TRAINING, "training")
    _positive_int(config["training"].get("batch_size"), "training.batch_size")
    _finite_float(
        config["training"].get("max_grad_norm"),
        "training.max_grad_norm",
        minimum=np.finfo(float).eps,
    )
    if not isinstance(config["training"].get("checkpoint_dir"), str):
        raise TypeError("training.checkpoint_dir 必须是字符串")
    return config


def _select_device(requested: str) -> torch.device:
    """解析设备配置，显式 CUDA 不可用时 fail fast。"""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置请求 CUDA，但当前 PyTorch 没有可用 CUDA")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的确定性种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synthetic_dataset(size: int, seed: int) -> TensorDataset:
    """生成离线 smoke test 使用的确定性 CIFAR-like 数据集。"""

    if size < 4:
        raise ValueError("synthetic size 至少为 4")
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(size, 3, 32, 32, generator=generator)
    fine = torch.arange(size, dtype=torch.long) % 10
    coarse = torch.tensor(
        [fine_to_coarse(int(value)) for value in fine], dtype=torch.long
    )
    return TensorDataset(images, fine, coarse)


def _indices_hash(train_indices: Iterable[int], val_indices: Iterable[int]) -> str:
    """计算 train/validation split 的稳定 SHA-256。"""

    payload = json.dumps(
        {"train": list(train_indices), "validation": list(val_indices)},
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_signature(config: dict) -> str:
    """计算允许调整 epoch/patience 后仍稳定的恢复配置签名。"""

    normalized = copy.deepcopy(config)
    for stage in normalized["stages"].values():
        stage.pop("epochs", None)
        stage.pop("patience", None)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _load_datasets(
    config: dict, smoke_test: bool, smoke_size: int
) -> Tuple[Iterable, Iterable, Dict[str, Any]]:
    """加载 synthetic 或真实 CIFAR-10 的 train/validation 数据集。"""

    if smoke_test:
        dataset = _synthetic_dataset(smoke_size, int(config["seed"]))
        train_count = max(2, int(len(dataset) * 0.75))
        train_indices = list(range(train_count))
        val_indices = list(range(train_count, len(dataset)))
        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            {
                "train_indices": train_indices,
                "validation_indices": val_indices,
                "split_hash": _indices_hash(train_indices, val_indices),
            },
        )
    transform_train = get_cifar_transforms(True)
    transform_eval = get_cifar_transforms(False)
    root = str(config["data"]["root"])
    train_dataset = CIFAR10WithCoarse(
        root, train=True, transform=transform_train, download=config["data"]["download"]
    )
    eval_dataset = CIFAR10WithCoarse(
        root, train=True, transform=transform_eval, download=False
    )
    train_indices, val_indices = split_indices(
        len(train_dataset),
        config["data"]["validation_fraction"],
        config["data"]["split_seed"],
    )
    return (
        Subset(train_dataset, train_indices),
        Subset(eval_dataset, val_indices),
        {
            "train_indices": list(train_indices),
            "validation_indices": list(val_indices),
            "split_hash": _indices_hash(train_indices, val_indices),
        },
    )


def _make_loaders(
    config: dict, train_dataset: Iterable, val_dataset: Iterable, seed: int
) -> Tuple[DataLoader, DataLoader]:
    """创建带显式 generator 和 worker seed 的 DataLoader。"""

    batch_size = int(config["training"]["batch_size"])
    workers = int(config["data"]["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": make_worker_init_fn(seed),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, generator=generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    if len(train_loader) == 0:
        raise ValueError(
            "drop_last=True 后训练 DataLoader 没有 batch；"
            f"train_size={len(train_dataset)}, batch_size={batch_size}"
        )
    return train_loader, val_loader


def _validate_batch_contract(config: dict) -> None:
    """在加载数据前验证 Stage A/C 每个完整 batch 至少有两个 valid。"""

    batch_size = int(config["training"]["batch_size"])
    for stage in ("A", "C"):
        ratio = float(config["stages"][stage]["valid_ratio"])
        if ratio > 0.0 and int(round(batch_size * ratio)) < 2:
            raise ValueError(
                f"Stage {stage} 的 batch_size={batch_size}, valid_ratio={ratio} "
                "无法保证至少两个 valid 样本"
            )


def _new_model(A: np.ndarray, b: np.ndarray, params: LWEParams) -> GatedResNet18:
    """创建结构固定的 CIFAR Gated ResNet-18。"""

    return GatedResNet18(A, b, params)


def _optimizer(model: GatedResNet18, stage: str, config: dict) -> torch.optim.Optimizer:
    """按阶段选择 student 可训练参数并创建 SGD。"""

    if stage == "A":
        parameters = [
            p
            for name, p in model.named_parameters()
            if not name.startswith("public_fc.")
        ]
        lr = float(config["optimizer"]["lr"])
    elif stage == "B":
        parameters = list(model.public_fc.parameters())
        lr = float(config["optimizer"]["joint_lr"])
    else:
        parameters = list(model.parameters())
        lr = float(config["optimizer"]["joint_lr"])
    return torch.optim.SGD(
        parameters,
        lr=lr,
        momentum=float(config["optimizer"]["momentum"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )


def _run_training(config: dict, args: argparse.Namespace) -> Dict[str, Any]:
    """执行 Stage A/B/C 链式训练并返回各阶段摘要。"""

    _seed_everything(int(config["seed"]))
    _validate_batch_contract(config)
    device = _select_device(config["device"])
    train_dataset, val_dataset, split_metadata = _load_datasets(
        config, args.smoke_test, args.smoke_size
    )
    if args.smoke_test and len(train_dataset) < int(config["training"]["batch_size"]):
        raise ValueError(
            "smoke 训练集小于 batch_size；请增大 --smoke-size 或减小 --batch-size"
        )
    train_loader, val_loader = _make_loaders(
        config, train_dataset, val_dataset, int(config["seed"])
    )
    params = LWEParams(
        n=int(config["lwe"]["n"]),
        m=int(config["lwe"]["m"]),
        sigma=float(config["lwe"]["sigma"]),
        error_threshold=float(config["lwe"]["error_threshold"]),
    )
    # 使用独立显式 RNG 确定性重建 keypair；secret 不写入普通 checkpoint/实验目录。
    key_rng = np.random.default_rng(int(config["seed"]) + 100)
    A, secret, b = generate_keypair(params, rng=key_rng)
    credential_generator = CredentialGenerator(
        A, secret, b, params, seed=int(config["seed"]) + 1
    )
    root_checkpoint = Path(config["training"]["checkpoint_dir"])
    root_checkpoint.mkdir(parents=True, exist_ok=True)
    (root_checkpoint / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root_checkpoint / "split_indices.json").write_text(
        json.dumps(split_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary: Dict[str, Any] = {
        "device": str(device),
        "mapping_version": DATA_MAPPING_VERSION,
        "lwe": {
            "n": params.n,
            "m": params.m,
            "sigma": params.sigma,
            "error_threshold": params.error_threshold,
        },
        "split": split_metadata,
        "keypair": {
            "A_sha256": hashlib.sha256(A.tobytes()).hexdigest(),
            "b_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
        },
    }
    checkpoint_metadata = {
        "config": config,
        "config_signature": _config_signature(config),
        "mapping_version": DATA_MAPPING_VERSION,
        "lwe": summary["lwe"],
        "A": A,
        "b": b,
        "split": split_metadata,
    }
    resume_payload = None
    resume_stage = None
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint 不存在: {resume_path}")
        resume_payload = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        resume_stage = resume_payload.get("stage")
        if resume_stage not in {"A", "B", "C"}:
            raise ValueError("resume checkpoint 的 stage 非法")
        if (
            resume_path.parent.parent != root_checkpoint
            and resume_path.parent != root_checkpoint
        ):
            raise ValueError("resume checkpoint 必须位于当前 checkpoint_dir 下")

    model_a = _new_model(A, b, params).to(device)
    stage_a = config["stages"]["A"]
    trainer_a = GatedResNetTrainer(
        model_a,
        train_loader,
        val_loader,
        credential_generator,
        _optimizer(model_a, "A", config),
        device,
        stage="A",
        valid_ratio=float(stage_a["valid_ratio"]),
        alpha=float(stage_a["alpha"]),
        beta_ce=float(stage_a["beta_ce"]),
        beta_kd=float(stage_a["beta_kd"]),
        temperature=float(config["kd"]["temperature"]),
        max_grad_norm=float(config["training"]["max_grad_norm"]),
        progress=not args.no_progress,
        checkpoint_metadata=checkpoint_metadata,
    )
    if resume_stage in {"B", "C"}:
        trainer_a.load_checkpoint(root_checkpoint / "stage_a" / "best.ckpt")
        summary["stage_a"] = dict(trainer_a.best_metrics)
    else:
        if resume_stage == "A":
            trainer_a.load_checkpoint(Path(args.resume))
        summary["stage_a"] = trainer_a.fit(
            int(stage_a["epochs"]),
            int(stage_a["patience"]),
            float(stage_a.get("min_delta", 0.0)),
            root_checkpoint / "stage_a",
        )
    stage_a_best = root_checkpoint / "stage_a" / "best.ckpt"
    if not stage_a_best.exists():
        raise RuntimeError("Stage A 未生成 best checkpoint")
    teacher = copy.deepcopy(model_a).to(device)
    stage_a_payload = torch.load(stage_a_best, map_location=device, weights_only=False)
    teacher.load_state_dict(stage_a_payload["model"])
    teacher.eval()
    teacher_identity = {
        "path": str(stage_a_best.resolve()),
        "sha256": checkpoint_sha256(stage_a_best),
    }

    model_b = _new_model(A, b, params).to(device)
    model_b.load_state_dict(stage_a_payload["model"])
    stage_b = config["stages"]["B"]
    trainer_b = GatedResNetTrainer(
        model_b,
        train_loader,
        val_loader,
        credential_generator,
        _optimizer(model_b, "B", config),
        device,
        stage="B",
        teacher=teacher,
        teacher_identity=teacher_identity,
        valid_ratio=float(stage_b["valid_ratio"]),
        alpha=float(stage_b["alpha"]),
        beta_ce=float(stage_b["beta_ce"]),
        beta_kd=float(stage_b["beta_kd"]),
        temperature=float(config["kd"]["temperature"]),
        max_grad_norm=float(config["training"]["max_grad_norm"]),
        progress=not args.no_progress,
        checkpoint_metadata=checkpoint_metadata,
    )
    if resume_stage == "C":
        trainer_b.load_checkpoint(root_checkpoint / "stage_b" / "best.ckpt")
        summary["stage_b"] = dict(trainer_b.best_metrics)
    else:
        if resume_stage == "B":
            trainer_b.load_checkpoint(Path(args.resume))
        summary["stage_b"] = trainer_b.fit(
            int(stage_b["epochs"]),
            int(stage_b["patience"]),
            float(stage_b.get("min_delta", 0.0)),
            root_checkpoint / "stage_b",
        )
    stage_b_best = root_checkpoint / "stage_b" / "best.ckpt"
    if not stage_b_best.exists():
        raise RuntimeError("Stage B 未生成 best checkpoint")
    model_c = _new_model(A, b, params).to(device)
    stage_b_payload = torch.load(stage_b_best, map_location=device, weights_only=False)
    model_c.load_state_dict(stage_b_payload["model"])
    stage_c = config["stages"]["C"]
    trainer_c = GatedResNetTrainer(
        model_c,
        train_loader,
        val_loader,
        credential_generator,
        _optimizer(model_c, "C", config),
        device,
        stage="C",
        teacher=teacher,
        teacher_identity=teacher_identity,
        protected_baseline=trainer_a.best_metric,
        max_protected_drop=float(stage_c.get("max_protected_drop", 0.03)),
        valid_ratio=float(stage_c["valid_ratio"]),
        alpha=float(stage_c["alpha"]),
        beta_ce=float(stage_c["beta_ce"]),
        beta_kd=float(stage_c["beta_kd"]),
        temperature=float(config["kd"]["temperature"]),
        max_grad_norm=float(config["training"]["max_grad_norm"]),
        progress=not args.no_progress,
        checkpoint_metadata=checkpoint_metadata,
    )
    if resume_stage == "C":
        trainer_c.load_checkpoint(Path(args.resume))
    summary["stage_c"] = trainer_c.fit(
        int(stage_c["epochs"]),
        int(stage_c["patience"]),
        float(stage_c.get("min_delta", 0.0)),
        root_checkpoint / "stage_c",
    )
    return summary


def main(argv: Any = None) -> None:
    """解析配置、执行可选 dry-run 或完整三阶段训练。"""

    parser = argparse.ArgumentParser(description="CAN Phase 2 Gated ResNet-18 trainer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/v2/train_gated_resnet18_cifar10.yaml"),
    )
    parser.add_argument("--stage-a-epochs", type=int)
    parser.add_argument("--stage-b-epochs", type=int)
    parser.add_argument("--stage-c-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="只解析并输出配置，不加载数据"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="使用离线 synthetic CIFAR-like 数据运行三阶段 smoke test",
    )
    parser.add_argument("--smoke-size", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--resume", type=Path, help="从当前 checkpoint_dir 下的 last.ckpt 恢复"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.download_data:
        config["data"]["download"] = True
    if args.device is not None:
        config["device"] = args.device
    if args.batch_size is not None:
        config["training"]["batch_size"] = _positive_int(args.batch_size, "batch-size")
    for stage, epochs in (
        ("A", args.stage_a_epochs),
        ("B", args.stage_b_epochs),
        ("C", args.stage_c_epochs),
    ):
        if epochs is not None:
            config["stages"][stage]["epochs"] = _positive_int(
                epochs, f"stage-{stage.lower()}-epochs"
            )
    if args.dry_run:
        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        return
    summary = _run_training(config, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
