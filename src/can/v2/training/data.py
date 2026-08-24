"""Phase 2 的 CIFAR-10 数据、标签映射和可复现 credential 生成。"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..crypto.lwe import LWEParams, V_ref

CIFAR2_MAPPING = {"vehicle": (0, 1, 8, 9), "animal": (2, 3, 4, 5, 6, 7)}
DATA_MAPPING_VERSION = "cifar10-animal-vehicle-v1"


def fine_to_coarse(fine_label: int) -> int:
    """将 CIFAR-10 的 0-9 标签映射为 vehicle=0、animal=1。"""

    if isinstance(fine_label, bool) or not isinstance(fine_label, int):
        raise TypeError("fine_label 必须是非 bool 整数")
    if not 0 <= fine_label < 10:
        raise ValueError("fine_label 必须位于 [0, 10)")
    return int(fine_label in CIFAR2_MAPPING["animal"])


class CIFAR10WithCoarse(Dataset):
    """为 CIFAR-10 样本同时提供图像、细粒度和粗粒度标签。"""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        base_dataset: Optional[Dataset] = None,
    ) -> None:
        """初始化数据集；测试可注入 base_dataset，避免网络下载。"""

        if not isinstance(root, str):
            raise TypeError("root 必须是字符串路径")
        if not isinstance(train, bool) or not isinstance(download, bool):
            raise TypeError("train 和 download 必须是 bool")
        if base_dataset is not None:
            self.cifar10 = base_dataset
        else:
            try:
                from torchvision import datasets
            except Exception as exc:
                raise RuntimeError(
                    "需要安装与 PyTorch 匹配的 torchvision，或注入 base_dataset"
                ) from exc
            self.cifar10 = datasets.CIFAR10(
                root=root, train=train, transform=transform, download=download
            )

    def __len__(self) -> int:
        """返回样本数量。"""

        return len(self.cifar10)

    def __getitem__(self, index: int) -> Tuple[Tensor, int, int]:
        """返回图像、CIFAR-10 标签和 CIFAR-2 标签。"""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index 必须是整数")
        image, fine_label = self.cifar10[index]
        if not isinstance(image, Tensor):
            raise TypeError("数据集 transform 必须返回 torch.Tensor 图像")
        fine = int(fine_label)
        return image, fine, fine_to_coarse(fine)


@dataclass(frozen=True)
class CredentialBatch:
    """保存 credential 值和仅用于审计采样器的预期有效 mask。"""

    values: np.ndarray
    expected_valid: np.ndarray

    def __post_init__(self) -> None:
        """校验 batch 的二维浮点 credential 与一维 bool 审计 mask。"""

        if not isinstance(self.values, np.ndarray) or self.values.ndim != 2:
            raise TypeError("CredentialBatch.values 必须是二维 ndarray")
        if self.values.dtype != np.float32:
            raise TypeError("CredentialBatch.values 必须使用 float32 dtype")
        if not np.isfinite(self.values).all():
            raise ValueError("CredentialBatch.values 必须全部有限")
        if not isinstance(
            self.expected_valid, np.ndarray
        ) or self.expected_valid.shape != (self.values.shape[0],):
            raise ValueError("CredentialBatch.expected_valid shape 与 values 不一致")
        if self.expected_valid.dtype != np.bool_:
            raise TypeError("CredentialBatch.expected_valid 必须是 bool ndarray")


class CredentialGenerator:
    """使用 V_ref rejection sampling 生成可验证的 valid/invalid credential。"""

    def __init__(
        self,
        A: np.ndarray,
        secret: np.ndarray,
        b: np.ndarray,
        params: LWEParams,
        seed: int = 0,
        max_attempts: int = 100,
    ) -> None:
        """保存 toy LWE 公共参数和确定性 NumPy Generator。"""

        if not isinstance(A, np.ndarray) or not isinstance(secret, np.ndarray):
            raise TypeError("A 和 secret 必须是 ndarray")
        if not isinstance(b, np.ndarray) or not isinstance(params, LWEParams):
            raise TypeError("b 和 params 类型非法")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise TypeError("max_attempts 必须是整数")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("seed 必须是整数")
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须大于 0")
        if (
            A.shape != (params.m, params.n)
            or b.shape != (params.m,)
            or secret.shape != (params.n,)
        ):
            raise ValueError("LWE A、b、secret 的 shape 与 params 不一致")
        if not all(np.issubdtype(value.dtype, np.floating) for value in (A, secret, b)):
            raise TypeError("LWE A、b、secret 必须使用浮点 dtype")
        if not all(np.isfinite(value).all() for value in (A, secret, b)):
            raise ValueError("LWE A、b、secret 必须全部有限")
        if V_ref({"vector": secret}, A, b, params) != 1:
            raise ValueError("secret 不是 V_ref 可接受的 valid credential")
        self.A = A.astype(np.float32, copy=True)
        self.secret = secret.astype(np.float32, copy=True)
        self.b = b.astype(np.float32, copy=True)
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.max_attempts = int(max_attempts)

    def generate(self, is_valid: bool) -> np.ndarray:
        """生成单个并由 V_ref 验证的 credential。"""

        if not isinstance(is_valid, (bool, np.bool_)):
            raise TypeError("is_valid 必须是 bool")
        if is_valid:
            return self.secret.copy()
        for _ in range(self.max_attempts):
            candidate = self.rng.normal(0.0, 1.0, self.params.n).astype(np.float32)
            if V_ref({"vector": candidate}, self.A, self.b, self.params) == 0:
                return candidate
        raise RuntimeError("无法在 max_attempts 内生成 invalid credential")

    def batch_generate(
        self, batch_size: int, valid_ratio: float, min_valid: int = 0
    ) -> CredentialBatch:
        """按比例生成 batch，并可强制每批至少包含指定数量 valid credential。"""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size 必须是整数")
        if isinstance(valid_ratio, (bool, np.bool_)) or not isinstance(
            valid_ratio, (int, float, np.integer, np.floating)
        ):
            raise TypeError("valid_ratio 必须是有限实数")
        ratio = float(valid_ratio)
        if batch_size <= 0 or not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("batch_size 或 valid_ratio 非法")
        if (
            isinstance(min_valid, bool)
            or not isinstance(min_valid, int)
            or min_valid < 0
        ):
            raise ValueError("min_valid 必须是非负整数")
        num_valid = int(round(batch_size * ratio))
        if min_valid > batch_size:
            raise ValueError("min_valid 不能大于 batch_size")
        if ratio > 0.0 and num_valid < min_valid:
            raise ValueError(
                f"valid_ratio={ratio} 在 batch_size={batch_size} 下只能生成 {num_valid} 个 valid，"
                f"小于要求的最小值 {min_valid}"
            )
        values = [self.generate(True) for _ in range(num_valid)]
        values.extend(self.generate(False) for _ in range(batch_size - num_valid))
        permutation = self.rng.permutation(batch_size)
        expected = np.array(
            [True] * num_valid + [False] * (batch_size - num_valid), dtype=bool
        )
        values_array = np.stack(values)[permutation]
        expected = expected[permutation]
        actual = np.array(
            [
                V_ref({"vector": row}, self.A, self.b, self.params) == 1
                for row in values_array
            ],
            dtype=bool,
        )
        if not np.array_equal(actual, expected):
            raise RuntimeError("credential 采样结果未通过 V_ref 一致性检查")
        return CredentialBatch(values_array, expected)

    def all_valid(self, batch_size: int) -> CredentialBatch:
        """生成全 valid 验证 batch。"""

        self._validate_batch_size(batch_size)
        return CredentialBatch(
            np.repeat(self.secret[None, :], batch_size, axis=0),
            np.ones(batch_size, dtype=bool),
        )

    def all_invalid(self, batch_size: int) -> CredentialBatch:
        """生成全 invalid 验证 batch。"""

        return self.batch_generate(batch_size, 0.0)

    def rng_state(self) -> Dict[str, object]:
        """返回 NumPy Generator 状态，用于确定性 checkpoint 恢复。"""

        return self.rng.bit_generator.state

    def set_rng_state(self, state: Dict[str, object]) -> None:
        """恢复 NumPy Generator 状态并拒绝非法状态。"""

        if not isinstance(state, dict):
            raise TypeError("credential RNG state 必须是字典")
        self.rng.bit_generator.state = state

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        """验证 credential batch 大小为正整数。"""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size 必须是整数")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")


def get_cifar_transforms(train: bool = True) -> Callable:
    """返回 CIFAR-10 标准训练/验证变换；torchvision 缺失时明确失败。"""

    if not isinstance(train, bool):
        raise TypeError("train 必须是 bool")
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("真实 CIFAR-10 训练需要安装兼容版本 torchvision") from exc
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose([transforms.ToTensor(), normalize])


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """创建同步 Python、NumPy 和 PyTorch 的 DataLoader worker 初始化函数。"""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed 必须是整数")

    def _init_worker(worker_id: int) -> None:
        """为单个 worker 设置稳定的三套随机种子。"""

        worker_seed = (int(seed) + int(worker_id)) % (2**32)
        import random

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_worker


def split_indices(
    length: int, validation_fraction: float, seed: int
) -> Tuple[Sequence[int], Sequence[int]]:
    """按固定 seed 划分 train/validation 索引并返回两个不重叠序列。"""

    if isinstance(length, bool) or not isinstance(length, int) or length <= 1:
        raise ValueError("length 必须是大于 1 的整数")
    if isinstance(validation_fraction, (bool, np.bool_)) or not isinstance(
        validation_fraction, (int, float, np.integer, np.floating)
    ):
        raise TypeError("validation_fraction 必须是有限实数")
    fraction = float(validation_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction 必须位于 (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed 必须是整数")
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(length)
    validation_size = min(length - 1, max(1, int(round(length * fraction))))
    validation = np.sort(permutation[:validation_size]).astype(np.int64).tolist()
    train = np.sort(permutation[validation_size:]).astype(np.int64).tolist()
    return train, validation


__all__ = [
    "CIFAR10WithCoarse",
    "CIFAR2_MAPPING",
    "DATA_MAPPING_VERSION",
    "CredentialBatch",
    "CredentialGenerator",
    "fine_to_coarse",
    "get_cifar_transforms",
    "make_worker_init_fn",
    "split_indices",
]
