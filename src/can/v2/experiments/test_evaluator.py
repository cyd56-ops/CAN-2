"""Phase 3 CIFAR-10 test split 评估器。

该模块只负责确定性计算评估指标，不负责 checkpoint 或 JSON 文件读写。
"""

from dataclasses import dataclass
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ..crypto.lwe import LWEParams
from ..models.gated_resnet import GatedResNet18, InferenceOutput
from ..training.data import CredentialGenerator


@dataclass
class _Confusion:
    """保存固定类别数的混淆矩阵计数。"""

    classes: int

    def __post_init__(self) -> None:
        """初始化全零混淆矩阵。"""

        self.matrix = torch.zeros((self.classes, self.classes), dtype=torch.long)

    def update(self, logits: Tensor, labels: Tensor) -> None:
        """累计一批 logits 与标签。"""

        if logits.ndim != 2 or logits.shape[1] != self.classes:
            raise ValueError("logits 类别维度不符合混淆矩阵配置")
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("logits 与 labels batch 大小不一致")
        if labels.dtype != torch.long or logits.device != labels.device:
            raise TypeError("labels 必须是同 device 的 torch.long Tensor")
        if logits.numel() == 0:
            return
        pred = logits.argmax(1)
        flat = labels * self.classes + pred
        self.matrix += torch.bincount(flat.detach().cpu(), minlength=self.classes**2).reshape(
            self.classes, self.classes
        )

    def accuracy(self) -> Optional[float]:
        """返回混淆矩阵对应的准确率。"""

        total = int(self.matrix.sum().item())
        return float(self.matrix.diag().sum().item() / total) if total else None

    def macro_f1(self) -> Optional[float]:
        """按有真实样本的类别计算 macro-F1。"""

        if not int(self.matrix.sum().item()):
            return None
        matrix = self.matrix.float()
        precision = matrix.diag() / matrix.sum(0).clamp_min(1)
        recall = matrix.diag() / matrix.sum(1).clamp_min(1)
        support = self.matrix.sum(1) > 0
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        return float(f1[support].mean().item())

    def per_class_accuracy(self) -> list:
        """返回每类准确率；空类使用 None。"""

        result = []
        for index, row in enumerate(self.matrix):
            total = int(row.sum().item())
            result.append(float(row[index].item() / total) if total else None)
        return result

    def balanced_accuracy(self) -> Optional[float]:
        """按有真实样本的类别计算 balanced accuracy。"""

        values = [value for value in self.per_class_accuracy() if value is not None]
        return float(np.mean(values)) if values else None


class TestSplitEvaluator:
    """在固定 test DataLoader 上执行三路径评估。"""

    def __init__(
        self,
        model: GatedResNet18,
        credential_generator: CredentialGenerator,
        params: LWEParams,
        device: torch.device,
        mixed_ratio: float = 0.5,
    ) -> None:
        """初始化评估器并校验运行参数。"""

        if not isinstance(model, GatedResNet18):
            raise TypeError("model 必须是 GatedResNet18")
        if not isinstance(credential_generator, CredentialGenerator):
            raise TypeError("credential_generator 类型非法")
        if not isinstance(params, LWEParams):
            raise TypeError("params 必须是 LWEParams")
        if not isinstance(device, torch.device):
            raise TypeError("device 必须是 torch.device")
        if not np.isfinite(mixed_ratio) or not 0.0 < float(mixed_ratio) < 1.0:
            raise ValueError("mixed_ratio 必须位于 (0, 1)")
        self.model = model.to(device)
        self.generator = credential_generator
        self.params = params
        self.device = device
        self.mixed_ratio = float(mixed_ratio)
        self._lut = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.long)

    @staticmethod
    def _stats(values: list) -> Dict[str, Optional[float]]:
        """计算有限误差范数的统计量。"""

        finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if finite.size == 0:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "min": float(finite.min()),
            "max": float(finite.max()),
        }

    def _run_uniform(self, loader: Iterable, valid: bool) -> Dict[str, object]:
        """评估全 valid 或全 invalid 路径。"""

        confusion = _Confusion(10 if valid else 2)
        norms = []
        reason_histogram: Dict[str, int] = {}
        rejected = 0
        total = 0
        coarse_correct = 0
        self.model.eval()
        with torch.inference_mode():
            for images, fine, coarse in loader:
                images = images.to(self.device)
                fine = torch.as_tensor(fine, device=self.device, dtype=torch.long)
                coarse = torch.as_tensor(coarse, device=self.device, dtype=torch.long)
                batch = images.shape[0]
                cb = self.generator.all_valid(batch) if valid else self.generator.all_invalid(batch)
                output = self.model(images, cb.values)
                if not isinstance(output, InferenceOutput):
                    raise TypeError("eval 模式必须返回 InferenceOutput")
                allow = output.decision.allow
                expected = torch.full_like(allow, valid)
                if not torch.equal(allow, expected):
                    raise AssertionError("Gate 判决与 credential 采样类型不一致")
                norms.extend(output.decision.evidence.error_norm.detach().cpu().tolist())
                reason_codes = output.decision.evidence.reason_code.detach().cpu()
                for code, count in zip(*torch.unique(reason_codes, return_counts=True)):
                    key = str(int(code.item()))
                    reason_histogram[key] = reason_histogram.get(key, 0) + int(count.item())
                rejected += int((~allow).sum().item()) if valid else int(allow.sum().item())
                total += batch
                indices = output.protected_indices if valid else output.public_indices
                logits = output.protected_logits if valid else output.public_logits
                labels = fine.index_select(0, indices) if valid else coarse.index_select(0, indices)
                confusion.update(logits, labels)
                if valid:
                    coarse_pred = self._lut.to(logits.device).index_select(0, logits.argmax(1))
                    coarse_labels = coarse.index_select(0, indices)
                    coarse_correct += int((coarse_pred == coarse_labels).sum().item())
        return {
            "accuracy": confusion.accuracy(),
            "macro_f1": confusion.macro_f1(),
            "balanced_accuracy": confusion.balanced_accuracy(),
            "per_class_accuracy": confusion.per_class_accuracy(),
            "confusion": confusion.matrix.tolist(),
            "total": total,
            "rejected": rejected,
            "error_norm_stats": self._stats(norms),
            "error_norms": norms,
            "reason_code_histogram": reason_histogram,
            "coarse_accuracy": coarse_correct / total if valid and total else None,
        }

    def evaluate(self, loader: Iterable) -> Dict[str, object]:
        """执行 protected、public 和 mixed routing 评估并返回结构化结果。"""

        protected = self._run_uniform(loader, True)
        public = self._run_uniform(loader, False)
        mixed_batches = 0
        mismatches = 0
        empty_valid = 0
        empty_invalid = 0
        self.model.eval()
        with torch.inference_mode():
            for images, _, _ in loader:
                images = images.to(self.device)
                num_valid = int(round(images.shape[0] * self.mixed_ratio))
                if num_valid < 2 or images.shape[0] - num_valid < 1:
                    raise ValueError("mixed 尾批必须至少包含两个 valid 和一个 invalid 样本")
                cb = self.generator.batch_generate(images.shape[0], self.mixed_ratio, min_valid=2)
                mixed = self.model(images, cb.values)
                if not isinstance(mixed, InferenceOutput):
                    raise TypeError("eval 模式必须返回 InferenceOutput")
                expected = torch.as_tensor(cb.expected_valid, device=self.device)
                mismatches += int((mixed.decision.allow != expected).sum().item())
                valid_mask = mixed.decision.allow
                invalid_mask = ~valid_mask
                expected_indices = torch.arange(images.shape[0], device=self.device)
                combined = torch.cat((mixed.protected_indices, mixed.public_indices)).sort().values
                coverage_complete = torch.equal(combined, expected_indices)
                disjoint = not bool(torch.isin(mixed.protected_indices, mixed.public_indices).any().item())
                if not coverage_complete or not disjoint:
                    raise AssertionError("mixed routing indices 未完整且互斥地覆盖 batch")
                if not bool(valid_mask.any().item()):
                    empty_valid += 1
                if not bool(invalid_mask.any().item()):
                    empty_invalid += 1
                if bool(valid_mask.any().item()):
                    ref = self.model(images[valid_mask], cb.values[valid_mask.detach().cpu().numpy()])
                    if not isinstance(ref, InferenceOutput):
                        raise TypeError("reference protected 输出类型错误")
                    torch.testing.assert_close(mixed.protected_logits, ref.protected_logits, atol=1e-5, rtol=1e-4)
                if bool(invalid_mask.any().item()):
                    ref = self.model(images[invalid_mask], cb.values[invalid_mask.detach().cpu().numpy()])
                    if not isinstance(ref, InferenceOutput):
                        raise TypeError("reference public 输出类型错误")
                    torch.testing.assert_close(mixed.public_logits, ref.public_logits, atol=1e-5, rtol=1e-4)
                mixed_batches += 1
        return {
            "authorized": {
                "protected_accuracy": protected["accuracy"],
                "protected_macro_f1": protected["macro_f1"],
                "protected_per_class_accuracy": protected["per_class_accuracy"],
                "protected_confusion": protected["confusion"],
                "protected_total": protected["total"],
            },
            "unauthorized": {
                "public_accuracy": public["accuracy"],
                "public_macro_f1": public["macro_f1"],
                "public_balanced_accuracy": public["balanced_accuracy"],
                "public_confusion": public["confusion"],
                "public_total": public["total"],
            },
            "capability": {
                "protected_coarse_accuracy": protected["coarse_accuracy"],
                "capability_gap_fine": protected["accuracy"] - 0.2 if protected["accuracy"] is not None else None,
                "unauthorized_fine_random_guess_baseline": {"value": 0.2, "is_analytic": True},
            },
            "gate": {
                "far": public["rejected"] / public["total"] if public["total"] else None,
                "frr": protected["rejected"] / protected["total"] if protected["total"] else None,
                "error_norm_stats": {"valid": protected["error_norm_stats"], "invalid": public["error_norm_stats"]},
                "min_margin": {
                    "valid": self._min_margin(protected["error_norms"]),
                    "invalid": self._min_margin(public["error_norms"]),
                    "all": self._min_margin(protected["error_norms"] + public["error_norms"]),
                },
                "reason_code_histogram": {"valid": protected["reason_code_histogram"], "invalid": public["reason_code_histogram"]},
                "valid_samples": protected["total"],
                "invalid_samples": public["total"],
                "distinct_valid_credentials": 1,
                "distinct_invalid_credentials": public["total"],
            },
            "mixed_batch": {
                "mixed_ratio": self.mixed_ratio,
                "batches": mixed_batches,
                "routing_mismatches": mismatches,
                "index_coverage_complete": True,
                "reference_routing_logits_allclose": True,
                "assert_close_atol": 1e-5,
                "assert_close_rtol": 1e-4,
                "empty_subbatch_skips": {"valid": empty_valid, "invalid": empty_invalid},
            },
        }

    def _min_margin(self, values: list) -> Optional[float]:
        """返回有限 error norm 到验证阈值的最小距离。"""

        finite = [abs(float(value) - float(self.params.error_threshold)) for value in values if np.isfinite(value)]
        return min(finite) if finite else None

    def measure_latency(self, images: Tensor, warmup: int = 20, iterations: int = 100) -> Dict[str, object]:
        """使用固定输入测量三种路由的 forward 延迟（不含数据搬运）。"""
        if images.ndim != 4 or images.shape[0] != 256:
            raise ValueError("latency 输入必须是 batch=256 的图像 Tensor")
        images = images.to(self.device)
        batches = {
            "all_valid": self.generator.all_valid(256),
            "all_invalid": self.generator.all_invalid(256),
            "mixed": self.generator.batch_generate(256, self.mixed_ratio, min_valid=2),
        }
        result = {}
        self.model.eval()
        with torch.inference_mode():
            for name, batch in batches.items():
                credential = batch.values
                for _ in range(warmup):
                    self.model(images, credential)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                samples = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    self.model(images, credential)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    samples.append((time.perf_counter() - start) * 1000.0)
                values = np.asarray(samples, dtype=np.float64)
                result[name] = {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)), "std_ms": float(values.std())}
        return result


__all__ = ["TestSplitEvaluator"]
