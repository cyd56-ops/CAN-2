"""Phase 3 evaluator 的离线单元测试。"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.experiments.test_evaluator import TestSplitEvaluator as Evaluator, _Confusion
from src.can.v2.models import GatedResNet18
from src.can.v2.training.data import CredentialGenerator


def test_confusion_metrics_ignore_empty_classes():
    """macro-F1 只对有真实样本的类别求平均，空类准确率为 None。"""
    acc = _Confusion(3)
    acc.update(torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), torch.tensor([0, 1]))
    assert acc.accuracy() == 1.0
    assert acc.per_class_accuracy() == [1.0, 1.0, None]
    assert acc.macro_f1() == 1.0


def test_confusion_rejects_wrong_shape():
    """混淆矩阵拒绝错误类别维度。"""
    with pytest.raises(ValueError):
        _Confusion(2).update(torch.zeros(1, 3), torch.zeros(1, dtype=torch.long))


def test_evaluator_rejects_invalid_mixed_ratio():
    """mixed ratio 必须严格位于 (0, 1)。"""
    params = LWEParams(n=8, m=16)
    A, secret, b = generate_keypair(params, rng=np.random.default_rng(3))
    model = GatedResNet18(A, b, params)
    generator = CredentialGenerator(A, secret, b, params, seed=4)
    for ratio in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            Evaluator(model, generator, params, torch.device("cpu"), ratio)


def test_evaluator_rejects_wrong_device_type():
    """评估器要求显式 torch.device。"""
    params = LWEParams(n=8, m=16)
    A, secret, b = generate_keypair(params, rng=np.random.default_rng(5))
    with pytest.raises(TypeError):
        Evaluator(
            GatedResNet18(A, b, params),
            CredentialGenerator(A, secret, b, params),
            params,
            "cpu",
        )


def test_aggregate_requires_three_distinct_stage_c(tmp_path: Path):
    """aggregate 拒绝 seed 数量不足或阶段错误。"""
    import scripts.eval_cifar10_test as cli

    records = []
    for seed in (1, 2):
        records.append({"seed": seed, "checkpoint": {"stage": "C"}, "eval_batch_size": 2, "mapping_version": "v1"})
    paths = []
    for idx, record in enumerate(records):
        path = tmp_path / f"r{idx}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(str(path))
    with pytest.raises(ValueError):
        cli._aggregate(paths, tmp_path / "out.json", False)


def test_evaluator_reports_capability_gate_and_mixed_metrics():
    """离线 synthetic batch 应产出完整能力、Gate 与 mixed 指标。"""

    torch.manual_seed(11)
    params = LWEParams(n=32, m=64)
    A, secret, b = generate_keypair(params, rng=np.random.default_rng(12))
    model = GatedResNet18(A, b, params).eval()
    generator = CredentialGenerator(A, secret, b, params, seed=13)
    images = torch.randn(4, 3, 32, 32)
    fine = torch.tensor([0, 2, 8, 5], dtype=torch.long)
    coarse = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, fine, coarse), batch_size=4)

    result = Evaluator(model, generator, params, torch.device("cpu"), 0.5).evaluate(loader)

    assert result["gate"]["far"] == 0.0
    assert result["gate"]["frr"] == 0.0
    assert result["gate"]["distinct_valid_credentials"] == 1
    assert set(result["gate"]["min_margin"]) == {"all", "valid", "invalid"}
    assert len(result["authorized"]["protected_per_class_accuracy"]) == 10
    assert 0.0 <= result["capability"]["protected_coarse_accuracy"] <= 1.0
    assert result["mixed_batch"]["routing_mismatches"] == 0
    assert result["mixed_batch"]["index_coverage_complete"] is True
    assert result["mixed_batch"]["reference_routing_logits_allclose"] is True
