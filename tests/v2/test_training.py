"""Phase 2 训练数据、损失、指标和 checkpoint 的离线单元测试。"""

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.can.v2.crypto.lwe import LWEParams, V_ref, generate_keypair
from src.can.v2.models import GatedResNet18
from src.can.v2.models.gated_resnet import TrainingOutput
from src.can.v2.training.data import (
    CredentialBatch,
    CIFAR10WithCoarse,
    CredentialGenerator,
    fine_to_coarse,
    make_worker_init_fn,
    split_indices,
    get_cifar_transforms,
)
from src.can.v2.training.loss import compute_training_loss
from src.can.v2.training.metrics import EvaluationMetricAccumulator, training_accuracy
from src.can.v2.training.trainer import GatedResNetTrainer, checkpoint_sha256


@pytest.fixture
def lwe_fixture():
    """创建确定性 toy LWE 参数。"""

    np.random.seed(20260824)
    torch.manual_seed(20260824)
    params = LWEParams(n=32, m=64, sigma=1.0)
    A, secret, b = generate_keypair(params)
    return params, A, secret, b


def _labels(batch: int = 4):
    """创建固定标签和 CIFAR-like 图像。"""

    images = torch.randn(batch, 3, 32, 32, generator=torch.Generator().manual_seed(7))
    fine = torch.tensor([0, 2, 8, 5][:batch], dtype=torch.long)
    coarse = torch.tensor([0, 1, 0, 1][:batch], dtype=torch.long)
    return images, fine, coarse


def _model_output(lwe_fixture, batch: int = 4, mixed: bool = True):
    """生成 TrainingOutput 测试样本。"""

    params, A, secret, b = lwe_fixture
    model = GatedResNet18(A, b, params)
    model.train()
    images, fine, coarse = _labels(batch)
    invalid = np.ones(params.n, dtype=np.float32) * 10.0
    credentials = (
        np.stack([secret, invalid, secret, invalid][:batch]) if mixed else invalid
    )
    output = model(images, credentials)
    return model, output, images, fine, coarse


class TestDatasetAndCredentials:
    """验证标签映射、离线 Dataset 和 credential 采样。"""

    @pytest.mark.parametrize(
        "fine, coarse", [(0, 0), (1, 0), (8, 0), (9, 0), (2, 1), (7, 1)]
    )
    def test_cifar_mapping(self, fine, coarse):
        """CIFAR-2 映射应与设计固定类别集合一致。"""

        assert fine_to_coarse(fine) == coarse

    @pytest.mark.parametrize("value", [-1, 10, True, "0"])
    def test_cifar_mapping_rejects_invalid(self, value):
        """映射函数拒绝错误类型和范围。"""

        with pytest.raises((TypeError, ValueError)):
            fine_to_coarse(value)

    def test_dataset_injected_base_does_not_download(self):
        """注入 fake dataset 时不应导入或访问 torchvision 网络。"""

        base = TensorDataset(torch.zeros(2, 3, 32, 32), torch.tensor([0, 2]))
        dataset = CIFAR10WithCoarse("unused", base_dataset=base)
        assert len(dataset) == 2
        image, fine, coarse = dataset[1]
        assert image.shape == (3, 32, 32)
        assert (fine, coarse) == (2, 1)

    def test_credential_generator_is_verified_and_deterministic(self, lwe_fixture):
        """invalid 均由 V_ref 确认，且相同 seed 序列一致。"""

        params, A, secret, b = lwe_fixture
        first = CredentialGenerator(A, secret, b, params, seed=9)
        second = CredentialGenerator(A, secret, b, params, seed=9)
        batch_a = first.batch_generate(8, 0.5)
        batch_b = second.batch_generate(8, 0.5)
        assert np.array_equal(batch_a.values, batch_b.values)
        assert np.array_equal(batch_a.expected_valid, batch_b.expected_valid)
        actual = np.array(
            [V_ref({"vector": row}, A, b, params) for row in batch_a.values], dtype=bool
        )
        assert np.array_equal(actual, batch_a.expected_valid)

    def test_credential_generator_retry_exhaustion(self, lwe_fixture):
        """达到尝试上限时稳定报告生成失败。"""

        params, A, secret, b = lwe_fixture
        generator = CredentialGenerator(A, secret, b, params, seed=1, max_attempts=1)

        class _AlwaysSecret:
            """让 invalid rejection sampling 每次都收到 valid secret。"""

            def normal(self, *_args, **_kwargs):
                """返回 valid secret，触发重试耗尽。"""

                return secret.copy()

            def permutation(self, values):
                """保持 batch 顺序。"""

                return np.arange(values)

        generator.rng = _AlwaysSecret()
        with pytest.raises(RuntimeError):
            generator.generate(False)

    def test_minimum_valid_contract(self, lwe_fixture):
        """要求训练安全 batch 至少包含两个 valid credential。"""

        params, A, secret, b = lwe_fixture
        generator = CredentialGenerator(A, secret, b, params, seed=2)
        with pytest.raises(ValueError):
            generator.batch_generate(2, 0.5, min_valid=2)
        batch = generator.batch_generate(4, 0.5, min_valid=2)
        assert int(batch.expected_valid.sum()) == 2

    def test_keypair_explicit_rng_is_reproducible(self):
        """keypair 只依赖显式 Generator，不受全局 RNG 消耗影响。"""

        params = LWEParams(n=32, m=64, sigma=1.0)
        first = generate_keypair(params, rng=np.random.default_rng(99))
        np.random.seed(123)
        np.random.randn(1000)
        second = generate_keypair(params, rng=np.random.default_rng(99))
        assert all(np.array_equal(left, right) for left, right in zip(first, second))

    def test_credential_batch_and_split_boundaries(self, lwe_fixture):
        """覆盖全 valid、全 invalid、RNG 状态和固定 split 边界。"""

        params, A, secret, b = lwe_fixture
        generator = CredentialGenerator(A, secret, b, params, seed=4)
        assert generator.all_valid(2).expected_valid.all()
        assert not generator.all_invalid(2).expected_valid.any()
        state = generator.rng_state()
        generator.set_rng_state(state)
        with pytest.raises((TypeError, ValueError)):
            generator.batch_generate(0, 0.5)
        with pytest.raises(ValueError):
            generator.batch_generate(4, 0.5, min_valid=5)
        train, valid = split_indices(10, 0.2, 3)
        assert len(train) == 8 and len(valid) == 2
        assert set(train).isdisjoint(valid)
        with pytest.raises(ValueError):
            split_indices(1, 0.2, 3)

    def test_worker_seed_function(self):
        """worker 初始化函数应拒绝非法 seed 并同步三套 RNG。"""

        with pytest.raises(TypeError):
            make_worker_init_fn(True)
        init = make_worker_init_fn(10)
        init(2)
        first = (np.random.rand(), torch.rand(1).item())
        init(2)
        second = (np.random.rand(), torch.rand(1).item())
        assert first == second

    def test_data_contract_rejections(self, lwe_fixture):
        """数据包装器、CredentialBatch 和 transform 参数非法时拒绝。"""

        with pytest.raises(TypeError):
            CIFAR10WithCoarse(1)
        base = TensorDataset(torch.zeros(1, 3, 32, 32), torch.tensor([0]))
        dataset = CIFAR10WithCoarse("unused", base_dataset=base)
        with pytest.raises(TypeError):
            dataset[True]
        with pytest.raises(TypeError):
            CredentialBatch(np.zeros((1, 2), dtype=np.float64), np.array([True]))
        with pytest.raises(ValueError):
            CredentialBatch(np.full((1, 2), np.nan, dtype=np.float32), np.array([True]))
        with pytest.raises(TypeError):
            get_cifar_transforms(1)
        with pytest.raises(TypeError):
            CredentialGenerator("A", np.zeros(32), np.zeros(64), lwe_fixture[0])
        with pytest.raises(ValueError):
            CredentialGenerator(lwe_fixture[1], lwe_fixture[2], lwe_fixture[3], lwe_fixture[0], max_attempts=0)
        generator = CredentialGenerator(*lwe_fixture[1:], lwe_fixture[0])
        with pytest.raises(TypeError):
            generator.generate(1)
        with pytest.raises(TypeError):
            generator.set_rng_state(None)
        with pytest.raises((TypeError, ValueError)):
            generator.batch_generate(2, 0.5, min_valid=True)
        with pytest.raises(TypeError):
            split_indices(10, "0.2", 1)

    def test_torchvision_adapter_paths(self, monkeypatch):
        """用最小 fake torchvision 覆盖真实数据适配分支且不联网。"""

        class _Transform:
            """可调用的 fake transform。"""

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __call__(self, image):
                return image

        class _Compose(_Transform):
            """保存 transform 列表的 fake Compose。"""

            def __init__(self, transforms):
                super().__init__(transforms)
                self.transforms = transforms

        fake_transforms = types.SimpleNamespace(
            Normalize=_Transform,
            RandomCrop=_Transform,
            RandomHorizontalFlip=_Transform,
            ToTensor=_Transform,
            Compose=_Compose,
        )

        class _FakeCifar:
            """不下载数据的 fake CIFAR-10。"""

            def __init__(self, **kwargs):
                self.samples = [(torch.zeros(3, 32, 32), 0)]

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                return self.samples[index]

        fake_datasets = types.SimpleNamespace(CIFAR10=_FakeCifar)
        monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace(transforms=fake_transforms, datasets=fake_datasets))
        assert isinstance(get_cifar_transforms(True), _Compose)
        assert isinstance(get_cifar_transforms(False), _Compose)
        dataset = CIFAR10WithCoarse("unused", download=False)
        assert len(dataset) == 1


class TestLoss:
    """验证 protected mask、空 batch 和知识蒸馏。"""

    def test_mixed_loss_masks_invalid_rows(self, lwe_fixture):
        """invalid protected 占位不应参与 CE。"""

        _, output, _, fine, coarse = _model_output(lwe_fixture)
        losses = compute_training_loss(
            output, fine, coarse, None, alpha=1.0, beta_ce=1.0, beta_kd=0.0
        )
        expected = torch.nn.functional.cross_entropy(
            output.protected_logits[output.decision.allow], fine[output.decision.allow]
        )
        assert torch.allclose(losses.protected, expected)

    def test_all_invalid_loss_is_graph_connected(self, lwe_fixture):
        """全 invalid protected loss 为图连接零值，不创建独立 leaf。"""

        model, output, _, fine, coarse = _model_output(lwe_fixture, mixed=False)
        losses = compute_training_loss(
            output, fine, coarse, None, alpha=1.0, beta_ce=0.0, beta_kd=0.0
        )
        assert losses.protected.requires_grad
        assert losses.protected.item() == 0.0
        losses.total.backward()
        assert model.conv1.weight.grad is not None

    def test_teacher_coarse_kd_and_temperature(self, lwe_fixture):
        """teacher 10-class logits 聚合为固定的 vehicle/animal KD。"""

        _, output, _, fine, coarse = _model_output(lwe_fixture)
        teacher = torch.zeros(4, 10)
        teacher[:, 0] = 5.0
        losses = compute_training_loss(
            output, fine, coarse, teacher, beta_ce=0.0, beta_kd=1.0, temperature=4.0
        )
        assert losses.public_kd.item() >= 0.0
        assert losses.total.requires_grad

    @pytest.mark.parametrize("labels", [torch.tensor([0, 1]), torch.tensor([0.0, 1.0])])
    def test_loss_rejects_bad_labels(self, lwe_fixture, labels):
        """loss 拒绝 batch、dtype 不符合契约的 labels。"""

        _, output, _, fine, coarse = _model_output(lwe_fixture)
        with pytest.raises((TypeError, ValueError)):
            compute_training_loss(output, labels, coarse, None)

    def test_loss_weight_and_teacher_contracts(self, lwe_fixture):
        """损失权重、temperature、teacher logits 的非法输入必须拒绝。"""

        _, output, _, fine, coarse = _model_output(lwe_fixture)
        for kwargs in ({"alpha": -1.0}, {"temperature": 0.0}, {"beta_kd": 1.0}):
            with pytest.raises((TypeError, ValueError)):
                compute_training_loss(output, fine, coarse, None, **kwargs)
        teacher = torch.zeros(4, 10)
        with pytest.raises(ValueError):
            compute_training_loss(output, fine, coarse, teacher[:2], beta_kd=1.0)
        with pytest.raises(ValueError):
            compute_training_loss(output, fine, coarse, torch.full_like(teacher, float("nan")), beta_kd=1.0)
        malformed = TrainingOutput(output.protected_logits[:, :9], output.public_logits, output.decision)
        with pytest.raises(ValueError):
            compute_training_loss(malformed, fine, coarse, None)
        with pytest.raises(TypeError):
            compute_training_loss(output, fine, coarse, None, alpha="x")


class TestMetrics:
    """验证空路由、混淆矩阵和训练态指标。"""

    def test_metrics_empty_and_sample_counts(self):
        """空子批不进入分母，指标按样本计数。"""

        metrics = EvaluationMetricAccumulator()
        metrics.update_protected(torch.empty(0, 10), torch.empty(0, dtype=torch.long))
        metrics.update_public(torch.tensor([[2.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))
        result = metrics.compute()
        assert result["protected_accuracy"] is None
        assert result["public_total"] == 2.0
        assert result["public_balanced_accuracy"] == 1.0
        with pytest.raises(ValueError):
            metrics.update_public(torch.zeros(1, 2), torch.tensor([2]))

    def test_training_accuracy(self, lwe_fixture):
        """训练态 accuracy 只统计 allow 的 protected 行。"""

        _, output, _, fine, coarse = _model_output(lwe_fixture)
        result = training_accuracy(output, fine, coarse)
        assert set(result) == {"protected_accuracy", "public_accuracy"}

    def test_metrics_contract_rejections(self):
        """指标模块拒绝类型、shape、dtype 和非有限输入。"""

        metrics = EvaluationMetricAccumulator()
        with pytest.raises(TypeError):
            metrics.update_public([[1.0, 0.0]], torch.tensor([0]))
        with pytest.raises(ValueError):
            metrics.update_public(torch.zeros(1, 3), torch.tensor([0]))
        with pytest.raises(TypeError):
            metrics.update_public(torch.zeros(1, 2), torch.tensor([0.0]))
        with pytest.raises(ValueError):
            metrics.update_public(torch.full((1, 2), float("inf")), torch.tensor([0]))


class TestTrainer:
    """验证稀疏评估和 checkpoint 恢复。"""

    def test_validate_aligns_sparse_indices(self, lwe_fixture):
        """验证函数必须按 indices 选择原始标签。"""

        params, A, secret, b = lwe_fixture
        model = GatedResNet18(A, b, params)
        images, fine, coarse = _labels(4)
        loader = DataLoader(TensorDataset(images, fine, coarse), batch_size=4)
        generator = CredentialGenerator(A, secret, b, params, seed=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        trainer = GatedResNetTrainer(
            model, loader, loader, generator, optimizer, torch.device("cpu")
        )
        metrics = trainer.validate()
        assert metrics["protected_total"] == 4.0
        assert metrics["public_total"] == 4.0

    def test_stage_b_requires_teacher_identity(self, lwe_fixture):
        """Stage B 不允许无 identity 的匿名 teacher。"""

        params, A, secret, b = lwe_fixture
        model = GatedResNet18(A, b, params)
        loader = DataLoader(TensorDataset(*_labels(4)), batch_size=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        with pytest.raises(ValueError, match="teacher_identity"):
            GatedResNetTrainer(
                model,
                loader,
                loader,
                CredentialGenerator(A, secret, b, params),
                optimizer,
                torch.device("cpu"),
                stage="B",
                teacher=GatedResNet18(A, b, params),
            )

    def _identity(self, tmp_path: Path):
        """创建可供 teacher identity 校验的临时文件。"""

        path = tmp_path / "teacher.ckpt"
        path.write_bytes(b"teacher")
        return {"path": str(path), "sha256": checkpoint_sha256(path)}

    def _trainer(self, lwe_fixture, stage="A", tmp_path=None):
        """创建小型 trainer 以覆盖阶段配置和 fit 路径。"""

        params, A, secret, b = lwe_fixture
        model = GatedResNet18(A, b, params)
        images, fine, coarse = _labels(4)
        loader = DataLoader(TensorDataset(images, fine, coarse), batch_size=4)
        kwargs = {"stage": stage, "progress": False}
        if stage in {"B", "C"}:
            kwargs.update(
                teacher=GatedResNet18(A, b, params),
                teacher_identity=self._identity(tmp_path),
                valid_ratio=0.0 if stage == "B" else 0.5,
            )
        if stage == "C":
            kwargs.update(protected_baseline=0.0, max_protected_drop=0.1)
        return GatedResNetTrainer(
            model,
            loader,
            loader,
            CredentialGenerator(A, secret, b, params, seed=3),
            torch.optim.SGD(model.parameters(), lr=0.001),
            torch.device("cpu"),
            **kwargs,
        )

    def test_stage_freeze_contract(self, lwe_fixture, tmp_path: Path):
        """A/B/C 的可训练参数和模块模式符合设计。"""

        a = self._trainer(lwe_fixture)
        a._configure_stage()
        assert not any(p.requires_grad for p in a.model.public_fc.parameters())
        b = self._trainer(lwe_fixture, "B", tmp_path)
        b._configure_stage()
        assert all(p.requires_grad for p in b.model.public_fc.parameters())
        c = self._trainer(lwe_fixture, "C", tmp_path)
        c._configure_stage()
        assert all(p.requires_grad for p in c.model.parameters())
        b.train_epoch(1, 1)
        assert b._teacher_logits(_labels(4)[0]).shape == (4, 10)

    def test_trainer_constructor_contract(self, lwe_fixture):
        """训练器构造参数的类型、范围和阶段约束均 fail fast。"""

        params, A, secret, b = lwe_fixture
        model = GatedResNet18(A, b, params)
        images, fine, coarse = _labels(4)
        loader = DataLoader(TensorDataset(images, fine, coarse), batch_size=4)
        generator = CredentialGenerator(A, secret, b, params)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        common = (model, loader, loader, generator, optimizer, torch.device("cpu"))
        with pytest.raises(ValueError):
            GatedResNetTrainer(*common, stage="X")
        with pytest.raises(ValueError):
            GatedResNetTrainer(*common, valid_ratio=2.0)
        with pytest.raises(ValueError):
            GatedResNetTrainer(*common, max_grad_norm=0.0)
        with pytest.raises(ValueError):
            GatedResNetTrainer(*common, protected_baseline=float("nan"))

    def test_teacher_identity_and_fit_contract(self, lwe_fixture, tmp_path: Path):
        """teacher hash、fit 参数和 monitor 约束必须 fail fast。"""

        identity = tmp_path / "teacher.ckpt"
        identity.write_bytes(b"teacher")
        bad = {"path": str(identity), "sha256": "0" * 64}
        params, A, secret, b = lwe_fixture
        model = GatedResNet18(A, b, params)
        images, fine, coarse = _labels(4)
        loader = DataLoader(TensorDataset(images, fine, coarse), batch_size=4)
        with pytest.raises(ValueError):
            GatedResNetTrainer(model, loader, loader, CredentialGenerator(A, secret, b, params), torch.optim.SGD(model.parameters(), lr=0.001), torch.device("cpu"), stage="B", teacher=GatedResNet18(A, b, params), teacher_identity=bad)
        trainer = self._trainer(lwe_fixture, tmp_path=tmp_path)
        with pytest.raises(ValueError):
            trainer.fit(0)
        with pytest.raises(ValueError):
            trainer.fit(1, patience=-1)
        with pytest.raises(ValueError):
            trainer.fit(1, min_delta=-1)
        trainer.current_epoch = 2
        with pytest.raises(ValueError):
            trainer.fit(1)

    def test_monitor_and_metadata_contracts(self, lwe_fixture, tmp_path: Path):
        """覆盖三阶段 checkpoint 选择和 LWE/split metadata 校验。"""

        trainer = self._trainer(lwe_fixture, tmp_path=tmp_path)
        assert trainer._monitor_value({"protected_accuracy": 0.5}) == 0.5
        trainer.stage = "B"
        assert trainer._monitor_value({"public_balanced_accuracy": 0.4}) == 0.4
        trainer = self._trainer(lwe_fixture, "C", tmp_path)
        trainer.protected_baseline = 0.5
        assert trainer._monitor_value({"protected_accuracy": 0.0, "public_balanced_accuracy": 0.5}) is None
        assert trainer._monitor_value({"protected_accuracy": 0.5, "public_balanced_accuracy": 0.5}) == 0.5
        trainer.checkpoint_metadata = {"mapping_version": "v1", "split": {"hash": "x"}, "lwe": {"n": 1}, "A": np.zeros((1, 1)), "b": np.zeros(1)}
        for metadata in ({}, {"mapping_version": "bad"}, {"split": {"hash": "bad"}}, {"lwe": {"n": 2}}, {"A": np.ones((1, 1))}, {"b": np.ones(1)}):
            with pytest.raises(ValueError):
                trainer._validate_checkpoint_metadata(metadata)

    def test_constructor_type_contracts(self, lwe_fixture):
        """训练器基础依赖类型错误必须在构造期报告。"""

        params, A, secret, b = lwe_fixture
        model = GatedResNet18(A, b, params)
        images, fine, coarse = _labels(4)
        loader = DataLoader(TensorDataset(images, fine, coarse), batch_size=4)
        generator = CredentialGenerator(A, secret, b, params)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        base = [model, loader, loader, generator, optimizer, torch.device("cpu")]
        for index, value in enumerate([object(), object(), object(), object(), object(), "cpu"]):
            args = list(base)
            args[index] = value
            with pytest.raises(TypeError):
                GatedResNetTrainer(*args)

    def test_fit_and_checkpoint_metadata_rejection(self, lwe_fixture, tmp_path: Path):
        """fit 保存 last/best，并拒绝不匹配的 checkpoint metadata。"""

        trainer = self._trainer(lwe_fixture, tmp_path=tmp_path)
        history = trainer.fit(1, checkpoint_dir=tmp_path)
        assert history["epoch"] == 1.0
        assert (tmp_path / "last.ckpt").is_file()
        assert (tmp_path / "best.ckpt").is_file()
        trainer.checkpoint_metadata = {"config_signature": "expected"}
        with pytest.raises(ValueError, match="config_signature"):
            trainer.load_checkpoint(tmp_path / "last.ckpt")

    def test_stage_c_requires_protected_constraint(self, lwe_fixture, tmp_path: Path):
        """Stage C 必须显式提供 baseline 和允许下降阈值。"""

        params, A, secret, b = lwe_fixture
        identity_path = tmp_path / "teacher.ckpt"
        identity_path.write_bytes(b"teacher")
        identity = {
            "path": str(identity_path),
            "sha256": checkpoint_sha256(identity_path),
        }
        model = GatedResNet18(A, b, params)
        loader = DataLoader(TensorDataset(*_labels(4)), batch_size=4)
        with pytest.raises(ValueError, match="protected_baseline"):
            GatedResNetTrainer(
                model,
                loader,
                loader,
                CredentialGenerator(A, secret, b, params),
                torch.optim.SGD(model.parameters(), lr=0.001),
                torch.device("cpu"),
                stage="C",
                teacher=GatedResNet18(A, b, params),
                teacher_identity=identity,
            )

    def test_checkpoint_roundtrip(self, lwe_fixture, tmp_path: Path):
        """checkpoint 应恢复模型、优化器、global step 和 RNG。"""

        params, A, secret, b = lwe_fixture
        model = GatedResNet18(A, b, params)
        images, fine, coarse = _labels(4)
        loader = DataLoader(TensorDataset(images, fine, coarse), batch_size=4)
        generator = CredentialGenerator(A, secret, b, params, seed=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        trainer = GatedResNetTrainer(
            model, loader, loader, generator, optimizer, torch.device("cpu")
        )
        trainer.global_step = 7
        path = tmp_path / "last.ckpt"
        trainer.save_checkpoint(path, {"config_hash": "x"})
        restored = GatedResNetTrainer(
            GatedResNet18(A, b, params),
            loader,
            loader,
            generator,
            torch.optim.SGD(GatedResNet18(A, b, params).parameters(), lr=0.001),
            torch.device("cpu"),
        )
        # optimizer/model 参数必须属于同一 restored model。
        restored.optimizer = torch.optim.SGD(restored.model.parameters(), lr=0.001)
        metadata = restored.load_checkpoint(path)
        assert metadata == {"config_hash": "x"}
        assert restored.global_step == 7
        assert checkpoint_sha256(path) == checkpoint_sha256(path)

    def test_empty_train_loader_fails_fast(self, lwe_fixture):
        """drop_last 丢弃全部样本时 trainer 不得产生空训练 checkpoint。"""

        params, A, secret, b = lwe_fixture
        model = GatedResNet18(A, b, params)
        images, fine, coarse = _labels(2)
        empty_loader = DataLoader(
            TensorDataset(images, fine, coarse), batch_size=4, drop_last=True
        )
        validation_loader = DataLoader(
            TensorDataset(images, fine, coarse), batch_size=2
        )
        trainer = GatedResNetTrainer(
            model,
            empty_loader,
            validation_loader,
            CredentialGenerator(A, secret, b, params, seed=3),
            torch.optim.SGD(model.parameters(), lr=0.001),
            torch.device("cpu"),
            progress=False,
        )
        with pytest.raises(ValueError, match="没有 batch"):
            trainer.train_epoch()

    def test_resume_is_deterministic(self, lwe_fixture, tmp_path: Path):
        """恢复后的下一批 credential、数据顺序和 optimizer step 与未中断运行一致。"""

        params, A, secret, b = lwe_fixture
        images, fine, coarse = _labels(4)

        def build() -> GatedResNetTrainer:
            """创建使用独立 DataLoader/credential RNG 的 Stage A trainer。"""

            model = GatedResNet18(A, b, params)
            loader = DataLoader(
                TensorDataset(images, fine, coarse),
                batch_size=4,
                shuffle=True,
                generator=torch.Generator().manual_seed(44),
            )
            return GatedResNetTrainer(
                model,
                loader,
                loader,
                CredentialGenerator(A, secret, b, params, seed=55),
                torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
                torch.device("cpu"),
                progress=False,
            )

        uninterrupted = build()
        uninterrupted.train_epoch(1, 2)
        path = tmp_path / "resume.ckpt"
        uninterrupted.save_checkpoint(path)
        uninterrupted.train_epoch(2, 2)

        resumed = build()
        resumed.load_checkpoint(path)
        resumed.train_epoch(2, 2)
        for expected, actual in zip(
            uninterrupted.model.state_dict().values(),
            resumed.model.state_dict().values(),
        ):
            assert torch.equal(expected, actual)
        assert uninterrupted.global_step == resumed.global_step == 2
