"""Phase 1.3 Gated ResNet-18 的架构、路由和输入契约测试。"""

from dataclasses import FrozenInstanceError
from typing import Dict, List, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.can.v2.crypto.lwe import (  # noqa: E402
    LWEParams,
    generate_keypair,
    generate_random_credential,
)
from src.can.v2.models.gated_resnet import (  # noqa: E402
    BasicBlock,
    GatedResNet18,
    InferenceOutput,
    TrainingOutput,
)


@pytest.fixture
def model_fixture() -> Tuple[GatedResNet18, LWEParams, np.ndarray, np.ndarray]:
    """创建确定性的模型、参数以及 valid/invalid credential。"""

    np.random.seed(20260823)
    torch.manual_seed(20260823)
    params = LWEParams(n=8, m=16, sigma=1.0)
    A, secret, b = generate_keypair(params)
    invalid = generate_random_credential(params, scale=10.0)
    model = GatedResNet18(A, b, params, temperature=5.0)
    return model, params, secret, invalid


def _images(batch_size: int) -> torch.Tensor:
    """生成确定性的 float32 CIFAR 图像 batch。"""

    generator = torch.Generator().manual_seed(1000 + batch_size)
    return torch.randn(batch_size, 3, 32, 32, generator=generator)


def _register_deep_pre_hooks(
    model: GatedResNet18,
) -> Tuple[Dict[str, List[int]], List[torch.utils.hooks.RemovableHandle]]:
    """记录 layer3/layer4 每次调用时收到的 batch size。"""

    calls: Dict[str, List[int]] = {"layer3": [], "layer4": []}

    def record_layer3(_module, inputs) -> None:
        """记录 layer3 的输入 batch size。"""

        calls["layer3"].append(inputs[0].shape[0])

    def record_layer4(_module, inputs) -> None:
        """记录 layer4 的输入 batch size。"""

        calls["layer4"].append(inputs[0].shape[0])

    handles = [
        model.layer3.register_forward_pre_hook(record_layer3),
        model.layer4.register_forward_pre_hook(record_layer4),
    ]
    return calls, handles


class TestArchitecture:
    """验证 CIFAR ResNet-18 的静态结构和 stage shape。"""

    def test_resnet18_block_configuration(self, model_fixture) -> None:
        """四个 stage 必须使用标准的 [2, 2, 2, 2] BasicBlock。"""

        model, _, _, _ = model_fixture

        assert model.conv1.kernel_size == (3, 3)
        assert model.conv1.stride == (1, 1)
        assert not hasattr(model, "maxpool")
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
            assert len(layer) == 2
            assert all(isinstance(block, BasicBlock) for block in layer)

    def test_cifar_stage_shapes(self, model_fixture) -> None:
        """各 stage 必须实现 32→32→16→8→4 的空间尺寸。"""

        model, _, secret, _ = model_fixture
        model.eval()
        shapes: Dict[str, torch.Size] = {}
        handles = []
        for name in ("conv1", "layer1", "layer2", "layer3", "layer4"):
            module = getattr(model, name)

            def save_shape(_module, _inputs, output, key=name) -> None:
                """保存指定 stage 的输出 shape。"""

                shapes[key] = output.shape

            handles.append(module.register_forward_hook(save_shape))

        try:
            with torch.no_grad():
                model(_images(2), secret)
        finally:
            for handle in handles:
                handle.remove()

        assert shapes == {
            "conv1": torch.Size([2, 64, 32, 32]),
            "layer1": torch.Size([2, 64, 32, 32]),
            "layer2": torch.Size([2, 128, 16, 16]),
            "layer3": torch.Size([2, 256, 8, 8]),
            "layer4": torch.Size([2, 512, 4, 4]),
        }

    @pytest.mark.parametrize("value", [0, -1, True, 1.5])
    def test_constructor_rejects_invalid_class_counts(
        self, model_fixture, value
    ) -> None:
        """分类 head 的类别数必须是非 bool 正整数。"""

        model, params, _, _ = model_fixture
        A = model.gate_layer.verifier.A.detach().cpu().numpy()
        b = model.gate_layer.verifier.b.detach().cpu().numpy()

        with pytest.raises((TypeError, ValueError)):
            GatedResNet18(A, b, params, num_classes_protected=value)
        with pytest.raises((TypeError, ValueError)):
            GatedResNet18(A, b, params, num_classes_public=value)

    def test_standard_weight_initialization(self, model_fixture) -> None:
        """卷积、BatchNorm 和线性层应具有非退化的标准初始化。"""

        model, _, _, _ = model_fixture

        assert torch.count_nonzero(model.conv1.weight).item() > 0
        assert torch.equal(model.bn1.weight, torch.ones_like(model.bn1.weight))
        assert torch.equal(model.bn1.bias, torch.zeros_like(model.bn1.bias))
        assert torch.count_nonzero(model.protected_fc.weight).item() > 0


class TestRouting:
    """验证训练态和推理态的批量条件路由。"""

    def test_training_mixed_batch_isolates_deep_path(self, model_fixture) -> None:
        """训练态 mixed batch 只能把 valid 子批送入深层。"""

        model, _, secret, invalid = model_fixture
        model.train()
        credentials = np.stack([secret, invalid, secret])
        calls, handles = _register_deep_pre_hooks(model)

        try:
            output = model(_images(3), credentials)
        finally:
            for handle in handles:
                handle.remove()

        assert isinstance(output, TrainingOutput)
        assert output.decision.allow.tolist() == [True, False, True]
        assert output.protected_logits.shape == (3, 10)
        assert output.public_logits.shape == (3, 2)
        assert torch.count_nonzero(output.protected_logits[1]).item() == 0
        assert calls == {"layer3": [2], "layer4": [2]}

    def test_training_all_invalid_returns_linked_zero_placeholders(
        self, model_fixture
    ) -> None:
        """全 invalid 训练 batch 应返回可反传零占位且深层零调用。"""

        model, _, _, invalid = model_fixture
        model.train()
        calls, handles = _register_deep_pre_hooks(model)
        protected_head_calls: List[int] = []
        head_handle = model.protected_fc.register_forward_pre_hook(
            lambda _module, inputs: protected_head_calls.append(inputs[0].shape[0])
        )

        try:
            output = model(_images(3).requires_grad_(True), invalid)
        finally:
            for handle in handles:
                handle.remove()
            head_handle.remove()

        assert isinstance(output, TrainingOutput)
        assert output.decision.allow.tolist() == [False, False, False]
        assert output.protected_logits.shape == (3, 10)
        assert output.protected_logits.requires_grad
        assert torch.count_nonzero(output.protected_logits).item() == 0
        assert calls == {"layer3": [], "layer4": []}
        assert protected_head_calls == []

        # Phase 2 在无 valid 样本时可构造与图相连的零 protected loss。
        output.protected_logits.sum().backward()

    def test_eval_mixed_batch_preserves_indices(self, model_fixture) -> None:
        """mixed batch 的稀疏 logits 必须保留递增的原 batch 索引。"""

        model, _, secret, invalid = model_fixture
        model.eval()
        credentials = np.stack([secret, invalid, secret, invalid])
        calls, handles = _register_deep_pre_hooks(model)

        try:
            with torch.no_grad():
                output = model(_images(4), credentials)
        finally:
            for handle in handles:
                handle.remove()

        assert isinstance(output, InferenceOutput)
        assert output.protected_indices.tolist() == [0, 2]
        assert output.public_indices.tolist() == [1, 3]
        assert output.protected_logits.shape == (2, 10)
        assert output.public_logits.shape == (2, 2)
        assert output.protected_indices.dtype == torch.long
        assert output.public_indices.dtype == torch.long
        assert calls == {"layer3": [2], "layer4": [2]}

    def test_eval_all_invalid_skips_protected_path(self, model_fixture) -> None:
        """全 invalid 推理 batch 必须返回稳定空 protected logits。"""

        model, _, _, invalid = model_fixture
        model.eval()
        calls, handles = _register_deep_pre_hooks(model)
        protected_head_calls: List[int] = []
        head_handle = model.protected_fc.register_forward_hook(
            lambda _module, _inputs, output: protected_head_calls.append(
                output.shape[0]
            )
        )

        try:
            with torch.no_grad():
                output = model(_images(3), invalid)
        finally:
            for handle in handles:
                handle.remove()
            head_handle.remove()

        assert output.protected_logits.shape == (0, 10)
        assert output.protected_indices.shape == (0,)
        assert output.public_logits.shape == (3, 2)
        assert output.public_indices.tolist() == [0, 1, 2]
        assert output.protected_logits.dtype == torch.float32
        assert calls == {"layer3": [], "layer4": []}
        assert protected_head_calls == []

    def test_eval_all_valid_skips_public_head(self, model_fixture) -> None:
        """全 valid 推理 batch 必须单次执行完整深层并返回空 public logits。"""

        model, _, secret, _ = model_fixture
        model.eval()
        calls, handles = _register_deep_pre_hooks(model)
        public_calls: List[int] = []
        head_handle = model.public_fc.register_forward_hook(
            lambda _module, _inputs, output: public_calls.append(output.shape[0])
        )

        try:
            with torch.no_grad():
                output = model(_images(3), secret)
        finally:
            for handle in handles:
                handle.remove()
            head_handle.remove()

        assert output.protected_logits.shape == (3, 10)
        assert output.protected_indices.tolist() == [0, 1, 2]
        assert output.public_logits.shape == (0, 2)
        assert output.public_indices.shape == (0,)
        assert calls == {"layer3": [3], "layer4": [3]}
        assert public_calls == []

    def test_malformed_credential_fails_closed(self, model_fixture) -> None:
        """credential 请求级错误应整批 deny 且深层零调用。"""

        model, _, _, _ = model_fixture
        model.eval()
        calls, handles = _register_deep_pre_hooks(model)

        try:
            with torch.no_grad():
                output = model(_images(2), "malformed")
        finally:
            for handle in handles:
                handle.remove()

        assert output.decision.allow.tolist() == [False, False]
        assert output.protected_logits.shape == (0, 10)
        assert output.public_indices.tolist() == [0, 1]
        assert calls == {"layer3": [], "layer4": []}

    def test_valid_logits_match_direct_protected_path(self, model_fixture) -> None:
        """valid 推理 logits 应与同一权重的直接 protected path 完全一致。"""

        model, _, secret, _ = model_fixture
        model.eval()
        images = _images(2)

        with torch.no_grad():
            routed = model(images, secret)
            shallow = model._forward_shallow(images)
            direct = model._forward_protected(shallow)

        assert torch.equal(routed.protected_logits, direct)

    def test_output_dataclasses_are_frozen(self, model_fixture) -> None:
        """结构化输出必须不可变，避免调用方篡改路由映射。"""

        model, _, secret, _ = model_fixture
        model.eval()
        with torch.no_grad():
            output = model(_images(1), secret)

        with pytest.raises(FrozenInstanceError):
            output.public_logits = torch.ones(1, 2)


class TestGradients:
    """验证软门控路径和两个分类 head 的梯度传播。"""

    def test_both_heads_receive_gradients_with_valid_sample(
        self, model_fixture
    ) -> None:
        """batch 含 valid 样本时浅层及两个 head 都应获得梯度。"""

        model, _, secret, invalid = model_fixture
        model.train()
        images = _images(2).requires_grad_(True)
        output = model(images, np.stack([secret, invalid]))
        valid_mask = output.decision.allow

        protected_loss = output.protected_logits[valid_mask].sum()
        public_loss = output.public_logits.sum()
        (protected_loss + public_loss).backward()

        assert images.grad is not None
        assert model.conv1.weight.grad is not None
        assert model.protected_fc.weight.grad is not None
        assert model.public_fc.weight.grad is not None
        assert list(model.gate_layer.parameters()) == []


class TestInputValidation:
    """验证图像输入在任何网络计算前被严格检查。"""

    @pytest.mark.parametrize(
        "image",
        [
            np.ones((1, 3, 32, 32), dtype=np.float32),
            torch.ones((3, 32, 32), dtype=torch.float32),
            torch.ones((1, 1, 32, 32), dtype=torch.float32),
            torch.ones((1, 3, 28, 28), dtype=torch.float32),
            torch.ones((1, 3, 32, 32), dtype=torch.float64),
            torch.full((1, 3, 32, 32), float("nan")),
            torch.full((1, 3, 32, 32), float("inf")),
        ],
    )
    def test_invalid_images_are_rejected(self, model_fixture, image) -> None:
        """错误类型、shape、dtype 和非有限图像必须稳定拒绝。"""

        model, _, secret, _ = model_fixture

        with pytest.raises((TypeError, ValueError)):
            model(image, secret)

    def test_empty_batch_rejected_before_layers(self, model_fixture) -> None:
        """B=0 必须在 stem、Gate 和两个 head 执行前被拒绝。"""

        model, params, _, _ = model_fixture
        calls: List[str] = []
        handles = [
            model.conv1.register_forward_hook(
                lambda _module, _inputs, _output: calls.append("conv1")
            ),
            model.gate_layer.register_forward_hook(
                lambda _module, _inputs, _output: calls.append("gate")
            ),
            model.protected_fc.register_forward_hook(
                lambda _module, _inputs, _output: calls.append("protected")
            ),
            model.public_fc.register_forward_hook(
                lambda _module, _inputs, _output: calls.append("public")
            ),
        ]

        try:
            with pytest.raises(ValueError):
                model(
                    torch.empty((0, 3, 32, 32), dtype=torch.float32),
                    torch.empty((0, params.n), dtype=torch.float32),
                )
        finally:
            for handle in handles:
                handle.remove()

        assert calls == []

    def test_image_device_mismatch_is_rejected(self, model_fixture) -> None:
        """图像与模型 device 不一致时必须在数值检查前拒绝。"""

        model, _, secret, _ = model_fixture
        meta_image = torch.empty((1, 3, 32, 32), device="meta")

        with pytest.raises(ValueError):
            model(meta_image, secret)
