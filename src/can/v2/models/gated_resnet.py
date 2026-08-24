"""在 CIFAR ResNet-18 中集成 Neural Gate Layer 的条件路由模型。"""

from dataclasses import dataclass
from typing import List, Type, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..crypto.lwe import LWEParams
from ..layers.gate_layer import AuthorizationDecision, GateLayer


@dataclass(frozen=True)
class TrainingOutput:
    """保存训练态完整 batch 输出及其授权决定。

    参数:
        protected_logits: 受保护分类 logits，shape 为 ``[B, protected_classes]``；
            未授权行是零占位，不得用于 protected loss 或指标。
        public_logits: 公开分类 logits，shape 为 ``[B, public_classes]``。
        decision: Gate Layer 提交的批量授权决定。
    """

    protected_logits: Tensor
    public_logits: Tensor
    decision: AuthorizationDecision


@dataclass(frozen=True)
class InferenceOutput:
    """保存推理态稀疏路由输出及其原 batch 索引。

    参数:
        protected_logits: valid 子批的受保护 logits。
        protected_indices: protected_logits 对应的原 batch 索引。
        public_logits: invalid 子批的公开 logits。
        public_indices: public_logits 对应的原 batch 索引。
        decision: Gate Layer 提交的批量授权决定。
    """

    protected_logits: Tensor
    protected_indices: Tensor
    public_logits: Tensor
    public_indices: Tensor
    decision: AuthorizationDecision


class BasicBlock(nn.Module):
    """实现 CIFAR ResNet-18 使用的标准两层残差块。"""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        """初始化残差块。

        参数:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            stride: 第一层卷积的步幅。
            downsample: 对齐残差分支 shape 的可选模块。
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """执行残差块前向传播。

        参数:
            x: 输入特征图。

        返回:
            与主分支和残差分支相加后的特征图。
        """

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class GatedResNet18(nn.Module):
    """将 Gate Layer 集成到 CIFAR ResNet-18 并执行凭据条件路由。"""

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        params: LWEParams,
        num_classes_protected: int = 10,
        num_classes_public: int = 2,
        temperature: float = 5.0,
    ) -> None:
        """初始化 Gated ResNet-18。

        参数:
            A: LWE 公钥矩阵，shape 为 ``[m, n]``。
            b: LWE 公钥向量，shape 为 ``[m]``。
            params: LWE 参数。
            num_classes_protected: 受保护 head 的类别数。
            num_classes_public: 公开 head 的类别数。
            temperature: Gate Layer 训练态软门控温度。
        """

        super().__init__()
        self.num_classes_protected = self._validate_class_count(
            num_classes_protected, "num_classes_protected"
        )
        self.num_classes_public = self._validate_class_count(
            num_classes_public, "num_classes_public"
        )

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, blocks=2, stride=2)
        self.gate_layer = GateLayer(A, b, params, temperature=temperature)
        self.layer3 = self._make_layer(BasicBlock, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.protected_fc = nn.Linear(512, self.num_classes_protected)
        self.public_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.public_fc = nn.Linear(128, self.num_classes_public)

        self._initialize_weights()

    @staticmethod
    def _validate_class_count(value: int, name: str) -> int:
        """验证分类 head 的类别数。

        参数:
            value: 待验证的类别数。
            name: 用于异常消息的参数名。

        返回:
            验证后的正整数类别数。
        """

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} 必须是非 bool 的整数")
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0")
        return value

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """构造一个 ResNet stage。

        参数:
            block: 残差块类型。
            out_channels: stage 输出通道数。
            blocks: stage 内残差块数量。
            stride: 第一个残差块的步幅。

        返回:
            顺序连接的残差 stage。
        """

        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers: List[nn.Module] = [
            block(self.in_channels, out_channels, stride, downsample)
        ]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """使用标准 ResNet 规则初始化卷积、BatchNorm 和线性层。"""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def _validate_input(self, x: Tensor) -> None:
        """在任何网络层执行前严格验证 CIFAR 图像 batch。

        参数:
            x: 预期为 ``float32 Tensor[B, 3, 32, 32]`` 的图像 batch。
        """

        if not isinstance(x, Tensor):
            raise TypeError("x 必须是 torch.Tensor")
        if x.ndim != 4:
            raise ValueError("x 必须是四维 Tensor[B, 3, 32, 32]")
        if x.shape[0] == 0:
            raise ValueError("x 的 batch 不能为空")
        if tuple(x.shape[1:]) != (3, 32, 32):
            raise ValueError("x 必须具有 shape [B, 3, 32, 32]")
        if x.dtype != torch.float32:
            raise TypeError("x 的 dtype 必须是 torch.float32")
        if x.device != self.conv1.weight.device:
            raise ValueError("x 与模型参数必须位于相同 device")
        if x.device != self.gate_layer.verifier.A.device:
            raise ValueError("x 与 Gate Layer buffer 必须位于相同 device")
        if not bool(torch.isfinite(x).all().item()):
            raise ValueError("x 必须全部为有限数值")

    def _forward_shallow(self, x: Tensor) -> Tensor:
        """提取 layer2 后的 CIFAR 浅层特征。"""

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        return self.layer2(x)

    def _forward_protected(self, features: Tensor) -> Tensor:
        """对已授权子批执行深层网络和受保护分类 head。"""

        deep_features = self.layer4(self.layer3(features))
        return self.protected_fc(self.avgpool(deep_features).flatten(1))

    def _forward_public(self, features: Tensor) -> Tensor:
        """对指定浅层特征执行公开分类 head。"""

        return self.public_fc(self.public_pool(features).flatten(1))

    def _protected_placeholders(self, shallow_features: Tensor) -> Tensor:
        """创建与浅层计算图相连的完整 batch 零占位 logits。"""

        linked_zeros = shallow_features.sum(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return linked_zeros.expand(-1, self.num_classes_protected) * 0.0

    def forward(
        self, x: Tensor, credential: Union[Tensor, np.ndarray]
    ) -> Union[TrainingOutput, InferenceOutput]:
        """验证 credential 并执行训练态或推理态条件路由。

        参数:
            x: ``float32 Tensor[B, 3, 32, 32]`` CIFAR 图像。
            credential: 单个 ``[n]`` 或批量 ``[B, n]`` LWE credential。

        返回:
            训练态返回完整 batch 的 ``TrainingOutput``；推理态返回携带
            原 batch 索引的 ``InferenceOutput``。
        """

        self._validate_input(x)
        shallow_features = self._forward_shallow(x)
        gated_features, decision = self.gate_layer(shallow_features, credential)

        valid_indices = torch.nonzero(decision.allow, as_tuple=False).flatten()
        invalid_indices = torch.nonzero(~decision.allow, as_tuple=False).flatten()

        if self.training:
            # 未授权行保持零占位，且不会进入深层 BatchNorm 或 protected head。
            protected_logits = self._protected_placeholders(shallow_features)
            if valid_indices.numel() > 0:
                valid_features = gated_features.index_select(0, valid_indices)
                valid_logits = self._forward_protected(valid_features)
                protected_logits = protected_logits.index_copy(
                    0, valid_indices, valid_logits
                )

            public_logits = self._forward_public(shallow_features)
            return TrainingOutput(protected_logits, public_logits, decision)

        # 推理态只执行被选中的路径；空路由返回 shape 稳定的二维空 Tensor。
        if valid_indices.numel() > 0:
            valid_features = gated_features.index_select(0, valid_indices)
            protected_logits = self._forward_protected(valid_features)
        else:
            protected_logits = shallow_features.new_empty(
                (0, self.num_classes_protected)
            )

        if invalid_indices.numel() > 0:
            invalid_features = shallow_features.index_select(0, invalid_indices)
            public_logits = self._forward_public(invalid_features)
        else:
            public_logits = shallow_features.new_empty((0, self.num_classes_public))

        return InferenceOutput(
            protected_logits=protected_logits,
            protected_indices=valid_indices,
            public_logits=public_logits,
            public_indices=invalid_indices,
            decision=decision,
        )
