"""将模型稀疏推理结果转换为不暴露内部验证证据的响应信封。"""

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
from torch import Tensor

from ..models.gated_resnet import InferenceOutput

PROTECTED_CLASSES = 10
PUBLIC_CLASSES = 2


@dataclass(frozen=True)
class ResponseEnvelope:
    """保存单样本的固定结构服务响应。

    参数:
        probabilities: 长度固定为 10 的不可变概率元组。public 响应只使用
            前两个位置，其余八个位置必须精确为零。
        prediction: 当前能力标签空间内的预测类别。
        capability_level: 当前返回的是 protected 还是 public 能力。
    """

    probabilities: Tuple[float, ...]
    prediction: int
    capability_level: Literal["protected", "public"]

    def __post_init__(self) -> None:
        """验证响应字段的类型、范围和固定结构不变量。"""

        if self.capability_level not in ("protected", "public"):
            raise ValueError("capability_level 不受支持")
        if not isinstance(self.probabilities, tuple):
            raise TypeError("probabilities 必须是 tuple")
        if len(self.probabilities) != PROTECTED_CLASSES:
            raise ValueError("probabilities 长度必须为 10")
        if not all(
            isinstance(value, float) and math.isfinite(value) and 0.0 <= value <= 1.0
            for value in self.probabilities
        ):
            raise ValueError("probabilities 必须是 [0, 1] 内的有限 float")
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("probabilities 之和必须为 1")
        if isinstance(self.prediction, bool) or not isinstance(self.prediction, int):
            raise TypeError("prediction 必须是非 bool 整数")

        class_count = (
            PROTECTED_CLASSES
            if self.capability_level == "protected"
            else PUBLIC_CLASSES
        )
        if not 0 <= self.prediction < class_count:
            raise ValueError("prediction 超出当前能力的类别范围")
        if self.capability_level == "public" and any(
            value != 0.0 for value in self.probabilities[PUBLIC_CLASSES:]
        ):
            raise ValueError("public probabilities 的后 8 位必须为 0")


def _validate_route(
    logits: object,
    indices: object,
    expected_classes: int,
    route_name: str,
) -> Tuple[Tensor, Tensor]:
    """验证单条稀疏路由的 logits 与 indices 契约。

    参数:
        logits: 待验证的二维浮点 logits。
        indices: logits 对应的原 batch 一维索引。
        expected_classes: 当前分类 head 的固定类别数。
        route_name: 用于内部异常定位的路由名称。

    返回:
        验证后的 ``(logits, indices)`` Tensor。
    """

    if not isinstance(logits, Tensor):
        raise TypeError(f"{route_name}_logits 必须是 Tensor")
    if logits.ndim != 2 or logits.shape[1] != expected_classes:
        raise ValueError(f"{route_name}_logits 必须具有 shape [N, {expected_classes}]")
    if logits.dtype != torch.float32:
        raise TypeError(f"{route_name}_logits 必须是 float32")
    if logits.device.type == "meta":
        raise ValueError(f"{route_name}_logits 不支持 meta device")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError(f"{route_name}_logits 必须全部有限")

    if not isinstance(indices, Tensor):
        raise TypeError(f"{route_name}_indices 必须是 Tensor")
    if indices.ndim != 1:
        raise ValueError(f"{route_name}_indices 必须是一维")
    if indices.dtype != torch.long:
        raise TypeError(f"{route_name}_indices 必须是 LongTensor")
    if indices.device.type == "meta":
        raise ValueError(f"{route_name}_indices 不支持 meta device")
    if logits.shape[0] != indices.shape[0]:
        raise ValueError(f"{route_name} logits 与 indices 行数不一致")
    if logits.device != indices.device:
        raise ValueError(f"{route_name} logits 与 indices device 不一致")
    return logits, indices


def _as_probabilities(logits: Tensor, capability_level: str) -> Tuple[float, ...]:
    """将单样本 logits 转为固定长度 10 的不可变概率元组。

    参数:
        logits: 单样本一维 logits。
        capability_level: ``protected`` 或 ``public``。

    返回:
        与 PyTorch storage 解耦的 Python float 元组。
    """

    route_probabilities = torch.softmax(logits, dim=0)
    probabilities = torch.zeros(
        PROTECTED_CLASSES,
        dtype=route_probabilities.dtype,
        device=route_probabilities.device,
    )
    if capability_level == "protected":
        probabilities.copy_(route_probabilities)
    else:
        probabilities[:PUBLIC_CLASSES].copy_(route_probabilities)
    return tuple(float(value) for value in probabilities.detach().cpu().tolist())


def to_response_envelope(
    output: InferenceOutput, batch_size: int
) -> List[ResponseEnvelope]:
    """将内部 ``InferenceOutput`` 转换为按原 batch 排序的公开响应。

    参数:
        output: 模型产生的稀疏 protected/public 推理结果。
        batch_size: 原始请求 batch 大小，允许零值供转换边界测试使用。

    返回:
        长度等于 ``batch_size`` 的响应列表。

    异常:
        TypeError: 输出、索引或 logits 类型不符合契约。
        ValueError: shape、device、有限性或索引覆盖不符合契约。
    """

    if not isinstance(output, InferenceOutput):
        raise TypeError("output 必须是 InferenceOutput")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size 必须是非 bool 整数")
    if batch_size < 0:
        raise ValueError("batch_size 不能为负数")

    protected_logits, protected_indices = _validate_route(
        output.protected_logits,
        output.protected_indices,
        PROTECTED_CLASSES,
        "protected",
    )
    public_logits, public_indices = _validate_route(
        output.public_logits,
        output.public_indices,
        PUBLIC_CLASSES,
        "public",
    )
    if protected_logits.device != public_logits.device:
        raise ValueError("protected/public 路由必须位于同一 device")

    # 长度与全集同时校验，拒绝重复、缺失、负数和越界索引。
    all_indices = torch.cat((protected_indices, public_indices))
    index_values = all_indices.detach().cpu().tolist()
    if len(index_values) != batch_size or set(index_values) != set(range(batch_size)):
        raise ValueError("路由 indices 必须无重复地完整覆盖原 batch")

    envelopes: List[Optional[ResponseEnvelope]] = [None] * batch_size
    for index, logits in zip(
        protected_indices.detach().cpu().tolist(), protected_logits
    ):
        envelopes[index] = ResponseEnvelope(
            probabilities=_as_probabilities(logits, "protected"),
            prediction=int(torch.argmax(logits).item()),
            capability_level="protected",
        )
    for index, logits in zip(public_indices.detach().cpu().tolist(), public_logits):
        envelopes[index] = ResponseEnvelope(
            probabilities=_as_probabilities(logits, "public"),
            prediction=int(torch.argmax(logits).item()),
            capability_level="public",
        )

    # 前置的行数与索引全集校验保证每个位置恰好写入一次。
    return [envelope for envelope in envelopes if envelope is not None]
