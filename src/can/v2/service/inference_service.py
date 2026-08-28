"""提供只暴露脱敏能力结果的可信进程内推理入口。"""

from typing import List

import torch
from torch import Tensor

from ..models.gated_resnet import GatedResNet18
from .response_envelope import ResponseEnvelope, to_response_envelope


class ServiceExecutionError(RuntimeError):
    """表示不包含内部验证细节的稳定服务执行失败。"""


class InferenceService:
    """封装 ``GatedResNet18`` 并返回固定结构响应信封。"""

    def __init__(
        self,
        model: GatedResNet18,
        device: torch.device,
        max_batch_size: int = 256,
    ) -> None:
        """初始化可信进程内服务入口。

        参数:
            model: 已加载的 GatedResNet18 模型。
            device: 模型和请求执行所在的 PyTorch device。
            max_batch_size: 单次服务请求允许的最大样本数。
        """

        if not isinstance(model, GatedResNet18):
            raise TypeError("model 必须是 GatedResNet18")
        if not isinstance(device, torch.device):
            raise TypeError("device 必须是 torch.device")
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise TypeError("max_batch_size 必须是非 bool 整数")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size 必须大于 0")

        model_devices = {value.device for value in model.parameters()}
        model_devices.update(value.device for value in model.buffers())
        if model_devices != {device}:
            raise ValueError("模型参数和 buffer 必须全部位于指定 device")

        self.model = model.eval()
        self.device = device
        self.max_batch_size = max_batch_size

    def _validate_request(self, images: object, credentials: object) -> int:
        """在模型执行前验证不可信图像和 credential batch。

        参数:
            images: 预期为 float32 Tensor[B, 3, 32, 32]。
            credentials: 预期为 float32 Tensor[B, n]。

        返回:
            验证后的 batch 大小。
        """

        if not isinstance(images, Tensor):
            raise TypeError("images 必须是 Tensor")
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError("images 必须具有 shape [B, 3, 32, 32]")
        if images.dtype != torch.float32:
            raise TypeError("images 必须是 float32")
        if not bool(torch.isfinite(images).all().item()):
            raise ValueError("images 必须全部有限")

        if not isinstance(credentials, Tensor):
            raise TypeError("credentials 必须是 Tensor")
        if credentials.ndim != 2:
            raise ValueError("credentials 必须具有 shape [B, n]")
        if credentials.dtype != torch.float32:
            raise TypeError("credentials 必须是 float32")
        if not bool(torch.isfinite(credentials).all().item()):
            raise ValueError("credentials 必须全部有限")

        batch_size = int(images.shape[0])
        if batch_size < 1 or batch_size > self.max_batch_size:
            raise ValueError("batch_size 必须位于 [1, max_batch_size]")
        if credentials.shape[0] != batch_size:
            raise ValueError("credentials 行数必须与 images batch 一致")
        if credentials.shape[1] != self.model.gate_layer.verifier.n:
            raise ValueError("credentials 维度与模型 LWE 参数不一致")
        return batch_size

    @torch.inference_mode()
    def infer(self, images: Tensor, credentials: Tensor) -> List[ResponseEnvelope]:
        """验证真实 credential 并返回不含内部验证证据的响应。

        参数:
            images: float32 CIFAR 图像 Tensor[B, 3, 32, 32]。
            credentials: 调用方提供的真实 float32 credential Tensor[B, n]。

        返回:
            按原 batch 顺序排列的固定结构响应列表。

        异常:
            TypeError: 请求字段类型或 dtype 错误。
            ValueError: 请求 shape、有限性或大小错误。
            ServiceExecutionError: 模型执行或内部输出契约失败。
        """

        batch_size = self._validate_request(images, credentials)
        try:
            device_images = images.to(self.device)
            device_credentials = credentials.to(self.device)
            output = self.model(device_images, device_credentials)
            return to_response_envelope(output, batch_size)
        except Exception:
            # 退出异常处理块后再抛出，避免通过 __cause__/__context__ 泄露内部异常。
            pass
        raise ServiceExecutionError("服务推理失败") from None
