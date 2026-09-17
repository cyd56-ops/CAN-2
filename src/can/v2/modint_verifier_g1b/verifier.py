"""G1-b 无参数 PyTorch verifier module。"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from ..modint_verifier_g1a.types import G1AParameters
from .kernel import verify_tensor_batch
from .types import G1BResult


class ModIntNeuralVerifier(nn.Module):
    """在计算图中执行固定整数关系判定的无参数 module。"""

    def __init__(self, parameters: G1AParameters) -> None:
        """注册冻结公开参数；不创建可训练 Parameter。"""
        super().__init__()
        self._parameters_ref = parameters
        self.register_buffer("public_A", torch.from_numpy(np.array(parameters.A, dtype="<i8", order="C", copy=True)), persistent=True)
        self.register_buffer("public_b", torch.from_numpy(np.array(parameters.b, dtype="<i8", order="C", copy=True)), persistent=True)
        self.requires_grad_(False)

    @property
    def parameters_ref(self) -> G1AParameters:
        """返回绑定的不可变 G1-a 参数。"""
        return self._parameters_ref

    def forward(self, credentials: object) -> G1BResult:
        """对 canonical credential 执行 CPU int64 verifier 并返回离散 evidence。"""
        return verify_tensor_batch(credentials, self._parameters_ref)
