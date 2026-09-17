"""M1a 冻结 shared/protected Expert 实现。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .types import CallLedger


class _FrozenExpert(nn.Module):
    """带真实调用计数的冻结 tiny FFN Expert。"""

    def __init__(self, d_model: int, seed: int, expert_id: str, kind: str) -> None:
        """按固定 seed 初始化两层 FFN，并关闭梯度。"""

        super().__init__()
        if d_model < 1 or not isinstance(seed, int):
            raise ValueError("d_model/seed 参数非法")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.linear1 = nn.Linear(d_model, 2 * d_model, bias=False)
        self.linear2 = nn.Linear(2 * d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear1.weight.copy_(
                torch.randn(self.linear1.weight.shape, generator=generator)
                / d_model**0.5
            )
            self.linear2.weight.copy_(
                torch.randn(self.linear2.weight.shape, generator=generator)
                / (2 * d_model) ** 0.5
            )
        self.expert_id = expert_id
        self.kind = kind
        self.forward_calls = 0
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        hidden: Tensor,
        request_ids: Tuple[str, ...],
        ledger: Optional[CallLedger] = None,
        batch_indices: Optional[Tuple[int, ...]] = None,
        case_id: str = "default",
    ) -> Tensor:
        """执行冻结 FFN；ledger 记录真实调用的原始 batch 索引。"""

        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise ValueError("hidden 必须是 Tensor[B,T,D]")
        if (
            hidden.shape[-1] != self.linear1.in_features
            or not hidden.dtype.is_floating_point
        ):
            raise ValueError("hidden 宽度或 dtype 与 Expert 不匹配")
        if not bool(torch.isfinite(hidden).all().item()):
            raise ValueError("hidden 必须全部有限")
        if len(request_ids) != hidden.shape[0]:
            raise ValueError("request_ids 与 hidden batch 不一致")
        if batch_indices is None:
            batch_indices = tuple(range(hidden.shape[0]))
        self.forward_calls += 1
        if ledger is not None:
            ledger.record(case_id, self.expert_id, self.kind, batch_indices)
        return self.linear2(torch.nn.functional.gelu(self.linear1(hidden)))


class SharedExpert(_FrozenExpert):
    """始终执行的通用 Shared E0。"""

    def __init__(self, d_model: int = 16, seed: int = 1001) -> None:
        """创建冻结 shared expert。"""

        super().__init__(d_model, seed, "E0", "shared")


class ProtectedExpert(_FrozenExpert):
    """仅授权行可达的 Protected Routed E1。"""

    def __init__(self, d_model: int = 16, seed: int = 1002) -> None:
        """创建冻结 protected expert。"""

        super().__init__(d_model, seed, "E1", "routed")
