"""预训练宿主插件的稳定边界类型。

G0 仅定义并用离线 tiny-host 验证该边界；真实 Qwen2 插件须在 P0 固定模型与
Transformers 版本后实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Tuple, runtime_checkable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HostSpec:
    """记录经核查的宿主结构与执行配置。"""

    model_type: str
    model_revision: str
    model_sha256: str
    num_hidden_layers: int
    cut_layer: int
    tied_embeddings: bool
    supports_kv_cache: bool
    attention_backend: str
    dtype: str

    def __post_init__(self) -> None:
        """拒绝空标识、类型混淆和非法 cut。"""

        for name in (
            "model_type",
            "model_revision",
            "model_sha256",
            "attention_backend",
            "dtype",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} 必须是非空字符串")
        if (
            isinstance(self.num_hidden_layers, bool)
            or not isinstance(self.num_hidden_layers, int)
            or self.num_hidden_layers < 2
        ):
            raise ValueError("num_hidden_layers 必须是至少 2 的整数")
        if (
            isinstance(self.cut_layer, bool)
            or not isinstance(self.cut_layer, int)
            or not 1 <= self.cut_layer < self.num_hidden_layers
        ):
            raise ValueError("cut_layer 必须位于完整 block 边界内")
        if not isinstance(self.tied_embeddings, bool) or not isinstance(
            self.supports_kv_cache, bool
        ):
            raise TypeError("宿主 capability 标志必须是 bool")


@dataclass(frozen=True)
class PrefixState:
    """保存 suffix 继续执行所需的完整逻辑状态。"""

    hidden: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    request_ids: Tuple[str, ...]
    host_state: Any = None

    def __post_init__(self) -> None:
        """验证 hidden、mask、position 与请求 batch 对齐。"""

        if not isinstance(self.hidden, Tensor) or self.hidden.ndim != 3:
            raise ValueError("hidden 必须是 Tensor[B,T,D]")
        if not self.hidden.dtype.is_floating_point:
            raise TypeError("hidden 必须使用浮点 dtype")
        if (
            not isinstance(self.attention_mask, Tensor)
            or self.attention_mask.dtype != torch.bool
        ):
            raise TypeError("attention_mask 必须是 BoolTensor")
        if (
            not isinstance(self.position_ids, Tensor)
            or self.position_ids.dtype != torch.long
        ):
            raise TypeError("position_ids 必须是 LongTensor")
        expected = self.hidden.shape[:2]
        if self.attention_mask.shape != expected or self.position_ids.shape != expected:
            raise ValueError("mask/position_ids 必须与 hidden 的 [B,T] 对齐")
        if len(self.request_ids) != self.hidden.shape[0]:
            raise ValueError("request_ids 必须与 hidden batch 对齐")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request_ids 必须唯一")
        if (
            self.attention_mask.device != self.hidden.device
            or self.position_ids.device != self.hidden.device
        ):
            raise ValueError("prefix 状态 Tensor 必须位于同一 device")


@runtime_checkable
class HostPlugin(Protocol):
    """描述 IdentitySplit 与 Gate adapter 共用的最小宿主接口。"""

    @property
    def spec(self) -> HostSpec:
        """返回冻结宿主规格。"""

    def prefix(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        request_ids: Tuple[str, ...],
    ) -> PrefixState:
        """执行 embedding 与 cut 前 blocks。"""

    def protected_suffix(
        self,
        state: PrefixState,
        request_ids: Tuple[str, ...],
    ) -> Any:
        """对合法子批执行 cut 后 blocks、norm 与原 head。"""

    def select_prefix_state(
        self,
        state: PrefixState,
        indices: Tensor,
        request_ids: Tuple[str, ...],
    ) -> PrefixState:
        """按原 batch 索引选取 suffix 所需的完整状态。"""


class TinyDecoderHost(nn.Module):
    """用于 G0 CPU 验收的最小 decoder-only prefix/suffix 宿主。

    该宿主只验证切分、恒等 hidden 和调用边界，不模拟 Qwen2 的内部 API，
    也不承担真实语言能力或 KV-cache 兼容性结论。
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 8,
        num_layers: int = 4,
        cut_layer: int = 2,
        model_revision: str = "tiny-g0-v1",
        call_observer: Optional[Callable[[str, str, Tuple[str, ...]], None]] = None,
    ) -> None:
        """构造确定性微型宿主。

        参数:
            vocab_size: token 词表大小。
            d_model: hidden 宽度。
            num_layers: 线性 block 总数。
            cut_layer: prefix 使用的 block 数量。
            model_revision: 绑定到 HostSpec 的本地版本标识。
            call_observer: 可选测试观测器，在真实模块执行点记录调用。
        """

        super().__init__()
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (vocab_size, d_model, num_layers)
        ):
            raise ValueError("vocab_size/d_model/num_layers 必须是正整数")
        if isinstance(cut_layer, bool) or not isinstance(cut_layer, int):
            raise TypeError("cut_layer 必须是整数")
        if not 1 <= cut_layer < num_layers:
            raise ValueError("cut_layer 必须位于完整 block 边界内")
        self._spec = HostSpec(
            model_type="tiny_decoder",
            model_revision=model_revision,
            model_sha256="local-tiny-host",
            num_hidden_layers=num_layers,
            cut_layer=cut_layer,
            tied_embeddings=False,
            supports_kv_cache=False,
            attention_backend="torch-linear-fixture",
            dtype="float32",
        )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if call_observer is not None and not callable(call_observer):
            raise TypeError("call_observer 必须可调用或为 None")
        self._call_observer = call_observer
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def spec(self) -> HostSpec:
        """返回冻结的 tiny 宿主规格。"""

        return self._spec

    def prefix(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        request_ids: Tuple[str, ...],
    ) -> PrefixState:
        """执行 embedding 和 cut 前 block，并返回完整逻辑状态。"""

        self._validate_inputs(input_ids, attention_mask, request_ids)
        self._record("embedding", "prefill", request_ids)
        hidden = self.embedding(input_ids)
        for index, block in enumerate(self.blocks[: self.spec.cut_layer]):
            self._record(f"block.{index}", "prefill", request_ids)
            hidden = torch.tanh(block(hidden))
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).expand(input_ids.shape[0], -1)
        return PrefixState(hidden, attention_mask, position_ids, request_ids)

    def protected_suffix(
        self,
        state: PrefixState,
        request_ids: Tuple[str, ...],
    ) -> Tensor:
        """执行 cut 后 block、norm 和原 lm_head。"""

        if not isinstance(state, PrefixState):
            raise TypeError("state 必须是 PrefixState")
        if tuple(request_ids) != state.request_ids:
            raise PermissionError("suffix request IDs 与 prefix 状态不匹配")
        hidden = state.hidden
        for index, block in enumerate(
            self.blocks[self.spec.cut_layer :], start=self.spec.cut_layer
        ):
            self._record(f"block.{index}", "protected", request_ids)
            hidden = torch.tanh(block(hidden))
        self._record("norm", "protected", request_ids)
        hidden = self.norm(hidden)
        self._record("lm_head", "protected", request_ids)
        return self.lm_head(hidden)

    def select_prefix_state(
        self,
        state: PrefixState,
        indices: Tensor,
        request_ids: Tuple[str, ...],
    ) -> PrefixState:
        """按索引选取 tiny-host 的 hidden、mask 与 position 状态。"""

        if not isinstance(state, PrefixState):
            raise TypeError("state 必须是 PrefixState")
        if (
            not isinstance(indices, Tensor)
            or indices.dtype != torch.long
            or indices.ndim != 1
        ):
            raise TypeError("indices 必须是一维 LongTensor")
        expected_ids = tuple(
            state.request_ids[int(index)] for index in indices.tolist()
        )
        if expected_ids != request_ids:
            raise PermissionError("选取索引与 request IDs 不一致")
        return PrefixState(
            hidden=state.hidden.index_select(0, indices),
            attention_mask=state.attention_mask.index_select(0, indices),
            position_ids=state.position_ids.index_select(0, indices),
            request_ids=request_ids,
            host_state=None,
        )

    def _record(self, component: str, stage: str, request_ids: Tuple[str, ...]) -> None:
        """只在真实模块调用点通知受信测试观测器。"""

        if self._call_observer is not None:
            self._call_observer(component, stage, request_ids)

    @staticmethod
    def _validate_inputs(
        input_ids: Tensor,
        attention_mask: Tensor,
        request_ids: Tuple[str, ...],
    ) -> None:
        """验证 tiny 宿主的输入 dtype、shape 和请求身份。"""

        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 Tensor[B,T]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids 必须是 LongTensor")
        if (
            not isinstance(attention_mask, Tensor)
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError("attention_mask 必须与 input_ids 对齐")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask 必须是 BoolTensor")
        if len(request_ids) != input_ids.shape[0] or len(set(request_ids)) != len(
            request_ids
        ):
            raise ValueError("request_ids 必须与输入 batch 对齐且唯一")


class _TinyCausalBlock(nn.Module):
    """提供真实 K/V 累积的单头 causal attention 测试 block。"""

    def __init__(self, d_model: int) -> None:
        """构造固定宽度的注意力投影。"""

        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward_full(self, hidden: Tensor, valid_mask: Tensor) -> Tensor:
        """使用完整 causal mask 计算一层 hidden。"""

        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_model)
        length = hidden.shape[1]
        causal = torch.tril(
            torch.ones((length, length), dtype=torch.bool, device=hidden.device)
        )
        allowed = causal[None] & valid_mask[:, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value)
        return torch.tanh(hidden + self.output(context))

    def forward_step(
        self,
        hidden: Tensor,
        valid_mask: Tensor,
        past_key: Optional[Tensor],
        past_value: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """追加当前 token K/V，并计算最新位置 hidden。"""

        query = self.query(hidden)
        current_key = self.key(hidden)
        current_value = self.value(hidden)
        key = (
            current_key
            if past_key is None
            else torch.cat((past_key, current_key), dim=1)
        )
        value = (
            current_value
            if past_value is None
            else torch.cat((past_value, current_value), dim=1)
        )
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_model)
        scores = scores.masked_fill(
            ~valid_mask[:, None, :], torch.finfo(scores.dtype).min
        )
        context = torch.matmul(torch.softmax(scores, dim=-1), value)
        return torch.tanh(hidden + self.output(context)), key, value


class TinyKVDecoderHost(nn.Module):
    """验证无 cache 与真实增量 K/V 等价性的离线 tiny decoder。"""

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 8,
        num_layers: int = 3,
        max_length: int = 16,
    ) -> None:
        """构造微型 causal decoder；所有参数冻结且 dropout 为零。"""

        super().__init__()
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in (vocab_size, d_model, num_layers, max_length)
        ):
            raise ValueError("tiny KV 配置必须是至少 2 的整数")
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList(
            [_TinyCausalBlock(d_model) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_length = max_length
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward_full(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """不使用 cache 计算完整 causal logits。"""

        self._validate_inputs(input_ids, attention_mask)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = (
            self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        )
        for block in self.blocks:
            hidden = block.forward_full(hidden, attention_mask)
        return self.lm_head(self.norm(hidden))

    def forward_incremental(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        """逐 token 计算 logits 并返回每层真实 K/V cache。"""

        self._validate_inputs(input_ids, attention_mask)
        batch_size, length = input_ids.shape
        keys: List[Optional[Tensor]] = [None] * len(self.blocks)
        values: List[Optional[Tensor]] = [None] * len(self.blocks)
        outputs = []
        for position in range(length):
            token = input_ids[:, position : position + 1]
            position_ids = torch.full(
                (batch_size, 1), position, dtype=torch.long, device=input_ids.device
            )
            hidden = self.token_embedding(token) + self.position_embedding(position_ids)
            current_mask = attention_mask[:, : position + 1]
            for layer, block in enumerate(self.blocks):
                hidden, keys[layer], values[layer] = block.forward_step(
                    hidden, current_mask, keys[layer], values[layer]
                )
            outputs.append(self.lm_head(self.norm(hidden)))
        assert all(value is not None for value in keys)
        assert all(value is not None for value in values)
        return (
            torch.cat(outputs, dim=1),
            tuple(value for value in keys if value is not None),
            tuple(value for value in values if value is not None),
        )

    def _validate_inputs(self, input_ids: Tensor, attention_mask: Tensor) -> None:
        """验证增量 fixture 的输入及右 padding 约束。"""

        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 Tensor[B,T]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids 必须是 LongTensor")
        if (
            not isinstance(attention_mask, Tensor)
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError("attention_mask 必须与 input_ids 对齐")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask 必须是 BoolTensor")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input batch 和序列不能为空")
        if input_ids.shape[1] > self.max_length:
            raise ValueError("输入超过 tiny KV max_length")
        if not bool(attention_mask[:, 0].all().item()):
            raise ValueError("每条序列首 token 必须有效")
        if bool((~attention_mask[:, :-1] & attention_mask[:, 1:]).any().item()):
            raise ValueError("attention_mask 必须使用右 padding")
