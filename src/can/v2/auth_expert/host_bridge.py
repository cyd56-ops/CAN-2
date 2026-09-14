"""复用 G0 tiny host 算子的 M0 prefix/suffix bridge。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..pretrained_gate.host import (
    HostPlugin,
    HostSpec,
    PrefixState,
    TinyDecoderHost,
    TinyKVDecoderHost,
)


class M0TinyHostBridge:
    """把 tiny host 的 prefix 与完整 protected suffix 暴露给 M0 Dispatcher。"""

    def __init__(self, host: HostPlugin, cut_layer: Optional[int] = None) -> None:
        """绑定冻结 tiny host；不复制注意力数学。"""

        if not isinstance(host, (TinyDecoderHost, TinyKVDecoderHost)):
            raise TypeError("host 必须实现 HostPlugin")
        if isinstance(cut_layer, bool) or (
            cut_layer is not None and (not isinstance(cut_layer, int) or cut_layer < 1)
        ):
            raise ValueError("cut_layer 必须是正整数或 None")
        if isinstance(host, TinyKVDecoderHost) and cut_layer is None:
            raise ValueError("TinyKVDecoderHost 必须显式提供 cut_layer")
        if isinstance(host, TinyKVDecoderHost) and cut_layer >= len(host.blocks):
            raise ValueError("cut_layer 必须小于 KV block 数")
        self.host = host
        self.cut_layer = cut_layer

    @property
    def spec(self) -> Optional[HostSpec]:
        """返回底层宿主规格。"""

        return getattr(self.host, "spec", None)

    def prefix_full(
        self, input_ids: Tensor, attention_mask: Tensor, request_ids: Tuple[str, ...]
    ) -> PrefixState:
        """执行 G0 prefix，并校验请求身份未被宿主改变。"""

        if isinstance(self.host, TinyKVDecoderHost):
            self.host._validate_inputs(input_ids, attention_mask)
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            hidden = (
                self.host.token_embedding(input_ids)
                + self.host.position_embedding(positions)[None]
            )
            assert self.cut_layer is not None
            for block in self.host.blocks[: self.cut_layer]:
                hidden = block.forward_full(hidden, attention_mask)
            state = PrefixState(
                hidden,
                attention_mask,
                positions.expand(input_ids.shape[0], -1),
                request_ids,
            )
        else:
            state = self.host.prefix(input_ids, attention_mask, request_ids)
        if state.request_ids != request_ids:
            raise PermissionError("host prefix 改变 request IDs")
        return state

    def suffix_full(self, state: PrefixState, request_ids: Tuple[str, ...]) -> Tensor:
        """执行已授权的 protected suffix/head。"""

        if isinstance(self.host, TinyKVDecoderHost):
            if state.request_ids != request_ids:
                raise PermissionError("suffix request IDs 与 prefix 状态不匹配")
            assert self.cut_layer is not None
            cut = self.cut_layer
            output, _, _ = self.suffix_with_cache(state, request_ids)
            return output
        return self.host.protected_suffix(state, request_ids)

    def suffix_with_cache(
        self, state: PrefixState, request_ids: Tuple[str, ...]
    ) -> Tuple[Tensor, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        """执行 TinyKV protected suffix，并返回本次生成的真实 K/V。"""

        if not isinstance(self.host, TinyKVDecoderHost):
            return self.host.protected_suffix(state, request_ids), (), ()
        if state.request_ids != request_ids:
            raise PermissionError("suffix request IDs 与 prefix 状态不匹配")
        assert self.cut_layer is not None
        keys = [None] * (len(self.host.blocks) - self.cut_layer)
        values = [None] * (len(self.host.blocks) - self.cut_layer)
        outputs = []
        for position in range(state.hidden.shape[1]):
            hidden = state.hidden[:, position : position + 1]
            valid_mask = state.attention_mask[:, : position + 1]
            for index, block in enumerate(self.host.blocks[self.cut_layer :]):
                hidden, keys[index], values[index] = block.forward_step(
                    hidden, valid_mask, keys[index], values[index]
                )
            outputs.append(self.host.lm_head(self.host.norm(hidden)))
        return (
            torch.cat(outputs, dim=1),
            tuple(value for value in keys if value is not None),
            tuple(value for value in values if value is not None),
        )

    def suffix_step(
        self,
        state: PrefixState,
        request_ids: Tuple[str, ...],
        past_keys: Tuple[Tensor, ...],
        past_values: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        """使用已有 suffix K/V 只计算当前最后一个 token。"""

        if not isinstance(self.host, TinyKVDecoderHost):
            raise ValueError("只有 TinyKV host 支持 suffix_step")
        if state.request_ids != request_ids or len(past_keys) != len(past_values):
            raise ValueError("suffix cache 与 request IDs/layers 不匹配")
        assert self.cut_layer is not None
        hidden = state.hidden[:, -1:]
        valid_mask = state.attention_mask
        keys, values = [], []
        for index, block in enumerate(self.host.blocks[self.cut_layer :]):
            hidden, key, value = block.forward_step(
                hidden, valid_mask, past_keys[index], past_values[index]
            )
            keys.append(key)
            values.append(value)
        return self.host.lm_head(self.host.norm(hidden)), tuple(keys), tuple(values)

    def select(
        self, state: PrefixState, indices: Tensor, request_ids: Tuple[str, ...]
    ) -> PrefixState:
        """通过底层 host 的安全索引接口选择合法子批。"""

        return self.host.select_prefix_state(state, indices, request_ids)
