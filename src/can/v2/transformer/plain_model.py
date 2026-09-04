"""不含 credential/Gate 的 Plain decoder-only Transformer 对照模型。"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from .model import DecoderBlock, TransformerConfig
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class PlainTransformerOutput:
    """保存普通模型两个 head 的完整 batch logits。"""

    protected_logits: Tensor
    public_logits: Tensor


@dataclass(frozen=True)
class PlainGenerationOutput:
    """保存显式 head 路由下的确定性生成结果。"""

    token_ids: Tuple[Tuple[int, ...], ...]
    head: str
    stop_reasons: Tuple[str, ...]
    cache_lengths: Tuple[int, ...]


class PlainDecoderTransformer(nn.Module):
    """提供与 CAN 同构、但不含 Gate/credential 的双 head Transformer。"""

    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        """初始化共享 prefix、public early-exit 与 protected full-path。"""
        super().__init__()
        self.config = config if config is not None else TransformerConfig()
        if not isinstance(self.config, TransformerConfig):
            raise TypeError("config 必须是 TransformerConfig")
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.position_embedding = nn.Embedding(
            self.config.max_seq_len, self.config.d_model
        )
        self.blocks = nn.ModuleList(
            DecoderBlock(self.config) for _ in range(self.config.num_layers)
        )
        self.public_norm = nn.LayerNorm(self.config.d_model)
        self.public_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self.protected_norm = nn.LayerNorm(self.config.d_model)
        self.protected_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """使用与 GatedDecoderTransformer 一致的权重初始化。"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_inputs(
        self, input_ids: Tensor, attention_mask: Optional[Tensor]
    ) -> Tensor:
        """验证 token batch，并返回规范化的右侧 padding mask。"""
        if not isinstance(input_ids, Tensor):
            raise TypeError("input_ids 必须是 Tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids 必须是非空二维 Tensor[B,T]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids 必须是 torch.long")
        if input_ids.device != self.token_embedding.weight.device:
            raise ValueError("input_ids 与模型参数必须位于同一 device")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("序列长度超过 max_seq_len")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("input_ids 包含词表外 token")
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        if not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask 必须是 Tensor 或 None")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask 必须与 input_ids shape 相同")
        if attention_mask.device != input_ids.device:
            raise ValueError("attention_mask 与 input_ids 必须位于同一 device")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask 必须是 BoolTensor")
        if bool((~attention_mask).all(dim=1).any()):
            raise ValueError("每条序列至少包含一个有效 token")
        seen_padding = (~attention_mask).cumsum(dim=1) > 0
        if bool((seen_padding & attention_mask).any()):
            raise ValueError("attention_mask 只允许右侧连续 padding")
        return attention_mask

    def _forward_blocks(
        self, hidden: Tensor, attention_mask: Tensor, start: int, end: int
    ) -> Tensor:
        """执行指定半开区间内的 causal Transformer blocks。"""
        seq_len = hidden.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        for block in self.blocks[start:end]:
            hidden = block(hidden, causal_mask, ~attention_mask)
        return hidden

    def _forward_prefix(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """计算 cut layer 之前的共享 token 表示。"""
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = (
            self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        )
        return self._forward_blocks(
            hidden, attention_mask, start=0, end=self.config.cut_layer
        )

    def _head_logits(self, prefix: Tensor, attention_mask: Tensor, head: str) -> Tensor:
        """从共享 prefix 计算指定 public 或 protected head 的 logits。"""
        if head == "protected":
            hidden = self._forward_blocks(
                prefix,
                attention_mask,
                start=self.config.cut_layer,
                end=self.config.num_layers,
            )
            return self.protected_head(self.protected_norm(hidden))
        if head == "public":
            return self.public_head(self.public_norm(prefix))
        raise ValueError("head 必须为 public 或 protected")

    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> PlainTransformerOutput:
        """计算两个 head 的完整 batch teacher-forced logits。"""
        mask = self._validate_inputs(input_ids, attention_mask)
        prefix = self._forward_prefix(input_ids, mask)
        return PlainTransformerOutput(
            protected_logits=self._head_logits(prefix, mask, "protected"),
            public_logits=self._head_logits(prefix, mask, "public"),
        )

    def direct_protected_logits(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """执行 protected full-path 的直接 teacher-forced 计算。"""
        mask = self._validate_inputs(input_ids, attention_mask)
        return self._head_logits(
            self._forward_prefix(input_ids, mask), mask, "protected"
        )

    def direct_public_logits(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """执行 public early-exit 的直接 teacher-forced 计算。"""
        mask = self._validate_inputs(input_ids, attention_mask)
        return self._head_logits(self._forward_prefix(input_ids, mask), mask, "public")

    def logits(
        self,
        input_ids: Tensor,
        head: str,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """按显式 head 名称计算 teacher-forced logits。"""
        if head == "protected":
            return self.direct_protected_logits(input_ids, attention_mask)
        if head == "public":
            return self.direct_public_logits(input_ids, attention_mask)
        raise ValueError("head 必须为 public 或 protected")

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        head: str,
        attention_mask: Optional[Tensor] = None,
        max_new_tokens: int = 16,
        eos_token_id: int = ByteTokenizer.eos_token_id,
        pad_token_id: int = ByteTokenizer.pad_token_id,
        cache_mode: str = "none",
    ) -> PlainGenerationOutput:
        """按显式 public/protected head 执行确定性 greedy 生成。"""
        if self.training:
            raise RuntimeError("generate 只能在 eval 模式调用")
        if head not in {"public", "protected"}:
            raise ValueError("head 必须为 public 或 protected")
        if cache_mode not in {"none", "kv"}:
            raise ValueError("cache_mode 必须为 none 或 kv")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens 必须是整数")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")
        mask = self._validate_inputs(input_ids, attention_mask)
        if cache_mode == "kv":
            return self._generate_kv(
                input_ids, mask, head, max_new_tokens, eos_token_id, pad_token_id
            )
        return self._generate_none(
            input_ids, mask, head, max_new_tokens, eos_token_id, pad_token_id
        )

    def _generate_none(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        head: str,
        max_new_tokens: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> PlainGenerationOutput:
        """通过每轮重算 prefix 执行 greedy 生成。"""
        lengths = attention_mask.sum(dim=1).tolist()
        sequences: List[List[int]] = [
            input_ids[row, : int(length)].tolist() for row, length in enumerate(lengths)
        ]
        stops = ["max_new_tokens"] * len(sequences)
        active = [True] * len(sequences)
        for row, sequence in enumerate(sequences):
            if sequence[-1] == eos_token_id:
                active[row], stops[row] = False, "eos"
            elif len(sequence) >= self.config.max_seq_len:
                active[row], stops[row] = False, "max_seq_len"
        for _ in range(max_new_tokens):
            if not any(active):
                break
            current_ids, current_mask = self._pad_sequences(
                sequences, pad_token_id, input_ids.device
            )
            prefix = self._forward_prefix(current_ids, current_mask)
            logits = self._head_logits(prefix, current_mask, head)
            for row, is_active in enumerate(active):
                if not is_active:
                    continue
                token_id = int(logits[row, len(sequences[row]) - 1].argmax().item())
                sequences[row].append(token_id)
                if token_id == eos_token_id:
                    active[row], stops[row] = False, "eos"
                elif len(sequences[row]) >= self.config.max_seq_len:
                    active[row], stops[row] = False, "max_seq_len"
        return PlainGenerationOutput(
            tuple(tuple(sequence) for sequence in sequences),
            head,
            tuple(stops),
            tuple(len(sequence) for sequence in sequences),
        )

    def _generate_kv(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        head: str,
        max_new_tokens: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> PlainGenerationOutput:
        """为每条序列维护独立 K/V cache 并执行 greedy 生成。"""
        del pad_token_id  # KV 路径按单条序列计算，不需要 padding token。
        end = self.config.num_layers if head == "protected" else self.config.cut_layer
        sequences = [
            input_ids[row, : int(attention_mask[row].sum())].tolist()
            for row in range(input_ids.shape[0])
        ]
        stops = ["max_new_tokens"] * len(sequences)
        cache_lengths = [len(sequence) for sequence in sequences]
        for row, sequence in enumerate(sequences):
            if sequence[-1] == eos_token_id:
                stops[row] = "eos"
                continue
            if len(sequence) >= self.config.max_seq_len:
                stops[row] = "max_seq_len"
                continue
            caches = [(None, None) for _ in range(end)]
            hidden: Optional[Tensor] = None
            for position, token_id in enumerate(sequence):
                hidden = (
                    self.token_embedding(
                        torch.tensor(
                            [[token_id]], dtype=torch.long, device=input_ids.device
                        )
                    )
                    + self.position_embedding(
                        torch.tensor(
                            [position], dtype=torch.long, device=input_ids.device
                        )
                    )[None]
                )
                for layer in range(end):
                    hidden, key, value = self.blocks[layer].forward_incremental(
                        hidden, *caches[layer]
                    )
                    caches[layer] = (key, value)
            assert hidden is not None
            for _ in range(max_new_tokens):
                if len(sequence) >= self.config.max_seq_len:
                    stops[row] = "max_seq_len"
                    break
                logits = (
                    self.protected_head(self.protected_norm(hidden))
                    if head == "protected"
                    else self.public_head(self.public_norm(hidden))
                )
                token_id = int(logits[0, -1].argmax().item())
                sequence.append(token_id)
                cache_lengths[row] = len(sequence)
                if token_id == eos_token_id:
                    stops[row] = "eos"
                    break
                position = len(sequence) - 1
                hidden = (
                    self.token_embedding(
                        torch.tensor(
                            [[token_id]], dtype=torch.long, device=input_ids.device
                        )
                    )
                    + self.position_embedding(
                        torch.tensor(
                            [position], dtype=torch.long, device=input_ids.device
                        )
                    )[None]
                )
                for layer in range(end):
                    hidden, key, value = self.blocks[layer].forward_incremental(
                        hidden, *caches[layer]
                    )
                    caches[layer] = (key, value)
        return PlainGenerationOutput(
            tuple(tuple(sequence) for sequence in sequences),
            head,
            tuple(stops),
            tuple(cache_lengths),
        )

    @staticmethod
    def _pad_sequences(
        sequences: List[List[int]], pad_token_id: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """把动态生成序列右侧填充为 batch Tensor。"""
        max_length = max(len(sequence) for sequence in sequences)
        padded = torch.full(
            (len(sequences), max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(padded, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            padded[row, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device=device
            )
            mask[row, : len(sequence)] = True
        return padded, mask


__all__ = [
    "PlainDecoderTransformer",
    "PlainGenerationOutput",
    "PlainTransformerOutput",
]
