"""Phase 5 的 credential-gated decoder-only Transformer。"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..crypto.lwe import LWEParams
from ..layers.gate_layer import AuthorizationDecision, GateLayer, ReasonCode
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class TransformerConfig:
    """定义 T0 decoder-only Transformer 的固定结构参数。"""

    vocab_size: int = 260
    max_seq_len: int = 256
    num_layers: int = 6
    cut_layer: int = 2
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """验证模型结构参数，拒绝含糊或不支持的配置。"""

        integer_fields = (
            "vocab_size",
            "max_seq_len",
            "num_layers",
            "cut_layer",
            "d_model",
            "num_heads",
            "d_ff",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是非 bool 整数")
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.vocab_size != ByteTokenizer.vocab_size:
            raise ValueError("T0 vocab_size 必须固定为 260")
        if self.cut_layer >= self.num_layers:
            raise ValueError("cut_layer 必须位于完整 Transformer block 中间")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout 必须是有限实数")
        if not np.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")


@dataclass(frozen=True)
class TransformerTrainingOutput:
    """保存训练态的完整 batch logits 与授权决定。"""

    protected_logits: Tensor
    public_logits: Tensor
    decision: AuthorizationDecision


@dataclass(frozen=True)
class TransformerInferenceOutput:
    """保存推理态的稀疏路由 logits、原索引与拒绝索引。"""

    protected_logits: Tensor
    protected_indices: Tensor
    public_logits: Tensor
    public_indices: Tensor
    rejected_indices: Tensor
    decision: AuthorizationDecision


@dataclass(frozen=True)
class GenerationOutput:
    """保存每条序列的确定性生成结果和一次路由计数。"""

    token_ids: Tuple[Tuple[int, ...], ...]
    capability_levels: Tuple[str, ...]
    stop_reasons: Tuple[str, ...]
    route_call_count: Tensor
    decision: AuthorizationDecision
    cache_lengths: Tuple[int, ...] = ()


class DecoderBlock(nn.Module):
    """实现 pre-norm causal self-attention Transformer block。"""

    def __init__(self, config: TransformerConfig) -> None:
        """根据模型配置初始化 attention 和前馈子层。"""

        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, hidden: Tensor, causal_mask: Tensor, padding_mask: Tensor
    ) -> Tensor:
        """执行带 causal mask 和 padding mask 的 decoder block。"""

        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = hidden + attended
        return hidden + self.ffn(self.ffn_norm(hidden))

    def forward_incremental(
        self,
        hidden: Tensor,
        past_key: Optional[Tensor] = None,
        past_value: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """执行单 token 增量 attention，并返回更新后的 K/V cache。"""
        normalized = self.attention_norm(hidden)
        qkv = F.linear(
            normalized, self.attention.in_proj_weight, self.attention.in_proj_bias
        )
        q, key, value = qkv.chunk(3, dim=-1)
        batch, steps, dim = q.shape
        heads = self.attention.num_heads
        head_dim = dim // heads
        q = q.view(batch, steps, heads, head_dim).transpose(1, 2)
        key = key.view(batch, steps, heads, head_dim).transpose(1, 2)
        value = value.view(batch, steps, heads, head_dim).transpose(1, 2)
        if past_key is not None:
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        weights = torch.matmul(q, key.transpose(-2, -1)) / (head_dim**0.5)
        weights = torch.softmax(weights, dim=-1)
        attended = (
            torch.matmul(weights, value).transpose(1, 2).reshape(batch, steps, dim)
        )
        attended = self.attention.out_proj(attended)
        output = hidden + attended
        output = output + self.ffn(self.ffn_norm(output))
        return output, key, value


class GatedDecoderTransformer(nn.Module):
    """在完整 Transformer block 中间执行 credential 条件路由。"""

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        params: LWEParams,
        config: Optional[TransformerConfig] = None,
        temperature: float = 5.0,
    ) -> None:
        """初始化共享前缀、Gate、public early-exit 和 protected full path。

        参数:
            A: LWE 公共矩阵 ``[m, n]``。
            b: LWE 公共向量 ``[m]``。
            params: 固定 toy LWE 参数。
            config: Transformer 结构；未提供时使用 T0 默认值。
            temperature: 训练态软门控温度。
        """

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
        self.gate_layer = GateLayer(A, b, params, temperature=temperature)
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
        """使用小型语言模型的稳定正态分布初始化可训练权重。"""

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_inputs(
        self, input_ids: Tensor, attention_mask: Optional[Tensor]
    ) -> Tensor:
        """验证 token batch，并返回规范化 Bool attention mask。"""

        if not isinstance(input_ids, Tensor):
            raise TypeError("input_ids 必须是 Tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids 必须是非空二维 Tensor[B, T]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids 必须是 torch.long")
        if input_ids.device != self.token_embedding.weight.device:
            raise ValueError("input_ids 与模型参数必须位于同一 device")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("序列长度超过 max_seq_len")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any().item()):
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
        if bool((~attention_mask).all(dim=1).any().item()):
            raise ValueError("每条序列至少包含一个有效 token")
        # T0 只接受右侧 padding，避免最后有效 token 定位含糊。
        seen_padding = (~attention_mask).cumsum(dim=1) > 0
        if bool((seen_padding & attention_mask).any().item()):
            raise ValueError("attention_mask 只允许右侧连续 padding")
        return attention_mask

    def _forward_blocks(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        start: int,
        end: int,
    ) -> Tensor:
        """执行指定半开区间内的 Transformer blocks。"""

        seq_len = hidden.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        padding_mask = ~attention_mask
        for block in self.blocks[start:end]:
            hidden = block(hidden, causal_mask, padding_mask)
        return hidden

    def _forward_prefix(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """计算 Gate cut 之前的共享 token 表示。"""

        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )
        return self._forward_blocks(
            hidden, attention_mask, start=0, end=self.config.cut_layer
        )

    def _forward_protected(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        """执行 Gate cut 后的 protected blocks 和 LM head。"""

        hidden = self._forward_blocks(
            hidden,
            attention_mask,
            start=self.config.cut_layer,
            end=self.config.num_layers,
        )
        return self.protected_head(self.protected_norm(hidden))

    def _forward_public(self, hidden: Tensor) -> Tensor:
        """从未门控的共享表示计算 public early-exit logits。"""

        return self.public_head(self.public_norm(hidden))

    def direct_protected_logits(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """绕过路由执行同 checkpoint 的 direct full-path 参考计算。"""

        normalized_mask = self._validate_inputs(input_ids, attention_mask)
        prefix = self._forward_prefix(input_ids, normalized_mask)
        return self._forward_protected(prefix, normalized_mask)

    def forward(
        self,
        input_ids: Tensor,
        credential: Union[Tensor, np.ndarray],
        attention_mask: Optional[Tensor] = None,
    ) -> Union[TransformerTrainingOutput, TransformerInferenceOutput]:
        """提交 credential 判决并执行训练态或推理态条件路由。"""

        normalized_mask = self._validate_inputs(input_ids, attention_mask)
        prefix = self._forward_prefix(input_ids, normalized_mask)
        # 复用现有 GateLayer：增加单例空间维，不改变 credential-only 判决。
        gated_4d, decision = self.gate_layer(prefix.unsqueeze(-1), credential)
        gated_prefix = gated_4d.squeeze(-1)
        valid_indices = torch.nonzero(decision.allow, as_tuple=False).flatten()
        parsed_invalid = decision.evidence.reason_code == int(
            ReasonCode.LWE_VERIFICATION_FAILED
        )
        public_indices = torch.nonzero(parsed_invalid, as_tuple=False).flatten()
        rejected_indices = torch.nonzero(
            ~(decision.allow | parsed_invalid), as_tuple=False
        ).flatten()

        if self.training:
            linked_zero = prefix.sum(dim=2, keepdim=True) * 0.0
            protected_logits = linked_zero.expand(
                -1, -1, self.config.vocab_size
            ).clone()
            if valid_indices.numel() > 0:
                protected_values = self._forward_protected(
                    gated_prefix.index_select(0, valid_indices),
                    normalized_mask.index_select(0, valid_indices),
                )
                protected_logits = protected_logits.index_copy(
                    0, valid_indices, protected_values
                )
            public_logits = self._forward_public(prefix)
            return TransformerTrainingOutput(
                protected_logits=protected_logits,
                public_logits=public_logits,
                decision=decision,
            )

        protected_logits = prefix.new_empty(
            (0, input_ids.shape[1], self.config.vocab_size)
        )
        if valid_indices.numel() > 0:
            protected_logits = self._forward_protected(
                gated_prefix.index_select(0, valid_indices),
                normalized_mask.index_select(0, valid_indices),
            )
        public_logits = prefix.new_empty(
            (0, input_ids.shape[1], self.config.vocab_size)
        )
        if public_indices.numel() > 0:
            public_logits = self._forward_public(prefix.index_select(0, public_indices))
        return TransformerInferenceOutput(
            protected_logits=protected_logits,
            protected_indices=valid_indices,
            public_logits=public_logits,
            public_indices=public_indices,
            rejected_indices=rejected_indices,
            decision=decision,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        credential: Union[Tensor, np.ndarray],
        attention_mask: Optional[Tensor] = None,
        max_new_tokens: int = 16,
        eos_token_id: int = ByteTokenizer.eos_token_id,
        pad_token_id: int = ByteTokenizer.pad_token_id,
        cache_mode: str = "none",
    ) -> GenerationOutput:
        """使用一次 credential route 执行确定性 greedy 生成。

        参数:
            input_ids: 右侧填充的 prompt token batch。
            credential: 单个或逐样本 credential。
            attention_mask: prompt 的有效 token mask。
            max_new_tokens: 每条序列最多生成的 token 数。
            eos_token_id: 停止 token ID。
            pad_token_id: 批处理填充 token ID。

        返回:
            不含填充位的逐样本 token、能力级别、停止原因和路由计数。
        """

        if cache_mode not in {"none", "kv"}:
            raise ValueError("cache_mode 必须为 none 或 kv")
        if self.training:
            raise RuntimeError("generate 只能在 eval 模式调用")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens 必须是整数")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")
        normalized_mask = self._validate_inputs(input_ids, attention_mask)
        if cache_mode == "kv":
            return self._generate_kv(
                input_ids,
                credential,
                normalized_mask,
                max_new_tokens,
                eos_token_id,
                pad_token_id,
            )
        prefix = self._forward_prefix(input_ids, normalized_mask)
        _, decision = self.gate_layer(prefix.unsqueeze(-1), credential)
        parsed_invalid = decision.evidence.reason_code == int(
            ReasonCode.LWE_VERIFICATION_FAILED
        )
        rejected = ~(decision.allow | parsed_invalid)

        lengths = normalized_mask.sum(dim=1).tolist()
        sequences: List[List[int]] = [
            input_ids[row, : int(length)].tolist() for row, length in enumerate(lengths)
        ]
        capabilities = [
            "protected" if bool(decision.allow[row].item()) else "public"
            for row in range(input_ids.shape[0])
        ]
        stop_reasons = ["max_new_tokens"] * input_ids.shape[0]
        active = [True] * input_ids.shape[0]
        route_counts = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )
        for row in range(input_ids.shape[0]):
            if bool(rejected[row].item()):
                capabilities[row] = "rejected"
                stop_reasons[row] = "invalid_credential_format"
                active[row] = False
                route_counts[row] = 0
            elif sequences[row][-1] == eos_token_id:
                stop_reasons[row] = "eos"
                active[row] = False
            elif len(sequences[row]) >= self.config.max_seq_len:
                stop_reasons[row] = "max_seq_len"
                active[row] = False

        for _ in range(max_new_tokens):
            if not any(active):
                break
            current_ids, current_mask = self._pad_sequences(
                sequences, pad_token_id, input_ids.device
            )
            current_prefix = self._forward_prefix(current_ids, current_mask)
            protected_rows = torch.nonzero(decision.allow, as_tuple=False).flatten()
            public_rows = torch.nonzero(parsed_invalid, as_tuple=False).flatten()
            next_logits = current_prefix.new_zeros(
                (input_ids.shape[0], self.config.vocab_size)
            )
            if protected_rows.numel() > 0:
                protected_values = self._forward_protected(
                    current_prefix.index_select(0, protected_rows),
                    current_mask.index_select(0, protected_rows),
                )
                for local_row, original_row in enumerate(protected_rows.tolist()):
                    next_logits[original_row] = protected_values[
                        local_row, len(sequences[original_row]) - 1
                    ]
            if public_rows.numel() > 0:
                public_values = self._forward_public(
                    current_prefix.index_select(0, public_rows)
                )
                for local_row, original_row in enumerate(public_rows.tolist()):
                    next_logits[original_row] = public_values[
                        local_row, len(sequences[original_row]) - 1
                    ]
            for row, is_active in enumerate(active):
                if not is_active:
                    continue
                token_id = int(torch.argmax(next_logits[row]).item())
                sequences[row].append(token_id)
                if token_id == eos_token_id:
                    active[row] = False
                    stop_reasons[row] = "eos"
                elif len(sequences[row]) >= self.config.max_seq_len:
                    active[row] = False
                    stop_reasons[row] = "max_seq_len"

        return GenerationOutput(
            token_ids=tuple(tuple(sequence) for sequence in sequences),
            capability_levels=tuple(capabilities),
            stop_reasons=tuple(stop_reasons),
            route_call_count=route_counts,
            decision=decision,
        )

    def _generate_kv(
        self,
        input_ids: Tensor,
        credential: Union[Tensor, np.ndarray],
        attention_mask: Tensor,
        max_new_tokens: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> GenerationOutput:
        """使用每条序列独立的增量 K/V cache 执行 greedy 生成。"""
        prefix = self._forward_prefix(input_ids, attention_mask)
        _, decision = self.gate_layer(prefix.unsqueeze(-1), credential)
        parsed_invalid = decision.evidence.reason_code == int(
            ReasonCode.LWE_VERIFICATION_FAILED
        )
        rejected = ~(decision.allow | parsed_invalid)
        sequences = [
            input_ids[i, : int(attention_mask[i].sum())].tolist()
            for i in range(input_ids.shape[0])
        ]
        capabilities = [
            "protected" if bool(decision.allow[i]) else "public"
            for i in range(input_ids.shape[0])
        ]
        stops = ["max_new_tokens"] * len(sequences)
        counts = torch.ones(len(sequences), dtype=torch.long, device=input_ids.device)
        cache_lengths = [0] * len(sequences)
        for i in range(len(sequences)):
            if bool(rejected[i]):
                capabilities[i], stops[i], counts[i], cache_lengths[i] = (
                    "rejected",
                    "invalid_credential_format",
                    0,
                    0,
                )
                continue
            end = (
                self.config.cut_layer
                if bool(parsed_invalid[i])
                else self.config.num_layers
            )
            caches = [(None, None) for _ in range(end)]
            hidden = None
            for pos, token in enumerate(sequences[i]):
                hidden = (
                    self.token_embedding(
                        torch.tensor([[token]], device=input_ids.device)
                    )
                    + self.position_embedding(
                        torch.tensor([pos], device=input_ids.device)
                    )[None]
                )
                for layer in range(end):
                    hidden, key, value = self.blocks[layer].forward_incremental(
                        hidden, *caches[layer]
                    )
                    caches[layer] = (key, value)
            assert hidden is not None
            cache_lengths[i] = len(sequences[i])
            active = True
            for _ in range(max_new_tokens):
                if not active or len(sequences[i]) >= self.config.max_seq_len:
                    stops[i] = (
                        "max_seq_len"
                        if len(sequences[i]) >= self.config.max_seq_len
                        else stops[i]
                    )
                    break
                logits = (
                    self.public_head(self.public_norm(hidden))
                    if end == self.config.cut_layer
                    else self.protected_head(self.protected_norm(hidden))
                )
                token = int(torch.argmax(logits[0, -1]).item())
                sequences[i].append(token)
                cache_lengths[i] += 1
                if token == eos_token_id:
                    active = False
                    stops[i] = "eos"
                    continue
                pos = len(sequences[i]) - 1
                hidden = (
                    self.token_embedding(
                        torch.tensor([[token]], device=input_ids.device)
                    )
                    + self.position_embedding(
                        torch.tensor([pos], device=input_ids.device)
                    )[None]
                )
                for layer in range(end):
                    hidden, key, value = self.blocks[layer].forward_incremental(
                        hidden, *caches[layer]
                    )
                    caches[layer] = (key, value)
        return GenerationOutput(
            tuple(tuple(s) for s in sequences),
            tuple(capabilities),
            tuple(stops),
            counts,
            decision,
            tuple(cache_lengths),
        )

    @staticmethod
    def _pad_sequences(
        sequences: List[List[int]], pad_token_id: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """把动态生成序列右侧填充为 batch Tensor。"""

        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(
                sequence, dtype=torch.long, device=device
            )
            attention_mask[row, :length] = True
        return input_ids, attention_mask


__all__ = [
    "DecoderBlock",
    "GatedDecoderTransformer",
    "GenerationOutput",
    "TransformerConfig",
    "TransformerInferenceOutput",
    "TransformerTrainingOutput",
]
