"""Phase 5 T1 的逐样本 reference routing 校验工具。"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .model import GatedDecoderTransformer, GenerationOutput
from ..layers.gate_layer import ReasonCode


@dataclass(frozen=True)
class ReferenceTrace:
    """记录混合 batch 与逐样本 reference 的差异。"""

    sample_count: int
    matched_sequences: int
    max_token_mismatch: int
    route_calls: int
    cache_mode: str = "none"
    status: str = "ok"
    first_divergence_positions: Tuple[Optional[int], ...] = ()
    longest_common_prefix_ratios: Tuple[float, ...] = ()
    token_agreement_rates: Tuple[float, ...] = ()
    max_abs_difference: Optional[float] = None
    stop_positions: Tuple[int, ...] = ()
    logits_allclose: bool = True
    assert_close_atol: float = 1e-5
    assert_close_rtol: float = 1e-4


@dataclass(frozen=True)
class MixedRoutingValidation:
    """封装 mixed routing 校验结果及空子批计数。"""

    trace: ReferenceTrace
    empty_protected: int = 0
    empty_public: int = 0


@dataclass(frozen=True)
class DirectReferenceEquivalence:
    """记录 routed protected logits 与 direct full-path 的等价性。"""

    logits_allclose: bool
    greedy_token_match: bool
    max_abs_difference: float
    first_divergence_positions: Tuple[Optional[int], ...]
    status: str = "ok"


def validate_generation_reference(
    model: GatedDecoderTransformer,
    input_ids: torch.Tensor,
    credential: torch.Tensor,
    max_new_tokens: int = 16,
    cache_mode: str = "none",
) -> MixedRoutingValidation:
    """比较 mixed batch 和逐样本生成，验证 token 序列无串扰。

    ``cache_mode=none`` 是当前交付模式；``kv`` 明确返回 blocked，避免把
    重计算误报为 KV-cache 结果。
    """

    if cache_mode not in {"none", "kv"}:
        raise ValueError("cache_mode 必须为 none 或 kv")
    if cache_mode == "kv":
        trace = ReferenceTrace(0, 0, 0, 0, "kv", "blocked")
        return MixedRoutingValidation(trace)
    if input_ids.ndim != 2 or credential.ndim != 2:
        raise ValueError("input_ids 和 credential 必须为二维 batch")
    if input_ids.shape[0] != credential.shape[0] or input_ids.shape[0] == 0:
        raise ValueError("输入 batch 维度不匹配或为空")
    model.eval()
    mixed = model.generate(input_ids, credential, max_new_tokens=max_new_tokens)
    decision = mixed.decision
    valid_mask = decision.allow
    invalid_mask = decision.evidence.reason_code == int(
        ReasonCode.LWE_VERIFICATION_FAILED
    )
    empty_protected = int(not bool(valid_mask.any().item()))
    empty_public = int(not bool(invalid_mask.any().item()))
    mixed_logits = model(input_ids, credential)
    max_abs_difference = 0.0
    logits_allclose = True
    matched = 0
    max_mismatch = 0
    divergences = []
    prefixes = []
    agreements = []
    stops = []
    for row in range(input_ids.shape[0]):
        single = model.generate(
            input_ids[row : row + 1],
            credential[row : row + 1],
            max_new_tokens=max_new_tokens,
        )
        single_logits = model(input_ids[row : row + 1], credential[row : row + 1])
        if bool(valid_mask[row].item()):
            local = int((mixed_logits.protected_indices == row).nonzero()[0].item())
            max_abs_difference = max(
                max_abs_difference,
                float(
                    (
                        mixed_logits.protected_logits[local]
                        - single_logits.protected_logits[0]
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
        elif bool(invalid_mask[row].item()):
            local = int((mixed_logits.public_indices == row).nonzero()[0].item())
            max_abs_difference = max(
                max_abs_difference,
                float(
                    (mixed_logits.public_logits[local] - single_logits.public_logits[0])
                    .abs()
                    .max()
                    .item()
                ),
            )
        if bool(valid_mask[row].item()) or bool(invalid_mask[row].item()):
            mixed_row = (
                mixed_logits.protected_logits[local]
                if bool(valid_mask[row].item())
                else mixed_logits.public_logits[local]
            )
            single_row = (
                single_logits.protected_logits[0]
                if bool(valid_mask[row].item())
                else single_logits.public_logits[0]
            )
            logits_allclose = logits_allclose and bool(
                torch.allclose(mixed_row, single_row, atol=1e-5, rtol=1e-4)
            )
        lhs = mixed.token_ids[row]
        rhs = single.token_ids[0]
        mismatch = sum(a != b for a, b in zip(lhs, rhs)) + abs(len(lhs) - len(rhs))
        max_mismatch = max(max_mismatch, mismatch)
        matched += int(mismatch == 0)
        first = next((i for i, (a, b) in enumerate(zip(lhs, rhs)) if a != b), None)
        divergences.append(first)
        common = 0
        for a, b in zip(lhs, rhs):
            if a != b:
                break
            common += 1
        prefixes.append(common / max(len(lhs), len(rhs), 1))
        agreements.append(1.0 - mismatch / max(len(lhs), len(rhs), 1))
        stops.append(len(lhs))
    trace = ReferenceTrace(
        sample_count=input_ids.shape[0],
        matched_sequences=matched,
        max_token_mismatch=max_mismatch,
        route_calls=int(mixed.route_call_count.sum().item()),
        cache_mode="none",
        status=(
            "ok" if matched == input_ids.shape[0] and logits_allclose else "mismatch"
        ),
        first_divergence_positions=tuple(divergences),
        longest_common_prefix_ratios=tuple(prefixes),
        token_agreement_rates=tuple(agreements),
        stop_positions=tuple(stops),
        max_abs_difference=max_abs_difference,
        logits_allclose=logits_allclose,
    )
    return MixedRoutingValidation(trace, empty_protected, empty_public)


def validate_direct_reference(
    model: GatedDecoderTransformer,
    input_ids: torch.Tensor,
    credential: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> DirectReferenceEquivalence:
    """比较 routed protected logits 与 direct full-path logits。"""
    model.eval()
    routed = model(input_ids, credential)
    if routed.protected_indices.numel() == 0:
        return DirectReferenceEquivalence(True, True, 0.0, tuple(), "not_applicable")
    indices = routed.protected_indices
    direct = model.direct_protected_logits(input_ids.index_select(0, indices))
    diff = float((routed.protected_logits - direct).abs().max().item())
    close = bool(torch.allclose(routed.protected_logits, direct, atol=atol, rtol=rtol))
    greedy = torch.argmax(routed.protected_logits, dim=-1) == torch.argmax(
        direct, dim=-1
    )
    positions = []
    for row in range(greedy.shape[0]):
        bad = torch.nonzero(~greedy[row], as_tuple=False).flatten()
        positions.append(int(bad[0].item()) if bad.numel() else None)
    return DirectReferenceEquivalence(
        close,
        bool(greedy.all().item()),
        diff,
        tuple(positions),
        "ok" if close and bool(greedy.all().item()) else "mismatch",
    )


__all__ = [
    "DirectReferenceEquivalence",
    "MixedRoutingValidation",
    "ReferenceTrace",
    "validate_direct_reference",
    "validate_generation_reference",
]
