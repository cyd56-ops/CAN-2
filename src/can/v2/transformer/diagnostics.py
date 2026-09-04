"""Phase 5 逐样本生成与 teacher-forced 诊断工具。"""

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from ..training.data import CredentialGenerator
from .normalization import classify_refusal, normalize_answer


def _first_divergence(
    generated: Sequence[int], expected: Sequence[int]
) -> Optional[int]:
    """返回生成 continuation 与目标 token 的首个差异位置。"""
    for index, (actual, target) in enumerate(zip(generated, expected)):
        if actual != target:
            return index
    if len(generated) != len(expected):
        return min(len(generated), len(expected))
    return None


def _teacher_forced(
    model: Any,
    prompt: Sequence[int],
    target: Sequence[int],
    head: str,
    device: torch.device,
    credential: Optional[Tensor],
) -> Dict[str, Any]:
    """计算答案 token 每个位置的 teacher-forced 正确性。"""
    full = torch.tensor([list(prompt) + list(target)], dtype=torch.long, device=device)
    mask = torch.ones_like(full, dtype=torch.bool)
    if credential is None:
        logits = model.logits(full, head, mask)
    elif head == "protected":
        logits = model.direct_protected_logits(full, mask)
    else:
        logits = model.direct_public_logits(full, mask)
    answer_logits = logits[0, len(prompt) - 1 : -1]
    labels = torch.tensor(target, dtype=torch.long, device=device)
    predictions = answer_logits.argmax(dim=-1)
    correct = (predictions == labels).tolist()
    return {
        "correct_by_position": [bool(value) for value in correct],
        "correct_count": int(sum(correct)),
        "target_token_count": len(target),
        "first_incorrect_position": next(
            (index for index, value in enumerate(correct) if not value), None
        ),
    }


def _route_diagnostic(
    model: Any,
    item: Any,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int,
    cache_mode: str,
    head: str,
    credential: Optional[Tensor],
) -> Dict[str, Any]:
    """生成一个 scope/head 组合的诊断记录。"""
    prompt = tokenizer.encode(
        item.prompt, add_bos=True, add_eos=False, max_length=model.config.max_seq_len
    )
    target = tokenizer.encode(
        item.answer, add_bos=False, add_eos=True, max_length=model.config.max_seq_len
    )
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    if credential is None:
        generated = model.generate(
            ids, head, max_new_tokens=max_new_tokens, cache_mode=cache_mode
        )
    else:
        generated = model.generate(
            ids,
            credential,
            max_new_tokens=max_new_tokens,
            cache_mode=cache_mode,
        )
    token_ids = list(generated.token_ids[0])
    continuation = token_ids[len(prompt) :]
    hit_eos = tokenizer.eos_token_id in continuation
    if hit_eos:
        continuation = continuation[: continuation.index(tokenizer.eos_token_id) + 1]
    teacher = _teacher_forced(model, prompt, target, head, device, credential)
    return {
        "head": head,
        "generated": tokenizer.decode(continuation),
        "expected": item.answer,
        "generated_normalized": normalize_answer(tokenizer.decode(continuation)),
        "expected_normalized": normalize_answer(item.answer),
        "exact_match": normalize_answer(tokenizer.decode(continuation))
        == normalize_answer(item.answer),
        "first_divergence_position": _first_divergence(continuation, target),
        "generated_token_count": len(continuation),
        "expected_token_count": len(target),
        "hit_eos": hit_eos,
        "stop_reason": generated.stop_reasons[0],
        "teacher_forced": teacher,
    }


def build_sample_diagnostics(
    model: Any,
    examples: Sequence[Any],
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int,
    cache_mode: str,
    credential_generator: Optional[CredentialGenerator] = None,
) -> Dict[str, Any]:
    """生成 train/validation 逐样本诊断记录，兼容 CAN 与 Plain 模型。"""
    model.eval()
    by_entity = {(item.entity_id, item.scope): item.answer for item in examples}
    records: List[Dict[str, Any]] = []
    for item in examples:
        routes: Dict[str, Any] = {}
        if credential_generator is None:
            if item.scope in {"public", "private"}:
                routes["protected"] = _route_diagnostic(
                    model,
                    item,
                    tokenizer,
                    device,
                    max_new_tokens,
                    cache_mode,
                    "protected",
                    None,
                )
            if item.scope in {"public", "refusal"}:
                routes["public"] = _route_diagnostic(
                    model,
                    item,
                    tokenizer,
                    device,
                    max_new_tokens,
                    cache_mode,
                    "public",
                    None,
                )
        else:
            if item.scope in {"public", "private"}:
                valid = credential_generator.generate(True)
                routes["protected"] = _route_diagnostic(
                    model,
                    item,
                    tokenizer,
                    device,
                    max_new_tokens,
                    cache_mode,
                    "protected",
                    torch.tensor(valid, dtype=torch.float32, device=device),
                )
            if item.scope in {"public", "refusal"}:
                invalid = credential_generator.generate(False)
                routes["public"] = _route_diagnostic(
                    model,
                    item,
                    tokenizer,
                    device,
                    max_new_tokens,
                    cache_mode,
                    "public",
                    torch.tensor(invalid, dtype=torch.float32, device=device),
                )
        record: Dict[str, Any] = {
            "sample_id": item.sample_id,
            "scope": item.scope,
            "entity_id": item.entity_id,
            "prompt": item.prompt,
            "answer": item.answer,
            "routes": routes,
        }
        if item.scope == "refusal" and "public" in routes:
            continuation = routes["public"]["generated_normalized"]
            record["refusal_class"] = classify_refusal(
                continuation,
                by_entity.get((item.entity_id, "refusal"), item.answer),
                by_entity.get((item.entity_id, "private"), ""),
                by_entity.get((item.entity_id, "public"), ""),
            )
        records.append(record)
    return {
        "schema_version": 1,
        "model_kind": "can" if credential_generator is not None else "plain",
        "sample_count": len(records),
        "records": records,
    }


__all__ = ["build_sample_diagnostics"]
