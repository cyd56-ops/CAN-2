"""P0-B 公开能力 fixture 的确定性生成与评分。"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .fixture import normalized_em, strict_em
from .real_types import GenerationRecord, GenerationRequest
from .types import FixtureCase, P0Error


def _safe_decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """解码 continuation，并拒绝控制字符和非法特殊 token。"""

    try:
        text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    except Exception as exc:  # pragma: no cover - 服务器 tokenizer 路径
        raise P0Error("generation_decode_failed", "tokenizer 无法解码输出") from exc
    if any(
        (ord(char) < 32 and char not in "\t\n\r") or ord(char) == 127 for char in text
    ):
        raise P0Error("generation_invalid_control_token", "输出含控制字符")
    return text


def _input_ids(tokenizer: Any, request: GenerationRequest) -> Any:
    """按原生 chat template 构造单样本输入。"""

    messages = [
        {"role": "system", "content": request.system_text},
        {"role": "user", "content": request.user_text},
    ]
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise P0Error("chat_template_missing", "tokenizer 没有原生 chat template")
    try:
        return apply_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    except Exception as exc:  # pragma: no cover - 服务器 tokenizer 路径
        raise P0Error(
            "chat_template_rejected", "chat template 拒绝冻结消息组合"
        ) from exc


def _tensor_to_list(value: Any) -> List[int]:
    """将一维/单 batch token tensor 转成 Python int 列表。"""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise P0Error("generation_tensor_invalid", "token 输出结构非法")
    return value


def generate_one(
    tokenizer: Any, model: Any, request: GenerationRequest
) -> GenerationRecord:
    """对单条 fixture 执行 greedy 生成并返回逐 token 诊断。"""

    inputs = _input_ids(tokenizer, request)
    if isinstance(inputs, dict):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
    else:
        input_ids = inputs
        attention_mask = None
    prompt_tokens = _tensor_to_list(input_ids)
    if len(prompt_tokens) > 256:
        raise P0Error("prompt_too_long", "prompt 超过冻结 256 token 上限")
    kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": request.max_new_tokens,
        "use_cache": request.use_cache,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    try:
        output = model.generate(**kwargs)
    except Exception as exc:  # pragma: no cover - 服务器模型路径
        raise P0Error("generation_failed", "模型生成失败") from exc
    output_tokens = _tensor_to_list(output)
    if len(output_tokens) < len(prompt_tokens):
        raise P0Error("generation_tensor_invalid", "输出短于 prompt")
    continuation = output_tokens[len(prompt_tokens) :]
    generated = _safe_decode(tokenizer, continuation)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop_reason = (
        "eos" if eos_id is not None and eos_id in continuation else "max_new_tokens"
    )
    matched = (
        strict_em(generated, request.expected_text)
        if request.metric == "strict_em"
        else normalized_em(generated, request.expected_text)
    )
    return GenerationRecord(
        request.case_id,
        generated,
        generated,
        request.expected_text,
        matched,
        tuple(prompt_tokens),
        tuple(continuation),
        stop_reason,
        request.use_cache,
    )


def requests_from_fixture(
    cases: Iterable[FixtureCase], use_cache: bool
) -> Tuple[GenerationRequest, ...]:
    """将严格 fixture 类型转换为生成请求。"""

    return tuple(
        GenerationRequest(
            case.case_id,
            case.system_text,
            case.user_text,
            case.expected_text,
            case.metric,
            case.max_new_tokens,
            use_cache,
        )
        for case in cases
    )


def evaluate_fixture(adapter: Any, cases: Iterable[FixtureCase]) -> Dict[str, Any]:
    """运行两种 cache 配置各三次 baseline，检查门槛和 token 确定性。"""

    case_list = tuple(cases)
    records: List[GenerationRecord] = []
    errors: List[Dict[str, str]] = []
    for use_cache in (False, True):
        for repeat in range(3):
            for request in requests_from_fixture(case_list, use_cache):
                try:
                    record = adapter.baseline_generate(request)
                    records.append(record)
                except P0Error as exc:
                    errors.append(
                        {
                            "case_id": request.case_id,
                            "use_cache": str(use_cache),
                            "repeat": str(repeat),
                            "code": exc.code,
                        }
                    )
    by_group: Dict[str, Dict[str, int]] = {}
    for case in case_list:
        group = case.case_id.split("-", 1)[0]
        group_records = [item for item in records if item.case_id == case.case_id]
        by_group.setdefault(group, {"correct": 0, "total": len(case_list) // 3})
        signatures = {
            (item.continuation_tokens, item.stop_reason) for item in group_records
        }
        expected_repetitions = 6
        if (
            len(group_records) == expected_repetitions
            and len(signatures) == 1
            and all(item.matched for item in group_records)
        ):
            by_group[group]["correct"] += 1
    thresholds = {"format": 7, "single": 7, "two": 5}
    groups = {
        "format": by_group.get("format", {"correct": 0, "total": 8}),
        "single": by_group.get("single", {"correct": 0, "total": 8}),
        "two": by_group.get("two", {"correct": 0, "total": 8}),
    }
    deterministic = not errors and all(
        len(
            {
                (item.continuation_tokens, item.stop_reason)
                for item in records
                if item.case_id == case.case_id and item.use_cache == use_cache
            }
        )
        == 1
        for case in case_list
        for use_cache in (False, True)
    )
    cache_equivalent = not errors and all(
        len(
            {
                (item.continuation_tokens, item.stop_reason)
                for item in records
                if item.case_id == case.case_id
            }
        )
        == 1
        for case in case_list
    )
    passed = (
        not errors
        and deterministic
        and cache_equivalent
        and all(groups[key]["correct"] >= thresholds[key] for key in groups)
    )
    return {
        "status": "passed" if passed else "failed",
        "groups": groups,
        "thresholds": thresholds,
        "records": [record.__dict__ for record in records],
        "record_objects": tuple(records),
        "errors": errors,
        "repeat_count": 3,
        "deterministic": deterministic,
        "cache_equivalent": cache_equivalent,
        "in_process_run_count": 6,
        "in_process_difference_count": 0 if deterministic else 1,
        "cache_difference_count": 0 if cache_equivalent else 1,
    }
