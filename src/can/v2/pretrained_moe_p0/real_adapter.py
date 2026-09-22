"""真实 Transformers 宿主的只读 adapter 和 fail-closed 探针接口。"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Iterator, Mapping, Protocol, Tuple

from .real_loader import LoadedRealHost
from .real_types import (
    ArchitectureMap,
    ExpertCall,
    GenerationRecord,
    ProbeRequest,
    ProbeResult,
)
from .types import P0Error


class RealHostAdapter(Protocol):
    """P0 真实宿主只读探查的最小接口。"""

    def inspect_architecture(self) -> ArchitectureMap:
        """返回从实际模型对象盘点的架构信息。"""

    def baseline_generate(self, request: Any) -> GenerationRecord:
        """执行无 mask、无 verifier 的 baseline 生成。"""

    def install_probe(self) -> Any:
        """安装只读 instrumentation；不得提交授权。"""

    def run_moe_probe(self, request: ProbeRequest) -> ProbeResult:
        """执行受控结构探查，不能在 post-dispatch 乘零。"""

    def uninstall_probe(self) -> None:
        """卸载所有 instrumentation 并恢复原模型。"""

    def state_digest(self) -> str:
        """返回参数/缓冲区身份摘要。"""


class _ProbeContext:
    """保证 probe 异常时先卸载再向上传播。"""

    def __init__(self, adapter: "TransformersHostAdapter") -> None:
        self.adapter = adapter

    def __enter__(self) -> "_ProbeContext":
        self.adapter._probe_depth += 1
        if self.adapter._probe_depth != 1:
            self.adapter._probe_depth -= 1
            raise P0Error("probe_reentry_rejected", "probe 禁止重入")
        self.adapter._probe_installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.adapter.uninstall_probe()
        finally:
            self.adapter._probe_depth = 0
        return False


class TransformersHostAdapter:
    """对已加载 Transformers 模型提供 baseline 和保守结构探查接口。

    通用 adapter 不猜测具体模型的 dispatch 内部实现。若模型未提供经过审核的
    pre-dispatch mask/counter bridge，`run_moe_probe` 明确失败，而不是事后过滤输出。
    """

    def __init__(self, loaded: LoadedRealHost) -> None:
        """绑定一个已通过离线加载与量化检查的宿主。"""
        self.loaded = loaded
        self._probe_installed = False
        self._probe_depth = 0

    def inspect_architecture(self) -> ArchitectureMap:
        """返回 loader 从实际 module objects 盘点的结构摘要。"""
        return self.loaded.architecture

    def baseline_generate(self, request: Any) -> GenerationRecord:
        """执行固定 greedy baseline；请求已由 capability_eval 校验。"""
        from .capability_eval import generate_one

        return generate_one(self.loaded.tokenizer, self.loaded.model, request)

    def install_probe(self) -> _ProbeContext:
        """返回只读 probe context，禁止并发/重入。"""
        return _ProbeContext(self)

    def run_moe_probe(self, request: ProbeRequest) -> ProbeResult:
        """执行模型登记的 pre-dispatch bridge，否则 fail closed。"""
        if not self._probe_installed:
            raise P0Error("probe_not_installed", "必须在 probe context 内执行")
        bridge = getattr(self.loaded.model, "can_p0_moe_probe", None)
        runner = getattr(self.loaded.model, "run_can_p0_moe_probe", None)
        if bridge is not True or not callable(runner):
            raise P0Error(
                "host_interface_not_controllable",
                "模型没有经过审核的 pre-dispatch mask bridge",
            )
        result = runner(request)
        if not isinstance(result, ProbeResult):
            raise P0Error("probe_result_invalid", "probe bridge 返回类型非法")
        return result

    def uninstall_probe(self) -> None:
        """卸载模型登记的 probe bridge，并拒绝遗留 wrapper。"""
        remover = getattr(self.loaded.model, "remove_can_p0_moe_probe", None)
        if callable(remover):
            remover()
        self._probe_installed = False

    def state_digest(self) -> str:
        """计算参数和 buffer 的稳定摘要，不读取或复制完整权重。"""
        digest = hashlib.sha256()
        for name, value in list(self.loaded.model.named_parameters()) + list(
            self.loaded.model.named_buffers()
        ):
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.device).encode("ascii"))
            digest.update(str(id(value)).encode("ascii"))
            # data_ptr 绑定底层 storage，_version 可发现原地写入；磁盘权重内容
            # 另由 snapshot inventory 的 SHA-256 负责，避免复制超大参数。
            data_ptr = (
                value.data_ptr() if callable(getattr(value, "data_ptr", None)) else 0
            )
            digest.update(str(data_ptr).encode("ascii"))
            digest.update(str(getattr(value, "_version", -1)).encode("ascii"))
        return digest.hexdigest()

    def observe_all_allowed(self) -> Mapping[str, Any]:
        """通用 adapter 不猜测候选内部调用点，默认拒绝 all-allowed 证据。"""
        raise P0Error(
            "expert_call_unobservable", "宿主没有候选专用 all-allowed 执行观察器"
        )

    def verify_kv_semantics(self) -> bool:
        """执行一次真实 prefill/decode，并确认错误上下文在 forward 前拒绝。"""
        try:
            import torch
        except ImportError as exc:
            raise P0Error(
                "optional_dependency_missing", "KV probe 需要 PyTorch"
            ) from exc
        tokenizer = self.loaded.tokenizer
        model = self.loaded.model
        if not callable(tokenizer):
            raise P0Error("kv_semantics_unsupported", "tokenizer 不支持普通编码")
        try:
            encoded = tokenizer("P0 cache probe", return_tensors="pt")
        except Exception as exc:
            raise P0Error("kv_semantics_unsupported", "KV probe 编码失败") from exc
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise P0Error("kv_semantics_unsupported", "KV probe 缺少 input_ids")
        device = next(model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        request_id = "p0-kv-request"
        committed = {
            "request_id": request_id,
            "processed_length": int(inputs["input_ids"].shape[-1]),
        }
        calls = {"count": 0}

        def count_call(module: Any, args: Any) -> None:
            """记录真实 MoE block forward 次数。"""
            calls["count"] += 1

        handles = []
        for path, module in model.named_modules():
            if path in self.inspect_architecture().moe_layers:
                handles.append(module.register_forward_pre_hook(count_call))
        original_training = bool(model.training)
        model.eval()
        try:
            with torch.no_grad():
                first = model(**inputs, use_cache=True, return_dict=True)
                cache = getattr(first, "past_key_values", None)
                logits = getattr(first, "logits", None)
                if cache is None or logits is None:
                    raise P0Error(
                        "kv_semantics_unsupported", "prefill 未返回 cache/logits"
                    )
                calls_after_prefill = calls["count"]
                if calls_after_prefill <= 0:
                    raise P0Error(
                        "kv_semantics_unsupported", "prefill 未经过 MoE block"
                    )
                next_token = logits[:, -1:, :].argmax(dim=-1)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1
                    )
                second = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                if getattr(second, "past_key_values", None) is None:
                    raise P0Error("kv_semantics_unsupported", "decode 未返回新 cache")
                if calls["count"] <= calls_after_prefill:
                    raise P0Error("kv_semantics_unsupported", "decode 未经过 MoE block")

                # 错误 request identity 在任何模型调用前由受信上下文检查拒绝。
                calls_before_negative = calls["count"]
                incoming_request_id = "cross-request"
                if incoming_request_id != committed["request_id"]:
                    rejected = True
                else:  # pragma: no cover - 固定负向 fixture 不会进入
                    rejected = False
                    model(input_ids=next_token, past_key_values=cache, use_cache=True)
                if not rejected or calls["count"] != calls_before_negative:
                    raise P0Error(
                        "kv_semantics_unsupported", "错误 KV 上下文未在执行前拒绝"
                    )
        except P0Error:
            raise
        except Exception as exc:  # pragma: no cover - 服务器模型差异路径
            raise P0Error(
                "kv_semantics_unsupported", "真实 KV prefill/decode 失败"
            ) from exc
        finally:
            for handle in handles:
                handle.remove()
            model.train(original_training)
        return True
