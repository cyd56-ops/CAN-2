"""Qwen1.5-MoE C1 的 grouped-mm 原生 backend adapter。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

from .grouped_mm_observer import GroupedMMObserver, validate_grouped_invocations
from .real_adapter import TransformersHostAdapter
from .real_types import ArchitectureMap, ProbeRequest, ProbeResult
from .types import P0Error


class QwenHostAdapter(TransformersHostAdapter):
    """使用 Qwen 原生 grouped-mm 调用记录验证 routed zero-call。"""

    def _moe_blocks(self) -> Tuple[Tuple[str, Any], ...]:
        """返回具有 router、packed experts 和 shared expert 的 Qwen MoE block。"""

        blocks: List[Tuple[str, Any]] = []
        for path, module in self.loaded.model.named_modules():
            experts = getattr(module, "experts", None)
            router = getattr(module, "gate", None) or getattr(module, "router", None)
            shared = getattr(module, "shared_expert", None) or getattr(
                module, "shared_experts", None
            )
            if experts is None or router is None or shared is None:
                continue
            if experts.__class__.__name__ == "Qwen2MoeExperts":
                blocks.append((path, module))
        if not blocks:
            raise P0Error(
                "host_architecture_unresolved", "未找到 Qwen grouped-mm MoE block"
            )
        return tuple(blocks)

    def inspect_architecture(self) -> ArchitectureMap:
        """确认 grouped-mm backend 后启用 offsets execution evidence。"""

        architecture = super().inspect_architecture()
        try:
            blocks = self._moe_blocks()
        except (AttributeError, P0Error):
            # 纯 CPU stand-in 或 backend 未解析时保持 fail-closed 的旧语义。
            return replace(
                architecture,
                supports_row_mask=False,
                supports_execution_counter=False,
            )
        block = blocks[0][1]
        router = getattr(block, "gate", None) or getattr(block, "router", None)
        experts = getattr(block, "experts", None)
        top_k = int(
            getattr(router, "top_k", 0)
            or getattr(self.loaded.model.config, "num_experts_per_tok", 0)
        )
        count = int(
            getattr(experts, "num_experts", 0)
            or getattr(self.loaded.model.config, "num_experts", 0)
            or architecture.routed_expert_count
        )
        grouped_backend = architecture.backend_id == "grouped_mm"
        return replace(
            architecture,
            native_top_k=top_k,
            routed_expert_count=count,
            supports_row_mask=grouped_backend,
            supports_execution_counter=grouped_backend,
        )

    @staticmethod
    def _validate_request(request: ProbeRequest, experts: int, top_k: int) -> None:
        """校验 Qwen grouped-mm probe 的 mask、top-k 和请求维度。"""

        batch, sequence, hidden = request.hidden_shape
        if min(batch, sequence, hidden, experts, top_k) <= 0:
            raise P0Error("probe_fixture_invalid", "Qwen probe 维度非法")
        if request.native_top_k != top_k:
            raise P0Error("topk_changed", "probe top-k 与宿主不一致")
        if len(request.request_ids) != batch or len(set(request.request_ids)) != batch:
            raise P0Error("request_binding_invalid", "request IDs 不完整或重复")
        if len(request.allowed_mask) != batch or len(request.padding_mask) != batch:
            raise P0Error("mask_shape_invalid", "mask batch 不匹配")
        for rows, padding in zip(request.allowed_mask, request.padding_mask):
            if len(rows) != sequence or len(padding) != sequence:
                raise P0Error("mask_shape_invalid", "mask sequence 不匹配")
            for allowed in rows:
                if len(allowed) != experts:
                    raise P0Error("mask_shape_invalid", "mask expert 维度不匹配")
                selected_count = sum(bool(value) for value in allowed)
                if 0 < selected_count < top_k:
                    raise P0Error(
                        "mask_insufficient_experts", "allowed expert 少于 native top-k"
                    )

    @staticmethod
    def _hook_selected(selected: List[Tuple[int, ...]]):
        """构造只读 router selected-ID 记录 hook。"""

        def hook(module: Any, args: Any, output: Any) -> None:
            """记录 Qwen router 返回的原生 selected indices。"""

            indices = (
                output[2]
                if isinstance(output, (tuple, list)) and len(output) >= 3
                else None
            )
            if indices is None or not hasattr(indices, "tolist"):
                raise P0Error(
                    "router_ledger_invalid", "Qwen router 未返回 selected indices"
                )
            selected.extend(
                tuple(int(item) for item in row) for row in indices.tolist()
            )

        return hook

    def _run_block(
        self,
        block_path: str,
        block: Any,
        request: ProbeRequest,
        allowed_mask: Any = None,
    ) -> Tuple[Any, Tuple[Tuple[int, ...], ...], Tuple[Any, ...], int]:
        """在单个 Qwen MoE block 上运行原生 grouped-mm probe。"""

        import torch

        experts = getattr(block, "experts", None)
        router = getattr(block, "gate", None) or getattr(block, "router", None)
        shared = getattr(block, "shared_expert", None) or getattr(
            block, "shared_experts", None
        )
        architecture = self.inspect_architecture()
        first_parameter = next(block.parameters())
        batch, sequence, hidden_size = request.hidden_shape
        hidden = torch.zeros(
            (batch, sequence, hidden_size),
            dtype=first_parameter.dtype,
            device=first_parameter.device,
        )
        flattened_mask = None
        if allowed_mask is not None:
            flattened_mask = torch.tensor(
                [row for rows in allowed_mask for row in rows],
                dtype=torch.bool,
                device=hidden.device,
            )
        selected: List[Tuple[int, ...]] = []
        shared_calls = {"count": 0}

        def shared_hook(module: Any, args: Any) -> None:
            """记录真实 shared expert forward。"""

            shared_calls["count"] += 1

        router_handle = router.register_forward_hook(self._hook_selected(selected))
        shared_handle = shared.register_forward_pre_hook(shared_hook)
        observer = GroupedMMObserver(
            layer_path=block_path,
            allowed_mask=flattened_mask,
            sentinel_index=int(getattr(experts, "num_experts", 0)),
            native_top_k=architecture.native_top_k,
        )
        original_training = bool(block.training)
        block.eval()
        try:
            with torch.no_grad(), observer:
                output = block(hidden)
        finally:
            router_handle.remove()
            shared_handle.remove()
            block.train(original_training)
        return output, tuple(selected), observer.invocations, shared_calls["count"]

    def run_moe_probe(self, request: ProbeRequest) -> ProbeResult:
        """用原生 top-k mask 和 grouped-mm offsets 执行 mixed probe。"""

        if not self._probe_installed:
            raise P0Error("probe_not_installed", "必须在 probe context 内执行")
        architecture = self.inspect_architecture()
        if not architecture.supports_execution_counter:
            raise P0Error(
                "expert_call_unobservable",
                "C1 backend 不是已审计的 grouped-mm 原生实现",
            )
        self._validate_request(
            request, architecture.routed_expert_count, architecture.native_top_k
        )
        block_path, block = self._moe_blocks()[0]
        output, selected, invocations, _ = self._run_block(
            block_path, block, request, request.allowed_mask
        )
        validate_grouped_invocations(invocations)
        batch, sequence, _ = request.hidden_shape
        valid_indices = tuple(
            index
            for index in range(batch * sequence)
            if request.padding_mask[index // sequence][index % sequence]
        )
        return ProbeResult(
            output_shape=tuple(int(value) for value in output.shape),
            selected_ids=selected,
            expert_calls=tuple(),
            original_indices=valid_indices,
            reassembled_indices=valid_indices,
            kv_bound=True,
            grouped_mm_invocations=invocations,
            execution_observer_kind="grouped_mm_offsets",
        )

    def observe_all_allowed(self) -> Dict[str, Any]:
        """比较原生 block 与 grouped-mm observer 下的 all-allowed 输出。"""

        import torch

        if not self.inspect_architecture().supports_execution_counter:
            raise P0Error(
                "expert_call_unobservable",
                "C1 backend 不是已审计的 grouped-mm 原生实现",
            )
        block_path, block = self._moe_blocks()[0]
        architecture = self.inspect_architecture()
        hidden_size = int(getattr(block.config, "hidden_size", 1))
        request = ProbeRequest(
            hidden_shape=(1, 2, hidden_size),
            allowed_mask=(
                (tuple(True for _ in range(architecture.routed_expert_count)),) * 2,
            ),
            native_top_k=architecture.native_top_k,
            padding_mask=((True, True),),
            request_ids=("p0-all-allowed",),
        )
        first_parameter = next(block.parameters())
        hidden = torch.zeros(
            request.hidden_shape,
            dtype=first_parameter.dtype,
            device=first_parameter.device,
        )
        original_training = bool(block.training)
        block.eval()
        try:
            with torch.no_grad():
                baseline = block(hidden)
            observed, selected, invocations, shared_calls = self._run_block(
                block_path, block, request, allowed_mask=None
            )
        finally:
            block.train(original_training)
        validate_grouped_invocations(invocations)
        counts = invocations[0].counts
        return {
            "output_exact": bool(torch.equal(baseline, observed)),
            "selected_ids": selected,
            "routed_actual_calls": sum(count > 0 for count in counts),
            "routed_actual_rows": sum(counts),
            "shared_actual_calls": shared_calls,
            "execution_observer_kind": "grouped_mm_offsets",
            "grouped_mm_invocations": invocations,
        }
