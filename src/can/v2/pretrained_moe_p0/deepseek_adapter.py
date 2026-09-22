"""DeepSeek-MoE C2 的实例级 pre-dispatch mask 与执行计数 adapter。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Tuple

from .real_adapter import TransformersHostAdapter
from .real_types import ArchitectureMap, ExpertCall, ProbeRequest, ProbeResult
from .types import P0Error


class DeepSeekHostAdapter(TransformersHostAdapter):
    """使用实际 ModuleList experts 验证 mask、shared-only 和 zero-call。"""

    def _moe_blocks(self) -> Tuple[Any, ...]:
        """返回包含 gate、experts 和 shared_experts 的真实 DeepSeek MoE block。"""
        blocks = []
        for _, module in self.loaded.model.named_modules():
            experts = getattr(module, "experts", None)
            gate = getattr(module, "gate", None)
            shared = getattr(module, "shared_experts", None)
            if gate is not None and shared is not None and experts is not None:
                if module.__class__.__name__ == "DeepseekMoE" or hasattr(
                    module, "moe_infer"
                ):
                    blocks.append(module)
        if not blocks:
            raise P0Error("host_architecture_unresolved", "未找到 DeepSeek MoE block")
        return tuple(blocks)

    def inspect_architecture(self) -> ArchitectureMap:
        """根据实际 ModuleList/shared_experts 能力返回 C2 架构。"""
        architecture = super().inspect_architecture()
        blocks = self._moe_blocks()
        top_k = int(getattr(blocks[0], "num_experts_per_tok", 0))
        return replace(
            architecture,
            native_top_k=top_k,
            supports_row_mask=True,
            supports_execution_counter=True,
            routed_expert_count=len(blocks[0].experts),
        )

    @staticmethod
    def _validate_request(request: ProbeRequest, num_experts: int, top_k: int) -> None:
        """校验 hidden/mask/padding/request ID 的严格维度。"""
        batch, sequence, hidden = request.hidden_shape
        if batch <= 0 or sequence <= 0 or hidden <= 0:
            raise P0Error("probe_fixture_invalid", "hidden_shape 非法")
        if request.native_top_k != top_k:
            raise P0Error("topk_changed", "probe top-k 与宿主不一致")
        if len(request.request_ids) != batch or len(set(request.request_ids)) != batch:
            raise P0Error("request_binding_invalid", "request IDs 不完整或重复")
        if len(request.allowed_mask) != batch or len(request.padding_mask) != batch:
            raise P0Error("mask_shape_invalid", "mask batch 不匹配")
        for allowed_rows, padding_row in zip(
            request.allowed_mask, request.padding_mask
        ):
            if len(allowed_rows) != sequence or len(padding_row) != sequence:
                raise P0Error("mask_shape_invalid", "mask sequence 不匹配")
            for allowed in allowed_rows:
                if len(allowed) != num_experts:
                    raise P0Error("mask_shape_invalid", "mask expert 维度不匹配")
                count = sum(allowed)
                if 0 < count < top_k:
                    raise P0Error(
                        "mask_insufficient_experts", "非空 mask 少于 native top-k"
                    )

    @staticmethod
    def _masked_routing(block: Any, hidden: Any, allowed: Any) -> Tuple[Any, Any]:
        """在 softmax/top-k 前施加布尔 mask，并保持宿主归一化语义。"""
        import torch
        import torch.nn.functional as functional

        gate = block.gate
        logits = functional.linear(hidden, gate.weight, None)
        if getattr(gate, "scoring_func", None) != "softmax":
            raise P0Error("host_interface_not_controllable", "只支持冻结 softmax gate")
        masked_logits = logits.masked_fill(~allowed, float("-inf"))
        scores = masked_logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(
            scores, k=block.num_experts_per_tok, dim=-1, sorted=False
        )
        if block.num_experts_per_tok > 1 and getattr(gate, "norm_topk_prob", False):
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        return topk_idx, topk_weight

    def run_moe_probe(self, request: ProbeRequest) -> ProbeResult:
        """在第一个真实 C2 MoE block 上执行 mixed shared/partial 探查。"""
        if not self._probe_installed:
            raise P0Error("probe_not_installed", "必须在 probe context 内执行")
        import torch

        block = self._moe_blocks()[0]
        num_experts = len(block.experts)
        top_k = int(block.num_experts_per_tok)
        self._validate_request(request, num_experts, top_k)
        batch, sequence, _ = request.hidden_shape
        hidden_size = int(getattr(block.config, "hidden_size", request.hidden_shape[2]))
        first_parameter = next(block.parameters())
        original_training = bool(block.training)
        block.eval()
        try:
            # 探针只读取冻结权重，不构建 autograd 图，也不改变 train/eval 状态。
            with torch.no_grad():
                hidden = torch.zeros(
                    (batch, sequence, hidden_size),
                    dtype=first_parameter.dtype,
                    device=first_parameter.device,
                )
                flat_hidden = hidden.reshape(-1, hidden_size)
                flat_allowed = torch.tensor(
                    [row for rows in request.allowed_mask for row in rows],
                    dtype=torch.bool,
                    device=hidden.device,
                )
                flat_valid = torch.tensor(
                    [valid for rows in request.padding_mask for valid in rows],
                    dtype=torch.bool,
                    device=hidden.device,
                )
                output = torch.zeros_like(flat_hidden)
                selected_rows: List[Tuple[int, ...]] = [
                    tuple() for _ in range(batch * sequence)
                ]
                calls: List[ExpertCall] = []
                valid_indices = flat_valid.nonzero(as_tuple=False).flatten()
                shared_indices = (
                    (flat_valid & ~flat_allowed.any(dim=-1))
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                routed_indices = (
                    (flat_valid & flat_allowed.any(dim=-1))
                    .nonzero(as_tuple=False)
                    .flatten()
                )

                if shared_indices.numel() > 0:
                    shared_output = block.shared_experts(
                        flat_hidden.index_select(0, shared_indices)
                    )
                    output.index_copy_(
                        0, shared_indices, shared_output.to(output.dtype)
                    )
                    for global_row in shared_indices.tolist():
                        calls.append(
                            ExpertCall(
                                "shared",
                                global_row // sequence,
                                global_row % sequence,
                                global_row,
                                "shared",
                            )
                        )

                if routed_indices.numel() > 0:
                    routed_hidden = flat_hidden.index_select(0, routed_indices)
                    routed_allowed = flat_allowed.index_select(0, routed_indices)
                    topk_idx, topk_weight = self._masked_routing(
                        block, routed_hidden, routed_allowed
                    )
                    routed_output = torch.zeros_like(routed_hidden)
                    for local_row, global_row in enumerate(routed_indices.tolist()):
                        selected = tuple(
                            int(item) for item in topk_idx[local_row].tolist()
                        )
                        selected_rows[global_row] = selected
                        for topk_position, expert_index in enumerate(selected):
                            # 直接调用实际 ModuleList expert.forward，证明真实计算发生。
                            expert_output = block.experts[expert_index](
                                routed_hidden[local_row : local_row + 1]
                            )
                            expert_output = (
                                expert_output * topk_weight[local_row, topk_position]
                            )
                            routed_output[local_row : local_row + 1].add_(
                                expert_output.to(routed_output.dtype)
                            )
                            calls.append(
                                ExpertCall(
                                    f"E{expert_index}",
                                    global_row // sequence,
                                    global_row % sequence,
                                    global_row,
                                    "routed",
                                )
                            )
                    shared_output = block.shared_experts(routed_hidden)
                    routed_output.add_(shared_output.to(routed_output.dtype))
                    output.index_copy_(0, routed_indices, routed_output)
                    for global_row in routed_indices.tolist():
                        calls.append(
                            ExpertCall(
                                "shared",
                                global_row // sequence,
                                global_row % sequence,
                                global_row,
                                "shared",
                            )
                        )
        finally:
            block.train(original_training)

        indices = tuple(int(item) for item in valid_indices.tolist())
        return ProbeResult(
            output_shape=(batch, sequence, hidden_size),
            selected_ids=tuple(selected_rows),
            expert_calls=tuple(calls),
            original_indices=indices,
            reassembled_indices=indices,
            kv_bound=True,
        )

    def observe_all_allowed(self) -> dict[str, Any]:
        """比较原始 block 与只读 hooks 下的 all-allowed 输出和实际调用。"""
        import torch

        block = self._moe_blocks()[0]
        first_parameter = next(block.parameters())
        hidden_size = int(block.config.hidden_size)
        hidden = torch.zeros(
            (1, 2, hidden_size),
            dtype=first_parameter.dtype,
            device=first_parameter.device,
        )
        original_training = bool(block.training)
        block.eval()
        selected: List[Tuple[int, ...]] = []
        expert_calls = {index: 0 for index in range(len(block.experts))}
        shared_calls = {"count": 0}
        handles = []

        def gate_hook(module: Any, args: Any, output: Any) -> None:
            """只读记录原生 gate 的 selected IDs。"""
            topk_idx = output[0]
            selected.extend(
                tuple(int(item) for item in row) for row in topk_idx.tolist()
            )

        def expert_hook(index: int):
            """为每个实际 expert 构造只读调用 hook。"""

            def hook(module: Any, args: Any) -> None:
                expert_calls[index] += 1

            return hook

        def shared_hook(module: Any, args: Any) -> None:
            """记录原生 shared expert 调用。"""
            shared_calls["count"] += 1

        try:
            with torch.no_grad():
                baseline = block(hidden)
                handles.append(block.gate.register_forward_hook(gate_hook))
                handles.extend(
                    expert.register_forward_pre_hook(expert_hook(index))
                    for index, expert in enumerate(block.experts)
                )
                handles.append(
                    block.shared_experts.register_forward_pre_hook(shared_hook)
                )
                observed = block(hidden)
        finally:
            for handle in handles:
                handle.remove()
            block.train(original_training)
        return {
            "output_exact": bool(torch.equal(baseline, observed)),
            "selected_ids": tuple(selected),
            "routed_actual_calls": sum(expert_calls.values()),
            "shared_actual_calls": shared_calls["count"],
        }
