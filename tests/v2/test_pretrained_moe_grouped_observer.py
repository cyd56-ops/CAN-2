"""C1 grouped-mm observer 的无权重正向与负向测试。"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from can.v2.pretrained_moe_p0.grouped_mm_observer import (
    GroupedMMInvocation,
    GroupedMMObserver,
    validate_grouped_invocations,
)
from can.v2.pretrained_moe_p0.qwen_adapter import QwenHostAdapter
from can.v2.pretrained_moe_p0.real_types import (
    ArchitectureMap,
    ProbeRequest,
    ProbeResult,
)
from can.v2.pretrained_moe_p0.structure_probe import (
    make_mixed_probe_request,
    validate_probe_result,
)
from can.v2.pretrained_moe_p0.types import P0Error


def _grouped_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造满足 CPU grouped-mm 对齐要求的确定性输入。"""

    torch.manual_seed(20261003)
    return (
        torch.randn(4, 16),
        torch.randn(3, 16, 16),
        torch.tensor([1, 3, 4], dtype=torch.int32),
    )


def test_observer_captures_actual_grouped_mm_offsets() -> None:
    """observer 必须捕获实际 aten grouped-mm invocation 的 offsets。"""

    inputs, weights, offsets = _grouped_inputs()
    observer = GroupedMMObserver(layer_path="layers.0.mlp", projection_stage="up")
    with observer:
        output = torch._grouped_mm(inputs, weights, offs=offsets)
    assert output.shape == (4, 16)
    assert len(observer.invocations) == 1
    invocation = observer.invocations[0]
    assert invocation.operator_name == "aten::_grouped_mm"
    assert invocation.offsets == (1, 3, 4)
    assert invocation.counts == (1, 2, 1)
    assert invocation.projection_stage == "up"


def test_observer_maps_grouped_rows_to_original_tokens() -> None:
    """native top-k 排序置换必须还原到原始 token 行。"""

    inputs, weights, offsets = _grouped_inputs()
    observer = GroupedMMObserver(
        layer_path="layers.0.mlp", projection_stage="up", native_top_k=2
    )
    with observer:
        torch.sort(torch.tensor([0, 1, 1, 2]))
        torch._grouped_mm(inputs, weights, offs=offsets)
    assert observer.invocations[0].original_rows_by_expert == ((0,), (0, 1), (1,))


def test_observer_requires_sort_mapping_when_native_topk_is_declared() -> None:
    """声明 native top-k 后缺失排序置换必须 fail closed。"""

    inputs, weights, offsets = _grouped_inputs()
    observer = GroupedMMObserver(
        layer_path="layers.0.mlp", projection_stage="up", native_top_k=2
    )
    with pytest.raises(P0Error, match="grouped_mm_offsets_invalid"):
        with observer:
            torch._grouped_mm(inputs, weights, offs=offsets)


def test_observer_supports_empty_expert_group() -> None:
    """count=0 的 expert 必须保留在完整 offsets 中。"""

    inputs, weights, _ = _grouped_inputs()
    offsets = torch.tensor([0, 2, 2], dtype=torch.int32)
    observer = GroupedMMObserver(layer_path="layers.0.mlp", projection_stage="down")
    with observer:
        torch._grouped_mm(inputs, weights, offs=offsets)
    assert observer.invocations[0].counts == (0, 2, 0)


@pytest.mark.parametrize(
    "offsets",
    [
        torch.tensor([2, 1, 4], dtype=torch.int32),
        torch.tensor([1, 3, 5], dtype=torch.int32),
        torch.tensor([1, 3, 4], dtype=torch.float32),
    ],
)
def test_observer_rejects_invalid_offsets(offsets: torch.Tensor) -> None:
    """非单调、超输入范围或非整数 offsets 必须 fail closed。"""

    inputs, weights, _ = _grouped_inputs()
    observer = GroupedMMObserver(layer_path="layers.0.mlp")
    with pytest.raises(P0Error, match="grouped_mm_offsets_invalid"):
        with observer:
            torch._grouped_mm(inputs, weights, offs=offsets)


def test_observer_masks_topk_before_original_operator() -> None:
    """test-only mask 应在原生 top-k operator 输入前生效。"""

    scores = torch.tensor([[0.1, 0.9, 0.2], [0.4, 0.3, 0.2]])
    allowed = torch.tensor([[True, False, True], [False, False, False]])
    observer = GroupedMMObserver(allowed_mask=allowed, sentinel_index=3)
    with observer:
        values, indices = torch.topk(scores, k=2, dim=-1)
    assert set(indices[0].tolist()).issubset({0, 2})
    assert indices[1].tolist() == [3, 3]
    assert values[1].tolist() == [1.0, 1.0]


def test_observer_reentry_and_stage_mismatch_are_rejected() -> None:
    """observer 重入及 up/down 结构不一致必须拒绝。"""

    observer = GroupedMMObserver()
    observer.__enter__()
    try:
        with pytest.raises(P0Error, match="probe_reentry_rejected"):
            observer.__enter__()
    finally:
        observer.__exit__(None, None, None)
    first = GroupedMMInvocation(
        0, "op", "layer", "up", 4, 3, (1, 3, 4), (1, 2, 1), "cpu", "int32"
    )
    second = replace(first, invocation_index=1, projection_stage="up")
    with pytest.raises(P0Error, match="grouped_mm_stage_mismatch"):
        validate_grouped_invocations((first, second))


def test_qwen_adapter_uses_native_grouped_mm_offsets() -> None:
    """Qwen stand-in 必须用原生 grouped operator 产生 execution evidence。"""

    class Qwen2MoeExperts(torch.nn.Module):
        """实现最小 Qwen grouped expert forward。"""

        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 3
            self.gate_up_proj = torch.nn.Parameter(torch.randn(3, 16, 32))
            self.down_proj = torch.nn.Parameter(torch.randn(3, 16, 16))

        def forward(self, hidden, top_k_index, top_k_weights):
            """按 Qwen grouped-mm 语义执行两次真实 grouped operator。"""
            num_tokens, top_k = hidden.shape[0], top_k_index.shape[-1]
            flat_ids = top_k_index.reshape(-1)
            flat_hidden = hidden.reshape(-1, 16)
            flat_weights = top_k_weights.reshape(-1)
            sorted_ids, permutation = torch.sort(flat_ids)
            grouped_hidden = flat_hidden[permutation // top_k]
            grouped_weights = flat_weights[permutation]
            counts = torch.histc(
                sorted_ids.float(),
                bins=self.num_experts,
                min=0,
                max=self.num_experts - 1,
            )
            offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
            sentinel = sorted_ids >= self.num_experts
            grouped_hidden = grouped_hidden.masked_fill(sentinel.unsqueeze(-1), 0.0)
            up = torch._grouped_mm(grouped_hidden, self.gate_up_proj, offs=offsets)
            up = up.masked_fill(sentinel.unsqueeze(-1), 0.0)
            gate, value = up.chunk(2, dim=-1)
            down_input = torch.sigmoid(gate) * value
            down = torch._grouped_mm(down_input, self.down_proj, offs=offsets)
            down = down.masked_fill(sentinel.unsqueeze(-1), 0.0)
            weighted = down * grouped_weights.unsqueeze(-1)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(permutation.numel())
            return weighted[inverse].view(num_tokens, top_k, 16).sum(dim=1)

    class Router(torch.nn.Module):
        """返回 Qwen router 的 logits、scores 和 indices。"""

        def __init__(self) -> None:
            super().__init__()
            self.top_k = 2
            self.weight = torch.nn.Parameter(torch.zeros(3, 16))

        def forward(self, hidden):
            """执行固定 top-k 路由。"""
            logits = torch.nn.functional.linear(hidden.reshape(-1, 16), self.weight)
            probs = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(probs, 2, dim=-1)
            values = values / values.sum(dim=-1, keepdim=True)
            return logits, values, indices

    class Shared(torch.nn.Module):
        """提供始终执行的 shared branch。"""

        def forward(self, hidden):
            """返回 shared 恒等输出。"""
            return hidden

    class Block(torch.nn.Module):
        """组合 Qwen router、shared expert 和 packed routed expert。"""

        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=16)
            self.gate = Router()
            self.experts = Qwen2MoeExperts()
            self.shared_expert = Shared()

        def forward(self, hidden):
            """执行 shared+routed 输出。"""
            _, scores, indices = self.gate(hidden)
            routed = self.experts(hidden, indices, scores).reshape_as(hidden)
            return routed + self.shared_expert(hidden)

    class Model(torch.nn.Module):
        """提供真实 adapter 所需的命名模块和配置。"""

        def __init__(self) -> None:
            super().__init__()
            self.block = Block()
            self.config = SimpleNamespace(
                _experts_implementation="grouped_mm",
                num_experts=3,
                num_experts_per_tok=2,
            )

    architecture = ArchitectureMap(
        "qwen2_moe",
        "Model",
        ("block",),
        ("block.gate",),
        ("block.shared_expert",),
        ("block.experts",),
        2,
        "grouped_mm",
        True,
        True,
        3,
    )
    adapter = QwenHostAdapter(SimpleNamespace(model=Model(), architecture=architecture))
    inspected = adapter.inspect_architecture()
    assert inspected.supports_execution_counter is True
    request = ProbeRequest(
        hidden_shape=(2, 1, 16),
        allowed_mask=(
            ((False, False, False),),
            ((True, True, True),),
        ),
        native_top_k=2,
        padding_mask=((True,), (True,)),
        request_ids=("r0", "r1"),
    )
    with adapter.install_probe():
        result = adapter.run_moe_probe(request)
    validate_probe_result(request, result, 3)
    assert result.execution_observer_kind == "grouped_mm_offsets"
    assert sum(result.grouped_mm_invocations[0].counts) == 2
    assert result.grouped_mm_invocations[0].counts[2] == 0
    assert result.grouped_mm_invocations[0].original_rows_by_expert == ((1,), (1,), ())


def test_grouped_mm_mapping_rejects_forbidden_original_row() -> None:
    """行映射把 shared-only token 归入 routed expert 时必须拒绝。"""

    request = make_mixed_probe_request(batch=2, sequence=2, experts=5, top_k=4)
    selected = ((5, 5, 5, 5), (5, 5, 5, 5), (0, 1, 2, 3), (0, 1, 2, 3))
    counts = (3, 3, 3, 3, 0)
    offsets = (3, 6, 9, 12, 12)
    bad_rows = ((0, 1, 2), (1, 2, 3), (1, 2, 3), (1, 2, 3), ())
    invocation = GroupedMMInvocation(
        0,
        "aten::_grouped_mm",
        "layers.0.mlp",
        "up",
        12,
        5,
        offsets,
        counts,
        "cpu",
        "torch.float32",
        bad_rows,
    )
    second = replace(invocation, invocation_index=1, projection_stage="down")
    result = ProbeResult(
        output_shape=(4, 8),
        selected_ids=selected,
        expert_calls=(),
        original_indices=(0, 1, 2, 3),
        reassembled_indices=(0, 1, 2, 3),
        kv_bound=True,
        grouped_mm_invocations=(invocation, second),
        execution_observer_kind="grouped_mm_offsets",
    )
    with pytest.raises(P0Error, match="forbidden_expert_called"):
        validate_probe_result(request, result, 5)
