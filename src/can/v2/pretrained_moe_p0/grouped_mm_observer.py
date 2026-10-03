"""C1 grouped-GEMM 原生调用的只读观测器。

观测器只在显式 probe context 内使用，通过 ``TorchDispatchMode`` 捕获真实
grouped operator 的 ``offsets`` 参数；它不替换 backend、不修改输入或输出，
也不根据 router 输出离线重算执行台账。
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from .types import P0Error


@dataclass(frozen=True)
class GroupedMMInvocation:
    """保存一次实际 grouped-matmul invocation 的可审计整数事实。"""

    invocation_index: int
    operator_name: str
    layer_path: str
    projection_stage: str
    input_rows: int
    expert_count: int
    offsets: Tuple[int, ...]
    counts: Tuple[int, ...]
    device: str
    dtype: str
    original_rows_by_expert: Tuple[Tuple[int, ...], ...] = ()


def _operator_name(function: Any) -> str:
    """返回 Torch operator 的稳定名称，兼容不同 PyTorch 版本。"""

    name = getattr(function, "name", None)
    if callable(name):
        try:
            value = name()
        except TypeError:
            value = None
        if isinstance(value, str):
            return value
    return str(function)


def _find_offsets(args: Sequence[Any], kwargs: Dict[str, Any]) -> Any:
    """从 grouped operator 参数中定位原生 offsets tensor。"""

    for key in ("offs", "offsets"):
        candidate = kwargs.get(key)
        if candidate is not None:
            return candidate
    if len(args) >= 3:
        return args[2]
    return None


def _counts_from_offsets(offsets: Tuple[int, ...]) -> Tuple[int, ...]:
    """把非递减累计 offsets 转换为逐 expert 行数并严格校验。"""

    if not offsets:
        raise P0Error("grouped_mm_offsets_invalid", "offsets 不能为空")
    if any(value < 0 for value in offsets):
        raise P0Error("grouped_mm_offsets_invalid", "offsets 不能为负数")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise P0Error("grouped_mm_offsets_invalid", "offsets 必须单调非递减")
    previous = 0
    counts = []
    for value in offsets:
        counts.append(value - previous)
        previous = value
    return tuple(counts)


class GroupedMMObserver(AbstractContextManager["GroupedMMObserver"]):
    """在局部 probe 中捕获实际 grouped-mm invocation。"""

    def __init__(
        self,
        *,
        layer_path: str = "",
        projection_stage: Optional[str] = None,
        allowed_mask: Any = None,
        sentinel_index: Optional[int] = None,
        native_top_k: Optional[int] = None,
    ) -> None:
        """创建 observer。

        参数:
            layer_path: 当前 probe 绑定的 MoE 层路径。
            projection_stage: 单层 probe 时预登记的 ``up`` 或 ``down``；
                未指定时按同层调用顺序推断。
        """

        if not isinstance(layer_path, str):
            raise TypeError("layer_path 必须是字符串")
        if projection_stage not in (None, "up", "down"):
            raise ValueError("projection_stage 必须是 up、down 或 None")
        self.layer_path = layer_path
        self._fixed_stage = projection_stage
        self._allowed_mask = allowed_mask
        self._sentinel_index = sentinel_index
        self._native_top_k = native_top_k
        self._mode: Any = None
        self._invocations: list[GroupedMMInvocation] = []
        self._stage_counts: Dict[str, int] = {}
        self._entered = False
        self._last_sorted_ids: Optional[Tuple[int, ...]] = None
        self._last_permutation: Optional[Tuple[int, ...]] = None

    @property
    def invocations(self) -> Tuple[GroupedMMInvocation, ...]:
        """返回已经捕获的不可变 invocation 列表。"""

        return tuple(self._invocations)

    def _stage_for(self, layer_path: str) -> str:
        """根据实际层路径和调用顺序确定 up/down 阶段。"""

        if self._fixed_stage is not None:
            return self._fixed_stage
        count = self._stage_counts.get(layer_path, 0)
        self._stage_counts[layer_path] = count + 1
        if count == 0:
            return "up"
        if count == 1:
            return "down"
        raise P0Error("grouped_mm_stage_mismatch", "同一 MoE 层出现多于两次 grouped-mm")

    def _capture(
        self, function: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        """读取实际 operator 参数并登记 offsets/counts。"""

        offsets_tensor = _find_offsets(args, kwargs)
        if offsets_tensor is None or not hasattr(offsets_tensor, "detach"):
            raise P0Error(
                "grouped_mm_offsets_invalid",
                "实际 grouped-mm invocation 缺少 offsets tensor",
            )
        if getattr(offsets_tensor, "ndim", None) != 1:
            raise P0Error("grouped_mm_offsets_invalid", "offsets 必须是一维 tensor")
        dtype = getattr(offsets_tensor, "dtype", None)
        dtype_name = str(dtype)
        if not (dtype_name.endswith("int32") or dtype_name.endswith("int64")):
            raise P0Error("grouped_mm_offsets_invalid", "offsets 必须是整数 tensor")
        try:
            offsets = tuple(
                int(value) for value in offsets_tensor.detach().cpu().tolist()
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise P0Error("grouped_mm_offsets_invalid", "offsets 无法读取") from exc
        counts = _counts_from_offsets(offsets)
        input_tensor = args[0] if args else None
        weight_tensor = args[1] if len(args) > 1 else None
        input_rows = int(getattr(input_tensor, "shape", (0,))[0])
        expert_count = int(getattr(weight_tensor, "shape", (len(offsets),))[0])
        if expert_count != len(offsets):
            raise P0Error(
                "grouped_mm_offsets_invalid",
                "offsets 长度与 grouped weight expert 数不一致",
            )
        if offsets[-1] > input_rows:
            raise P0Error("grouped_mm_offsets_invalid", "offsets 末值超过输入行数")
        original_rows_by_expert: Tuple[Tuple[int, ...], ...] = ()
        if self._native_top_k is not None:
            if self._native_top_k <= 0:
                raise P0Error("grouped_mm_offsets_invalid", "native top-k 必须为正数")
            if (
                self._last_sorted_ids is None
                or self._last_permutation is None
                or len(self._last_sorted_ids) != input_rows
                or len(self._last_permutation) != input_rows
            ):
                raise P0Error(
                    "grouped_mm_offsets_invalid",
                    "grouped-mm 缺少与当前输入行对应的排序置换",
                )
            if tuple(sorted(self._last_permutation)) != tuple(range(input_rows)):
                raise P0Error(
                    "grouped_mm_offsets_invalid",
                    "排序置换不是当前 grouped 输入行的完整排列",
                )
            if any(
                left > right
                for left, right in zip(self._last_sorted_ids, self._last_sorted_ids[1:])
            ):
                raise P0Error(
                    "grouped_mm_offsets_invalid", "expert 排序结果不是非递减序列"
                )
            rows = []
            previous = 0
            for expert_index, count in enumerate(counts):
                segment = self._last_sorted_ids[previous : previous + count]
                if any(item != expert_index for item in segment):
                    raise P0Error(
                        "grouped_mm_offsets_invalid",
                        "offsets 分组与实际排序 expert ID 不一致",
                    )
                rows.append(
                    tuple(
                        self._last_permutation[position] // self._native_top_k
                        for position in range(previous, previous + count)
                    )
                )
                previous += count
            if any(item < expert_count for item in self._last_sorted_ids[previous:]):
                raise P0Error(
                    "grouped_mm_offsets_invalid",
                    "offsets 未覆盖实际 grouped 输入中的 routed expert 行",
                )
            original_rows_by_expert = tuple(rows)
        layer_path = self.layer_path
        stage = self._stage_for(layer_path)
        self._invocations.append(
            GroupedMMInvocation(
                invocation_index=len(self._invocations),
                operator_name=_operator_name(function),
                layer_path=layer_path,
                projection_stage=stage,
                input_rows=input_rows,
                expert_count=expert_count,
                offsets=offsets,
                counts=counts,
                device=str(getattr(offsets_tensor, "device", "unknown")),
                dtype=dtype_name,
                original_rows_by_expert=original_rows_by_expert,
            )
        )

    def __enter__(self) -> "GroupedMMObserver":
        """安装局部 Torch dispatch observer。"""

        if self._entered:
            raise P0Error("probe_reentry_rejected", "grouped-mm observer 禁止重入")
        try:
            import torch
            from torch.utils._python_dispatch import TorchDispatchMode
        except (
            ImportError,
            AttributeError,
        ) as exc:  # pragma: no cover - optional runtime
            raise P0Error(
                "grouped_mm_invocation_unobservable",
                "当前 PyTorch 不提供可用的 TorchDispatchMode",
            ) from exc

        observer = self

        class _Mode(TorchDispatchMode):
            """只读捕获 grouped-mm operator 的局部 dispatch mode。"""

            def __torch_dispatch__(
                self,
                function: Any,
                types: Tuple[Any, ...] = (),
                args: Tuple[Any, ...] = (),
                kwargs: Optional[Dict[str, Any]] = None,
            ) -> Any:
                """捕获 offsets 后原样调用原始 operator。"""

                actual_kwargs = dict(kwargs or {})
                name = _operator_name(function).lower()
                if "sort" in name and args and getattr(args[0], "ndim", None) == 1:
                    output = function(*args, **actual_kwargs)
                    values = getattr(output, "values", None)
                    indices = getattr(output, "indices", None)
                    if (
                        values is None
                        and isinstance(output, (tuple, list))
                        and len(output) == 2
                    ):
                        values, indices = output[0], output[1]
                    if (
                        values is not None
                        and indices is not None
                        and str(getattr(args[0], "dtype", "")).endswith(
                            ("int32", "int64")
                        )
                        and getattr(values, "ndim", None) == 1
                        and getattr(indices, "ndim", None) == 1
                        and tuple(values.shape) == tuple(indices.shape)
                    ):
                        try:
                            observer._last_sorted_ids = tuple(
                                int(value) for value in values.detach().cpu().tolist()
                            )
                            observer._last_permutation = tuple(
                                int(value) for value in indices.detach().cpu().tolist()
                            )
                        except (TypeError, ValueError, RuntimeError) as exc:
                            raise P0Error(
                                "grouped_mm_offsets_invalid",
                                "无法读取 grouped-mm 的排序置换",
                            ) from exc
                    return output
                if "topk" in name and observer._allowed_mask is not None:
                    return observer._masked_topk(function, args, actual_kwargs)
                if "grouped_mm" in name:
                    observer._capture(function, args, actual_kwargs)
                return function(*args, **actual_kwargs)

        self._mode = _Mode()
        self._mode.__enter__()
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        """卸载局部 dispatch mode，并拒绝吞掉异常。"""

        try:
            if self._mode is not None:
                self._mode.__exit__(exc_type, exc, traceback)
        finally:
            self._mode = None
            self._entered = False
        return False

    def _masked_topk(
        self, function: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """在原生 top-k operator 输入前应用 test-only allowed mask。"""

        if not args or not hasattr(args[0], "masked_fill"):
            raise P0Error("host_interface_not_controllable", "top-k 输入不可屏蔽")
        allowed = self._allowed_mask
        if tuple(args[0].shape) != tuple(allowed.shape):
            raise P0Error(
                "mask_shape_invalid", "router top-k 行数与 allowed mask 不匹配"
            )
        masked_args = list(args)
        masked_args[0] = args[0].masked_fill(~allowed, float("-inf"))
        output = function(*tuple(masked_args), **kwargs)
        values = getattr(output, "values", None)
        indices = getattr(output, "indices", None)
        if values is None and isinstance(output, (tuple, list)) and len(output) == 2:
            values, indices = output[0], output[1]
        if values is None or indices is None:
            raise P0Error("host_interface_not_controllable", "top-k 返回结构不可审计")
        all_false = ~allowed.any(dim=-1)
        if bool(all_false.any().item()):
            sentinel = self._sentinel_index
            if sentinel is None:
                raise P0Error("mask_shape_invalid", "全拒绝行缺少 sentinel expert")
            indices = indices.masked_fill(all_false.unsqueeze(-1), int(sentinel))
            # Qwen router 后续可能重新归一化 top-k 权重；避免 0/0 产生 NaN。
            values = values.masked_fill(all_false.unsqueeze(-1), 1.0)
        try:
            return output._replace(values=values, indices=indices)
        except AttributeError:
            if isinstance(output, tuple):
                return (values, indices)
            return type(output)(values, indices)


def validate_grouped_invocations(
    invocations: Sequence[GroupedMMInvocation],
) -> None:
    """验证单层 grouped-mm up/down invocation 的结构一致性。"""

    if len(invocations) != 2:
        raise P0Error(
            "grouped_mm_stage_mismatch",
            "每个受测 MoE 层必须捕获恰好 up/down 两次 grouped-mm",
        )
    first, second = invocations
    if (first.projection_stage, second.projection_stage) != ("up", "down"):
        raise P0Error("grouped_mm_stage_mismatch", "grouped-mm 阶段不是 up/down")
    if first.layer_path != second.layer_path:
        raise P0Error("grouped_mm_stage_mismatch", "up/down 不属于同一 MoE 层")
    if first.offsets != second.offsets or first.counts != second.counts:
        raise P0Error("grouped_mm_stage_mismatch", "up/down offsets 不一致")
    if (
        first.input_rows != second.input_rows
        or first.expert_count != second.expert_count
    ):
        raise P0Error("grouped_mm_stage_mismatch", "up/down 输入或 expert 数不一致")
    if (
        first.original_rows_by_expert != second.original_rows_by_expert
        or len(first.original_rows_by_expert) != first.expert_count
    ):
        raise P0Error(
            "grouped_mm_stage_mismatch",
            "up/down 缺少一致的原始 token 行映射",
        )
