"""请求级 KV-cache 绑定与增量步预检。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class CacheBinding:
    """描述一次请求的受信 cache 状态元数据。"""

    request_id: str
    execution_config_id: str
    model_sha256: str
    policy_id: str
    route: str
    cut_layer: int
    processed_valid_tokens: int
    physical_kv_length: int
    valid_mask: Tuple[bool, ...]
    position: int
    active: bool = True
    _registry_seal: object = None


class CacheStateError(RuntimeError):
    """表示增量步在 prefix 前发现 cache 状态不匹配。"""

    def __init__(self, reason: str) -> None:
        """保存稳定的脱敏错误码。"""

        if not isinstance(reason, str) or not reason:
            raise ValueError("reason 必须是非空字符串")
        self.reason = reason
        super().__init__(f"cache state validation failed: {reason}")


def validate_cache_binding(
    binding: CacheBinding,
    *,
    request_id: str,
    execution_config_id: str,
    model_sha256: str,
    policy_id: str,
    route: str,
    cut_layer: int,
    expected_processed_valid_tokens: int,
    expected_position: int,
    expected_valid_mask: Optional[Tuple[bool, ...]] = None,
) -> None:
    """在任何增量计算前校验 cache 与受信请求状态一致。"""

    if not isinstance(binding, CacheBinding):
        raise TypeError("binding 必须是 CacheBinding")
    expected = {
        "request_id": request_id,
        "execution_config_id": execution_config_id,
        "model_sha256": model_sha256,
        "policy_id": policy_id,
        "route": route,
        "cut_layer": cut_layer,
        "processed_valid_tokens": expected_processed_valid_tokens,
        "position": expected_position,
    }
    for name, value in expected.items():
        if getattr(binding, name) != value:
            raise CacheStateError(name)
    if not binding.active:
        raise CacheStateError("inactive_request")
    if binding.physical_kv_length < binding.processed_valid_tokens:
        raise CacheStateError("physical_length")
    if binding.processed_valid_tokens < 0 or binding.position < 0:
        raise CacheStateError("negative_position")
    if len(binding.valid_mask) != binding.physical_kv_length:
        raise CacheStateError("valid_mask_length")
    if any(not isinstance(value, bool) for value in binding.valid_mask):
        raise CacheStateError("valid_mask_type")
    if sum(binding.valid_mask) != binding.processed_valid_tokens:
        raise CacheStateError("valid_mask_count")
    if expected_valid_mask is not None and binding.valid_mask != expected_valid_mask:
        raise CacheStateError("valid_mask")


def bind_cache(
    request_id: str,
    execution_config_id: str,
    model_sha256: str,
    policy_id: str,
    route: str,
    cut_layer: int,
    processed_valid_tokens: int,
    physical_kv_length: int,
    valid_mask: Tuple[bool, ...],
    position: int,
    registry_seal: object = None,
) -> CacheBinding:
    """创建一次受信 cache 绑定；不接受外部 Tensor/cache 对象。"""

    return CacheBinding(
        request_id=request_id,
        execution_config_id=execution_config_id,
        model_sha256=model_sha256,
        policy_id=policy_id,
        route=route,
        cut_layer=cut_layer,
        processed_valid_tokens=processed_valid_tokens,
        physical_kv_length=physical_kv_length,
        valid_mask=valid_mask,
        position=position,
        _registry_seal=registry_seal,
    )


@dataclass(frozen=True)
class CacheHandle:
    """表示 registry 内部 cache 状态的不透明句柄。"""

    request_id: str
    _token: object
    _registry_seal: object


@dataclass
class _CacheEntry:
    """保存 registry 私有的绑定和每层 K/V 张量。"""

    binding: CacheBinding
    keys: Tuple[Tensor, ...] = ()
    values: Tuple[Tensor, ...] = ()


class CacheRegistry:
    """创建、保存并原子预检一次 generate 生命周期的 cache 元数据。"""

    def __init__(
        self,
        execution_config_id: str,
        model_sha256: str,
        policy_id: str,
        cut_layer: int,
        layer_count: int = 1,
    ) -> None:
        """绑定不可跨用的模型、配置、policy 和 cut。"""

        for name, value in (
            ("execution_config_id", execution_config_id),
            ("model_sha256", model_sha256),
            ("policy_id", policy_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} 必须是非空字符串")
        if (
            isinstance(cut_layer, bool)
            or not isinstance(cut_layer, int)
            or cut_layer < 0
        ):
            raise ValueError("cut_layer 必须是非负整数")
        if (
            isinstance(layer_count, bool)
            or not isinstance(layer_count, int)
            or layer_count < 1
        ):
            raise ValueError("layer_count 必须是正整数")
        self.execution_config_id = execution_config_id
        self.model_sha256 = model_sha256
        self.policy_id = policy_id
        self.cut_layer = cut_layer
        self.layer_count = layer_count
        self._registry_seal = object()
        self._entries: Dict[object, _CacheEntry] = {}

    def create_protected(
        self,
        request_id: str,
        processed_valid_tokens: int,
        physical_kv_length: int,
        position: int,
    ) -> CacheHandle:
        """为一次已授权 PROTECTED 请求创建内部 cache 绑定。"""

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id 必须是非空字符串")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (processed_valid_tokens, physical_kv_length, position)
        ):
            raise ValueError("cache 长度和位置必须是非负整数")
        if physical_kv_length < processed_valid_tokens:
            raise ValueError("物理 K/V 长度不能小于有效 token 数")
        token = object()
        binding = bind_cache(
            request_id,
            self.execution_config_id,
            self.model_sha256,
            self.policy_id,
            "protected",
            self.cut_layer,
            processed_valid_tokens,
            physical_kv_length,
            (True,) * processed_valid_tokens
            + (False,) * (physical_kv_length - processed_valid_tokens),
            position,
            self._registry_seal,
        )
        self._entries[token] = _CacheEntry(binding)
        return CacheHandle(request_id, token, self._registry_seal)

    def create_protected_kv(
        self,
        request_id: str,
        keys: Tuple[Tensor, ...],
        values: Tuple[Tensor, ...],
        valid_mask: Tuple[bool, ...],
        position: int,
    ) -> CacheHandle:
        """为真实增量宿主保存每层 K/V，并由 mask 推导长度。

        该方法只供受信 host plugin 使用；调用方不能提交自定义 route 或长度。
        """

        if not isinstance(valid_mask, tuple) or not valid_mask:
            raise ValueError("valid_mask 必须是非空 bool tuple")
        if any(not isinstance(value, bool) for value in valid_mask):
            raise TypeError("valid_mask 元素必须是 bool")
        self._validate_kv(keys, values, len(valid_mask))
        handle = self.create_protected(
            request_id,
            sum(valid_mask),
            len(valid_mask),
            position,
        )
        self._entries[handle._token] = _CacheEntry(
            replace(self._entries[handle._token].binding, valid_mask=valid_mask),
            keys,
            values,
        )
        return handle

    def preflight_batch(
        self,
        handles: Tuple[CacheHandle, ...],
        request_ids: Tuple[str, ...],
        processed_valid_tokens: Tuple[int, ...],
        positions: Tuple[int, ...],
    ) -> Tuple[CacheBinding, ...]:
        """先验证完整活动 batch，再允许本步读取或更新任何 cache。"""

        if not isinstance(handles, tuple) or not handles:
            raise CacheStateError("empty_batch")
        if not (
            len(handles)
            == len(request_ids)
            == len(processed_valid_tokens)
            == len(positions)
        ):
            raise CacheStateError("batch_size")
        if len(set(request_ids)) != len(request_ids):
            raise CacheStateError("duplicate_request_id")
        validated = []
        for handle, request_id, processed, position in zip(
            handles, request_ids, processed_valid_tokens, positions
        ):
            binding = self._resolve(handle)
            if handle.request_id != request_id:
                raise CacheStateError("handle_request_id")
            validate_cache_binding(
                binding,
                request_id=request_id,
                execution_config_id=self.execution_config_id,
                model_sha256=self.model_sha256,
                policy_id=self.policy_id,
                route="protected",
                cut_layer=self.cut_layer,
                expected_processed_valid_tokens=processed,
                expected_position=position,
            )
            validated.append(binding)
        return tuple(validated)

    def preflight_kv_batch(
        self,
        handles: Tuple[CacheHandle, ...],
        request_ids: Tuple[str, ...],
        valid_masks: Tuple[Tuple[bool, ...], ...],
        positions: Tuple[int, ...],
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        """原子核对整批 cache 后返回 K/V；任一行失败则不返回部分状态。"""

        if not (
            isinstance(handles, tuple)
            and isinstance(request_ids, tuple)
            and isinstance(valid_masks, tuple)
            and isinstance(positions, tuple)
        ):
            raise TypeError("cache batch 参数必须是 tuple")
        if not handles or not (
            len(handles) == len(request_ids) == len(valid_masks) == len(positions)
        ):
            raise CacheStateError("batch_size")
        if len(set(request_ids)) != len(request_ids):
            raise CacheStateError("duplicate_request_id")
        entries = []
        for handle, request_id, valid_mask, position in zip(
            handles, request_ids, valid_masks, positions
        ):
            entry = self._resolve_entry(handle)
            if not isinstance(valid_mask, tuple) or any(
                not isinstance(value, bool) for value in valid_mask
            ):
                raise CacheStateError("valid_mask_type")
            if handle.request_id != request_id:
                raise CacheStateError("handle_request_id")
            validate_cache_binding(
                entry.binding,
                request_id=request_id,
                execution_config_id=self.execution_config_id,
                model_sha256=self.model_sha256,
                policy_id=self.policy_id,
                route="protected",
                cut_layer=self.cut_layer,
                expected_processed_valid_tokens=sum(valid_mask),
                expected_position=position,
                expected_valid_mask=valid_mask,
            )
            self._validate_kv(entry.keys, entry.values, len(valid_mask))
            entries.append(entry)
        return (
            tuple(
                torch.cat([entry.keys[layer] for entry in entries], dim=0)
                for layer in range(self.layer_count)
            ),
            tuple(
                torch.cat([entry.values[layer] for entry in entries], dim=0)
                for layer in range(self.layer_count)
            ),
        )

    def finish(self, handle: CacheHandle) -> None:
        """结束请求并使句柄不可继续复用。"""

        entry = self._resolve_entry(handle)
        entry.binding = replace(entry.binding, active=False)

    def clear(self) -> None:
        """清理本次 generate 生命周期的全部 cache 状态。"""

        self._entries.clear()

    def _resolve_entry(self, handle: CacheHandle) -> _CacheEntry:
        """验证不透明句柄来源并取得内部状态。"""

        if not isinstance(handle, CacheHandle):
            raise TypeError("handle 必须是 CacheHandle")
        if handle._registry_seal is not self._registry_seal:
            raise CacheStateError("registry")
        if handle._token not in self._entries:
            raise CacheStateError("unknown_handle")
        entry = self._entries[handle._token]
        if entry.binding._registry_seal is not self._registry_seal:
            raise CacheStateError("binding_source")
        return entry

    def read_kv(self, handle: CacheHandle) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        """由受信 runtime 读取已预检的 K/V；不接受外部 cache 对象。"""

        entry = self._resolve_entry(handle)
        validate_cache_binding(
            entry.binding,
            request_id=entry.binding.request_id,
            execution_config_id=self.execution_config_id,
            model_sha256=self.model_sha256,
            policy_id=self.policy_id,
            route="protected",
            cut_layer=self.cut_layer,
            expected_processed_valid_tokens=entry.binding.processed_valid_tokens,
            expected_position=entry.binding.position,
            expected_valid_mask=entry.binding.valid_mask,
        )
        return entry.keys, entry.values

    def _resolve(self, handle: CacheHandle) -> CacheBinding:
        """兼容返回 binding 的内部辅助方法。"""

        return self._resolve_entry(handle).binding

    def _replace_for_test(self, handle: CacheHandle, **changes: object) -> None:
        """仅供受信测试注入错误元数据，不属于服务公开入口。"""

        entry = self._resolve_entry(handle)
        entry.binding = replace(entry.binding, **changes)

    def _replace_kv_for_test(
        self,
        handle: CacheHandle,
        keys: Tuple[Tensor, ...],
        values: Tuple[Tensor, ...],
    ) -> None:
        """仅供受信测试注入错误 K/V，不属于服务公开入口。"""

        entry = self._resolve_entry(handle)
        entry.keys = keys
        entry.values = values

    def _validate_kv(
        self,
        keys: Tuple[Tensor, ...],
        values: Tuple[Tensor, ...],
        physical_length: int,
    ) -> None:
        """验证每层 K/V 的数量、shape、dtype、device 与物理长度。"""

        if not isinstance(keys, tuple) or not isinstance(values, tuple):
            raise TypeError("keys/values 必须是 tuple")
        if len(keys) != self.layer_count or len(values) != self.layer_count:
            raise CacheStateError("layer_count")
        devices = set()
        for key, value in zip(keys, values):
            if not isinstance(key, Tensor) or not isinstance(value, Tensor):
                raise TypeError("K/V 必须是 Tensor")
            if key.ndim != 3 or key.shape != value.shape:
                raise CacheStateError("kv_shape")
            if key.shape[0] != 1:
                raise CacheStateError("kv_batch")
            if key.shape[-2] != physical_length:
                raise CacheStateError("physical_length")
            if key.dtype != value.dtype or not key.dtype.is_floating_point:
                raise CacheStateError("kv_dtype")
            if key.device != value.device:
                raise CacheStateError("kv_device")
            devices.add(key.device)
        if len(devices) != 1:
            raise CacheStateError("cross_layer_device")
