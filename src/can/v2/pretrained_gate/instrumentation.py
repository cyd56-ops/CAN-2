"""G0 受信测试使用的实际调用记录器。"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from .types import CallRecord, TrustedRequestContext


class CallLedger:
    """按真实 module call 记录组件、阶段和参与请求身份。"""

    def __init__(self, context: TrustedRequestContext) -> None:
        """绑定一次 batch 的完整受信请求上下文。"""

        if not isinstance(context, TrustedRequestContext):
            raise TypeError("context 必须是 TrustedRequestContext")
        self._allowed_ids = frozenset(context.request_ids)
        self._records: list[CallRecord] = []

    def record(
        self,
        component: str,
        stage: str,
        request_ids: Iterable[str],
    ) -> None:
        """记录一次真实调用，不允许未知或重复请求身份。"""

        if not isinstance(component, str) or not component:
            raise ValueError("component 必须是非空字符串")
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage 必须是非空字符串")
        rows = tuple(request_ids)
        if not rows or any(not isinstance(value, str) or not value for value in rows):
            raise ValueError("调用记录必须包含非空 request IDs")
        if len(set(rows)) != len(rows):
            raise ValueError("一次调用中的 request IDs 不得重复")
        if any(value not in self._allowed_ids for value in rows):
            raise PermissionError("调用记录包含当前 batch 之外的 request ID")
        self._records.append(CallRecord(component, stage, rows))

    @property
    def records(self) -> Tuple[CallRecord, ...]:
        """返回不可变调用记录快照。"""

        return tuple(self._records)

    def count(self, component: str, request_id: Optional[str] = None) -> int:
        """统计组件总调用数或指定请求参与的调用数。"""

        if not isinstance(component, str) or not component:
            raise ValueError("component 必须是非空字符串")
        if request_id is not None and (
            not isinstance(request_id, str) or not request_id
        ):
            raise ValueError("request_id 必须是非空字符串或 None")
        return sum(
            1
            for record in self._records
            if record.component == component
            and (request_id is None or request_id in record.request_ids)
        )


class InstrumentedModuleCall:
    """在宿主实际调用点统一写入调用台账。"""

    def __init__(self, ledger: CallLedger, component: str) -> None:
        """绑定调用台账和稳定组件名称。"""

        if not isinstance(ledger, CallLedger):
            raise TypeError("ledger 必须是 CallLedger")
        if not isinstance(component, str) or not component:
            raise ValueError("component 必须是非空字符串")
        self.ledger = ledger
        self.component = component

    def record(self, stage: str, request_ids: Iterable[str]) -> None:
        """在被测模块实际执行处记录一次调用。"""

        self.ledger.record(self.component, stage, request_ids)
