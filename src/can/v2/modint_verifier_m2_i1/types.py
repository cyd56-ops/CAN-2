"""I1 集成层的固定标识、错误和数值范围。"""

from __future__ import annotations

I1_EXECUTION_CONFIG_ID = "i1-g1b-m2-integration-v1"
I1_PROTOCOL_ID = "i1-g1b-m2-v1"
FP32_EXACT_INTEGER_LIMIT = 2**24


class I1Error(ValueError):
    """表示 I1 输入、证据或集成配置违反冻结契约。"""

    def __init__(self, code: str, message: str) -> None:
        """使用稳定错误码构造异常。"""
        self.code = code
        super().__init__(f"{code}: {message}")
