"""CAN V2 可信进程内服务适配层。"""

from .inference_service import InferenceService, ServiceExecutionError
from .response_envelope import ResponseEnvelope, to_response_envelope

__all__ = [
    "InferenceService",
    "ResponseEnvelope",
    "ServiceExecutionError",
    "to_response_envelope",
]
