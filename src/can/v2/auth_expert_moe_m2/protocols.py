"""M2 verifier 的公开结构化接口。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from ..pretrained_gate.types import VerificationEvidence


@runtime_checkable
class M2VerifierProtocol(Protocol):
    """定义 M2 AuthExpert 可接受的最小 verifier 契约。"""

    def __call__(self, credential: Tensor) -> VerificationEvidence:
        """验证批量 credential 并返回结构化 evidence。"""

    def validate_evidence(self, evidence: VerificationEvidence) -> int:
        """验证 evidence 来源和完整性，并返回 batch 大小。"""
