"""把 G1-b 模整数结果适配为 M2 可验证 evidence。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor, nn

from ..modint_verifier_g1a.parameters import parameter_digest
from ..modint_verifier_g1a.types import G1AParameters
from ..modint_verifier_g1b.kernel import derive_integer_bounds
from ..modint_verifier_g1b.manifest import load_g1b_manifest, validate_g1b_manifest
from ..modint_verifier_g1b.types import G1BResult
from ..modint_verifier_g1b.verifier import ModIntNeuralVerifier
from ..pretrained_gate.types import EvidenceReason, VerificationEvidence
from .types import FP32_EXACT_INTEGER_LIMIT, I1Error

_REASON_MAP = {
    "RELATION_WITHIN_BOUND": EvidenceReason.SUCCESS,
    "RELATION_OUT_OF_BOUND": EvidenceReason.RELATION_FAILED,
}


def _require_sha256(value: object, name: str) -> str:
    """验证并返回小写 SHA-256 字符串。"""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise I1Error("I1_DIGEST_SCHEMA", f"{name} 必须是小写 SHA-256")
    return value


def _evidence_tag(
    verified: Tensor,
    error_norm: Tensor,
    reason_code: Tensor,
    seal: object,
) -> str:
    """计算 adapter 私有 evidence 完整性标签。"""
    digest = hashlib.sha256()
    digest.update(str(id(seal)).encode("ascii"))
    for value in (verified, error_norm, reason_code):
        detached = value.detach().to(device="cpu").contiguous()
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(str(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


class G1BVerifierAdapter(nn.Module):
    """将固定 G1-b CPU verifier 接到 M2 verifier protocol。"""

    def __init__(
        self,
        parameters: G1AParameters,
        g1b_manifest: Dict[str, Any],
        g1b_manifest_sha256: str,
    ) -> None:
        """绑定公开参数、已验证 manifest 和原始 bytes 摘要。"""
        super().__init__()
        if not isinstance(parameters, G1AParameters):
            raise I1Error("I1_PARAMETER_TYPE", "parameters 必须是 G1AParameters")
        self.g1b_manifest_sha256 = _require_sha256(
            g1b_manifest_sha256, "g1b_manifest_sha256"
        )
        if not isinstance(g1b_manifest, dict):
            raise I1Error("I1_MANIFEST_TYPE", "G1-b manifest 必须是 dict")
        canonical_manifest = json.dumps(
            g1b_manifest,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(canonical_manifest).hexdigest() != self.g1b_manifest_sha256:
            raise I1Error(
                "I1_MANIFEST_DIGEST_MISMATCH", "G1-b manifest 内容与摘要不匹配"
            )
        manifest = validate_g1b_manifest(dict(g1b_manifest))
        expected_parameter = parameter_digest(parameters)
        if manifest["source_parameter_sha256"] != expected_parameter:
            raise I1Error(
                "I1_PARAMETER_DIGEST_MISMATCH", "公开参数与 G1-b manifest 不匹配"
            )
        bounds = derive_integer_bounds(parameters)
        if manifest["integer_bounds"] != bounds.to_dict():
            raise I1Error(
                "I1_INTEGER_BOUNDS_MISMATCH", "整数上界与 G1-b manifest 不匹配"
            )
        if manifest["backend_id"] != "g1b-torch-int64-cpu-v1":
            raise I1Error("I1_BACKEND_MISMATCH", "I1 只接受已验收 CPU int64 backend")
        upper_bound = parameters.config.q // 2
        if upper_bound > FP32_EXACT_INTEGER_LIMIT:
            raise I1Error("I1_FP32_DIAGNOSTIC_LOSSY", "整数诊断值不能无损转换为 FP32")

        self.verifier = ModIntNeuralVerifier(parameters)
        self.parameter_sha256 = expected_parameter
        self.max_abs_residual_upper_bound = upper_bound
        self.fp32_lossless_verified = True
        self._evidence_seal = object()
        self._issued: Dict[int, VerificationEvidence] = {}
        self.requires_grad_(False)

    @classmethod
    def from_manifest(
        cls,
        parameters: G1AParameters,
        manifest_path: Path,
        expected_sha256: str,
    ) -> "G1BVerifierAdapter":
        """按原始 manifest bytes 摘要构造受绑定 adapter。"""
        digest = _require_sha256(expected_sha256, "expected_sha256")
        manifest = load_g1b_manifest(Path(manifest_path), digest)
        return cls(parameters, manifest, digest)

    def forward(self, credential: Tensor) -> VerificationEvidence:
        """验证原生 canonical int64 credential 并返回 M2 evidence。"""
        if not isinstance(credential, Tensor):
            raise I1Error("I1_INPUT_TYPE", "credential 必须是 Tensor")
        if credential.dtype is not torch.int64:
            raise I1Error("I1_INPUT_DTYPE", "credential 必须是原生 torch.int64")
        if credential.device.type != "cpu":
            raise I1Error("I1_INPUT_DEVICE", "当前 I1 adapter 只接受 CPU credential")
        if not credential.is_contiguous():
            raise I1Error("I1_INPUT_LAYOUT", "credential 必须是 contiguous")
        result = self.verifier(credential)
        return self._adapt_result(result)

    def _adapt_result(self, result: G1BResult) -> VerificationEvidence:
        """执行固定 reason 和整数诊断字段映射。"""
        if not isinstance(result, G1BResult):
            raise I1Error("I1_RESULT_TYPE", "G1-b 必须返回 G1BResult")
        source = result.evidence
        if source.protocol_id != self.verifier.parameters_ref.config.protocol_id:
            raise I1Error("I1_PROTOCOL_MISMATCH", "G1-b evidence protocol 漂移")
        if source.parameter_digest != self.parameter_sha256:
            raise I1Error("I1_PARAMETER_DIGEST_MISMATCH", "G1-b evidence 参数摘要漂移")
        if result.backend_id != "g1b-torch-int64-cpu-v1":
            raise I1Error("I1_BACKEND_MISMATCH", "G1-b evidence backend 漂移")
        try:
            mapped_reasons = tuple(_REASON_MAP[value] for value in source.reason_code)
        except KeyError as exc:
            raise I1Error("I1_UNKNOWN_REASON", "未知 G1-b reason code") from exc
        if any(
            value > self.max_abs_residual_upper_bound
            for value in source.max_abs_residual
        ):
            raise I1Error("I1_RESIDUAL_RANGE", "G1-b residual 超出解析上界")

        verified = torch.tensor(source.accepted, dtype=torch.bool)
        error_norm = torch.tensor(source.max_abs_residual, dtype=torch.float32)
        reason_code = torch.tensor(
            [int(value) for value in mapped_reasons], dtype=torch.long
        )
        tag = _evidence_tag(verified, error_norm, reason_code, self._evidence_seal)
        evidence = VerificationEvidence(
            verified=verified,
            error_norm=error_norm,
            reason_code=reason_code,
            _verifier_seal=self._evidence_seal,
            _integrity_tag=tag,
        )
        self._issued[id(evidence)] = evidence
        return evidence

    def validate_evidence(self, evidence: VerificationEvidence) -> int:
        """验证 adapter evidence 的来源、字段和完整性。"""
        if not isinstance(evidence, VerificationEvidence):
            raise I1Error("I1_EVIDENCE_TYPE", "evidence 类型非法")
        if self._issued.get(id(evidence)) is not evidence:
            raise I1Error("I1_EVIDENCE_SOURCE", "evidence 未由当前 adapter 签发")
        if evidence._verifier_seal is not self._evidence_seal:
            raise I1Error("I1_EVIDENCE_SOURCE", "evidence seal 不匹配")
        tensors = (evidence.verified, evidence.error_norm, evidence.reason_code)
        if any(not isinstance(value, Tensor) or value.ndim != 1 for value in tensors):
            raise I1Error("I1_EVIDENCE_SHAPE", "evidence 字段必须是一维 Tensor")
        if len({value.shape[0] for value in tensors}) != 1 or not tensors[0].numel():
            raise I1Error("I1_EVIDENCE_SHAPE", "evidence batch 不能为空且长度必须一致")
        if evidence.verified.dtype is not torch.bool:
            raise I1Error("I1_EVIDENCE_DTYPE", "verified 必须是 bool")
        if evidence.error_norm.dtype is not torch.float32:
            raise I1Error("I1_EVIDENCE_DTYPE", "error_norm 必须是 float32")
        if evidence.reason_code.dtype is not torch.long:
            raise I1Error("I1_EVIDENCE_DTYPE", "reason_code 必须是 int64")
        if len({value.device for value in tensors}) != 1:
            raise I1Error("I1_EVIDENCE_DEVICE", "evidence 字段必须位于同一 device")
        expected_tag = _evidence_tag(*tensors, self._evidence_seal)
        if evidence._integrity_tag != expected_tag:
            raise I1Error("I1_EVIDENCE_TAMPERED", "evidence 内容被修改")
        valid_codes = {int(EvidenceReason.SUCCESS), int(EvidenceReason.RELATION_FAILED)}
        if any(
            int(value) not in valid_codes for value in evidence.reason_code.tolist()
        ):
            raise I1Error("I1_EVIDENCE_REASON", "evidence reason 不在 I1 映射集合")
        success = evidence.reason_code == int(EvidenceReason.SUCCESS)
        if not torch.equal(evidence.verified, success):
            raise I1Error("I1_EVIDENCE_SEMANTICS", "verified 与 reason 不一致")
        if not bool(torch.isfinite(evidence.error_norm).all().item()):
            raise I1Error("I1_EVIDENCE_NUMERIC", "整数诊断映射必须全部有限")
        if bool((evidence.error_norm < 0).any().item()):
            raise I1Error("I1_EVIDENCE_NUMERIC", "残差诊断不能为负")
        if bool(
            (evidence.error_norm > float(self.max_abs_residual_upper_bound))
            .any()
            .item()
        ):
            raise I1Error("I1_EVIDENCE_NUMERIC", "残差诊断超出解析上界")
        return int(evidence.verified.shape[0])
