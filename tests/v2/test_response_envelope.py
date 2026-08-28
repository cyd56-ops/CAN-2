"""Phase 3.6 response envelope 与进程内服务入口测试。"""

import numpy as np
import pytest
import torch

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.models.gated_resnet import GatedResNet18, InferenceOutput
from src.can.v2.service import ResponseEnvelope, to_response_envelope


@pytest.fixture()
def model_and_credentials():
    """构造小参数 CPU 模型及一组真实 valid/invalid credential。"""
    params = LWEParams(n=32, m=64)
    A, secret, b = generate_keypair(params, rng=np.random.default_rng(20))
    model = GatedResNet18(A, b, params).eval()
    invalid = np.random.default_rng(21).normal(size=params.n).astype(np.float32)
    credentials = torch.from_numpy(np.stack([secret.astype(np.float32), invalid]))
    return model, credentials


def _output(model, credentials):
    """获取 deterministic inference output。"""
    with torch.inference_mode():
        return model(torch.randn(2, 3, 32, 32), credentials)


def test_envelope_fields_are_homogeneous(model_and_credentials):
    """protected/public 响应字段和概率长度一致。"""
    model, credentials = model_and_credentials
    output = _output(model, credentials)
    envelopes = to_response_envelope(output, 2)
    assert [e.capability_level for e in envelopes] == ["protected", "public"]
    assert all(len(e.probabilities) == 10 for e in envelopes)
    assert all(isinstance(e.prediction, int) for e in envelopes)


def test_envelope_preserves_batch_order(model_and_credentials):
    """稀疏 indices 转换后保持原 batch 顺序。"""
    model, credentials = model_and_credentials
    output = _output(model, credentials)
    swapped = InferenceOutput(
        output.protected_logits,
        output.protected_indices,
        output.public_logits,
        output.public_indices,
        output.decision,
    )
    envelopes = to_response_envelope(swapped, 2)
    assert envelopes[0].capability_level == "protected"
    assert envelopes[1].capability_level == "public"


def test_envelope_probabilities_and_prediction(model_and_credentials):
    """概率等于对应 head softmax，prediction 等于 argmax。"""
    model, credentials = model_and_credentials
    output = _output(model, credentials)
    envelopes = to_response_envelope(output, 2)
    expected = torch.softmax(output.protected_logits[0], dim=0)
    assert envelopes[0].probabilities == pytest.approx(expected.tolist())
    assert envelopes[0].prediction == int(output.protected_logits[0].argmax())
    public = torch.softmax(output.public_logits[0], dim=0)
    assert envelopes[1].probabilities[:2] == pytest.approx(public.tolist())
    assert envelopes[1].probabilities[2:] == (0.0,) * 8
    assert envelopes[1].prediction == int(output.public_logits[0].argmax())


@pytest.mark.parametrize("field", ["duplicate", "missing", "out_of_range"])
def test_envelope_rejects_bad_indices(model_and_credentials, field):
    """拒绝重复、缺失或越界路由索引。"""
    model, credentials = model_and_credentials
    output = _output(model, credentials)
    public_indices = output.public_indices.clone()
    if field == "duplicate":
        public_indices[0] = output.protected_indices[0]
    elif field == "missing":
        public_indices[0] = public_indices[0] + 1
    else:
        public_indices[0] = 2
    broken = InferenceOutput(
        output.protected_logits,
        output.protected_indices,
        output.public_logits,
        public_indices,
        output.decision,
    )
    with pytest.raises(ValueError):
        to_response_envelope(broken, 2)


def test_envelope_rejects_row_mismatch(model_and_credentials):
    """拒绝 logits 与 indices 行数不一致。"""
    model, credentials = model_and_credentials
    output = _output(model, credentials)
    broken = InferenceOutput(
        output.protected_logits[:0],
        output.protected_indices,
        output.public_logits,
        output.public_indices,
        output.decision,
    )
    with pytest.raises(ValueError):
        to_response_envelope(broken, 2)


def test_envelope_rejects_non_finite_and_wrong_types(model_and_credentials):
    """拒绝非有限 logits、错误 dtype 和错误输出类型。"""
    model, credentials = model_and_credentials
    output = _output(model, credentials)
    nonfinite = output.protected_logits.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError):
        to_response_envelope(
            InferenceOutput(
                nonfinite,
                output.protected_indices,
                output.public_logits,
                output.public_indices,
                output.decision,
            ),
            2,
        )
    with pytest.raises(TypeError):
        to_response_envelope(object(), 2)


def test_response_envelope_invariants():
    """响应 dataclass 拒绝错误长度、概率和类别范围。"""
    with pytest.raises(ValueError):
        ResponseEnvelope((1.0, 0.0), 0, "public")
    with pytest.raises(ValueError):
        ResponseEnvelope((0.2,) * 10, 0, "protected")
    with pytest.raises(ValueError):
        ResponseEnvelope((0.5, 0.5) + (0.1,) * 8, 0, "public")
    with pytest.raises(ValueError):
        ResponseEnvelope((1.0,) + (0.0,) * 9, 2, "public")


@pytest.mark.parametrize(
    "value, error",
    [
        ([1.0] * 10, TypeError),
        ((float("nan"),) + (0.0,) * 9, ValueError),
        ((1.1,) + (0.0,) * 9, ValueError),
    ],
)
def test_response_envelope_rejects_field_contracts(value, error):
    """覆盖概率容器类型、非有限值和范围校验。"""
    with pytest.raises(error):
        ResponseEnvelope(value, 0, "protected")


def test_response_envelope_rejects_capability_prediction_and_public_padding():
    """拒绝未知能力、bool prediction 和 public 非零填充。"""
    protected = (1.0,) + (0.0,) * 9
    with pytest.raises(ValueError):
        ResponseEnvelope(protected, 0, "unknown")
    with pytest.raises(TypeError):
        ResponseEnvelope(protected, True, "protected")
    with pytest.raises(ValueError):
        ResponseEnvelope((0.5, 0.4, 0.1) + (0.0,) * 7, 0, "public")


@pytest.mark.parametrize(
    "mutation",
    [
        "logits_type",
        "logits_shape",
        "logits_dtype",
        "indices_type",
        "indices_shape",
        "indices_dtype",
    ],
)
def test_envelope_rejects_route_contracts(model_and_credentials, mutation):
    """拒绝路由 logits/indices 的错误类型、shape 和 dtype。"""
    model, credentials = model_and_credentials
    output = _output(model, credentials)
    protected_logits, protected_indices = (
        output.protected_logits,
        output.protected_indices,
    )
    if mutation == "logits_type":
        protected_logits = [[1.0] * 10]
    elif mutation == "logits_shape":
        protected_logits = protected_logits[:, :1]
    elif mutation == "logits_dtype":
        protected_logits = protected_logits.double()
    elif mutation == "indices_type":
        protected_indices = [0]
    elif mutation == "indices_shape":
        protected_indices = protected_indices.unsqueeze(1)
    else:
        protected_indices = protected_indices.float()
    broken = InferenceOutput(
        protected_logits,
        protected_indices,
        output.public_logits,
        output.public_indices,
        output.decision,
    )
    with pytest.raises((TypeError, ValueError)):
        to_response_envelope(broken, 2)


def test_envelope_rejects_invalid_batch_size_and_device(model_and_credentials):
    """覆盖 batch_size 和 protected/public device 一致性校验。"""
    model, credentials = model_and_credentials
    output = _output(model, credentials)
    with pytest.raises(TypeError):
        to_response_envelope(output, True)
    with pytest.raises(ValueError):
        to_response_envelope(output, -1)
    broken = InferenceOutput(
        output.protected_logits,
        output.protected_indices,
        output.public_logits.to("meta"),
        output.public_indices.to("meta"),
        output.decision,
    )
    with pytest.raises(ValueError):
        to_response_envelope(broken, 2)
