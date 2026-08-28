"""Phase 3.6 可信进程内推理服务集成测试。"""

import numpy as np
import pytest
import torch

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.models.gated_resnet import GatedResNet18
from src.can.v2.service import InferenceService, ServiceExecutionError


@pytest.fixture()
def model_and_credentials():
    """构造 CPU 模型及真实 valid/invalid credential。"""
    params = LWEParams(n=32, m=64)
    A, secret, b = generate_keypair(params, rng=np.random.default_rng(30))
    model = GatedResNet18(A, b, params).eval()
    invalid = np.random.default_rng(31).normal(size=params.n).astype(np.float32)
    credentials = torch.from_numpy(np.stack([secret.astype(np.float32), invalid]))
    return model, credentials


def test_service_returns_envelopes_and_hides_internal_fields(model_and_credentials):
    """服务返回原生 envelope，不携带内部验证证据。"""
    model, credentials = model_and_credentials
    result = InferenceService(model, torch.device("cpu")).infer(
        torch.randn(2, 3, 32, 32), credentials
    )
    assert len(result) == 2
    assert not hasattr(result[0], "decision")
    assert not hasattr(result[0], "indices")
    assert isinstance(result[0].probabilities, tuple)
    assert [item.capability_level for item in result] == ["protected", "public"]


@pytest.mark.parametrize("valid", [True, False])
def test_service_handles_homogeneous_credential_batches(model_and_credentials, valid):
    """全 valid 与全 invalid batch 均返回字段和概率长度同构的响应。"""
    model, credentials = model_and_credentials
    source = credentials[0 if valid else 1]
    batch_credentials = source.unsqueeze(0).repeat(2, 1)
    result = InferenceService(model, torch.device("cpu")).infer(
        torch.randn(2, 3, 32, 32), batch_credentials
    )
    expected = "protected" if valid else "public"
    assert all(item.capability_level == expected for item in result)
    assert all(len(item.probabilities) == 10 for item in result)


def test_service_rejects_request_contracts(model_and_credentials):
    """拒绝空 batch、超限、dtype、有限性和 batch 对齐错误。"""
    model, credentials = model_and_credentials
    service = InferenceService(model, torch.device("cpu"), max_batch_size=2)
    good = torch.randn(2, 3, 32, 32)
    cases = [
        (good[:1], credentials),
        (good[:0], credentials[:0]),
        (good, credentials[:1]),
        (good.double(), credentials),
        (torch.full_like(good, float("inf")), credentials),
        (torch.randn(3, 3, 32, 32), torch.zeros(3, 32)),
    ]
    for images, creds in cases:
        with pytest.raises((TypeError, ValueError)):
            service.infer(images, creds)


def test_service_wraps_model_failures_without_internal_details(
    model_and_credentials, monkeypatch
):
    """模型异常转为稳定错误，不返回内部错误文本。"""
    model, credentials = model_and_credentials
    service = InferenceService(model, torch.device("cpu"))

    def fail_forward(*args, **kwargs):
        """构造内部模型异常。"""
        raise RuntimeError("error_norm internal detail")

    monkeypatch.setattr(model, "forward", fail_forward)
    with pytest.raises(ServiceExecutionError, match="服务推理失败") as error:
        service.infer(torch.randn(2, 3, 32, 32), credentials)
    assert "error_norm" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_service_constructor_contracts(model_and_credentials):
    """拒绝错误 model、device 和 batch 上限。"""
    model, _ = model_and_credentials
    with pytest.raises(TypeError):
        InferenceService(model, "cpu")
    with pytest.raises(ValueError):
        InferenceService(model, torch.device("cpu"), max_batch_size=0)
    with pytest.raises(TypeError):
        InferenceService(model, torch.device("cpu"), max_batch_size=True)
    with pytest.raises(TypeError):
        InferenceService(object(), torch.device("cpu"))


def test_service_constructor_rejects_device_mismatch(model_and_credentials):
    """拒绝模型存储与指定执行 device 不一致。"""
    model, _ = model_and_credentials
    with pytest.raises(ValueError):
        InferenceService(model.to("meta"), torch.device("cpu"))


def test_service_rejects_non_tensor_and_bad_credential_shape(model_and_credentials):
    """拒绝非 Tensor 和错误 credential 维度。"""
    model, credentials = model_and_credentials
    service = InferenceService(model, torch.device("cpu"))
    images = torch.randn(2, 3, 32, 32)
    with pytest.raises(TypeError):
        service.infer(object(), credentials)
    with pytest.raises(TypeError):
        service.infer(images, object())
    with pytest.raises(ValueError):
        service.infer(images, credentials[:, :31])


def test_service_rejects_bad_image_requests(model_and_credentials):
    """拒绝错误图像 shape、dtype、非有限值和 batch 上限。"""
    model, credentials = model_and_credentials
    service = InferenceService(model, torch.device("cpu"), max_batch_size=2)
    with pytest.raises(ValueError):
        service.infer(torch.randn(2, 3, 31, 32), credentials)
    with pytest.raises(TypeError):
        service.infer(torch.randn(2, 3, 32, 32).double(), credentials)
    with pytest.raises(ValueError):
        service.infer(torch.full((2, 3, 32, 32), float("inf")), credentials)
    with pytest.raises(ValueError):
        service.infer(torch.randn(3, 3, 32, 32), torch.zeros(3, 32))


def test_service_rejects_bad_credential_tensor(model_and_credentials):
    """拒绝 credential ndim、dtype 和非有限值。"""
    model, credentials = model_and_credentials
    service = InferenceService(model, torch.device("cpu"))
    images = torch.randn(2, 3, 32, 32)
    with pytest.raises(ValueError):
        service.infer(images, credentials[0])
    with pytest.raises(TypeError):
        service.infer(images, credentials.double())
    with pytest.raises(ValueError):
        service.infer(images, torch.full_like(credentials, float("nan")))
