"""Neural Gate Layer 的单元测试。

测试覆盖 LWE 差分验证、批量证据、训练软门控、推理硬门控、输入拒绝、
特征门控和梯度传播。所有随机测试均使用显式种子。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.can.v2.crypto.lwe import V_ref  # noqa: E402
from src.can.v2.crypto.lwe import (
    LWEParams,
    generate_keypair,
    generate_random_credential,
)
from src.can.v2.layers.gate_layer import (  # noqa: E402
    AuthorizationCoordinator,
    AuthorizationDecision,
    FeatureGate,
    GateLayer,
    LWEVerifier,
    ReasonCode,
    VerificationEvidence,
)


@pytest.fixture
def gate_fixture():
    """创建确定性的 Gate Layer、参数和有效 credential。"""

    np.random.seed(20260823)
    torch.manual_seed(20260823)
    params = LWEParams(n=16, m=32, sigma=1.0)
    A, secret, b = generate_keypair(params)
    gate = GateLayer(A, b, params, temperature=5.0)
    return gate, params, A, secret, b


def _features(batch_size: int, dtype=None):
    """生成确定性的浅层特征测试输入。"""

    selected_dtype = torch.float32 if dtype is None else dtype
    values = torch.arange(batch_size * 24, dtype=selected_dtype)
    return values.reshape(batch_size, 3, 2, 4).requires_grad_(True)


class TestLWEVerifier:
    """测试批量 LWE 验证器和参考实现一致性。"""

    def test_valid_and_invalid_credential(self, gate_fixture):
        """合法和非法 credential 应产生稳定的验证证据。"""

        gate, params, _, secret, _ = gate_fixture
        np.random.seed(1)
        invalid = generate_random_credential(params, scale=5.0)
        credentials = torch.from_numpy(np.stack([secret, invalid]))

        evidence = gate.verifier(credentials)

        assert evidence.verified.tolist() == [True, False]
        assert evidence.reason_code.tolist() == [
            int(ReasonCode.SUCCESS),
            int(ReasonCode.LWE_VERIFICATION_FAILED),
        ]
        assert evidence.error_norm.shape == (2,)

    def test_verify_matches_v_ref(self, gate_fixture):
        """PyTorch 验证逻辑应与 NumPy V_ref 完全一致。"""

        gate, params, A, secret, b = gate_fixture
        rng = np.random.RandomState(7)
        credentials = [secret]
        credentials.extend(
            rng.randn(params.n).astype(np.float32) * 5.0 for _ in range(100)
        )
        batch = torch.from_numpy(np.stack(credentials))

        evidence = gate.verifier(batch)
        expected = [V_ref({"vector": value}, A, b, params) for value in credentials]

        assert evidence.verified.to(torch.int64).tolist() == expected

    def test_mixed_non_finite_batch(self, gate_fixture):
        """非有限行应独立拒绝且不污染同 batch 的合法行。"""

        gate, params, _, secret, _ = gate_fixture
        np.random.seed(2)
        invalid = generate_random_credential(params, scale=5.0)
        nan_value = secret.copy()
        nan_value[0] = np.nan
        inf_value = secret.copy()
        inf_value[0] = np.inf
        batch = torch.from_numpy(np.stack([secret, invalid, nan_value, inf_value]))

        evidence = gate.verifier(batch)

        assert evidence.verified.tolist() == [True, False, False, False]
        assert evidence.reason_code.tolist() == [
            int(ReasonCode.SUCCESS),
            int(ReasonCode.LWE_VERIFICATION_FAILED),
            int(ReasonCode.NON_FINITE),
            int(ReasonCode.NON_FINITE),
        ]
        assert torch.isfinite(evidence.error_norm[:2]).all()
        assert torch.isinf(evidence.error_norm[2:]).all()

    def test_threshold_boundary_is_strict(self):
        """误差恰等于阈值时必须拒绝。"""

        params = LWEParams(n=1, m=1, error_threshold=2.0)
        A = np.ones((1, 1), dtype=np.float32)
        b = np.zeros(1, dtype=np.float32)
        verifier = LWEVerifier(A, b, params)
        credentials = torch.tensor([[1.999], [2.0], [2.001]], dtype=torch.float32)

        evidence = verifier(credentials)

        assert evidence.verified.tolist() == [True, False, False]

    def test_verifier_is_stateless(self, gate_fixture):
        """重复验证不得修改 buffer 或改变相同输入的证据。"""

        gate, _, _, secret, _ = gate_fixture
        credential = torch.from_numpy(secret).unsqueeze(0)
        state_before = {
            name: value.clone() for name, value in gate.verifier.state_dict().items()
        }

        first = gate.verifier(credential)
        second = gate.verifier(credential)

        assert torch.equal(first.verified, second.verified)
        assert torch.equal(first.error_norm, second.error_norm)
        assert torch.equal(first.reason_code, second.reason_code)
        for name, value in state_before.items():
            assert torch.equal(gate.verifier.state_dict()[name], value)


class TestGateModes:
    """测试训练软门控和推理硬门控。"""

    def test_training_mode_uses_soft_gate(self, gate_fixture):
        """训练时合法和格式合法的非法样本应获得连续软门控。"""

        gate, params, _, secret, _ = gate_fixture
        np.random.seed(3)
        invalid = generate_random_credential(params, scale=5.0)
        features = _features(2)
        gate.train()

        gated, decision = gate(features, np.stack([secret, invalid]))

        assert decision.allow.tolist() == [True, False]
        assert decision.gate_signal[0] > 0.7
        assert 0.0 <= decision.gate_signal[1] < 0.3
        expected = features * decision.gate_signal[:, None, None, None]
        assert torch.equal(gated, expected)

    def test_eval_mode_hard_blocks_invalid(self, gate_fixture):
        """推理时非法 credential 必须产生严格全零特征。"""

        gate, params, _, secret, _ = gate_fixture
        np.random.seed(4)
        invalid = generate_random_credential(params, scale=5.0)
        features = _features(2)
        gate.eval()

        gated, decision = gate(features, np.stack([secret, invalid]))

        assert decision.allow.tolist() == [True, False]
        assert decision.gate_signal.tolist() == [1.0, 0.0]
        assert torch.equal(gated[0], features[0])
        assert torch.count_nonzero(gated[1]).item() == 0

    def test_eval_forward_matches_v_ref(self, gate_fixture):
        """GateLayer 推理决定应端到端匹配 V_ref。"""

        gate, params, A, secret, b = gate_fixture
        rng = np.random.RandomState(9)
        values = [secret]
        values.extend(rng.randn(params.n).astype(np.float32) * 5.0 for _ in range(20))
        batch = np.stack(values)
        gate.eval()

        _, decision = gate(_features(len(values)), batch)
        expected = [V_ref({"vector": value}, A, b, params) for value in values]

        assert decision.allow.to(torch.int64).tolist() == expected
        assert decision.gate_signal.to(torch.int64).tolist() == expected

    def test_mode_switch(self, gate_fixture):
        """train/eval 切换应同步作用于内部协调器。"""

        gate, _, _, secret, _ = gate_fixture
        features = _features(1)

        gate.train()
        _, training_decision = gate(features, secret)
        gate.eval()
        _, eval_decision = gate(features, secret)

        assert gate.coordinator.training is False
        assert 0.0 < training_decision.gate_signal.item() < 1.0
        assert eval_decision.gate_signal.item() == 1.0

    def test_credential_is_reusable(self, gate_fixture):
        """无状态阶段应允许相同 credential 产生一致决定。"""

        gate, _, _, secret, _ = gate_fixture
        features = _features(1)
        gate.eval()

        first_features, first_decision = gate(features, secret)
        second_features, second_decision = gate(features, secret)

        assert torch.equal(first_features, second_features)
        assert torch.equal(first_decision.allow, second_decision.allow)
        assert torch.equal(first_decision.gate_signal, second_decision.gate_signal)


class TestInputValidation:
    """测试请求级拒绝、逐行拒绝和空 batch。"""

    @pytest.mark.parametrize(
        ("credential", "reason"),
        [
            (np.ones((2, 2, 2), dtype=np.float32), ReasonCode.INVALID_SHAPE),
            (np.ones((2, 2, 2), dtype=np.int64), ReasonCode.INVALID_SHAPE),
            (np.ones(16, dtype=np.int64), ReasonCode.WRONG_DTYPE),
            (np.ones(16, dtype=object), ReasonCode.WRONG_DTYPE),
            (np.ones(15, dtype=np.float32), ReasonCode.DIMENSION_MISMATCH),
            ("invalid", ReasonCode.WRONG_DTYPE),
        ],
    )
    def test_request_error_returns_batch_rejection(
        self, gate_fixture, credential, reason
    ):
        """credential 请求级错误应返回全 batch 的结构化拒绝。"""

        gate, _, _, _, _ = gate_fixture
        features = _features(3)

        gated, decision = gate(features, credential)

        assert decision.allow.tolist() == [False, False, False]
        assert decision.gate_signal.tolist() == [0.0, 0.0, 0.0]
        assert decision.evidence.reason_code.tolist() == [int(reason)] * 3
        assert torch.isinf(decision.evidence.error_norm).all()
        assert torch.count_nonzero(gated).item() == 0

    def test_batch_mismatch_returns_rejection(self, gate_fixture):
        """无法广播的 credential batch 应整批拒绝。"""

        gate, params, _, _, _ = gate_fixture
        credential = np.ones((2, params.n), dtype=np.float32)

        _, decision = gate(_features(3), credential)

        assert (decision.evidence.reason_code == ReasonCode.DIMENSION_MISMATCH).all()

    def test_single_credential_broadcast(self, gate_fixture):
        """单个 credential 应广播到 feature batch。"""

        gate, _, _, secret, _ = gate_fixture
        gate.eval()

        gated, decision = gate(_features(3), secret)

        assert decision.allow.tolist() == [True, True, True]
        assert gated.shape == (3, 3, 2, 4)

    def test_empty_batch(self, gate_fixture):
        """空 batch 应返回 batch 维为零的所有结果。"""

        gate, params, _, _, _ = gate_fixture
        features = torch.empty((0, 3, 2, 4), dtype=torch.float32)
        credential = torch.empty((0, params.n), dtype=torch.float32)

        gated, decision = gate(features, credential)

        assert gated.shape == features.shape
        assert decision.allow.shape == (0,)
        assert decision.gate_signal.shape == (0,)
        assert decision.evidence.error_norm.shape == (0,)

    @pytest.mark.parametrize(
        "features",
        [
            torch.ones((2, 3, 4), dtype=torch.float32),
            torch.ones((2, 3, 2, 4), dtype=torch.int64),
            torch.full((2, 3, 2, 4), float("nan")),
        ],
    )
    def test_invalid_features_raise(self, gate_fixture, features):
        """非法 features 应在进入验证器前抛出稳定异常。"""

        gate, _, _, secret, _ = gate_fixture

        with pytest.raises((TypeError, ValueError)):
            gate(features, secret)


class TestConstructionAndGradients:
    """测试构造期约束、dtype 保持和梯度行为。"""

    @pytest.mark.parametrize("temperature", [0.0, -1.0, np.inf, np.nan, True])
    def test_reject_invalid_temperature(self, gate_fixture, temperature):
        """temperature 必须是有限正实数。"""

        _, params, A, _, b = gate_fixture

        with pytest.raises((TypeError, ValueError)):
            GateLayer(A, b, params, temperature=temperature)

    def test_reject_invalid_public_arrays(self, gate_fixture):
        """A、b 的错误类型、形状和非有限值应被拒绝。"""

        _, params, A, _, b = gate_fixture
        bad_A = A.copy()
        bad_A[0, 0] = np.nan

        with pytest.raises(ValueError):
            GateLayer(bad_A, b, params)
        with pytest.raises(ValueError):
            GateLayer(A[:, :-1], b, params)
        with pytest.raises(TypeError):
            GateLayer(A.tolist(), b, params)
        with pytest.raises(TypeError):
            GateLayer(A.astype(np.int64), b, params)

    def test_reject_invalid_params(self, gate_fixture):
        """验证器和协调器必须拒绝错误的参数对象。"""

        _, _, A, _, b = gate_fixture

        with pytest.raises(TypeError):
            LWEVerifier(A, b, object())
        with pytest.raises(TypeError):
            AuthorizationCoordinator(object())

    def test_output_dtype_and_gradient_preserved(self, gate_fixture):
        """特征门控应保持 dtype，并把梯度回传到浅层特征。"""

        gate, _, _, secret, _ = gate_fixture
        features = _features(2, dtype=torch.float64)
        gate.train()

        gated, decision = gate(features, secret)
        gated.sum().backward()

        assert gated.dtype == features.dtype
        assert gated.device == features.device
        assert features.grad is not None
        expected_grad = decision.gate_signal.to(torch.float64)[:, None, None, None]
        assert torch.allclose(features.grad, expected_grad.expand_as(features))

    def test_gate_has_no_trainable_parameters(self, gate_fixture):
        """Gate Layer 应只包含 buffer，不包含可训练参数。"""

        gate, _, _, _, _ = gate_fixture

        assert list(gate.parameters()) == []
        buffers = dict(gate.named_buffers())
        assert set(buffers) == {"verifier.A", "verifier.b"}

    def test_coordinator_rejects_malformed_evidence(self, gate_fixture):
        """协调器应拒绝非结构化 evidence，避免静默类型混淆。"""

        _, params, _, _, _ = gate_fixture
        coordinator = AuthorizationCoordinator(params)

        with pytest.raises(TypeError):
            coordinator({"verified": True})

    @pytest.mark.parametrize(
        "evidence",
        [
            VerificationEvidence(True, torch.ones(1), torch.zeros(1, dtype=torch.long)),
            VerificationEvidence(
                torch.ones((1, 1), dtype=torch.bool),
                torch.ones(1),
                torch.zeros(1, dtype=torch.long),
            ),
            VerificationEvidence(
                torch.ones(2, dtype=torch.bool),
                torch.ones(1),
                torch.zeros(1, dtype=torch.long),
            ),
            VerificationEvidence(
                torch.ones(1),
                torch.ones(1),
                torch.zeros(1, dtype=torch.long),
            ),
            VerificationEvidence(
                torch.ones(1, dtype=torch.bool),
                torch.ones(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            ),
            VerificationEvidence(
                torch.ones(1, dtype=torch.bool),
                torch.ones(1),
                torch.zeros(1, dtype=torch.int32),
            ),
        ],
    )
    def test_coordinator_rejects_invalid_evidence_fields(self, gate_fixture, evidence):
        """协调器必须拒绝类型、rank、batch 或 dtype 非法的证据。"""

        _, params, _, _, _ = gate_fixture
        coordinator = AuthorizationCoordinator(params)

        with pytest.raises((TypeError, ValueError)):
            coordinator(evidence)

    def test_verifier_rejects_non_normalized_inputs(self, gate_fixture):
        """验证器包内接口必须拒绝未规范化的 credential。"""

        gate, params, _, _, _ = gate_fixture
        invalid_values = [
            np.ones((1, params.n), dtype=np.float32),
            torch.ones(params.n),
            torch.ones((1, params.n), dtype=torch.float64),
        ]

        for value in invalid_values:
            with pytest.raises((TypeError, ValueError)):
                gate.verifier(value)

    def test_feature_gate_rejects_invalid_decision(self):
        """FeatureGate 必须拒绝错误对象、rank 和 batch。"""

        feature_gate = FeatureGate()
        features = torch.ones((2, 3, 2, 2))
        evidence = VerificationEvidence(
            torch.ones(2, dtype=torch.bool),
            torch.ones(2),
            torch.zeros(2, dtype=torch.long),
        )

        with pytest.raises(TypeError):
            feature_gate(features, object())

        rank_decision = AuthorizationDecision(
            evidence.verified, torch.ones((2, 1)), evidence
        )
        with pytest.raises(ValueError):
            feature_gate(features, rank_decision)

        batch_decision = AuthorizationDecision(
            evidence.verified, torch.ones(1), evidence
        )
        with pytest.raises(ValueError):
            feature_gate(features, batch_decision)

    def test_gate_rejects_tensor_integer_and_non_tensor_features(self, gate_fixture):
        """GateLayer 必须拒绝 Tensor 整数 credential 和非 Tensor features。"""

        gate, params, _, secret, _ = gate_fixture
        _, decision = gate(_features(1), torch.ones(params.n, dtype=torch.int64))
        assert decision.evidence.reason_code.item() == ReasonCode.WRONG_DTYPE

        with pytest.raises(TypeError):
            gate(np.ones((1, 3, 2, 2), dtype=np.float32), secret)

    def test_negative_stride_numpy_credential_is_rejected(self, gate_fixture):
        """torch 无法共享的负 stride NumPy 输入应稳定映射为类型拒绝。"""

        gate, _, _, secret, _ = gate_fixture
        reversed_secret = secret[::-1]

        _, decision = gate(_features(1), reversed_secret)

        assert decision.evidence.reason_code.item() == ReasonCode.WRONG_DTYPE

    def test_meta_device_contract(self, gate_fixture):
        """device 不一致应在验证器和组合层边界被拒绝。"""

        gate, params, _, secret, _ = gate_fixture
        meta_credential = torch.empty((1, params.n), device="meta")
        meta_features = torch.empty((1, 3, 2, 2), device="meta")

        with pytest.raises(ValueError):
            gate.verifier(meta_credential)
        with pytest.raises(ValueError):
            gate(meta_features, secret)
