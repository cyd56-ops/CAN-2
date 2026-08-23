"""LWE 密码原语的单元测试

测试覆盖：
1. 参数配置
2. 密钥生成
3. 有效 credential 验证
4. 无效 credential 检测
5. 篡改检测
6. 边界情况
7. 参考验证器一致性
"""

import pytest
import numpy as np
from src.can.v2.crypto.lwe import (
    LWEParams,
    generate_keypair,
    verify,
    V_ref,
    compute_error_norm,
    generate_random_credential,
    generate_perturbed_credential,
)


class TestLWEParams:
    """测试 LWE 参数配置"""

    def test_default_params(self):
        """测试默认参数"""
        params = LWEParams()
        assert params.n == 128
        assert params.m == 256
        assert params.q == 8380417.0
        assert params.sigma == 1.0
        assert params.secret_bound == 2.0
        # 验证阈值应该自动计算
        assert params.error_threshold is not None
        assert params.error_threshold > 0

    def test_custom_params(self):
        """测试自定义参数"""
        params = LWEParams(n=64, m=128, sigma=2.0)
        assert params.n == 64
        assert params.m == 128
        assert params.sigma == 2.0

    def test_threshold_auto_calculation(self):
        """测试阈值自动计算"""
        params = LWEParams(m=256, sigma=1.0)
        # 阈值应该约为 sigma * sqrt(m) * 3
        expected = 1.0 * np.sqrt(256) * 3.0
        assert abs(params.error_threshold - expected) < 1e-5

    def test_manual_threshold(self):
        """测试手动设置阈值"""
        params = LWEParams(error_threshold=100.0)
        assert params.error_threshold == 100.0


class TestKeyGeneration:
    """测试密钥生成"""

    def test_generate_keypair_shape(self):
        """测试密钥对的形状"""
        params = LWEParams(n=64, m=128)
        A, secret, b = generate_keypair(params)

        assert A.shape == (128, 64)
        assert secret.shape == (64,)
        assert b.shape == (128,)

    def test_generate_keypair_dtype(self):
        """测试密钥对的数据类型"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        assert A.dtype == np.float32
        assert secret.dtype == np.float32
        assert b.dtype == np.float32

    def test_secret_bound(self):
        """测试秘密向量的范数约束"""
        params = LWEParams(secret_bound=10.0)
        A, secret, b = generate_keypair(params)

        # 秘密向量的每个元素应该在 [-bound/2, bound/2] 内
        assert np.all(secret >= -params.secret_bound / 2)
        assert np.all(secret <= params.secret_bound / 2)

    def test_generate_keypair_randomness(self):
        """测试密钥生成的随机性"""
        params = LWEParams()

        # 生成两对密钥
        A1, secret1, b1 = generate_keypair(params)
        A2, secret2, b2 = generate_keypair(params)

        # 应该不同
        assert not np.array_equal(A1, A2)
        assert not np.array_equal(secret1, secret2)
        assert not np.array_equal(b1, b2)

    def test_lwe_relation(self):
        """测试 LWE 关系：b ≈ A*s"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        # 计算 A*s
        As = np.dot(A, secret)

        # 残差应该是小噪声 e
        residual = b - As
        residual_norm = np.linalg.norm(residual)

        # 残差范数应该约为 sigma * sqrt(m)
        expected_norm = params.sigma * np.sqrt(params.m)
        # 允许 5 倍标准差的波动
        assert residual_norm < expected_norm * 5


class TestVerify:
    """测试验证函数"""

    def test_verify_valid_credential(self):
        """测试有效 credential 应该验证通过"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        # 使用正确的 secret 进行验证
        is_valid = verify(secret, A, b, params)
        assert is_valid is True

    def test_verify_invalid_credential(self):
        """测试无效 credential 应该验证失败"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        # 使用随机的 invalid credential
        invalid_cred = generate_random_credential(params, scale=5.0)

        is_valid = verify(invalid_cred, A, b, params)
        assert is_valid is False

    def test_verify_zero_credential(self):
        """测试零向量 credential 应该验证失败"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        zero_cred = np.zeros(params.n, dtype=np.float32)
        is_valid = verify(zero_cred, A, b, params)
        assert is_valid is False

    def test_verify_wrong_dimension(self):
        """测试错误维度的 credential 应该验证失败"""
        params = LWEParams(n=64)
        A, secret, b = generate_keypair(params)

        # 使用错误维度的 credential
        wrong_cred = np.random.randn(32).astype(np.float32)
        is_valid = verify(wrong_cred, A, b, params)
        assert is_valid is False

    def test_verify_multiple_times(self):
        """测试多次验证应该一致"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        # 多次验证应该得到相同结果
        results = [verify(secret, A, b, params) for _ in range(10)]
        assert all(results)


class TestTamperDetection:
    """测试篡改检测"""

    def test_tampered_credential(self):
        """测试篡改 credential 应该验证失败"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        # 篡改 secret（修改一个元素）
        tampered = secret.copy()
        tampered[0] += 5.0

        is_valid = verify(tampered, A, b, params)
        assert is_valid is False

    def test_small_perturbation(self):
        """测试小扰动的 credential 是否仍然通过"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        # 添加非常小的扰动（应该在误差容忍范围内）
        perturbed = generate_perturbed_credential(secret, noise_level=0.1)

        is_valid = verify(perturbed, A, b, params)
        # 小扰动可能通过，也可能失败，取决于具体情况
        # 这里我们只测试函数不会崩溃
        assert isinstance(is_valid, bool)

    def test_large_perturbation(self):
        """测试大扰动的 credential 应该验证失败"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        # 添加大扰动
        perturbed = generate_perturbed_credential(secret, noise_level=10.0)

        is_valid = verify(perturbed, A, b, params)
        assert is_valid is False

    def test_tampered_public_key(self):
        """测试使用错误的公钥应该验证失败"""
        params = LWEParams()
        A1, secret1, b1 = generate_keypair(params)
        A2, secret2, b2 = generate_keypair(params)

        # 使用 secret1 验证 (A2, b2)
        is_valid = verify(secret1, A2, b2, params)
        assert is_valid is False


class TestComputeErrorNorm:
    """测试误差范数计算"""

    def test_error_norm_valid_credential(self):
        """测试有效 credential 的误差范数应该很小"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        error_norm = compute_error_norm(secret, A, b)

        # 误差应该约为 sigma * sqrt(m)
        expected = params.sigma * np.sqrt(params.m)
        # 允许 5 倍标准差的波动
        assert error_norm < expected * 5

    def test_error_norm_invalid_credential(self):
        """测试无效 credential 的误差范数应该很大"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        invalid_cred = generate_random_credential(params, scale=10.0)
        error_norm = compute_error_norm(invalid_cred, A, b)

        # 误差应该远大于阈值
        assert error_norm > params.error_threshold


class TestReferenceVerifier:
    """测试参考验证器 V_ref"""

    def test_v_ref_valid_credential(self):
        """测试 V_ref 对有效 credential 返回 1"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        credential_dict = {'vector': secret}
        result = V_ref(credential_dict, A, b, params)
        assert result == 1

    def test_v_ref_invalid_credential(self):
        """测试 V_ref 对无效 credential 返回 0"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        invalid_cred = generate_random_credential(params, scale=5.0)
        credential_dict = {'vector': invalid_cred}
        result = V_ref(credential_dict, A, b, params)
        assert result == 0

    def test_v_ref_missing_vector(self):
        """测试 V_ref 对缺失 'vector' 字段返回 0"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        credential_dict = {}
        result = V_ref(credential_dict, A, b, params)
        assert result == 0

    def test_v_ref_consistency_with_verify(self):
        """测试 V_ref 与 verify 的一致性"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        # 测试多个 credentials
        credentials = [
            secret,
            generate_random_credential(params, scale=1.0),
            generate_random_credential(params, scale=5.0),
            np.zeros(params.n, dtype=np.float32),
        ]

        for cred in credentials:
            v_ref_result = V_ref({'vector': cred}, A, b, params)
            verify_result = verify(cred, A, b, params)

            # V_ref 应该与 verify 一致
            assert v_ref_result == (1 if verify_result else 0)


class TestBoundaryValues:
    """测试边界情况"""

    def test_small_params(self):
        """测试较小的参数"""
        params = LWEParams(n=16, m=32, sigma=1.0)
        A, secret, b = generate_keypair(params)

        # 应该仍然能够验证通过
        is_valid = verify(secret, A, b, params)
        assert is_valid is True

    def test_large_params(self):
        """测试较大的参数"""
        params = LWEParams(n=256, m=512, sigma=5.0)
        A, secret, b = generate_keypair(params)

        # 应该仍然能够验证通过
        is_valid = verify(secret, A, b, params)
        assert is_valid is True

    def test_tight_threshold(self):
        """测试较紧的阈值"""
        params = LWEParams(error_threshold=10.0)  # 很小的阈值
        A, secret, b = generate_keypair(params)

        # 有效 credential 可能无法通过（取决于噪声）
        is_valid = verify(secret, A, b, params)
        # 只测试不崩溃
        assert isinstance(is_valid, bool)

    def test_loose_threshold(self):
        """测试较松的阈值"""
        params = LWEParams(error_threshold=1000.0)  # 很大的阈值
        A, secret, b = generate_keypair(params)

        # 有效 credential 应该通过
        is_valid = verify(secret, A, b, params)
        assert is_valid is True


class TestHelperFunctions:
    """测试辅助函数"""

    def test_generate_random_credential(self):
        """测试生成随机 credential"""
        params = LWEParams(n=64)

        cred = generate_random_credential(params, scale=2.0)
        assert cred.shape == (64,)
        assert cred.dtype == np.float32

    def test_generate_perturbed_credential(self):
        """测试生成扰动 credential"""
        params = LWEParams(n=64)
        A, secret, b = generate_keypair(params)

        perturbed = generate_perturbed_credential(secret, noise_level=1.0)
        assert perturbed.shape == secret.shape
        assert perturbed.dtype == np.float32
        # 应该与原始 secret 不同
        assert not np.array_equal(perturbed, secret)


class TestStatisticalProperties:
    """测试统计特性"""

    def test_noise_distribution(self):
        """测试噪声的统计特性"""
        params = LWEParams()

        # 生成多个密钥对，测试噪声分布
        error_norms = []
        for _ in range(100):
            A, secret, b = generate_keypair(params)
            error_norm = compute_error_norm(secret, A, b)
            error_norms.append(error_norm)

        # 平均误差应该约为 sigma * sqrt(m)
        mean_error = np.mean(error_norms)
        expected = params.sigma * np.sqrt(params.m)

        # 允许 20% 的误差
        assert abs(mean_error - expected) < expected * 0.2

    def test_verification_success_rate(self):
        """测试有效 credential 的验证成功率"""
        params = LWEParams()

        # 生成多个密钥对并验证
        success_count = 0
        total = 100

        for _ in range(total):
            A, secret, b = generate_keypair(params)
            if verify(secret, A, b, params):
                success_count += 1

        # 成功率应该接近 100%
        success_rate = success_count / total
        assert success_rate >= 0.95  # 至少 95% 成功率

    def test_false_positive_rate(self):
        """测试无效 credential 的误判率（应该很低）"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        # 测试多个随机 credentials
        false_positive_count = 0
        total = 100

        for _ in range(total):
            invalid_cred = generate_random_credential(params, scale=5.0)
            if verify(invalid_cred, A, b, params):
                false_positive_count += 1

        # 误判率应该很低（理论上应该为 0）
        false_positive_rate = false_positive_count / total
        assert false_positive_rate < 0.05  # 小于 5%


class TestExceptionHandling:
    """异常处理测试（覆盖防御性代码）"""

    def test_verify_with_mismatched_A_shape(self):
        """测试 A 维度不匹配时返回 False"""
        params = LWEParams(n=128, m=256)
        A_wrong = np.random.randn(100, 128).astype(np.float32)  # m=100 而非 256
        _, secret, b = generate_keypair(params)

        # 应该返回 False 而非抛出异常
        assert verify(secret, A_wrong, b, params) is False

    def test_verify_with_mismatched_b_shape(self):
        """测试 b 维度不匹配时返回 False"""
        params = LWEParams(n=128, m=256)
        A, secret, _ = generate_keypair(params)
        b_wrong = np.random.randn(100).astype(np.float32)  # m=100 而非 256

        # 应该返回 False 而非抛出异常
        assert verify(secret, A, b_wrong, params) is False

    def test_verify_with_invalid_types(self):
        """测试无效类型时返回 False"""
        params = LWEParams()
        A, _, b = generate_keypair(params)

        # 传入 None
        assert verify(None, A, b, params) is False

        # 传入字符串
        assert verify("invalid", A, b, params) is False

        # 传入列表（会尝试转换，但维度不匹配）
        assert verify([1, 2, 3], A, b, params) is False

    def test_V_ref_with_non_array_credential(self):
        """测试 V_ref 能处理非数组输入"""
        params = LWEParams()
        A, secret, b = generate_keypair(params)

        # 传入 Python list（应该自动转换）
        credential_dict = {'vector': secret.tolist()}
        result = V_ref(credential_dict, A, b, params)
        assert result == 1  # 应该通过验证

    def test_V_ref_with_exception_triggers(self):
        """测试 V_ref 的异常处理"""
        params = LWEParams()
        A, _, b = generate_keypair(params)

        # 传入完全无效的数据
        assert V_ref({'vector': None}, A, b, params) == 0
        assert V_ref({'vector': "invalid"}, A, b, params) == 0
        assert V_ref({}, A, b, params) == 0  # 缺少 'vector' key
