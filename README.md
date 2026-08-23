# CAN - Capability Authentication Network

**基于 LWE 的模型内生安全研究项目**

---

## 项目目标

研究并实现"认证神经元层"（Gate Layer），使 AI 模型具备内生的能力分级访问控制：

- **无认证**: 只能使用浅层神经元（弱化能力）
- **有认证**: 解锁完整神经网络能力

核心思想：将密码学验证（LWE）嵌入神经网络，形成模型内部的安全门控。

---

## 当前状态

### ✅ Phase 1.1: LWE 密码原语 (Completed)

**实现**:
- `src/can/v2/crypto/lwe.py`: 完整的 LWE 实现
  - `generate_keypair()`: 生成 (A, secret, b)
  - `verify()`: 验证 credential
  - `V_ref()`: 参考验证器（模型集成接口）
  - `compute_error_norm()`: 误差分析工具

**测试**: 
- `tests/v2/test_lwe.py`: 33 个单元测试，100% 通过
- 误判率 < 0.01%
- 验证成功率 > 99%

**参数** (Toy Profile):
- n=128 (秘密维度)
- m=256 (公钥维度)
- σ=1.0 (噪声标准差)
- 阈值: 48.0

**关键决策**:
- 使用浮点运算（可微分，神经网络友好）
- 不归一化矩阵 A（保证 valid/invalid 误差分离度）
- 详见 `docs/V2_LWE_IMPLEMENTATION.md`

---

## 项目结构

```
E:/CAN/
├── src/can/v2/
│   └── crypto/
│       ├── __init__.py
│       └── lwe.py              # LWE 核心实现
├── tests/v2/
│   └── test_lwe.py             # LWE 测试套件 (33 tests)
├── docs/
│   └── V2_LWE_IMPLEMENTATION.md  # 实现决策文档
├── paper/                      # 参考论文
│   ├── shamir_crypto_dnn.pdf
│   └── ...
└── draw/                       # 设计图
    └── 认证神经元层框架图.png
```

---

## 下一步计划

### Phase 1.2: PyTorch 集成
- [ ] 实现 `LWEVerifierLayer(nn.Module)`
- [ ] 单元测试：前向传播、梯度检查
- [ ] Benchmark：latency、memory

### Phase 1.3: 门控路由器
- [ ] 实现 `GatedRouter`: credential → gate → public/protected path
- [ ] 集成测试：完整的 verify → route 流程

### Phase 2: 实验验证
- [ ] C2: 两路模型（CIFAR-10）
- [ ] 对比实验：external verifier vs. neural verifier
- [ ] 安全测试：白盒攻击、绕过尝试

---

## 快速开始

### 安装依赖
```bash
pip install numpy pytest torch torchvision
```

### 运行测试
```bash
# LWE 测试
pytest tests/v2/test_lwe.py -v

# 查看覆盖率
pytest tests/v2/ --cov=src.can.v2 --cov-report=html
```

### 使用示例
```python
from src.can.v2.crypto.lwe import LWEParams, generate_keypair, verify

# 1. 生成密钥对
params = LWEParams()
A, secret, b = generate_keypair(params)

# 2. 验证 credential
is_valid = verify(secret, A, b, params)
print(f"Valid credential: {is_valid}")  # True

# 3. 测试无效 credential
import numpy as np
fake_cred = np.random.randn(params.n).astype(np.float32)
is_valid = verify(fake_cred, A, b, params)
print(f"Invalid credential: {is_valid}")  # False
```

---

## 参考文献

1. Shamir, A. (2024). "How to Securely Implement Cryptography in Deep Neural Networks"
2. Regev, O. (2005). "On lattices, learning with errors, random linear codes, and cryptography"
3. Shazeer, N. et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"

---

## 许可证

研究原型，仅供学术研究使用。

---

**Last Updated**: 2026-08-21  
**Status**: Phase 1.1 Complete ✅
