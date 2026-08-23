# 设计方案文档

本文档记录项目所有阶段的设计方案，每个新功能的实现都必须先在此文档中添加设计方案，代码实现将严格遵循此文档。

## 文档结构

- 每个方案包含：设计目标、架构设计、接口定义、实现步骤、测试要求、风险和限制
- 方案状态标记：`[PROPOSED]` → `[APPROVED]` → `[IMPLEMENTED]` → `[COMPLETED]`
- 已实现的方案保留在文档中，便于后续维护和审查

---

## Phase 1.2: Neural Gate Layer [PROPOSED]

**状态**：[PROPOSED]  
**提出时间**：2026-08-23  
**依赖**：Phase 1.1 LWE 密码原语（已完成）

### 设计目标

实现可微分的 Gate Layer，将 LWE 验证逻辑嵌入神经网络计算图中间，实现以下功能：
1. 训练模式：软路由（可微分，支持梯度回传）
2. 推理模式：硬路由（fail-closed，真正不执行深层）
3. 与 LWE 参考实现的差分测试通过（验证逻辑正确性）

### 核心挑战

**问题**：LWE 验证是 NumPy 实现的非线性操作（矩阵乘法 + L2 范数 + 阈值判断），如何编译为可微分的 PyTorch 计算？

**关键约束**：
- 训练时必须可微分（软阈值）
- 推理时必须精确验证（硬阈值，与 V_ref 一致）
- A, b 参数不参与梯度更新（冻结）
- 支持批量处理（batch dimension）

### 架构设计

#### 1. 数据流

```
Input: (shallow_features, credential)
    ↓
[LWE 验证层]
    - 计算误差范数: ||b - As||₂
    - 软/硬阈值判断
    ↓
Output: gate_signal ∈ [0, 1] (训练) 或 {0, 1} (推理)
```

#### 2. 模块结构

```python
class GateLayer(nn.Module):
    """Gate Layer：嵌入 LWE 验证的门控层
    
    训练模式：软路由（sigmoid 软化，可微分）
    推理模式：硬路由（阈值判断，fail-closed）
    """
    
    def __init__(self, A: np.ndarray, b: np.ndarray, params: LWEParams):
        """初始化 Gate Layer
        
        参数:
            A: LWE 公钥矩阵 [m, n]
            b: LWE 公钥向量 [m]
            params: LWE 参数（threshold, temperature）
        """
        # A, b 存储为 nn.Buffer（不参与梯度更新）
        # temperature: 软化参数（训练时使用）
    
    def forward(self, shallow_features: Tensor, credential: Tensor) -> Tensor:
        """前向传播
        
        参数:
            shallow_features: 浅层特征 [B, C, H, W]（暂不使用，Phase 1.3 融合）
            credential: LWE secret [B, n]
        
        返回:
            gate_signal: [B]，训练时 ∈ [0, 1]，推理时 ∈ {0, 1}
        """
        # 1. 计算误差范数: error = ||b - A @ credential.T||₂
        # 2. 训练模式：gate = sigmoid((threshold - error) / temperature)
        # 3. 推理模式：gate = (error < threshold).float()
    
    def verify(self, credential: Tensor) -> Tensor:
        """精确验证（与 V_ref 一致的逻辑）
        
        用于差分测试和推理模式。
        """
```

### 接口定义

#### 输入
- `shallow_features`: `Tensor[B, C, H, W]`（Phase 1.2 暂不使用，直接传入）
- `credential`: `Tensor[B, n]` 或 `np.ndarray[B, n]`（LWE secret vector）

#### 输出
- `gate_signal`: `Tensor[B]`
  - 训练模式：`∈ [0, 1]`（连续值）
  - 推理模式：`∈ {0, 1}`（离散值）

#### 参数存储
- `A`: `nn.Buffer[m, n]`（LWE 公钥矩阵，冻结）
- `b`: `nn.Buffer[m]`（LWE 公钥向量，冻结）
- `threshold`: `float`（误差阈值，默认 48.0）
- `temperature`: `float`（软化温度，默认 10.0）

### 实现细节

#### 1. NumPy 到 PyTorch 的转换

**LWE 验证逻辑（NumPy 参考实现）**：
```python
def V_ref(credential, A, b, params):
    error = b - A @ credential
    error_norm = np.linalg.norm(error, ord=2)
    return error_norm < params.threshold
```

**PyTorch 神经编译版本**：
```python
def forward(self, credential):
    # credential: [B, n]
    # A: [m, n], b: [m]
    
    # 批量计算：b - A @ credential.T → [m, B]
    residual = self.b.unsqueeze(1) - torch.matmul(self.A, credential.T)
    
    # L2 范数：[B]
    error_norm = torch.norm(residual, p=2, dim=0)
    
    if self.training:
        # 软路由：sigmoid((threshold - error) / temperature)
        gate_signal = torch.sigmoid((self.threshold - error_norm) / self.temperature)
    else:
        # 硬路由：error < threshold
        gate_signal = (error_norm < self.threshold).float()
    
    return gate_signal
```

#### 2. 软硬路由切换

| 模式 | 公式 | 性质 |
|------|------|------|
| 训练（软路由） | `sigmoid((threshold - error) / T)` | 可微分，梯度连续 |
| 推理（硬路由） | `(error < threshold).float()` | 精确判断，fail-closed |

**温度参数 T 的作用**：
- T 较大（如 10.0）：软化明显，梯度平滑
- T 较小（如 1.0）：接近硬阈值，但仍可微
- 推理时直接使用硬阈值（不需要 T）

#### 3. 差分测试设计

**目标**：验证 `GateLayer.verify()` 与 `V_ref()` 的一致性

```python
def test_differential_verify():
    """差分测试：GateLayer.verify() vs V_ref()"""
    params = LWEParams()
    A, secret, b = generate_keypair(params)
    gate_layer = GateLayer(A, b, params)
    
    # 测试 100 个随机 credential
    for _ in range(100):
        credential = generate_random_credential(params)
        
        # PyTorch 版本
        gate_signal = gate_layer.verify(torch.from_numpy(credential))
        
        # NumPy 参考版本
        expected = V_ref(credential, A, b, params)
        
        # 断言一致
        assert gate_signal.item() == float(expected)
```

### 实现步骤

**Step 1**：创建 `src/can/v2/layers/gate_layer.py`
- 实现 `GateLayer` 类
- 实现 `forward()` 和 `verify()` 方法
- 支持 NumPy 和 PyTorch 输入类型转换

**Step 2**：创建 `tests/v2/test_gate_layer.py`
- 测试 valid/invalid credential 的路由行为
- 测试训练/推理模式切换
- 差分测试：`verify()` vs `V_ref()`
- 批量处理测试
- 形状和类型测试

**Step 3**：运行测试并验证
```bash
pytest tests/v2/test_gate_layer.py -v --cov=src/can/v2/layers
```

**Step 4**：更新 `PROJECT_WORKLOG.md`
- 标记 Phase 1.2 完成
- 记录测试结果
- 更新"唯一下一步"

### 测试要求

#### 功能测试
- [x] `test_valid_credential_training_mode`: valid credential → gate_signal > 0.7
- [x] `test_valid_credential_eval_mode`: valid credential → gate_signal = 1.0
- [x] `test_invalid_credential_training_mode`: invalid credential → gate_signal < 0.3
- [x] `test_invalid_credential_eval_mode`: invalid credential → gate_signal = 0.0

#### 差分测试
- [x] `test_verify_matches_V_ref`: `verify()` 与 `V_ref()` 100% 一致（100 次随机测试）

#### 边界测试
- [x] `test_threshold_boundary`: error = threshold ± ε 的行为
- [x] `test_temperature_effect`: 不同温度参数的软化效果

#### 形状测试
- [x] `test_batch_processing`: 批量输入 [B, n] → 批量输出 [B]
- [x] `test_single_sample`: 单样本输入 [n] → 标量输出

#### 异常处理
- [x] `test_wrong_dimension`: credential 维度不匹配 → 抛出异常
- [x] `test_invalid_type`: 非法输入类型 → 抛出异常

**目标覆盖率**：≥ 95%

### 性能指标

| 指标 | 目标 |
|------|------|
| Valid credential → gate_signal (训练) | > 0.7 |
| Valid credential → gate_signal (推理) | = 1.0 |
| Invalid credential → gate_signal (训练) | < 0.3 |
| Invalid credential → gate_signal (推理) | = 0.0 |
| 差分测试一致性 | 100% |
| 测试覆盖率 | ≥ 95% |

### 风险和限制

#### 1. 软硬路由语义一致性

**风险**：训练时的软路由（sigmoid）和推理时的硬路由（threshold）可能产生语义差异。

**示例**：
- 训练时：error = 47.5 → gate_signal = 0.52（sigmoid 软化）
- 推理时：error = 47.5 → gate_signal = 1.0（硬阈值）

**缓解措施**：
- 使用较小的 temperature（如 5.0）减少差异
- 训练后期使用 temperature annealing（逐步降低温度）
- 差分测试覆盖边界情况

#### 2. 梯度消失问题

**风险**：sigmoid 在远离 threshold 的区域梯度接近 0。

**缓解措施**：
- 使用合适的 temperature 参数
- 监控训练时的梯度范数
- 如果出现梯度消失，考虑使用 straight-through estimator

#### 3. 浮点精度差异

**风险**：PyTorch 和 NumPy 的浮点运算可能有微小差异。

**缓解措施**：
- 差分测试使用宽松的容差（如 `atol=1e-5`）
- 使用相同的数值类型（float32）
- 记录并报告差异范围

#### 4. Phase 1.2 未实现特征融合

**限制**：当前版本 `shallow_features` 参数暂不使用，直接传入但不参与计算。

**原因**：Phase 1.2 重点验证 LWE 验证的神经编译，特征融合留到 Phase 1.3 Gated ResNet-18。

### 后续扩展（Phase 1.3）

Phase 1.3 将实现 Gated ResNet-18，届时需要：
1. 将 `shallow_features` 与 credential 融合（如 concat + MLP）
2. 集成到 ResNet-18 的计算图中间
3. 实现完整的条件路由（deep path vs public path）

---

## Phase 1.1: LWE 密码原语 [COMPLETED]

**状态**：[COMPLETED]  
**完成时间**：2026-08-21  
**测试结果**：✅ 38/38 通过，100% 覆盖率

### 设计目标

实现 LWE (Learning With Errors) 密码原语，提供密钥生成、验证和参考实现。

### 实现内容

- `LWEParams`：参数类（n=128, m=256, σ=1.0, threshold=48.0）
- `generate_keypair()`：生成 (A, secret, b) where b = As + e
- `verify(secret, A, b, params)`：验证 ||b - As|| < threshold
- `compute_error_norm()`：计算 L2 误差范数
- `V_ref(credential, A, b, params) → {0, 1}`：参考验证器

### 测试结果

- 测试覆盖率：100% (55/55 statements)
- 测试通过率：38/38
- 误差分布：valid ~16, invalid ~900, threshold=48
- 假阳性率：0%（100 次随机测试）

### 文档

- 实现代码：`src/can/v2/crypto/lwe.py`
- 测试代码：`tests/v2/test_lwe.py`
- 决策文档：`docs/V2_LWE_IMPLEMENTATION.md`

---

## 附录：设计方案模板

```markdown
## Phase X.Y: [功能名称] [状态]

**状态**：[PROPOSED/APPROVED/IMPLEMENTED/COMPLETED]  
**提出时间**：YYYY-MM-DD  
**依赖**：[前置任务]

### 设计目标
[简要描述实现目标]

### 核心挑战
[技术难点和关键约束]

### 架构设计
[模块结构、数据流、接口定义]

### 实现步骤
[分步骤的实现计划]

### 测试要求
[测试用例列表和覆盖率目标]

### 性能指标
[量化的性能目标]

### 风险和限制
[已知风险和缓解措施]
```
