# 设计方案文档

本文档记录项目所有阶段的设计方案，每个新功能的实现都必须先在此文档中添加设计方案，代码实现将严格遵循此文档。

## 文档结构

- 每个方案包含：设计目标、架构设计、接口定义、实现步骤、测试要求、风险和限制
- 方案状态标记：`[PROPOSED]` → `[APPROVED]` → `[IMPLEMENTED]` → `[COMPLETED]`
- 已实现的方案保留在文档中，便于后续维护和审查


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

## Phase 1.2: Neural Gate Layer [REVISION-1]

**状态**：[REVISION-1]（修订中，待批准）  
**提出时间**：2026-08-23  
**修订时间**：2026-08-23（修复 Codex 审阅发现的 9 个问题）  
**依赖**：Phase 1.1 LWE 密码原语（已完成）

**修订原因**（基于 Codex 审阅）：
1. ❌ **[P1] 浅层特征未融合**：违反项目核心架构"融合浅层特征与 credential"
2. ❌ **[P1] API 不兼容**：`params.threshold` 应为 `params.error_threshold`，`V_ref()` 需要字典输入
3. ❌ **[P1] 单样本接口不兼容**：`[n]` 输入会导致广播错误
4. ❌ **[P1] 缺少输入规范化**：未处理 NaN/Inf/错误类型，违反 fail-closed 要求
5. ⚠️ **[P1] Replay 安全问题**：静态 credential 可重放（记录但不阻塞，当前阶段不防御）
6. ❌ **[P1] Toy LWE 伪造风险未披露**：可通过最小二乘伪造，必须明确声明
7. ❌ **[P2] Fail-closed 声明过度**：Gate Layer 只产生信号，不直接控制深层执行
8. ❌ **[P2] 测试状态标记混乱**：`[PROPOSED]` 但测试已标 `[x]`
9. ❌ **[P2] 差分测试策略矛盾**：布尔结果 100% 一致 vs `atol=1e-5` 矛盾

### 设计目标

实现可微分的 Gate Layer，**融合浅层特征与 LWE credential 验证结果**，实现以下功能：

1. **特征融合**：将 shallow_features 的全局池化与 credential 验证的误差范数融合（可训练）
2. **训练模式**：软路由（sigmoid 软化，可微分，支持梯度回传）
3. **推理模式**：硬路由判定（产生 0/1 信号，配合 Phase 1.3 Gated ResNet 实现 fail-closed）
4. **差分测试**：LWE 验证逻辑与 `V_ref()` 一致（远离阈值时严格一致，阈值附近容差比较误差范数）

**明确限制和风险披露**：
- ⚠️ **Toy LWE 安全性**：当前实现使用无模运算浮点 LWE，m>n，threshold=48 宽松。攻击者可通过最小二乘逼近伪造 credential。**本实现仅用于神经编译演示，不具有 LWE 困难性假设的密码学安全性**。生产部署需升级到整数模运算 + 更大参数 + 更紧阈值。
- Phase 1.2 的 Gate Layer 只产生门控信号（0/1），不直接控制深层执行。真正的 fail-closed（深层零调用）由 Phase 1.3 Gated ResNet 实现并通过 forward hook 验证。
- 当前使用静态 credential（可重放），不防御 replay 攻击。SECURITY.md 明确当前阶段不在范围内。

### 核心挑战

**问题 1**：如何融合异构信息（CNN 特征 vs 密码验证结果）？

**解决方案**：
```python
# 1. 浅层特征全局池化 → 标量特征向量
feature_vec = global_avg_pool(shallow_features)  # [B, C] → [B, d]

# 2. LWE 验证 → 误差范数（标量）
error_norm = ||b - As||₂  # [B]

# 3. 可学习融合（小型 MLP）
fused = MLP([feature_vec, error_norm])  # [B, d+1] → [B, 1]

# 4. 软/硬阈值
gate_signal = sigmoid(fused) or hard_threshold(fused)
```

**问题 2**：如何将 NumPy LWE 验证编译为 PyTorch 并保持 API 兼容？

**解决方案**：
- 使用 `params.error_threshold`（不是 `threshold`）
- `V_ref()` 差分测试传入 `{'vector': credential}`
- 统一输入规范化为 `[B, n]`，输出 `[B]`

### 架构设计

#### 1. 数据流

```
Input: (shallow_features [B,C,H,W], credential [B,n])
    ↓
[输入规范化]
    - 验证形状、dtype、有限性
    - 统一转换为 [B, *]
    ↓
[特征提取]
    - shallow_features → global_avg_pool → [B, C]
    - credential → LWE 验证 → error_norm [B]
    ↓
[可学习融合层]
    - concat([features, error_norm]) → [B, C+1]
    - MLP(2 层) → [B, 1]
    ↓
[软/硬阈值]
    - 训练：sigmoid(fused / temperature) → [B]
    - 推理：(fused > learned_threshold).float() → [B]
    ↓
Output: gate_signal ∈ [0,1] (训练) 或 {0,1} (推理)
```

#### 2. 模块结构

```python
class GateLayer(nn.Module):
    """Gate Layer：融合浅层特征与 LWE 验证的门控层
    
    训练模式：软路由（sigmoid 软化，可微分）
    推理模式：硬路由判定（产生 0/1 信号，不直接控制深层执行）
    
    安全声明：
    - 当前使用 toy LWE（无模运算，可被最小二乘伪造）
    - 仅用于神经编译演示，不具有密码学安全性
    - Gate Layer 只产生信号，fail-closed 由 Gated ResNet 实现
    """
    
    def __init__(self, A: np.ndarray, b: np.ndarray, params: LWEParams,
                 feature_dim: int, hidden_dim: int = 64,
                 temperature: float = 5.0):
        """初始化 Gate Layer
        
        参数:
            A: LWE 公钥矩阵 [m, n]，float32
            b: LWE 公钥向量 [m]，float32
            params: LWE 参数（包含 error_threshold）
            feature_dim: 浅层特征通道数（如 ResNet layer2 输出 C）
            hidden_dim: 融合 MLP 的隐藏层维度
            temperature: 软化温度（训练时使用）
        
        存储:
            - A, b: nn.Buffer（float32，冻结，不参与梯度）
            - error_threshold: float（来自 params.error_threshold）
            - fusion_mlp: nn.Sequential（可训练）
            - temperature: float（训练时软化参数）
        """
        super().__init__()
        
        # LWE 参数（冻结）
        self.register_buffer('A', torch.from_numpy(A).float())
        self.register_buffer('b', torch.from_numpy(b).float())
        self.error_threshold = params.error_threshold
        self.temperature = temperature
        
        # 可学习融合网络
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),  # +1 for error_norm
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输入规范化参数
        self.feature_dim = feature_dim
        self.n = A.shape[1]  # credential 维度
        
    def _validate_and_normalize_credential(self, credential: Union[Tensor, np.ndarray]) -> Tensor:
        """规范化并验证 credential 输入（fail-closed on invalid）
        
        参数:
            credential: [B, n] 或 [n]，Tensor 或 np.ndarray
        
        返回:
            Tensor[B, n]，float32，在正确 device 上
        
        异常:
            ValueError: 非法输入（非有限、错误形状、类型混淆）
        """
        # 类型转换
        if isinstance(credential, np.ndarray):
            credential = torch.from_numpy(credential)
        elif not isinstance(credential, Tensor):
            raise ValueError(f"credential 必须是 Tensor 或 np.ndarray，得到 {type(credential)}")
        
        # 转换为 float32
        if credential.dtype not in [torch.float32, torch.float64, torch.float16]:
            raise ValueError(f"credential 必须是浮点类型，得到 {credential.dtype}")
        credential = credential.float()
        
        # 形状规范化
        if credential.ndim == 1:
            credential = credential.unsqueeze(0)  # [n] → [1, n]
        elif credential.ndim != 2:
            raise ValueError(f"credential 必须是 1D 或 2D，得到 {credential.ndim}D")
        
        # 维度检查
        if credential.shape[1] != self.n:
            raise ValueError(f"credential 维度不匹配：期望 {self.n}，得到 {credential.shape[1]}")
        
        # 有限性检查
        if not torch.isfinite(credential).all():
            raise ValueError("credential 包含 NaN 或 Inf")
        
        # Device 一致性
        credential = credential.to(self.A.device)
        
        return credential
    
    def _compute_lwe_error_norm(self, credential: Tensor) -> Tensor:
        """计算 LWE 验证的误差范数（与 V_ref 一致的逻辑）
        
        参数:
            credential: [B, n]，已规范化
        
        返回:
            error_norm: [B]
        """
        # b - A @ credential.T → [m, B]
        residual = self.b.unsqueeze(1) - torch.matmul(self.A, credential.T)
        
        # L2 范数（按列）→ [B]
        error_norm = torch.norm(residual, p=2, dim=0)
        
        return error_norm
    
    def forward(self, shallow_features: Tensor, credential: Union[Tensor, np.ndarray]) -> Tensor:
        """前向传播：融合浅层特征与 credential 验证
        
        参数:
            shallow_features: [B, C, H, W]，浅层特征图
            credential: [B, n] 或 [n]，LWE secret vector
        
        返回:
            gate_signal: [B]，训练时 ∈ [0,1]，推理时 ∈ {0,1}
        
        异常:
            ValueError: 输入验证失败（fail-closed）
        """
        B = shallow_features.shape[0]
        
        # 1. 输入规范化与验证
        credential = self._validate_and_normalize_credential(credential)
        
        # Batch 维度一致性
        if credential.shape[0] == 1 and B > 1:
            credential = credential.expand(B, -1)  # 广播
        elif credential.shape[0] != B:
            raise ValueError(f"Batch 维度不匹配：features {B} vs credential {credential.shape[0]}")
        
        # 2. 特征提取
        # 浅层特征全局平均池化 → [B, C]
        feature_vec = F.adaptive_avg_pool2d(shallow_features, 1).squeeze(-1).squeeze(-1)
        
        # LWE 验证误差范数 → [B]
        error_norm = self._compute_lwe_error_norm(credential)
        
        # 3. 特征融合
        # concat → [B, C+1]
        fused_input = torch.cat([feature_vec, error_norm.unsqueeze(1)], dim=1)
        
        # MLP → [B, 1] → [B]
        fused_score = self.fusion_mlp(fused_input).squeeze(1)
        
        # 4. 软/硬阈值
        if self.training:
            # 训练：sigmoid 软化
            gate_signal = torch.sigmoid(fused_score / self.temperature)
        else:
            # 推理：硬阈值（learned threshold 隐含在 MLP 中）
            gate_signal = (fused_score > 0.0).float()
        
        return gate_signal
    
    def verify(self, credential: Union[Tensor, np.ndarray]) -> bool:
        """精确 LWE 验证（与 V_ref 一致，用于差分测试）
        
        仅验证 credential，不涉及特征融合。
        
        参数:
            credential: [n]，单个 credential
        
        返回:
            bool: True 表示验证通过
        """
        try:
            credential = self._validate_and_normalize_credential(credential)
            error_norm = self._compute_lwe_error_norm(credential)
            return (error_norm < self.error_threshold).item()
        except (ValueError, RuntimeError):
            return False  # fail-closed
```

### 接口定义

#### 输入（严格验证，fail-closed on invalid）

- `shallow_features`: `Tensor[B, C, H, W]`
  - 类型：`torch.float32`
  - 约束：有限值，C == feature_dim
  
- `credential`: `Tensor[B, n]` 或 `np.ndarray[B, n]` 或 `[n]`（单样本）
  - 类型：float32/float64（自动转换为 float32）
  - 约束：有限值，n == LWEParams.n
  - 单样本 `[n]` 自动广播为 `[B, n]`

**拒绝的输入**：
- 非有限值（NaN, Inf）
- 布尔、整数、复数类型
- 错误形状（> 2D，维度不匹配）
- 类型混淆（字典、列表、字符串）

#### 输出

- `gate_signal`: `Tensor[B]`
  - 训练模式：`∈ [0, 1]`（连续值）
  - 推理模式：`∈ {0, 1}`（离散值）

#### 异常

- `ValueError`：输入验证失败（形状、类型、有限性、维度不匹配）
  - 调用方必须捕获并路由到公开 head（fail-closed）

### 测试要求

**测试状态标记说明**：`[ ]` 未实现，`[x]` 已实现并通过

#### 功能测试

- [ ] `test_valid_credential_training_mode`: valid credential → gate_signal > 0.5
- [ ] `test_valid_credential_eval_mode`: valid credential → gate_signal = 1.0
- [ ] `test_invalid_credential_training_mode`: invalid credential → gate_signal < 0.5
- [ ] `test_invalid_credential_eval_mode`: invalid credential → gate_signal = 0.0
- [ ] `test_feature_fusion`: 不同 shallow_features 影响 gate_signal

#### 差分测试（LWE 验证逻辑）

- [ ] `test_verify_matches_V_ref_far_from_threshold`: 远离阈值时严格一致（100 次）
- [ ] `test_verify_near_threshold`: 阈值附近（±5.0）误差范数容差 `atol=1e-5`
- [ ] `test_verify_at_threshold`: error_norm == threshold 时统一拒绝

#### 输入规范化与异常处理

- [ ] `test_single_sample_broadcast`: `[n]` → `[1, n]` → `[B, n]` 广播
- [ ] `test_batch_processing`: `[B, n]` → `[B]` 正确
- [ ] `test_reject_nan`: NaN → ValueError
- [ ] `test_reject_inf`: Inf → ValueError
- [ ] `test_reject_wrong_dimension`: n != params.n → ValueError
- [ ] `test_reject_wrong_dtype`: int/bool/complex → ValueError
- [ ] `test_reject_wrong_shape`: 3D/0D → ValueError
- [ ] `test_verify_exception_returns_false`: 异常时 `verify()` 返回 False

#### 可训练性测试

- [ ] `test_fusion_mlp_gradients`: MLP 参数接收梯度
- [ ] `test_A_b_frozen`: A, b 不接收梯度

**目标覆盖率**：≥ 95%

### 实现步骤

**Step 1**：创建 `src/can/v2/layers/gate_layer.py`（约 200 行）
- 实现 `GateLayer` 类（含完整输入验证）
- 实现 `forward()`, `verify()`, `_validate_and_normalize_credential()`, `_compute_lwe_error_norm()`

**Step 2**：创建 `tests/v2/test_gate_layer.py`（约 300 行）
- 功能测试（valid/invalid，训练/推理）
- 差分测试（远离阈值/阈值附近/恰好阈值）
- 输入规范化测试（单样本/批量/异常）
- 可训练性测试

**Step 3**：运行测试
```bash
pytest tests/v2/test_gate_layer.py -v --cov=src/can/v2/layers --cov-report=term-missing
```

**Step 4**：更新文档
- `PROJECT_WORKLOG.md`：Phase 1.2 完成，记录测试结果
- `DESIGN_PROPOSALS.md`：状态改为 `[IMPLEMENTED]`

### 性能指标

| 指标 | 目标 | 说明 |
|------|------|------|
| Valid credential → gate_signal (训练) | > 0.5 | 放宽阈值（融合层学习） |
| Valid credential → gate_signal (推理) | = 1.0 | 硬判定 |
| Invalid credential → gate_signal (训练) | < 0.5 | 放宽阈值 |
| Invalid credential → gate_signal (推理) | = 0.0 | 硬判定 |
| LWE 验证逻辑一致性（远离阈值） | 100% | 严格匹配 V_ref |
| LWE 验证逻辑一致性（阈值附近） | 误差范数 atol=1e-5 | 浮点容差 |
| 测试覆盖率 | ≥ 95% | 行覆盖 |
| MLP 参数可训练 | 是 | 接收梯度 |
| A, b 冻结 | 是 | 不接收梯度 |

### 风险和限制

#### 1. Toy LWE 伪造风险（明确披露）

**风险**：当前实现使用无模运算浮点 LWE（m=256, n=128, threshold=48）。攻击者可通过最小二乘求解 `min ||A*x - b||` 得到伪造 credential。

**实验验证**：
```python
# 攻击：最小二乘伪造
x_fake = np.linalg.lstsq(A, b, rcond=None)[0]
error_fake = np.linalg.norm(b - A @ x_fake)
# error_fake 可能 < 48，伪造成功
```

**缓解措施**：
- **Phase 1-2**：明确标注"仅用于神经编译演示，不具有密码学安全性"
- **Phase 3-4（可选升级）**：
  - 整数模运算（q = 2^32）
  - 增大参数（n=256, m=512）
  - 收紧阈值（3σ → 2σ）
  - Rejection sampling

**文档要求**：所有提及 Gate Layer 的地方都必须附带安全声明。

#### 2. Fail-closed 范围限定

**限制**：Phase 1.2 的 Gate Layer 只产生 gate_signal ∈ {0,1}，不直接控制深层执行。

**真正的 fail-closed**（Phase 1.3 实现）：
```python
# Gated ResNet-18
if gate_signal > 0.5:
    deep_feat = self.layer4(self.layer3(shallow_feat))
else:
    # 真正不执行深层（通过 forward hook 验证调用计数 = 0）
    deep_feat = None
```

**Phase 1.2 声明**："产生硬路由判定信号"，不声称"实现 fail-closed"。

#### 3. Replay 攻击（记录但不阻塞）

**风险**：当前使用静态 credential（LWE secret vector），可被重放。

**为什么不阻塞 Phase 1.2**：
- SECURITY.md 明确当前阶段不防御 replay
- 防御需要 nonce/challenge/时效/会话绑定，超出架构验证范围
- Phase 3-4 可添加 challenge-response 扩展

**记录到 PROJECT_WORKLOG.md**："已知限制：静态 credential，不防御 replay"。

#### 4. 软硬路由语义一致性

**风险**：训练时 sigmoid 软化，推理时硬阈值，可能产生语义差异。

**缓解措施**：
- 使用较小 temperature（5.0 而非 10.0）
- 训练后期 temperature annealing
- 监控训练/推理模式下 gate_signal 分布差异

#### 5. 特征融合的可训练性

**风险**：MLP 可能学习到"忽略 error_norm，只看 feature"的捷径。

**缓解措施**：
- L_gate 损失监督：BCE(gate_signal, is_valid_credential)
- 监控 MLP 权重：error_norm 输入的权重不应接近 0
- 消融实验：只用 error_norm vs 完整融合的性能对比

### 后续工作（Phase 1.3）

Phase 1.3 Gated ResNet-18 将实现：
1. 集成 Gate Layer 到 ResNet 计算图中间（layer2 之后）
2. 条件路由：`if gate_signal > 0.5: deep_path else: public_path`
3. Forward hook 验证：invalid credential 时 layer3/layer4 调用计数 = 0
4. 完整的 fail-closed 验证

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
