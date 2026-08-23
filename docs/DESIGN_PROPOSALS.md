# 设计方案文档

本文档记录项目所有阶段的设计方案，每个新功能的实现都必须先在此文档中添加设计方案，代码实现将严格遵循此文档。

## 文档结构

- 每个方案包含：设计目标、架构设计、接口定义、实现步骤、测试要求、风险和限制
- 方案状态标记：`[PROPOSED]` → `[APPROVED]` → `[IMPLEMENTED]` → `[COMPLETED]`
- 已实现的方案保留在文档中，便于后续维护和审查

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

## Phase 1.2: Neural Gate Layer [REVISION-4]

**状态**：[REVISION-4]（修订中，待批准）  
**提出时间**：2026-08-23  
**修订时间**：2026-08-23（基于 Codex 第四轮审阅，移除 One-Time Credential）  
**依赖**：Phase 1.1 LWE 密码原语（已完成）

**修订原因**（基于 Codex 第四轮审阅的根本性问题）：
1. ❌ 派生 credential 无法通过 LWE 验证（数学上不可行）
2. ❌ 训练需要重复使用 credential，与 replay 检测冲突（架构矛盾）
3. ❌ Replay 防御需要复杂的状态管理、持久化、并发控制（超出核心目标）
4. ✅ 决策：Phase 1-2 不实现 replay 防御，专注于"LWE 验证的神经编译"

**用户决策**：修改安全模型，Phase 1-2 明确不防御 replay 攻击

**新的架构**（保留 Revision 3 的组件分离，移除 replay 检测）：
```
credential → LWE验证 → VerificationEvidence
VerificationEvidence → 协调器 → AuthorizationDecision
shallow_features + AuthorizationDecision → 特征门控 → gated_features
```

### 设计目标

实现具有清晰授权边界的 Gate Layer，**无状态、可训练、专注于 LWE 验证**：

1. **验证证据生成**：credential → LWE 验证 → VerificationEvidence（无副作用）
2. **授权决策**：VerificationEvidence → 协调器 → AuthorizationDecision（唯一决策点）
3. **特征门控**：shallow_features × gate_signal → gated_features（应用授权结果）
4. **无状态设计**：不维护 replay 状态，训练和推理使用相同逻辑

**明确限制和安全披露**：
- ⚠️ **Toy LWE 安全性**：无模运算，m>n，threshold=48 宽松。可通过最小二乘伪造。**仅用于神经编译演示，不具有密码学安全性**。
- ⚠️ **不防御 replay 攻击**：Phase 1-2 使用静态 credential，credential 可以被重复使用。研究重点是验证"LWE 验证可以编译为神经网络"，replay 防御留到 Phase 3-4。
- Gate Layer 包含：验证器（evidence）、协调器（decision）、特征门控（gating）三个组件。

### 核心架构

#### 1. 数据流

```
Input: (image, credential)
    ↓
[Shallow Layers] → shallow_features [B, C, H, W]
    ↓
[LWE Verifier] ← credential [B, n]
    ↓ LWE 验证（无副作用）
    ↓
VerificationEvidence {
    lwe_verified: bool,
    error_norm: float,
    reason: ReasonCode
}
    ↓
[Authorization Coordinator]
    ↓ 唯一授权决策点
    ↓
AuthorizationDecision {
    allow: bool,
    gate_signal: float  # 训练时软值，推理时 {0, 1}
}
    ↓
[Feature Gate]
    ↓
gated_features = shallow_features × gate_signal
    ↓
[条件路由]（Phase 1.3）
    if allow:
        deep_features = layer4(layer3(gated_features))
    else:
        public_output = public_head(shallow_features)
```

#### 2. 模块划分

**LWEVerifier**：验证器（只产生证据，无副作用）
- 输入：credential
- 输出：VerificationEvidence（结构化证据）
- 职责：LWE 验证，不做授权决策，不维护状态

**AuthorizationCoordinator**：协调器（唯一授权决策点）
- 输入：VerificationEvidence
- 输出：AuthorizationDecision
- 职责：根据证据提交授权决策

**FeatureGate**：特征门控（应用授权结果）
- 输入：shallow_features, AuthorizationDecision
- 输出：gated_features
- 职责：将授权结果应用到特征（element-wise 乘法）

**GateLayer**：组合层（nn.Module）
- 包含：LWEVerifier + AuthorizationCoordinator + FeatureGate
- 对外接口：forward(shallow_features, credential) → (gated_features, decision)

### 接口定义

#### 数据结构（Tensor-based，支持 batch、GPU、autograd）

```python
from enum import IntEnum
from dataclasses import dataclass
import torch
from torch import Tensor

class ReasonCode(IntEnum):
    """拒绝原因码（IntEnum 便于转换为 LongTensor）"""
    SUCCESS = 0
    LWE_VERIFICATION_FAILED = 1
    INVALID_SHAPE = 2
    NON_FINITE = 3
    WRONG_DTYPE = 4
    DIMENSION_MISMATCH = 5

@dataclass(frozen=True)
class VerificationEvidence:
    """批量 LWE 验证证据（Tensor-based）
    
    所有字段都是 Tensor，支持 batch、GPU 和 autograd。
    禁止使用 .item() 或返回 Python list。
    """
    verified: Tensor       # BoolTensor[B]，LWE 验证是否通过
    error_norm: Tensor     # FloatTensor[B]，LWE 误差范数
    reason_code: Tensor    # LongTensor[B]，拒绝原因码

@dataclass(frozen=True)
class AuthorizationDecision:
    """批量授权决策（Tensor-based）
    
    所有字段都是 Tensor，支持 batch、GPU 和 autograd。
    """
    allow: Tensor          # BoolTensor[B]，是否允许访问深层
    gate_signal: Tensor    # FloatTensor[B]，门控信号（训练时 ∈ [0,1]，推理时 ∈ {0,1}）
    evidence: VerificationEvidence  # 关联的证据
```

#### 非法输入的 Fail-Closed 数值语义

对于形状、类型、有限性验证失败的样本，统一产生：
```python
verified = torch.tensor([False])
error_norm = torch.tensor([float('inf')])  # +inf 表示"必然拒绝"
reason_code = torch.tensor([对应的 ReasonCode])
allow = torch.tensor([False])
gate_signal = torch.tensor([0.0])
```

**关键原则**：
- 不得让 `NaN` 进入 sigmoid 或后续特征计算
- `error_norm = +inf` 保证经过 sigmoid 后 gate_signal ≈ 0
- 所有失败样本的 gate_signal 严格为 0.0

#### 输入约束

**credential**：`Tensor[B, n]` 或 `np.ndarray[B, n]` 或 `[n]`（单样本）
- 类型：float32/float64（自动转换为 float32）
- 约束：有限值，n == LWEParams.n
- 单样本 `[n]` 自动扩展为 `[1, n]`，然后广播到 `[B, n]`（B 从 shallow_features 推断）

**shallow_features**：`Tensor[B, C, H, W]`
- 类型：float32/float64/float16
- 约束：
  - 必须是 4D Tensor
  - 必须是浮点 dtype
  - 必须全部有限（无 NaN/Inf）
  - Batch 维度 B 必须与 credential 一致
  - Device 必须与 Gate Layer 的 buffer (A, b) 兼容

**拒绝的输入**：
- credential: 非有限值、非浮点类型、错误形状、维度不匹配
- shallow_features: 非 Tensor、非 4D、非浮点 dtype、非有限值、batch 不一致

#### 输出

**gated_features**：`Tensor[B, C, H, W]`
- 与 shallow_features 相同的 shape、dtype、device
- `gated_features = shallow_features * gate_signal[:, None, None, None]`

**decision**：`AuthorizationDecision`
- `allow`: BoolTensor[B]
- `gate_signal`: FloatTensor[B]
- `evidence`: VerificationEvidence

#### Mixed Batch 契约

**必须支持**：
- 同一 batch 中同时存在 valid、invalid 和格式错误的 credential
- 每个样本独立生成 verified、reason_code、allow 和 gate_signal
- 输出顺序与输入顺序严格一致
- Invalid 样本不影响 valid 样本
- 空 batch (B=0) 返回空 Tensor

#### 模块接口

**LWEVerifier**：
```python
class LWEVerifier(nn.Module):
    def __init__(self, A: np.ndarray, b: np.ndarray, params: LWEParams):
        """初始化 LWE 验证器
        
        参数:
            A: [m, n]，LWE 公钥矩阵，float32
            b: [m]，LWE 公钥向量，float32
            params: LWE 参数（包含 error_threshold）
        """
        super().__init__()
        # 存储为 nn.Buffer（不可训练，但参与 device 转移）
        self.register_buffer('A', torch.from_numpy(A).float())
        self.register_buffer('b', torch.from_numpy(b).float())
        self.error_threshold = params.error_threshold
        self.n = A.shape[1]
        self.m = A.shape[0]
    
    def forward(self, credential: Union[Tensor, np.ndarray]) -> VerificationEvidence:
        """验证 credential，返回结构化证据（无副作用，可重复调用）
        
        参数:
            credential: [B, n] 或 [n]
        
        返回:
            VerificationEvidence（所有字段都是 Tensor[B]）
        """
        ...
```

**AuthorizationCoordinator**：
```python
class AuthorizationCoordinator(nn.Module):
    def __init__(self, params: LWEParams, temperature: float = 5.0):
        """初始化授权协调器
        
        参数:
            params: LWE 参数（包含 error_threshold）
            temperature: 训练时软化温度（默认 5.0）
        """
        super().__init__()
        self.error_threshold = params.error_threshold
        self.temperature = temperature
    
    def forward(self, evidence: VerificationEvidence) -> AuthorizationDecision:
        """根据证据做出授权决策（唯一授权决策点）
        
        训练模式：软门控，gate_signal = sigmoid(normalized_margin / temperature)
        推理模式：硬判定，gate_signal = evidence.verified.float()
        
        参数:
            evidence: VerificationEvidence
        
        返回:
            AuthorizationDecision
        """
        # allow 的唯一来源
        allow = evidence.verified & (evidence.reason_code == ReasonCode.SUCCESS)
        
        if self.training:
            # 训练：软门控（可微分）
            # 使用归一化 margin 避免饱和
            normalized_margin = (self.error_threshold - evidence.error_norm) / self.error_threshold
            gate_signal = torch.sigmoid(normalized_margin / self.temperature)
        else:
            # 推理：硬判定
            gate_signal = allow.to(evidence.error_norm.dtype)
        
        return AuthorizationDecision(
            allow=allow,
            gate_signal=gate_signal,
            evidence=evidence
        )
```

**FeatureGate**：
```python
class FeatureGate(nn.Module):
    def forward(self, shallow_features: Tensor, decision: AuthorizationDecision) -> Tensor:
        """应用门控到特征（保持 shape、dtype、device）
        
        参数:
            shallow_features: [B, C, H, W]
            decision: AuthorizationDecision（gate_signal [B]）
        
        返回:
            gated_features: [B, C, H, W]
        """
        # gate_signal [B] → [B, 1, 1, 1]
        gate = decision.gate_signal[:, None, None, None]
        
        # Element-wise 乘法
        gated_features = shallow_features * gate
        
        # 保持原始的 shape、dtype、device
        return gated_features
```

**GateLayer**（组合）：
```python
class GateLayer(nn.Module):
    """Gate Layer：LWE 验证 + 授权决策 + 特征门控
    
    关键特性：
    - 无状态：可重复调用（训练需要）
    - 可微分：梯度通过 gated_features 回传给浅层网络
    - 不可训练：A, b 冻结，无可训练参数
    - Batch 友好：所有操作基于 Tensor
    """
    
    def __init__(self, A: np.ndarray, b: np.ndarray, params: LWEParams,
                 temperature: float = 5.0):
        super().__init__()
        self.verifier = LWEVerifier(A, b, params)
        self.coordinator = AuthorizationCoordinator(params, temperature)
        self.feature_gate = FeatureGate()
    
    def forward(self, shallow_features: Tensor, credential: Union[Tensor, np.ndarray]) -> Tuple[Tensor, AuthorizationDecision]:
        """前向传播
        
        参数:
            shallow_features: [B, C, H, W]
            credential: [B, n] 或 [n]
        
        返回:
            gated_features: [B, C, H, W]
            decision: AuthorizationDecision
        
        训练模式：
        - 完整计算，软门控
        - 梯度通过 gated_features 回传给 shallow_features
        
        推理模式：
        - 硬判定，gate_signal ∈ {0, 1}
        - Phase 1.3 可根据 decision.allow 提前拒绝
        """
        # 1. 验证器：产生证据（无副作用）
        evidence = self.verifier(credential)
        
        # 2. 协调器：做出授权决策（唯一授权决策点）
        decision = self.coordinator(evidence)
        
        # 3. 特征门控：应用决策
        gated_features = self.feature_gate(shallow_features, decision)
        
        return gated_features, decision
```

### 软门控温度校准

#### 当前误差分布

根据 Phase 1.1 测试结果：
```
valid credential error_norm ≈ 16
invalid credential error_norm ≈ 900
threshold = 48
```

#### 温度参数分析

**原始方案**（可能饱和）：
```python
gate_signal = sigmoid((threshold - error_norm) / temperature)

# Valid: sigmoid((48 - 16) / 5) = sigmoid(6.4) ≈ 0.998（接近饱和）
# Invalid: sigmoid((48 - 900) / 5) = sigmoid(-170) ≈ 0（饱和）
```

**改进方案**（归一化 margin）：
```python
normalized_margin = (threshold - error_norm) / threshold
gate_signal = sigmoid(normalized_margin / temperature)

# Valid: sigmoid((48 - 16) / 48 / 5) = sigmoid(0.133) ≈ 0.533
# Invalid: sigmoid((48 - 900) / 48 / 5) = sigmoid(-3.55) ≈ 0.028
```

**参数选择**：
- `temperature = 5.0`（默认，提供合理的梯度）
- `temperature = 1.0`（更接近硬阈值，但仍可微）
- 训练后期可使用 temperature annealing（逐步降低温度）

#### 监控指标

实现时必须记录：
- Valid/Invalid gate_signal 分布
- 门控 margin 分布
- shallow_features 的梯度范数
- Coordinator 输出的梯度范数

### 可训练性说明

**Gate Layer 是可微分门控，不是可训练验证器。**

**无可训练参数**：
- `A`, `b` 存储为 `nn.Buffer`（冻结，不参与梯度更新）
- `error_threshold`, `temperature` 是超参数（不参与梯度）
- LWEVerifier、AuthorizationCoordinator、FeatureGate 都没有 `nn.Parameter`

**梯度流动**：
```
Loss → gated_features → gate_signal → evidence.error_norm → ...
     ↓
shallow_features （可以接收梯度）

A, b 不接收梯度（buffer 默认 requires_grad=False）
```

**测试要求**：
- [ ] `test_A_b_no_grad`: A, b 不接收梯度
- [ ] `test_gated_features_backward`: 梯度可以回传到 shallow_features
- [ ] `test_no_trainable_parameters`: Gate Layer 的 `parameters()` 为空

### 实现步骤

**Step 1**：创建 `src/can/v2/layers/gate_layer.py`（约 250 行）
- 实现 `ReasonCode`, `VerificationEvidence`, `AuthorizationDecision`
- 实现 `LWEVerifier`（验证器，无副作用）
- 实现 `AuthorizationCoordinator`（协调器）
- 实现 `FeatureGate`（特征门控）
- 实现 `GateLayer`（组合层）

**Step 2**：创建 `tests/v2/test_gate_layer.py`（约 300 行）
- 功能测试（valid/invalid，训练/推理）
- 差分测试（LWE 验证逻辑）
- 授权边界测试（验证器不做决策，协调器唯一决策）
- 特征门控测试
- Batch 处理测试

**Step 3**：运行测试
```bash
pytest tests/v2/test_gate_layer.py -v --cov=src/can/v2/layers
```

**Step 4**：更新文档
- `PROJECT_WORKLOG.md`：Phase 1.2 完成
- `SECURITY.md`：已更新（明确不防御 replay）
- `docs/DESIGN_PROPOSALS.md`：状态改为 `[IMPLEMENTED]`

### 测试要求

**测试状态标记**：`[ ]` 未实现，`[x]` 已实现并通过

#### LWE 验证测试
- [ ] `test_valid_credential_lwe_verification`: valid credential → lwe_verified=True
- [ ] `test_invalid_credential_lwe_verification`: invalid credential → lwe_verified=False
- [ ] `test_verify_matches_V_ref`: LWE 验证逻辑与 `V_ref()` 100% 一致

#### 授权边界测试
- [ ] `test_verifier_only_produces_evidence`: LWEVerifier 输出是 VerificationEvidence
- [ ] `test_verifier_no_side_effects`: Verifier 无状态变更（可重复调用）
- [ ] `test_coordinator_makes_decision`: AuthorizationCoordinator 输出是 AuthorizationDecision
- [ ] `test_evidence_does_not_contain_authorization`: VerificationEvidence 无授权能力
- [ ] `test_coordinator_is_sole_decision_maker`: 只有协调器产生 allow 字段

#### 特征门控测试
- [ ] `test_feature_gate_allows_when_authorized`: allow=True → gated_features = shallow_features
- [ ] `test_feature_gate_blocks_when_denied`: allow=False → gated_features ≈ 0
- [ ] `test_feature_gate_shape_preserved`: 门控后形状不变
- [ ] `test_gate_signal_soft_in_training`: 训练模式 gate_signal ∈ [0, 1]
- [ ] `test_gate_signal_hard_in_eval`: 推理模式 gate_signal ∈ {0, 1}

#### 端到端测试
- [ ] `test_valid_credential`: valid → allow=True, gated_features = features
- [ ] `test_invalid_credential`: invalid → allow=False, gated_features ≈ 0
- [ ] `test_reason_code_correctness`: 不同失败原因返回正确的 ReasonCode
- [ ] `test_credential_reusable`: 相同 credential 可重复使用（训练需要）

#### 输入验证测试
- [ ] `test_reject_nan`: NaN credential → reason=NON_FINITE
- [ ] `test_reject_inf`: Inf credential → reason=NON_FINITE
- [ ] `test_reject_wrong_dimension`: n != params.n → reason=DIMENSION_MISMATCH
- [ ] `test_reject_wrong_dtype`: int/bool credential → reason=WRONG_DTYPE
- [ ] `test_reject_wrong_shape`: 3D credential → reason=INVALID_SHAPE
- [ ] `test_single_sample_broadcast`: [n] → [1, n] 正确

#### Batch 处理测试
- [ ] `test_batch_processing`: [B, n] → [B] 正确
- [ ] `test_batch_evidence_list`: 逐样本产生 evidence
- [ ] `test_batch_decision_list`: 逐样本产生 decision
- [ ] `test_mixed_batch`: batch 中部分 valid、部分 invalid

#### 训练/推理模式测试
- [ ] `test_training_mode_soft_gate`: 训练模式 gate_signal 连续
- [ ] `test_eval_mode_hard_gate`: 推理模式 gate_signal 离散
- [ ] `test_mode_switch`: train() / eval() 切换正确

**目标覆盖率**：≥ 95%

### 性能指标

| 指标 | 目标 | 说明 |
|------|------|------|
| Valid credential → allow (训练) | True, gate_signal > 0.5 | 软判定 |
| Valid credential → allow (推理) | True, gate_signal = 1.0 | 硬判定 |
| Invalid credential → allow | False | LWE 验证失败 |
| LWE 验证一致性 | 100% | 与 `V_ref()` 一致 |
| Credential 可重用 | 是 | 训练需要重复使用 |
| 测试覆盖率 | ≥ 95% | 行覆盖 |

### 风险和限制

#### 1. Toy LWE 伪造风险（明确披露）

**风险**：无模运算，m>n，threshold=48 宽松，可通过最小二乘伪造。

**缓解措施**：
- **Phase 1-2**：明确标注"仅用于神经编译演示"
- **Phase 3-4（可选升级）**：整数模运算 + 更大参数 + 更紧阈值

#### 2. 不防御 Replay 攻击（明确限制）

**限制**：Phase 1-2 使用静态 credential，可被重复使用和重放。

**理由**：
- 研究重点是验证"LWE 验证可以编译为神经网络"
- Replay 防御需要复杂的状态管理、持久化、并发控制
- 训练需要重复使用 credential，与 One-Time Credential 冲突

**Phase 3-4 升级路径**：
- **Challenge-Response**：交互式，密码学安全
- **Time-based Nonce**：非交互式，简单
- **训练/推理分离**：训练不检测 replay

**记录到 SECURITY.md**："Phase 1-2 不防御 replay 攻击。Replay 防御留到 Phase 3-4。"

#### 3. Fail-closed 范围限定

**Phase 1.2 范围**：Gate Layer 产生 `AuthorizationDecision` 和 `gated_features`。

**Phase 1.3 范围**：Gated ResNet 根据 `decision.allow` 控制深层执行，通过 forward hook 验证深层零调用。

#### 4. 软硬路由语义一致性

**风险**：训练时 sigmoid 软化，推理时硬阈值，边界附近可能不一致。

**缓解措施**：
- Temperature = 5.0（较小，减少差异）
- 训练后期 temperature annealing
- 差分测试覆盖边界情况

#### 5. 特征门控的可微分性

**训练模式**：
```python
# gate_signal 是软值（sigmoid）
gated_features = shallow_features * gate_signal.view(B, 1, 1, 1)
# 梯度可以通过 gate_signal 回传到模型主干
```

**推理模式**：
```python
# gate_signal 是硬值 {0, 1}
gated_features = shallow_features * gate_signal.view(B, 1, 1, 1)
# gate_signal = 0 时，梯度为 0（但推理不需要梯度）
```

**注意**：
- Gate Layer 本身无可训练参数（A, b 冻结）
- 梯度通过 gate_signal 传递给模型主干（layer1, layer2）
- 训练时的软化保证梯度流动

### 与 Codex 审阅的对照

#### 解决的根本性问题

1. ✅ **移除 One-Time Credential**：避免派生 credential 无法通过 LWE 验证
2. ✅ **移除 Replay 检测**：避免训练时 credential 重用冲突
3. ✅ **无状态设计**：Verifier 无副作用，可重复调用
4. ✅ **保留授权边界**：Verifier → evidence，Coordinator → decision
5. ✅ **保留特征门控**：满足"融合浅层特征与 credential"目标

#### 解决的技术问题

6. ✅ **Batch 接口兼容**：逐样本处理 evidence 和 decision
7. ✅ **训练可行性**：credential 可重复使用
8. ✅ **安全模型一致**：SECURITY.md 明确不防御 replay

### 总结

**Revision 4 的核心变更**：
1. **移除 One-Time Credential**：避免根本性技术问题
2. **移除 Replay 检测**：专注于 LWE 验证的神经编译
3. **保留授权边界分离**：Verifier → Coordinator → FeatureGate
4. **无状态设计**：训练和推理使用相同逻辑
5. **修改安全模型**：明确 Phase 1-2 不防御 replay

**与 Revision 3 的对比**：
- Revision 3：credential → LWE验证 + Replay检查 → evidence → decision
- Revision 4：credential → LWE验证 → evidence → decision（无 replay）

**优势**：
- ✅ 解决 Codex 指出的所有根本性问题
- ✅ 架构清晰，无状态，可训练
- ✅ 专注于核心目标："LWE 验证的神经编译"
- ✅ 为 Phase 3-4 的 replay 防御打好基础

---
