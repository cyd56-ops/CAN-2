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

## Phase 1.2: Neural Gate Layer [COMPLETED]

**状态**：[COMPLETED]（Revision 5，2026-08-23）
**提出时间**：2026-08-23  
**修订时间**：2026-08-23（统一软门控公式、批量错误语义与 dtype/device 契约）
**依赖**：Phase 1.1 LWE 密码原语（已完成）

**修订原因**（基于 Codex 第四轮审阅的根本性问题）：
1. ❌ 派生 credential 无法通过 LWE 验证（数学上不可行）
2. ❌ 训练需要重复使用 credential，与 replay 检测冲突（架构矛盾）
3. ❌ Replay 防御需要复杂的状态管理、持久化、并发控制（超出核心目标）
4. ✅ 决策：Phase 1-2 不实现 replay 防御，专注于"LWE 验证的神经编译"

**用户决策**：修改安全模型，Phase 1-2 明确不防御 replay 攻击

**Revision 5 修订内容**：
1. 统一采用原始 margin 软门控公式，并重新校准 temperature
2. 区分训练软衰减与推理严格清零
3. 明确 mixed batch、请求级错误和空 batch 的处理职责
4. 固化 credential、feature、buffer 的 dtype/device 契约

**新的架构**（保留 Revision 3 的组件分离，移除 replay 检测）：
```
credential → LWE验证 → VerificationEvidence
VerificationEvidence → 协调器 → AuthorizationDecision
shallow_features + AuthorizationDecision → 特征门控 → gated_features
```

### 设计目标

实现具有清晰授权边界的 Gate Layer，**无状态、可微分、专注于 LWE 验证**：

1. **验证证据生成**：credential → LWE 验证 → VerificationEvidence（无副作用）
2. **授权决策**：VerificationEvidence → 协调器 → AuthorizationDecision（唯一决策点）
3. **特征门控**：shallow_features × gate_signal → gated_features（应用授权结果）
4. **无状态设计**：不维护 replay 状态，训练和推理共享同一确定性验证逻辑

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
    verified: BoolTensor[B],
    error_norm: FloatTensor[B],
    reason_code: LongTensor[B]
}
    ↓
[Authorization Coordinator]
    ↓ 唯一授权决策点
    ↓
AuthorizationDecision {
    allow: BoolTensor[B],
    gate_signal: FloatTensor[B]  # 训练时软值，推理时 {0, 1}
}
    ↓
[Feature Gate]
    ↓
gated_features = shallow_features × gate_signal
    ↓
[条件路由]（Phase 1.3）
    valid_indices = nonzero(allow)
    invalid_indices = nonzero(~allow)
    deep_features = layer4(layer3(gated_features[valid_indices]))
    public_output = public_head(shallow_features[invalid_indices])
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
- 协调器先根据 `reason_code` 建立成功 mask；失败样本不进入 sigmoid
- 所有输入验证失败样本的 gate_signal 严格为 0.0
- 格式合法但 LWE 验证失败的样本在训练模式保留软 gate；推理模式严格为 0.0
- 非法请求由 `GateLayer` 按 `shallow_features.shape[0]` 构造整批拒绝 evidence

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

**设备和数值类型规则**：
- credential 在验证前统一移动到 `A.device` 并转换为 `torch.float32`
- `shallow_features.device` 必须等于 `A.device`，不隐式移动特征
- FeatureGate 在乘法前将 gate_signal 转为 shallow_features 的 dtype 和 device
- gated_features 必须保持 shallow_features 的 shape、dtype 和 device

**拒绝的输入**：
- credential: 非有限值、非浮点类型、错误形状、维度不匹配
- shallow_features: 非 Tensor、非 4D、非浮点 dtype、非有限值、batch 不一致

**错误返回契约**：
- credential 的结构、dtype、维度或有限性错误：GateLayer 根据有效的 shallow_features batch 构造逐样本拒绝 evidence，并返回全零 gated_features
- shallow_features 非有限但仍是合法 4D 浮点 Tensor：抛出 `ValueError`，不得继续进入公开或深层 head
- shallow_features 非 Tensor、错误 rank、错误 dtype 或错误 device：抛出 `TypeError`/`ValueError`，此时无法承诺正常输出 shape
- 上述 feature 异常属于请求级失败，不进入 LWEVerifier、AuthorizationCoordinator 或 FeatureGate

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
- 同一 batch 中同时存在 valid、invalid 和**包含 NaN/Inf** 的 credential
- 每个样本独立生成 verified、reason_code、allow 和 gate_signal
- 输出顺序与输入顺序严格一致
- Invalid 样本不影响 valid 样本
- 空 batch (B=0) 返回空 Tensor

**请求级错误（整批拒绝）**：
- 错误的 dtype（如 int64, bool）
- 错误的 rank（如 3D, 0D）
- 错误的末维（n != params.n）

credential 请求级错误由 `GateLayer` 捕获并按 `B = shallow_features.shape[0]` 返回长度为 B 的拒绝 evidence；`LWEVerifier` 不负责猜测无法从 credential 推断的 batch 大小。

**稳定检查顺序**：输入类型 → rank → dtype → 末维 → batch → finite。若同时存在多个请求级错误，返回最先命中的 reason code。

**理由**：同一个 `Tensor[B, n]` 的所有行共享 shape 和 dtype，无法在同一 batch 内表示异构格式。逐样本异构需要 list 输入，会破坏向量化，当前阶段不支持。

**Mixed batch 定义**：指同一 `Tensor[B, n]` 中，某些行是 valid credential，某些行是 invalid credential，某些行包含 NaN/Inf。

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
    
    def forward(self, credential: Tensor) -> VerificationEvidence:
        """验证 credential，返回结构化证据（无副作用，可重复调用）
        
        参数:
            credential: 已规范化的 float32 Tensor[B, n]
        
        返回:
            VerificationEvidence（所有字段都是 Tensor[B]）
        """
        ...
```

`LWEVerifier` 只接收已经完成请求级规范化的 `[B, n]` float32 Tensor。它对每行建立 finite mask，只对有限行计算残差；非有限行直接产生 `verified=False`、`error_norm=+inf` 和 `NON_FINITE`，避免 NaN/Inf 进入矩阵乘法结果与 sigmoid。

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
        
        训练模式：软门控，gate_signal = sigmoid((threshold - error_norm) / temperature)
        推理模式：硬判定，gate_signal = evidence.verified.float()
        
        参数:
            evidence: VerificationEvidence
        
        返回:
            AuthorizationDecision
        """
        # allow 的唯一来源
        allow = evidence.verified & (evidence.reason_code == ReasonCode.SUCCESS)
        
        parsed = (
            (evidence.reason_code == ReasonCode.SUCCESS)
            | (evidence.reason_code == ReasonCode.LWE_VERIFICATION_FAILED)
        )

        if self.training:
            # 训练：仅让成功解析的样本贡献软门控，非法输入严格为 0。
            safe_error_norm = torch.where(
                parsed,
                evidence.error_norm,
                torch.full_like(evidence.error_norm, self.error_threshold),
            )
            soft_gate = torch.sigmoid(
                (self.error_threshold - safe_error_norm) / self.temperature
            )
            gate_signal = torch.where(parsed, soft_gate, torch.zeros_like(soft_gate))
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
        B = shallow_features.shape[0]
        
        # 验证 batch 一致性
        if decision.gate_signal.shape[0] != B:
            raise ValueError(
                f"Batch 不一致：shallow_features {B} vs decision {decision.gate_signal.shape[0]}"
            )
        
        # gate_signal [B] → 与特征相同 dtype/device 的 [B, 1, 1, 1]
        gate = decision.gate_signal.to(
            device=shallow_features.device,
            dtype=shallow_features.dtype,
        )[:, None, None, None]
        
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
        # GateLayer 先完整验证 shallow_features，再读取 B；结构或 device
        # 错误直接抛出，保证不会进入 verifier/coordinator/feature gate。
        B = shallow_features.shape[0]

        if B == 0:
            # 空 batch 保持所有输出 Tensor 的 batch 维为 0。
            credential = torch.empty(
                (0, self.verifier.n),
                device=self.verifier.A.device,
                dtype=torch.float32,
            )
        
        # 处理单 credential 广播
        if isinstance(credential, np.ndarray):
            credential = torch.from_numpy(credential)
        
        if B > 0 and credential.ndim == 1:
            # [n] → [B, n] 广播到与 shallow_features 相同的 batch
            credential = credential.unsqueeze(0).expand(B, -1)
        elif B > 1 and credential.ndim == 2 and credential.shape[0] == 1:
            # [1, n] → [B, n]
            credential = credential.expand(B, -1)
        
        # credential 按类型、rank、dtype、末维、batch、finite 的顺序检查；
        # 必须先拒绝整数、布尔和复数，再进行 float32 转换。
        # 请求级错误在此构造长度为 B 的拒绝 evidence。
        credential = credential.to(
            device=self.verifier.A.device,
            dtype=torch.float32,
        )

        # 1. 验证器：产生证据（无副作用）
        evidence = self.verifier(credential)
        
        # 2. 协调器：做出授权决策（唯一授权决策点）
        decision = self.coordinator(evidence)
        
        # 3. 特征门控：应用决策
        gated_features = self.feature_gate(shallow_features, decision)
        
        return gated_features, decision
```

实际实现应将上述请求级检查封装为 `_validate_and_normalize_request()`；示例中的 `.to(float32)` 只在 dtype 已确认是允许的浮点类型后执行。credential 请求级失败由 GateLayer 直接构造拒绝 decision 和全零 gated_features，不再调用后续组件。`LWEVerifier`、`AuthorizationCoordinator` 和 `FeatureGate` 视为包内组件，对外公开 API 仅为 `GateLayer.forward()`，避免调用方绕过组合流程。

#### 构造期约束

- `A` 必须是有限的二维浮点 NumPy 数组，shape 严格等于 `(params.m, params.n)`
- `b` 必须是有限的一维浮点 NumPy 数组，shape 严格等于 `(params.m,)`
- `params.error_threshold` 必须是有限正数
- `temperature` 必须是有限正数
- 构造后 A、b 注册为 float32 buffer，不出现在 `parameters()` 中

### 软门控温度校准

#### 当前误差分布

根据 Phase 1.1 测试结果：
```
valid credential error_norm ≈ 16
invalid credential error_norm ≈ 900
threshold = 48
```

#### 温度参数分析

**设计目标**：Valid credential → gate ≈ 1.0，Invalid credential → gate ≈ 0.0

根据 Phase 1.1 测试结果：
```
valid credential error_norm ≈ 16
invalid credential error_norm ≈ 900
threshold = 48
```

**方案**：使用原始公式（避免过度软化）
```python
gate_signal = sigmoid((threshold - error_norm) / temperature)

# 测试不同温度：
temperature = 1.0:
  Valid: sigmoid((48 - 16) / 1.0) = sigmoid(32) ≈ 1.0 ✓
  Invalid: sigmoid((48 - 900) / 1.0) = sigmoid(-852) ≈ 0.0 ✓

temperature = 5.0:
  Valid: sigmoid((48 - 16) / 5.0) = sigmoid(6.4) ≈ 0.998 ✓
  Invalid: sigmoid((48 - 900) / 5.0) = sigmoid(-170) ≈ 0.0 ✓

temperature = 10.0:
  Valid: sigmoid((48 - 16) / 10.0) = sigmoid(3.2) ≈ 0.96 ✓
  Invalid: sigmoid((48 - 900) / 10.0) = sigmoid(-85) ≈ 0.0 ✓
```

**推荐**：原始 margin 公式配合 `temperature = 5.0`（默认）
- Valid credential 产生接近 1.0 的 gate_signal
- Invalid credential 产生接近 0.0 的 gate_signal
- valid gate 接近 1，使浅层主干梯度基本不衰减
- 训练/推理差异小

**注意**：Gate Layer 本身无可训练参数，软化的主要目的是提供梯度给浅层网络，而不是为了 Gate 自身的学习。因此 sigmoid 轻微饱和是可接受的。

#### 监控指标

实现时必须记录：
- **Valid/Invalid gate_signal 分布**：确认 valid ≈ 1.0, invalid ≈ 0.0
- **训练/推理 gate_signal 差异**：确认差异小
- **Shallow_features 的梯度范数**：确认梯度流动正常
- **训练/推理输出差异**：最终模型输出的差异

### 可微分性说明

**Gate Layer 是可微分门控，不是可训练验证器。**

**无可训练参数**：
- `A`, `b` 存储为 `nn.Buffer`（冻结，不参与梯度更新）
- `error_threshold`, `temperature` 是超参数（不参与梯度）
- LWEVerifier、AuthorizationCoordinator、FeatureGate 都没有 `nn.Parameter`

**梯度流动**：
```
Loss → gated_features
     ↓ (∂Loss/∂gated_features × gate_signal)
shallow_features （接收梯度）

gate_signal 本身不参与参数更新（Gate Layer 无可训练参数）
A, b 不接收梯度（buffer 默认 requires_grad=False）
Credential 通常不需要梯度（外部输入）
```

**准确说明**：
- Gate Layer 提供可微分的门控信号
- 梯度通过 `gated_features = shallow_features * gate_signal` 回传
- 浅层网络（layer1, layer2）接收到的梯度是 `∂Loss/∂gated_features × gate_signal`
- Gate_signal 的数值影响梯度大小，但 Gate 本身不学习

**测试要求**：
- [x] `test_A_b_no_grad`: A, b 不接收梯度
- [x] `test_gated_features_backward`: 梯度可以回传到 shallow_features
- [x] `test_no_trainable_parameters`: Gate Layer 的 `parameters()` 为空

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
- [x] `test_valid_credential_lwe_verification`: valid credential → lwe_verified=True
- [x] `test_invalid_credential_lwe_verification`: invalid credential → lwe_verified=False
- [x] `test_verify_matches_V_ref`: LWE 验证逻辑与 `V_ref()` 100% 一致

#### 授权边界测试
- [x] `test_verifier_only_produces_evidence`: LWEVerifier 输出是 VerificationEvidence
- [x] `test_verifier_no_side_effects`: Verifier 无状态变更（可重复调用）
- [x] `test_coordinator_makes_decision`: AuthorizationCoordinator 输出是 AuthorizationDecision
- [x] `test_evidence_does_not_contain_authorization`: VerificationEvidence 无授权能力
- [x] `test_coordinator_is_sole_decision_maker`: 只有协调器产生 allow 字段

#### 特征门控测试
- [x] `test_feature_gate_allows_in_eval`: 推理 allow=True → gated_features = shallow_features
- [x] `test_feature_gate_blocks_in_eval`: 推理 allow=False → gated_features 严格为 0
- [x] `test_feature_gate_soft_in_training`: 训练时 gated_features = shallow_features × soft gate
- [x] `test_feature_gate_shape_preserved`: 门控后形状不变
- [x] `test_gate_signal_soft_in_training`: 训练模式 gate_signal ∈ [0, 1]
- [x] `test_gate_signal_hard_in_eval`: 推理模式 gate_signal ∈ {0, 1}

#### 端到端测试
- [x] `test_valid_credential_eval`: valid → allow=True, gated_features = features
- [x] `test_invalid_credential_eval`: invalid → allow=False, gated_features 严格为 0
- [x] `test_invalid_credential_training`: invalid → allow=False，但格式合法样本保留可计算的软 gate
- [x] `test_reason_code_correctness`: 不同失败原因返回正确的 ReasonCode
- [x] `test_credential_reusable`: 相同 credential 可重复使用（训练需要）

#### 输入验证测试
- [x] `test_reject_nan`: NaN credential → reason=NON_FINITE
- [x] `test_reject_inf`: Inf credential → reason=NON_FINITE
- [x] `test_reject_wrong_dimension`: n != params.n → reason=DIMENSION_MISMATCH
- [x] `test_reject_wrong_dtype`: int/bool credential → reason=WRONG_DTYPE
- [x] `test_reject_wrong_shape`: 3D credential → reason=INVALID_SHAPE
- [x] `test_single_sample_broadcast`: [n] → [1, n] 正确
- [x] `test_feature_reject_nan_inf`: 非有限 shallow_features → ValueError 且不调用后续组件
- [x] `test_feature_reject_wrong_dtype_rank`: 非浮点或非 4D features → TypeError/ValueError
- [x] `test_feature_credential_batch_mismatch`: batch 不一致 → 批量拒绝 decision 和全零 gated_features
- [x] `test_device_contract`: features、credential、buffer 的 device 契约正确

#### Batch 处理测试
- [x] `test_batch_processing`: [B, n] → [B] 正确
- [x] `test_batch_evidence_tensor`: evidence 各字段均为 Tensor[B]
- [x] `test_batch_decision_tensor`: decision 各字段均为 Tensor[B]
- [x] `test_mixed_batch`: batch 中部分 valid、部分 invalid
- [x] `test_mixed_non_finite_batch`: 非有限行严格拒绝且不影响其他行
- [x] `test_empty_batch`: B=0 时所有输出保持空 batch

#### 构造期和边界测试
- [x] `test_constructor_rejects_invalid_A_b`: 错误类型、shape 或非有限 A/b 被拒绝
- [x] `test_constructor_rejects_invalid_threshold_temperature`: 非有限或非正参数被拒绝
- [x] `test_threshold_boundary`: 等于阈值时拒绝，阈值两侧行为与 `V_ref()` 一致
- [x] `test_output_dtype_device_preserved`: gated_features 保持输入 dtype/device

#### 训练/推理模式测试
- [x] `test_training_mode_soft_gate`: 训练模式 gate_signal 连续
- [x] `test_eval_mode_hard_gate`: 推理模式 gate_signal 离散
- [x] `test_mode_switch`: train() / eval() 切换正确

**目标覆盖率**：≥ 95%

### 性能指标

| 指标 | 目标 | 说明 |
|------|------|------|
| Valid credential → allow (训练) | True, gate_signal > 0.7 | 软判定 |
| Valid credential → allow (推理) | True, gate_signal = 1.0 | 硬判定 |
| Invalid credential → allow | False | LWE 验证失败 |
| 非法输入 → gate_signal | = 0.0 | 训练和推理均严格拒绝 |
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

6. ✅ **Batch 接口兼容**：使用 Tensor[B] 向量化处理 evidence 和 decision
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
- ✅ 架构清晰、无状态、可微分
- ✅ 专注于核心目标："LWE 验证的神经编译"
- ✅ 为 Phase 3-4 的 replay 防御打好基础

---
# Phase 1.3: Gated ResNet-18 设计方案（Revision 1）

**状态**：[COMPLETED]（Codex 实现，Claude 已验收）
**提出时间**：2026-08-23
**修订时间**：2026-08-23
**依赖**：
- Phase 1.1: LWE 密码原语（已完成）
- Phase 1.2: Neural Gate Layer（已完成）

---

## 设计目标

将 Gate Layer 集成到 ResNet-18 架构中，实现基于 credential 的条件路由：
- Valid credential → 执行深层网络 → fine-grained 输出（10-class）
- Invalid credential → 只执行浅层网络 → coarse 输出（2-class）

**核心验证**：
1. **Fail-closed**：Invalid credential 不执行深层网络（通过 forward hook 验证）
2. **能力分级**：Valid 输出 fine-grained，Invalid 输出 coarse
3. **训练可行性**：训练使用软门控并返回两个完整 batch logits
4. **Mixed batch**：推理输出保留原 batch indices，不丢失样本对应关系

**规范来源**：本节是 Phase 1.3 的唯一规范方案。`docs/PHASE_1.3_GATED_RESNET.md` 仅作为索引，不复制设计正文。

---

## 核心架构

### 1. 网络结构

```
Input: CIFAR image [B,3,32,32], credential
    ↓
[Conv3x3 + BN + ReLU] → [B, 64, 32, 32]
    ↓
[Layer1: 2 blocks] → [B, 64, 32, 32]
    ↓
[Layer2: 2 blocks] → shallow_features [B, 128, 16, 16]
    ↓
[Gate Layer] ← credential [B, n]
    ↓
gated_features [B, 128, 16, 16], decision.allow [B]
    ↓
    ├─ allow=True ──→ [Layer3: 2 blocks] → [Nv, 256, 8, 8]
    │                      ↓
    │                  [Layer4: 2 blocks] → [Nv, 512, 4, 4]
    │                      ↓
    │                  [AvgPool + FC] → protected_logits [Nv, 10]
    │
    └─ allow=False ──→ [Public Head] → public_logits [Ni, 2]
```

### 2. 关键设计决策

**Gate 位置**：Layer2 之后
- CIFAR stem 不使用 ImageNet 的 7x7 stride=2 和 maxpool
- Layer1 保持 32x32，Layer2 完成第一次空间下采样到 16x16
- Layer3 + Layer4：深层特征提取（高级特征，受保护）
- 理由：layer2 后已有 128 通道表征，同时仍保留足够空间信息供 public head 使用

**两个输出头**：
- **Protected Head**：10-class（CIFAR-10 完整分类）
- **Public Head**：2-class（粗粒度分类，如 animal vs vehicle）

**训练模式**：
- Gate Layer 使用软 gate_signal，valid 子批的 gated_features 保持可微分
- `decision.allow` 仍由确定性 LWE 验证产生，不把授权边界作为可学习路由；软值只调节已授权 valid 特征的幅度
- 根据 `decision.allow` 选择 valid 子批进入深层，invalid 样本不得进入 layer3/layer4 或污染其 BatchNorm 统计量
- Protected logits 散射回完整 batch；invalid 行只是与计算图相连的零占位，必须由 Phase 2 使用 mask 排除
- Public head 对完整 batch 执行

**推理模式**：
- 硬路由：根据 decision.allow 决定执行哪条路径
- 使用 indices 向量化拆分 valid/invalid 子批，不逐样本执行
- Valid 子批只执行 protected path，Invalid 子批只执行 public path
- 空子批返回 shape 稳定的二维空 Tensor，不返回 `None`

---

## 接口定义

### GatedResNet18 类

```python
class GatedResNet18(nn.Module):
    """将 Gate Layer 集成到 CIFAR ResNet-18 并执行凭据条件路由。

    架构:
        - Shallow: conv1 + layer1 + layer2
        - Gate: LWE 验证 + 特征门控
        - Deep: layer3 + layer4
        - Protected Head: 10-class classifier
        - Public Head: 2-class classifier

    训练模式:
        - valid 子批使用软门控并执行 protected head
        - public head 对完整 batch 执行
        - 返回 TrainingOutput

    推理模式:
        - 硬路由，按 allow mask 拆分 batch
        - 返回稀疏 logits 及其原 batch indices
    """

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        params: LWEParams,
        num_classes_protected: int = 10,
        num_classes_public: int = 2,
        temperature: float = 5.0,
    ) -> None:
        """初始化 Gated ResNet-18

        参数:
            A, b: LWE 公钥
            params: LWE 参数
            num_classes_protected: Protected head 类别数（默认 10）
            num_classes_public: Public head 类别数（默认 2）
            temperature: Gate layer 温度参数

        约束:
            num_classes_protected 和 num_classes_public 必须是非 bool 的正整数
            A、b、params 和 temperature 复用 GateLayer 的严格校验
        """
        ...

    def forward(
        self,
        x: Tensor,
        credential: Union[Tensor, np.ndarray]
    ) -> Union[TrainingOutput, InferenceOutput]:
        """前向传播

        参数:
            x: [B, 3, 32, 32]，CIFAR 输入图像
            credential: [B, n] 或 [n]，LWE credential

        返回（训练模式）:
            protected_logits: [B, 10]
            public_logits: [B, 2]
            decision: AuthorizationDecision

        返回（推理模式）:
            InferenceOutput，其中 Nv + Ni = B，indices 保留原 batch 位置
        """
        ...
```

### 输入约束

**x (image)**：`Tensor[B, 3, H, W]`
- 类型：必须是 `torch.Tensor` 且 dtype 严格为 `torch.float32`
- 约束：本阶段严格要求四维、B≥1、C=3、H=W=32、全部有限
- Device：必须与模型参数和 Gate Layer buffer 相同
- B=0 在进入 stem 前以 `ValueError` 拒绝，避免训练态 BatchNorm 产生未定义或版本相关行为

**credential**：`Tensor[B, n]` 或 `[n]`
- 与 Gate Layer 相同的约束
- credential 的格式或 batch 错误由 Gate Layer 按请求级 fail-closed 处理：整批 deny，深层零调用；image 自身非法则直接拒绝请求，不执行任何 head

### 输出数据结构

```python
@dataclass(frozen=True)
class TrainingOutput:
    """训练输出：两个 head 都返回完整 batch logits。"""
    protected_logits: Tensor  # [B, num_classes_protected]
    public_logits: Tensor     # [B, num_classes_public]
    decision: AuthorizationDecision

@dataclass(frozen=True)
class InferenceOutput:
    """推理输出：稀疏 logits 通过 indices 映射回原 batch。"""
    protected_logits: Tensor   # [Nv, num_classes_protected]
    protected_indices: Tensor  # LongTensor[Nv]
    public_logits: Tensor      # [Ni, num_classes_public]
    public_indices: Tensor     # LongTensor[Ni]
    decision: AuthorizationDecision
```

**稳定契约**：
- `Nv + Ni == B`
- indices 递增且互不重叠，并完整覆盖 `[0, B)`
- 全 valid 时 `public_logits.shape == [0, num_classes_public]`
- 全 invalid 时 `protected_logits.shape == [0, num_classes_protected]`
- 输入 B=0 不产生输出，在任何网络层执行前抛出 `ValueError`

### 输出语义

**训练模式**：
- `protected_logits`: Tensor[B, num_classes_protected]
- `public_logits`: Tensor[B, num_classes_public]
- `decision`: AuthorizationDecision
- `protected_logits[~decision.allow]` 是零占位，不具有分类语义，禁止进入 protected loss 或指标

**推理模式**：
- 返回 `InferenceOutput`
- 调用方必须使用 indices 将预测映射回原输入顺序

---

## 实现细节

### 1. 浅层网络（共享）

使用 CIFAR ResNet-18 的前半部分：
```python
self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
self.bn1 = nn.BatchNorm2d(64)
self.relu = nn.ReLU(inplace=True)
self.layer1 = self._make_layer(64, 2, stride=1)
self.layer2 = self._make_layer(128, 2, stride=2)
```

本阶段在项目内实现标准 `BasicBlock`、downsample 和 `_make_layer`，不新增 torchvision 依赖。实现必须使用 ResNet-18 的 `[2,2,2,2]` block 配置、Kaiming 初始化和标准残差连接；测试与无 Gate 的同构 CIFAR ResNet baseline 比较参数结构和 direct protected logits。

### 2. Gate Layer 集成

```python
self.gate_layer = GateLayer(A, b, params, temperature)
```

### 3. 深层网络（受保护）

```python
self.layer3 = self._make_layer(256, 2, stride=2)
self.layer4 = self._make_layer(512, 2, stride=2)
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
self.protected_fc = nn.Linear(512, num_classes_protected)
```

### 4. 公开网络

```python
# 从 shallow_features [B, 128, 16, 16] 到 2-class
self.public_pool = nn.AdaptiveAvgPool2d((1, 1))
self.public_fc = nn.Linear(128, num_classes_public)
```

### 5. Forward 实现

```python
def forward(self, x, credential):
    self._validate_input(x)

    # 1. 浅层特征提取
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.layer1(x)
    shallow_features = self.layer2(x)  # [B, 128, 16, 16]

    # 2. Gate Layer
    gated_features, decision = self.gate_layer(shallow_features, credential)

    # 3. 条件路由
    valid_indices = torch.nonzero(decision.allow, as_tuple=False).flatten()
    invalid_indices = torch.nonzero(~decision.allow, as_tuple=False).flatten()

    if self.training:
        # 训练：仅 valid 子批进入带 BatchNorm 的深层，防止 invalid 样本污染统计量。
        # 先构造与浅层计算图相连的零占位，再散射 valid logits。
        protected_logits = shallow_features.sum(dim=(1, 2, 3)).unsqueeze(1)
        protected_logits = protected_logits.expand(
            -1, self.num_classes_protected
        ) * 0.0
        if valid_indices.numel() > 0:
            valid_features = gated_features.index_select(0, valid_indices)
            deep_features = self.layer4(self.layer3(valid_features))
            valid_logits = self.protected_fc(
                self.avgpool(deep_features).flatten(1)
            )
            protected_logits = protected_logits.index_copy(
                0, valid_indices, valid_logits
            )

        # Public head 对完整 batch 执行。
        public_pooled = self.public_pool(shallow_features).flatten(1)
        public_logits = self.public_fc(public_pooled)

        return TrainingOutput(protected_logits, public_logits, decision)

    # 推理：按 allow mask 向量化拆分 batch，并保留原 batch indices。
    if valid_indices.numel() > 0:
        valid_features = gated_features.index_select(0, valid_indices)
        deep_features = self.layer4(self.layer3(valid_features))
        protected_logits = self.protected_fc(
            self.avgpool(deep_features).flatten(1)
        )
    else:
        # 全 invalid 时绝不调用 layer3/layer4。
        protected_logits = shallow_features.new_empty(
            (0, self.num_classes_protected)
        )

    if invalid_indices.numel() > 0:
        invalid_features = shallow_features.index_select(0, invalid_indices)
        public_logits = self.public_fc(
            self.public_pool(invalid_features).flatten(1)
        )
    else:
        public_logits = shallow_features.new_empty(
            (0, self.num_classes_public)
        )

    return InferenceOutput(
        protected_logits=protected_logits,
        protected_indices=valid_indices,
        public_logits=public_logits,
        public_indices=invalid_indices,
        decision=decision,
    )
```

### 6. 训练职责边界

`GatedResNet18.forward()` 只负责产生 logits 和路由证据，不在模型内计算 CE、蒸馏或 gate regularization。Phase 2 的训练模块必须遵循：

- protected loss 只在 valid credential 样本上计算，禁止 invalid 的零/软衰减特征污染 protected CE
- invalid 样本不进入 layer3/layer4，因而也不能更新深层 BatchNorm running statistics
- public loss 使用 CIFAR-2 标签，可按实验设计在全部样本或指定 public 样本上计算
- Gate Layer 没有可训练参数，不设置无意义的 gate regularization
- 若一个训练 batch 没有 valid 样本，protected loss 返回与图相连的零标量；没有目标样本时不得对空 Tensor 调用 CrossEntropy
- 至少有一个 valid 样本时 protected head 执行；public head 始终对完整非空 batch 执行

---

## Fail-Closed 验证

### Forward Hook 机制

验证 invalid credential 不执行深层网络：

```python
class LayerCallCounter:
    """记录层的调用次数"""
    def __init__(self):
        self.count = 0

    def __call__(self, module, input, output):
        self.count += 1

# 使用
layer3_counter = LayerCallCounter()
layer4_counter = LayerCallCounter()

model.layer3.register_forward_hook(layer3_counter)
model.layer4.register_forward_hook(layer4_counter)

# 测试
model.eval()
with torch.no_grad():
    output = model(image, invalid_credential)

assert layer3_counter.count == 0, "Layer3 should not be called for invalid credential"
assert layer4_counter.count == 0, "Layer4 should not be called for invalid credential"
```

Forward hook / pre-hook 必须覆盖三种情况：
- 全 invalid：layer3/layer4 调用次数均为 0
- mixed batch：layer3/layer4 各调用 1 次，layer3 输入 batch size 严格等于 `Nv = decision.allow.sum()`
- 全 valid：layer3/layer4 各调用 1 次，layer3 输入 batch size 等于 B

仅检查“被调用过”不足以证明 invalid 样本未进入深层。训练态和推理态都要验证上述选择规则。

---

## 测试要求

**测试状态标记**：`[ ]` 未实现，`[x]` Codex 开发侧已实现并通过

### 架构测试
- [x] `test_model_architecture`: 验证网络结构正确
- [x] `test_cifar_stage_shapes`: 验证 32→32→16→8→4 的 stage shape
- [x] `test_output_shapes`: 验证训练和推理输出 dataclass 形状正确
- [x] `test_forward_pass_no_error`: 前向传播无错误
- [x] `test_constructor_validation`: 类别数、A/b、temperature 非法时拒绝
- [x] `test_input_validation`: image channel、shape、dtype、device、finite 检查

### 路由测试
- [x] `test_valid_credential_protected_path`: Valid → protected output
- [x] `test_invalid_credential_public_path`: Invalid → public output
- [x] `test_training_mode_selective_protected_path`: 训练模式仅 valid 子批进入深层，public head 处理完整 batch
- [x] `test_eval_mode_single_path`: 推理模式只执行一条路径
- [x] `test_mixed_batch_indices_preserve_order`: 稀疏 logits indices 映射正确
- [x] `test_all_valid_empty_public_output`: 全 valid 返回稳定空 public Tensor
- [x] `test_all_invalid_empty_protected_output`: 全 invalid 返回稳定空 protected Tensor
- [x] `test_empty_batch_rejected_before_layers`: B=0 在任何网络层调用前被拒绝

### Fail-Closed 验证
- [x] `test_invalid_credential_no_deep_layer_call`: Invalid → layer3/layer4 调用计数 = 0
- [x] `test_valid_credential_deep_layer_called`: Valid → layer3/layer4 被调用
- [x] `test_mixed_batch_selective_execution`: layer3 输入 batch size == valid 样本数
- [x] `test_all_valid_deep_batch_size`: layer3/layer4 各调用一次且输入 batch size == B
- [x] `test_malformed_credential_no_deep_layer_call`: credential 请求级错误 → 深层零调用
- [x] `test_training_invalid_samples_do_not_reach_deep_bn`: 训练态 mixed batch 的深层只接收 valid 子批

### 梯度测试
- [x] `test_backward_pass_training_mode`: 训练模式梯度传播正常
- [x] `test_gate_gradient_flow`: 梯度通过 gate 传递到浅层
- [x] `test_both_heads_receive_gradients_with_valid_sample`: batch 至少含一个 valid 样本时两个 head 均可接收梯度

### Baseline 等价性测试
- [x] `test_valid_logits_match_direct_protected_path`: eval valid 输出与绕过路由后的 direct protected logits 一致
- [x] `test_resnet18_block_configuration`: block 配置严格为 [2,2,2,2]

**移至 Phase 2/3 的实验指标**：
- Valid accuracy 接近训练后的 CIFAR ResNet-18 baseline
- Public head 的 CIFAR-2 accuracy
- latency、吞吐量和 GPU 内存

**目标覆盖率**：≥ 90%

**Codex 开发侧结果（2026-08-23）**：25/25 测试通过，模型模块行覆盖率 99%；完整 `tests/v2` 为 106/106 通过。CPU-only，CUDA/GPU 路径尚未实测。

---

## 实现步骤

**Step 1**：创建 `src/can/v2/models/gated_resnet.py`（约 300 行）
- 实现 `GatedResNet18` 类
- 浅层、深层、两个 head
- 训练/推理的条件路由逻辑

**Step 2**：创建 `tests/v2/test_gated_resnet.py`（约 250 行）
- 架构测试
- 路由测试
- Fail-closed 验证
- 梯度测试

**Step 3**：运行测试
```bash
python -m pytest tests/v2/test_gated_resnet.py -v --cov=src/can/v2/models --cov-config=.coveragerc --cov-report=term-missing
python -m pytest tests/v2/ -v
```

**Step 4**：更新文档
- `PROJECT_WORKLOG.md`：Phase 1.3 完成
- `docs/DESIGN_PROPOSALS.md`：添加 Phase 1.3 记录

---

## 风险和限制

### 1. 推理模式的动态 shape

**风险**：mixed batch 会产生随 Nv/Ni 变化的稀疏 logits shape，可能增加编译器 graph specialization，并要求调用方显式处理 indices。

**当前策略**：优先保证 fail-closed 和样本映射正确性；Phase 3 再基准测试 eager、`torch.compile` 和不同 valid ratio 下的性能。不得为了固定 shape 而让 invalid 样本进入深层。

### 2. Public Head 的训练

**问题**：Public head 需要学习粗粒度分类（2-class），但训练数据标签是 10-class

**解决方案**：
- CIFAR-10 → CIFAR-2 映射：
  - vehicle (0)：airplane, automobile, ship, truck，即原类别 `{0, 1, 8, 9}`
  - animal (1)：bird, cat, deer, dog, frog, horse，即原类别 `{2, 3, 4, 5, 6, 7}`

或者使用知识蒸馏（Phase 2）

### 3. 训练时的损失函数（Phase 2）

**约束**：Phase 1.3 不实现损失函数。Phase 2 使用显式 mask：
```python
valid_idx = torch.nonzero(decision.allow).flatten()
loss_protected = masked_cross_entropy(protected_logits, fine_labels, valid_idx)
loss_public = cross_entropy(public_logits, coarse_labels)
loss = alpha * loss_protected + beta * loss_public
```

不得加入 gate regularization：当前 Gate Layer 没有可训练参数，该项不能优化 Gate。

---

## 与设计文档的对照

### 符合 Phase 1 目标

- [x] 实现 Gate Layer 在计算图中间的架构
- [x] 验证密码验证逻辑可以嵌入 ResNet-18
- [x] Fail-closed：invalid credential 不执行深层

### Phase 2 预留

- 训练流程（知识蒸馏、损失函数）
- 数据集准备（CIFAR-10 → CIFAR-2 映射）
- 性能评估和优化

---

## 总结

**Phase 1.3 的核心目标**：
1. 集成 Gate Layer 到 ResNet-18
2. 实现条件路由（valid/invalid）
3. 验证 fail-closed（forward hook）
4. 确保训练可行性（梯度流动）

**不包括**：
- 完整的训练流程
- 性能优化
- 生产部署

Phase 1.3 已完成 Codex 开发侧实现与验证，待 Claude 按项目流程独立验收后进入 Phase 2。

---

## Phase 2: 训练流程（Revision 1）[DESIGN REVIEW]

**状态**：[REVISION-1]（修订中，待批准）
**提出时间**：2026-08-23
**修订时间**：2026-08-23
**依赖**：Phase 1.3 Gated ResNet-18（Claude 已验收）

### 修订目标与规范优先级

本节及 Phase 2.1-2.4 是 Phase 2 的唯一规范来源，并覆盖工作日志中的旧 TODO 摘要。Revision 1 必须解决：

1. Public head 的 coarse CE + knowledge distillation 完整契约
2. invalid credential 的可验证生成与确定性随机数
3. `TrainingOutput` mask 和 `InferenceOutput` indices 的严格消费方式
4. 分阶段训练、可复现 checkpoint 和多种子实验指标

### 范围与依赖

- Phase 2 原型只使用 CIFAR-10，输出 10-class protected 与 2-class public；CIFAR-100 留到后续扩展
- 运行时依赖：PyTorch 2.0+、torchvision（与 PyTorch 版本匹配）、PyYAML、tqdm
- 当前环境尚未安装 torchvision；实现前必须显式安装并记录版本，禁止静默 fallback
- 单元测试必须使用注入的 fake dataset / tensor dataset，禁止联网下载 CIFAR-10
- 数据下载是训练脚本的显式操作，默认 `download=False`；缺失数据时给出明确错误

### 三阶段训练策略

1. **Stage A - Protected baseline**：使用全 valid credential 训练 shallow + deep + protected head，public head 不参与 loss
2. **Stage B - Public distillation**：加载 Stage A 最优 checkpoint 作为冻结 teacher；student 顶层保持 training 输出协议，但冻结并逐个 `eval()` shallow/deep/protected 模块，仅 public head 为 `train()`，credential 全 invalid
3. **Stage C - Joint fine-tuning**：解冻 student，使用较小学习率联合优化 masked protected loss 与 public CE + KD；teacher 始终 `eval()`、`requires_grad_(False)`

每个阶段具有独立 epoch、学习率、冻结策略和 best-checkpoint 指标。禁止在没有 Stage A teacher checkpoint 的情况下声称执行 knowledge distillation。

### 默认 Epoch 与 Early Stopping

默认训练预算是初始配置，不是性能保证：

| 阶段 | 默认 epoch 上限 | Early stopping 监控项 | 默认 patience | 阶段初始化 |
|---|---:|---|---:|---|
| Stage A | 20 | protected validation accuracy（最大化） | 5 | 新建 GatedResNet-18 |
| Stage B | 60 | public validation balanced accuracy（最大化） | 10 | Stage A best student 权重 |
| Stage C | 20 | 满足 protected degradation 约束后的 public balanced accuracy | 5 | Stage B best student 权重 |

- `20/60/20` 仅作为 CIFAR-10 原型的默认上限，允许通过严格配置覆盖
- 每阶段至少完成 1 个 epoch；early stopping 只读取 validation set，不读取 test set
- patience 以“完整验证周期无改进次数”计数；改进方向、`min_delta` 和 tie-break 规则必须写入配置
- 先执行每阶段 1 epoch smoke benchmark，记录时间、显存和指标是否有限，再决定是否运行完整预算
- Stage A 必须使用全 valid credential；设置全 invalid 会导致 protected path 零调用和零 protected loss，属于配置错误

### Teacher 生命周期与管理

Teacher 不是 torchvision 预训练的标准 ResNet-18，也不是外部模型。它固定来源于本实验 **Stage A protected baseline 的 best checkpoint**，使用与 student 相同的 GatedResNet-18 架构、CIFAR 预处理、类别顺序和 LWE 公共参数。

1. **生成**：Stage A 每次验证后按 protected validation accuracy 更新 `stage_a/best.ckpt`；Stage A 结束后将该文件晋升为唯一 teacher source，并生成包含 SHA-256、schema version、architecture id、配置 hash、数据 split hash 和 LWE public-parameter hash 的 teacher manifest
2. **Stage B 初始化**：student 从 `stage_a/best.ckpt` 初始化；teacher 创建为独立模型实例，加载同一 checkpoint，随后执行 `eval()` 和 `requires_grad_(False)`。仅 student public head 可训练
3. **Stage C 初始化**：student 从 `stage_b/best.ckpt` 初始化；teacher 仍固定为原始 `stage_a/best.ckpt`，禁止替换为 Stage B student 或当前 Stage C student
4. **推理适配**：teacher 对同一批预处理图像使用 valid credential 做全 valid 推理；只有在 `protected_indices == torch.arange(B)` 时才返回 `[B, 10]` teacher logits
5. **恢复训练**：Stage B/C checkpoint 只保存 teacher 的规范相对路径、SHA-256 和 manifest identity，不复制一份隐式 teacher。恢复时必须从受信任实验目录加载并逐项验证；文件缺失、hash 不一致、架构/类别映射/LWE 公共参数不一致时 fail fast
6. **生命周期结束**：在 Stage C、最终验证和结果审计完成前不得删除或覆盖 Stage A teacher checkpoint；实验 manifest 永久记录 teacher identity。普通 checkpoint 不保存 secret，valid credential 仍由受限的独立输入提供

不支持“找不到 teacher 时自动下载 torchvision 权重”“自动使用最新 checkpoint”或“退化为无 KD 继续训练”。若要比较外部标准 ResNet-18 teacher，必须作为独立消融实验另行设计和标记。

### 全局训练可视化要求

**强制要求**：所有训练相关代码必须遵循统一的进度条设计规范。

#### 进度条规范

**要求**：
1. **唯一性**：训练过程中有且仅有**一个**进度条显示
2. **信息完整**：进度条必须显示：
   - 当前 epoch/总 epoch
   - 当前 batch/总 batch
   - 训练进度百分比
   - 关键指标（loss、accuracy）
   - ETA（预计剩余时间）
3. **库选择**：使用 `tqdm` 库实现
4. **禁止**：不得在训练循环中使用 `print()` 打印进度信息

#### 实现示例

```python
from tqdm import tqdm

def train_epoch(self, epoch: int, total_epochs: int):
    """训练一个 epoch"""
    self.model.train()
    
    # 创建唯一的进度条
    pbar = tqdm(
        self.train_loader,
        desc=f"Epoch {epoch}/{total_epochs}",
        unit="batch",
        leave=True,
        ncols=100,
    )
    
    epoch_metrics = TrainingMetricAccumulator()
    
    for batch_idx, (images, fine_labels, coarse_labels) in enumerate(pbar):
        # 训练细节由可测试的单 batch 方法负责。
        batch_metrics = self._train_batch(images, fine_labels, coarse_labels)
        epoch_metrics.update(batch_metrics, images.shape[0])
        
        # 更新进度条显示（不创建新的进度条）
        pbar.set_postfix({
            'loss': f'{batch_metrics.loss:.4f}',
            'acc_p': f'{epoch_metrics.protected_accuracy:.2f}%',
            'acc_c': f'{epoch_metrics.public_accuracy:.2f}%',
        })
    
    pbar.close()
    return epoch_metrics.compute()

def train(self, num_epochs: int):
    """完整训练流程"""
    for epoch in range(1, num_epochs + 1):
        train_metrics = self.train_epoch(epoch, num_epochs)
        val_metrics = self.validate()
        
        # 使用 tqdm.write 代替 print（不破坏进度条）
        tqdm.write(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
            f"protected_acc={val_metrics['protected_accuracy']:.2f}%"
        )
```

#### 禁止的做法

```python
# ❌ 错误：多个进度条
for epoch in tqdm(range(num_epochs)):
    for batch in tqdm(train_loader):  # 嵌套进度条，违反规范
        ...

# ❌ 错误：使用 print 破坏进度条
for batch in tqdm(train_loader):
    print(f"Loss: {loss}")  # 会破坏进度条显示

# ❌ 错误：没有进度条
for batch in train_loader:
    ...  # 没有任何进度显示
```

#### 正确的做法

```python
# ✅ 正确：单一进度条 + tqdm.write
for epoch in range(1, num_epochs + 1):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        # 训练逻辑
        pbar.set_postfix({'loss': loss.item()})
    
    # 使用 tqdm.write 输出 epoch 总结
    tqdm.write(f"Epoch {epoch} completed: loss={epoch_loss:.4f}")
```

#### 验收标准

- [ ] 训练脚本运行时终端只显示一个进度条
- [ ] 进度条包含 epoch、batch、百分比信息
- [ ] 进度条实时显示 loss 和 accuracy
- [ ] 没有使用 `print()` 破坏进度条显示
- [ ] Epoch 总结使用 `tqdm.write()` 输出

**注意**：此规范适用于 Phase 2 及之后所有涉及训练的代码。

---

## Phase 2.1: 数据与 Credential 准备 [IMPLEMENTED - CODE COMPLETE]

**状态**：[REVISION-1]
**提出时间**：2026-08-23
**依赖**：Phase 1.3 Gated ResNet-18（Claude 已验收）、兼容版本 torchvision

### 设计目标

1. CIFAR-10 → CIFAR-2 映射
2. 提供 fine-grained 和 coarse 标签
3. Credential 生成策略
4. 标准的 PyTorch Dataset/DataLoader 接口

### CIFAR-2 类别映射

```python
CIFAR10_CLASSES = [
    'airplane',   # 0
    'automobile', # 1
    'bird',       # 2
    'cat',        # 3
    'deer',       # 4
    'dog',        # 5
    'frog',       # 6
    'horse',      # 7
    'ship',       # 8
    'truck',      # 9
]

CIFAR2_MAPPING = {
    'vehicle': [0, 1, 8, 9],  # airplane, automobile, ship, truck
    'animal': [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
}

def fine_to_coarse(fine_label: int) -> int:
    """将 CIFAR-10 标签映射到 CIFAR-2"""
    if isinstance(fine_label, bool) or not isinstance(fine_label, int):
        raise TypeError("fine_label 必须是非 bool 整数")
    if not 0 <= fine_label < 10:
        raise ValueError("fine_label 必须位于 [0, 10)")
    return 0 if fine_label in {0, 1, 8, 9} else 1
```

### Dataset 实现

```python
class CIFAR10WithCoarse(Dataset):
    """CIFAR-10 dataset with both fine and coarse labels

    返回:
        image: Tensor[3, 32, 32]
        fine_label: int (0-9)
        coarse_label: int (0-1)
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, int]:
        image, fine_label = self.cifar10[idx]
        coarse_label = fine_to_coarse(fine_label)
        return image, fine_label, coarse_label
```

### 数据划分与 DataLoader 复现契约

- CIFAR-10 官方 train set 使用固定 seed 分为 train/validation；官方 test set 只用于最终报告
- split indices 必须保存到实验目录并写入 checkpoint 元数据，不得每次运行重新随机划分
- DataLoader 使用显式 `torch.Generator`，并通过 `worker_init_fn` 同步 Python、NumPy 和 PyTorch worker seed
- 记录 batch size、shuffle、drop_last、num_workers、数据增强和 torchvision 版本
- Stage A/C 的 valid 子批可能只有 1 个样本；深层 BatchNorm 在训练态对此不稳定。训练采样器必须保证每个 batch 至少 2 个 valid 样本，或显式采用冻结深层 BatchNorm 统计量的配置；默认采用前者并 fail fast

### Credential 生成器

```python
@dataclass(frozen=True)
class CredentialBatch:
    """保存 credential 值和仅用于实验审计的预期有效性。"""
    values: np.ndarray         # float32 [B, n]
    expected_valid: np.ndarray # bool [B]

class CredentialGenerator:
    """为训练生成 valid/invalid credential

    策略:
        - Valid: 使用真实的 secret
        - Invalid: 使用 rejection sampling，且必须由 V_ref 确认拒绝
    """

    def __init__(
        self,
        A: np.ndarray,
        secret: np.ndarray,
        b: np.ndarray,
        params: LWEParams,
        seed: int,
        max_attempts: int = 100,
    ):
        self.A = A
        self.secret = secret  # Valid credential
        self.b = b
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.max_attempts = max_attempts

    def generate(self, is_valid: bool) -> np.ndarray:
        """生成单个 credential"""
        if is_valid:
            return self.secret.copy()
        for _ in range(self.max_attempts):
            candidate = self.rng.normal(
                0.0, 1.0, size=self.params.n
            ).astype(np.float32)
            if V_ref({"vector": candidate}, self.A, self.b, self.params) == 0:
                return candidate
        raise RuntimeError("在 max_attempts 内无法生成已验证的 invalid credential")

    def batch_generate(
        self,
        batch_size: int,
        valid_ratio: float = 0.5,
    ) -> np.ndarray:
        """生成一个 batch 的 credentials

        参数:
            batch_size: batch 大小
            valid_ratio: valid credential 的比例（默认 0.5）

        返回:
            credentials: [B, n]
        """
        # batch_size、valid_ratio 必须严格校验；训练默认保证至少 2 个 valid。
        num_valid = int(round(batch_size * valid_ratio))
        num_invalid = batch_size - num_valid

        credentials = []
        for _ in range(num_valid):
            credentials.append(self.generate(is_valid=True))
        for _ in range(num_invalid):
            credentials.append(self.generate(is_valid=False))

        credentials_array = np.stack(credentials, axis=0)
        permutation = self.rng.permutation(batch_size)
        expected_valid = np.concatenate([
            np.ones(num_valid, dtype=np.bool_),
            np.zeros(num_invalid, dtype=np.bool_),
        ])[permutation]
        return CredentialBatch(
            values=credentials_array[permutation],
            expected_valid=expected_valid,
        )
```

`CredentialBatch.expected_valid` 只用于测试采样器和记录实验构成，不能传给模型、不能创建 `AuthorizationDecision`，也不能替代 `output.decision.allow`。每个生成 batch 必须断言参考验证结果与 `expected_valid` 完全一致。

### 数据增强

```python
def get_cifar_transforms(train: bool = True):
    """CIFAR-10 标准数据增强"""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])
```

### 交付物

**文件**：`src/can/v2/training/data.py`

**依赖清单**：新增项目可审查的依赖声明，固定与当前 PyTorch 兼容的 torchvision、PyYAML 和 tqdm 版本；不得只在个人环境中手工安装而不记录。

**导出**：
- `CIFAR10WithCoarse`: Dataset 类
- `CredentialGenerator`: Credential 生成器
- `fine_to_coarse()`: 标签映射函数
- `get_cifar_transforms()`: 数据增强
- `CIFAR2_MAPPING`: 类别映射字典

**测试**：`tests/v2/test_training.py`（当前实现将 Phase 2.1-2.3 离线测试合并于此文件）
- [ ] `test_cifar2_mapping`: 标签映射正确
- [ ] `test_cifar2_mapping_rejects_invalid_input`: 非法类型和范围拒绝
- [ ] `test_dataset_length`: 数据集长度正确
- [ ] `test_dataset_returns_three_items`: 返回 (image, fine, coarse)
- [ ] `test_credential_generator_valid`: Valid credential 生成
- [ ] `test_credential_generator_invalid`: 所有 invalid 均经 V_ref 确认拒绝
- [ ] `test_credential_generator_retry_exhaustion`: 达到 max_attempts 后稳定失败
- [ ] `test_batch_generate_ratio`: expected 与实际验证比例一致
- [ ] `test_batch_generate_deterministic`: 相同 seed 产生相同序列
- [ ] `test_dataset_tests_do_not_download`: 单元测试不触发网络访问
- [ ] `test_dataloader_worker_reproducibility`: 多 worker 固定种子可复现

---

## Phase 2.2: 监督损失与知识蒸馏 [IMPLEMENTED - CODE COMPLETE]

**状态**：[REVISION-1]
**提出时间**：2026-08-23
**依赖**：Phase 2.1 数据集准备

### 设计目标

1. Protected loss 只在 valid 样本上计算
2. Public loss 在所有样本上计算 coarse CE + teacher KD
3. 处理全 invalid batch 的边界情况
4. 严格验证 labels、logits、mask 和 loss 权重
5. teacher 冻结且只提供证据，不参与反向传播

### 损失函数实现

```python
@dataclass(frozen=True)
class LossOutput:
    """保存可审计的训练损失分项。"""
    total: Tensor
    protected: Tensor
    public_ce: Tensor
    public_kd: Tensor

def compute_training_loss(
    output: TrainingOutput,
    fine_labels: Tensor,
    coarse_labels: Tensor,
    teacher_fine_logits: Optional[Tensor],
    alpha: float = 1.0,
    beta_ce: float = 0.1,
    beta_kd: float = 0.9,
    temperature: float = 4.0,
) -> LossOutput:
    """计算训练损失

    参数:
        output: GatedResNet18 的 TrainingOutput
        fine_labels: [B] CIFAR-10 标签 (0-9)
        coarse_labels: [B] CIFAR-2 标签 (0-1)
        alpha: protected loss 权重（默认 1.0）
        teacher_fine_logits: 冻结 teacher 的 [B, 10] logits；Stage A 为 None
        alpha/beta_ce/beta_kd: 三项有限非负权重
        temperature: KD 的有限正温度

    返回:
        结构化 LossOutput，分别携带 protected、public_ce、public_kd 和 total

    约束:
        - Protected loss 只在 output.decision.allow == True 的样本上计算
        - 全 invalid batch 时 loss_protected = 0，不影响梯度
        - Public CE 和 KD 在完整 batch 上计算
        - Stage B/C 必须提供 teacher_fine_logits；Stage A 显式设置 beta_ce=beta_kd=0
    """
    validate_loss_inputs(...)
    valid_mask = output.decision.allow

    # Protected loss: 只对 valid 样本计算
    if bool(valid_mask.any().item()):
        valid_protected_logits = output.protected_logits[valid_mask]
        valid_fine_labels = fine_labels[valid_mask]
        loss_protected = F.cross_entropy(
            valid_protected_logits,
            valid_fine_labels,
            reduction='mean'
        )
    else:
        # 从模型输出构造与 shallow graph 相连的零值。
        loss_protected = output.protected_logits.sum() * 0.0

    # Public loss: 完整 batch
    loss_public_ce = F.cross_entropy(
        output.public_logits,
        coarse_labels,
        reduction='mean'
    )

    if teacher_fine_logits is None:
        loss_public_kd = output.public_logits.sum() * 0.0
    else:
        # 将 teacher 的 10-class logits 聚合为 vehicle/animal 两类 logits。
        teacher_coarse_logits = torch.stack([
            torch.logsumexp(teacher_fine_logits[:, [0, 1, 8, 9]], dim=1),
            torch.logsumexp(teacher_fine_logits[:, [2, 3, 4, 5, 6, 7]], dim=1),
        ], dim=1)
        loss_public_kd = F.kl_div(
            F.log_softmax(output.public_logits / temperature, dim=1),
            F.softmax(teacher_coarse_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)

    total_loss = (
        alpha * loss_protected
        + beta_ce * loss_public_ce
        + beta_kd * loss_public_kd
    )
    return LossOutput(total_loss, loss_protected, loss_public_ce, loss_public_kd)
```

Teacher logits 必须在 `torch.inference_mode()` 中由冻结的 Stage A 最优 checkpoint 产生，并与 student 使用同一批预处理后的图像。禁止从 student 的 public logits 构造 teacher 目标。`teacher_coarse_logits` 的类别顺序固定为 `[vehicle, animal]`。

Teacher adapter 使用冻结的 Gated ResNet Stage A 副本和 valid credential 做全 valid 推理，要求 `protected_indices == arange(B)` 后返回 protected logits；任何缺失或乱序立即失败。Teacher checkpoint、LWE 公共参数和 valid credential 必须与 student 实验配置匹配。

输入校验必须覆盖：`TrainingOutput` 类型、三个 batch 大小、labels 为同 device 的 `torch.long` 一维 Tensor、fine/coarse 范围、所有 logits 有限、`decision.allow` 为同 device 的 BoolTensor，以及权重/temperature 的类型和有限性。未知或不一致输入直接抛出稳定异常。

### 损失权重选择

**默认配置**：
- Stage A：`alpha=1.0, beta_ce=0.0, beta_kd=0.0`
- Stage B：`alpha=0.0, beta_ce=0.5, beta_kd=0.5, T=4.0`
- Stage C 初始建议：`alpha=1.0, beta_ce=0.25, beta_kd=0.25, T=4.0`

权重属于实验超参数，不把上述初始值写成性能结论。Stage C 必须监控 protected baseline degradation，并通过 validation 指标选择 checkpoint。

### 交付物

**文件**：`src/can/v2/training/loss.py`

**导出**：
- `compute_training_loss()`: 损失计算函数

**测试**：`tests/v2/test_training.py`（当前实现将 Phase 2.1-2.3 离线测试合并于此文件）
- [ ] `test_loss_all_valid`: 全 valid batch
- [ ] `test_loss_all_invalid`: 全 invalid batch（loss_protected = 0）
- [ ] `test_loss_mixed_batch`: Mixed batch
- [ ] `test_loss_weights`: alpha/beta_ce/beta_kd 权重影响
- [ ] `test_teacher_logits_aggregate_to_cifar2`: teacher 类别聚合顺序正确
- [ ] `test_kd_temperature_scaling`: KD 使用 T² 缩放
- [ ] `test_loss_rejects_malformed_inputs`: dtype、shape、device、范围和非有限值拒绝
- [ ] `test_loss_gradient_flow`: 梯度流动正确
- [ ] `test_total_loss_requires_grad`: 存在加权可训练项时 total loss 可反传；零权重分项不强制要求独立梯度

---

## Phase 2.3: 训练器实现 [IMPLEMENTED - CODE COMPLETE]

**状态**：[REVISION-1]
**提出时间**：2026-08-23
**依赖**：Phase 2.1 数据集准备，Phase 2.2 损失函数

### 设计目标

1. 封装完整训练循环
2. 遵循进度条规范（唯一进度条）
3. 支持验证和指标记录
4. 检查点保存/加载
5. 显式执行 Stage A/B/C 冻结策略和 teacher 生命周期
6. 保证 CPU/GPU、恢复训练和多 worker 数据加载可复现

### Trainer 类核心方法

```python
class GatedResNetTrainer:
    """执行分阶段、可复现的 Gated ResNet-18 训练。"""

    def train_epoch(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """训练一个 epoch（遵循进度条规范）

        必须使用唯一的 tqdm 进度条，显示:
        - Epoch {epoch}/{total_epochs}
        - 实时 loss、accuracy
        - ETA

        返回:
            样本加权的 loss、protected accuracy 和 public accuracy
        """
        self._configure_stage_mode()
        accumulator = TrainingMetricAccumulator()

        # 唯一的进度条
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{total_epochs}",
            unit="batch",
            leave=True,
            ncols=120,
        )

        for images, fine_labels, coarse_labels in pbar:
            # 生成 credentials
            credential_batch = self.credential_gen.batch_generate(
                batch_size=images.size(0),
                valid_ratio=self.valid_ratio,
            )
            images = images.to(self.device, non_blocking=True)
            fine_labels = fine_labels.to(self.device, non_blocking=True)
            coarse_labels = coarse_labels.to(self.device, non_blocking=True)

            # 前向传播
            output = self.model(images, credential_batch.values)
            teacher_logits = self._teacher_logits(images)

            # 计算损失
            losses = compute_training_loss(
                output,
                fine_labels,
                coarse_labels,
                teacher_logits,
                **self.stage_loss_weights,
            )

            # 反向传播
            self.optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(losses.total):
                raise FloatingPointError("训练 loss 出现 NaN 或 Inf")
            losses.total.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_parameters, self.max_grad_norm
                )
            self.optimizer.step()

            accumulator.update_training(output, fine_labels, coarse_labels, losses)

            # 更新进度条显示（不创建新进度条）
            pbar.set_postfix({
                'loss': f'{losses.total.item():.4f}',
                'acc_p': f'{accumulator.protected_accuracy:.2f}%',
                'acc_c': f'{accumulator.public_accuracy:.2f}%',
            })

        pbar.close()
        return accumulator.compute()

    def validate(self) -> Dict[str, float]:
        """验证模型性能

        返回:
            {'protected_accuracy': float, 'public_accuracy': float,
             'public_balanced_accuracy': float, 'public_macro_f1': float}
        """
        self.model.eval()
        metrics = EvaluationMetricAccumulator()

        with torch.inference_mode():
            for images, fine_labels, coarse_labels in self.val_loader:
                images = images.to(self.device)
                fine_labels = fine_labels.to(self.device)
                coarse_labels = coarse_labels.to(self.device)

                # 同一批图像先做全 valid 推理，评估完整 protected 样本集。
                valid_batch = self.validation_credentials.all_valid(images.shape[0])
                output = self.model(images, valid_batch.values)
                protected_targets = fine_labels.index_select(
                    0, output.protected_indices
                )
                metrics.update_protected(
                    output.protected_logits, protected_targets
                )

                # 再做全 invalid 推理，评估相同图像的完整 public 样本集。
                invalid_batch = self.validation_credentials.all_invalid(
                    images.shape[0]
                )
                output = self.model(images, invalid_batch.values)
                public_targets = coarse_labels.index_select(
                    0, output.public_indices
                )
                metrics.update_public(output.public_logits, public_targets)

        return metrics.compute()

    def train(self, num_epochs: int):
        """完整训练流程

        每个 epoch:
        1. 训练
        2. 验证
        3. 学习率调度
        4. 使用 tqdm.write 输出总结（不破坏进度条）
        """
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch, num_epochs)
            val_metrics = self.validate()

            self._step_scheduler(val_metrics)
            self._save_last_checkpoint(epoch, train_metrics, val_metrics)
            self._save_best_checkpoint_if_improved(epoch, val_metrics)

            # 使用 tqdm.write 输出（不使用 print）
            tqdm.write(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Acc: protected={val_metrics['protected_accuracy']}, "
                f"public_balanced={val_metrics['public_balanced_accuracy']}"
            )
```

### 模式、指标与 scheduler 契约

- Stage A：student 整体 `train()`，credential 全 valid；只优化 protected 参数路径
- Stage B：student 顶层必须 `train()` 以返回 `TrainingOutput`，随后把 conv/bn/layer1-4/protected head/Gate 逐个设为 `eval()`；仅 public head 为 `train()`，除 public head 外全部 `requires_grad=False`，防止 BatchNorm 漂移和深层执行
- Stage C：student 整体 `train()`；teacher 始终 `eval()` 且冻结
- 训练 protected accuracy 的分子/分母只统计 `decision.allow=True` 样本；public accuracy 统计完整训练 batch
- 验证必须处理空 protected/public 子批，空子批不更新对应分母；epoch 结束分母为零时返回结构化 `None`，不得报告 0% 冒充观测值
- accuracy、balanced accuracy、macro-F1 必须按样本累计混淆矩阵后计算，不能平均 batch accuracy
- 主验证对每个图像分别执行全 valid 和全 invalid 两次推理，使 protected/public 指标基于同一完整 validation set；固定 mixed pass 仅作为 indices 与 fail-closed 诊断，不参与主准确率
- 普通 scheduler 每 epoch step；`ReduceLROnPlateau` 使用明确的 validation metric step，不允许一个 epoch 调用两次

Best checkpoint 规则必须在配置中固定：Stage A 以 protected validation accuracy 最大为准；Stage B 以 public balanced accuracy 最大为准；Stage C 先要求 protected 相对 Stage A baseline 的下降不超过配置上限，再在满足约束的 checkpoint 中选择 public balanced accuracy 最大者。禁止查看 test set 后改变规则。

### Checkpoint 与恢复契约

每个 checkpoint 至少保存：

- schema version、stage、epoch、global step、best metrics
- student model、teacher identity/checkpoint hash、optimizer、scheduler state
- 完整解析后的配置、LWEParams、A/b；secret 不写入普通 checkpoint，由用户明确指定的受限 credential 文件提供
- train/validation split indices 及其 hash、CIFAR mapping version、依赖版本
- Python `random`、NumPy Generator、PyTorch CPU/CUDA 和 DataLoader generator RNG state

保存使用“同目录临时文件 + 原子替换”；加载支持显式 `map_location`，验证 schema/config/LWE shape 后才恢复。默认只加载本项目自己生成且受信任的 checkpoint；不得对不可信 pickle checkpoint 调用 `torch.load()`。恢复测试必须证明下一批 credential、数据顺序和一次 optimizer step 与未中断运行一致。

### 交付物

**文件**：`src/can/v2/training/trainer.py`

**导出**：
- `GatedResNetTrainer`: 训练器类

**测试**：`tests/v2/test_training.py`（当前实现将 Phase 2.1-2.3 离线测试合并于此文件）
- [ ] `test_trainer_initialization`: 初始化正确
- [ ] `test_train_epoch_returns_metrics`: 返回指标
- [ ] `test_stage_freeze_contract`: A/B/C 参数与 BatchNorm 模式正确
- [ ] `test_stage_initialization_chain`: Stage B 从 A best、Stage C 从 B best 初始化
- [ ] `test_teacher_is_frozen_stage_a_best`: teacher 固定为独立的 Stage A best 副本
- [ ] `test_teacher_manifest_validation`: teacher 路径、hash、架构、mapping 和 LWE 公共参数均校验
- [ ] `test_teacher_missing_or_mismatch_fails`: teacher 缺失或 identity 不一致时停止训练
- [ ] `test_teacher_not_replaced_on_resume`: Stage B/C 恢复后 teacher identity 保持不变
- [ ] `test_validate_sparse_indices_alignment`: 稀疏 logits 与原标签严格对齐
- [ ] `test_validate_empty_route`: 空路由不污染指标分母
- [ ] `test_metrics_use_sample_counts`: 指标按样本而非 batch 平均
- [ ] `test_checkpoint_save_load`: 状态、配置和 map_location 正确
- [ ] `test_resume_is_deterministic`: 恢复后的数据、credential 和 optimizer step 一致
- [ ] `test_non_finite_loss_aborts`: NaN/Inf loss 立即停止
- [ ] `test_progress_bar_compliance`: 遵循进度条规范

---

## Phase 2.4: 配置与训练脚本 [IMPLEMENTED - CODE COMPLETE]

**状态**：[REVISION-1]
**提出时间**：2026-08-23
**依赖**：Phase 2.1-2.3

### 设计目标

1. 命令行训练入口
2. 超参数配置
3. 自动保存检查点
4. 严格配置校验、确定性恢复和环境记录

### 脚本实现

**文件**：`scripts/train_gated_resnet.py`

**功能**：
- 命令行参数解析
- 数据加载
- 模型初始化
- 训练循环
- 结果保存
- 三阶段配置和 `--resume` 恢复
- 显式 `--download-data`，默认不下载
- `--device auto|cpu|cuda`；显式请求 CUDA 但不可用时 fail fast，不静默切换 CPU

### 配置文件

**文件**：`configs/v2/train_gated_resnet18_cifar10.yaml`

配置必须至少包含：schema version、全局 seed、数据路径/split seed/worker 数、LWE 参数引用、Stage A/B/C epochs 与 optimizer/scheduler/early stopping、loss 权重/KD temperature、valid ratio、gradient clipping、验证频率、checkpoint 目录和设备。默认 `stage_a.epochs=20`、`stage_b.epochs=60`、`stage_c.epochs=20`，默认 patience 分别为 `5/10/5`。解析后拒绝未知字段、重复 YAML key、错误类型、非有限数值和越界比例；命令行只覆盖显式列出的字段，并把最终解析配置保存到实验目录。

**使用示例**：
```bash
# 基本训练
python scripts/train_gated_resnet.py

# 自定义参数
python scripts/train_gated_resnet.py \
    --stage-a-epochs 20 \
    --stage-b-epochs 60 \
    --stage-c-epochs 20 \
    --batch-size 128 \
    --lr 0.1 \
    --alpha 1.0 \
    --beta-ce 0.25 \
    --beta-kd 0.25 \
    --kd-temperature 4.0 \
    --device cuda
```

### 交付物

**文件**：`scripts/train_gated_resnet.py`

**验收标准**：
- [ ] 脚本可执行
- [ ] 参数可配置
- [ ] 未知/重复/非法配置 fail fast
- [ ] 默认 Stage A/B/C epoch 上限和 patience 为 20/60/20、5/10/5
- [ ] Stage B/C checkpoint 固定记录并校验 Stage A teacher identity
- [ ] 单元测试和默认启动不访问网络
- [ ] 遵循进度条规范
- [ ] 原子保存 last/best checkpoint 并支持确定性恢复
- [ ] 输出训练总结

---

## Phase 2 总结

**核心目标**：训练 Gated ResNet-18，实现能力分级

**交付物**：
1. Phase 2.1: 数据集准备（CIFAR-10 + CIFAR-2）
2. Phase 2.2: 损失函数（protected + public）
3. Phase 2.3: 训练器（遵循进度条规范）
4. Phase 2.4: 训练脚本

**功能验收**：
- all-invalid 验证中 protected 路径零调用，public indices 完整覆盖 batch
- mixed 验证中 logits、indices、labels 映射严格一致
- Stage A/B/C 冻结策略、KD teacher 和恢复训练契约均有确定性测试
- 新增训练模块行覆盖率目标 ≥ 90%，完整 `tests/v2` 通过

**实验验收**：
- 至少运行 3 个预先声明的随机种子，报告均值、标准差和每次原始结果
- Protected：报告 top-1 accuracy，并报告相对同构无 Gate CIFAR ResNet-18 baseline 的绝对下降；初始目标为下降不超过 3 个百分点
- Public：同时报告 accuracy、balanced accuracy、macro-F1 和 2×2 confusion matrix，避免 CIFAR-2 的 40/60 类别不平衡误导
- 报告 Stage A/B/C 学习曲线、valid ratio 敏感性和最终选择 checkpoint 的规则
- 官方 test set 只在配置和 checkpoint 选择冻结后评估一次，不用于调参

上述阈值是原型实验目标而非保证；未达到时如实记录，不得通过更换 test seed 或反复查看 test set 选择结果。

**资源预估**：实现和 CPU 单元测试不要求 GPU；正式 CIFAR-10 三阶段训练建议 1 张 ≥8 GB CUDA GPU。单 seed 的时间需要先用 1 epoch smoke benchmark 实测后写入 `PROJECT_WORKLOG.md`，再决定完整 epochs；不得沿用未经当前硬件测量的时间估计。

**明确限制**：Phase 2 仍使用 toy LWE、静态可重用 credential，不解决 replay 或白盒攻击，不构成生产安全系统。所有训练代码必须遵循进度条规范（任一时刻唯一进度条 + `tqdm.write`）。

---
