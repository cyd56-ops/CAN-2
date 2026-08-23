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

## Phase 1.2: Neural Gate Layer [REVISION-3]

**状态**：[REVISION-3]（修订中，待批准）  
**提出时间**：2026-08-23  
**修订时间**：2026-08-23（基于 Codex 第三轮审阅，添加 One-Time Credential）  
**依赖**：Phase 1.1 LWE 密码原语（已完成）

**修订原因**（基于 Codex 第三轮审阅的 4 个 P1 问题）：
1. ❌ Replay 是阻塞问题却计划推迟（逻辑矛盾）
2. ❌ 验证器直接产生"硬授权判定"（违反授权边界原则）
3. ❌ 未满足"融合浅层特征与 credential"（与项目目标不一致）
4. ❌ 异常处理不是结构化 fail-closed

**用户决策**：采用 **One-Time Credential** 机制防御 replay

**新的架构**：
```
credential → LWE验证 + Replay检查 → VerificationEvidence
VerificationEvidence → 协调器 → AuthorizationDecision
shallow_features + AuthorizationDecision → 特征门控 → gated_features
```

### 设计目标

实现具有 One-Time Credential 防御机制的 Gate Layer，遵循授权边界原则：

1. **验证证据生成**：credential → LWE 验证 + Replay 检查 → VerificationEvidence
2. **授权决策**：VerificationEvidence → 协调器 → AuthorizationDecision
3. **特征门控**：shallow_features × AuthorizationDecision → gated_features
4. **One-Time Credential**：每个 credential 只能使用一次，防御 replay 攻击

**明确限制和安全披露**：
- ⚠️ **Toy LWE 安全性**：无模运算，m>n，threshold=48 宽松。可通过最小二乘伪造。**仅用于神经编译演示，不具有密码学安全性**。
- ⚠️ **One-Time Credential 存储**：Phase 1.2 使用内存集合（不持久化），服务器重启后丢失。Phase 3-4 可升级到持久化存储或 Bloom Filter。
- Gate Layer 包含：验证器（evidence）、协调器（decision）、特征门控（gating）三个组件。

### 核心架构

#### 1. 数据流

```
Input: (image, credential)
    ↓
[Shallow Layers] → shallow_features [B, C, H, W]
    ↓
[LWE Verifier] ← credential [B, n]
    ↓ (1) LWE 验证 (2) Replay 检查
    ↓
VerificationEvidence {
    lwe_verified: bool,
    replay_detected: bool,
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

**LWEVerifier**：验证器（只产生证据）
- 输入：credential
- 输出：VerificationEvidence（结构化证据）
- 职责：LWE 验证 + Replay 检查，不做授权决策

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

### One-Time Credential 机制

#### Credential 生成策略

**方案**：从 master secret 派生多个一次性 credential

```python
def derive_credential(master_secret: np.ndarray, nonce: int, params: LWEParams) -> np.ndarray:
    """从 master secret 和 nonce 派生一次性 credential
    
    参数:
        master_secret: [n]，主密钥
        nonce: int，单调递增的序号（0, 1, 2, ...）
        params: LWE 参数
    
    返回:
        derived_credential: [n]，派生的一次性 credential
    """
    # 方法 1：Hash-based derivation
    data = master_secret.tobytes() + nonce.to_bytes(8, 'big')
    hash_bytes = hashlib.sha256(data).digest()
    
    # 从哈希中提取 n 个 float32 值
    derived = np.frombuffer(hash_bytes[:params.n * 4], dtype=np.float32)
    
    # 归一化到 secret_bound
    derived = derived / np.linalg.norm(derived) * params.secret_bound
    
    return derived
```

**使用流程**：
```python
# 初始化：生成 master keypair
A, master_secret, b = generate_keypair(params)

# 客户端：生成第 i 个一次性 credential
credential_i = derive_credential(master_secret, nonce=i, params)

# 服务器：验证（同样派生然后验证）
# 或者：客户端发送 (nonce, credential)，服务器验证 credential 是否从 nonce 正确派生
```

#### Replay 检测机制

```python
class ReplayDetector:
    """Replay 检测器：维护已使用 credential 的集合"""
    
    def __init__(self, storage_type='memory'):
        """初始化
        
        参数:
            storage_type: 'memory' (Phase 1.2) 或 'persistent' (Phase 3-4)
        """
        if storage_type == 'memory':
            self.used_credentials = set()  # 内存集合
        elif storage_type == 'bloom':
            # Bloom Filter（内存优化，Phase 3-4）
            from pybloom_live import BloomFilter
            self.used_credentials = BloomFilter(capacity=1000000, error_rate=0.001)
        else:
            raise ValueError(f"Unsupported storage_type: {storage_type}")
    
    def compute_credential_hash(self, credential: np.ndarray) -> str:
        """计算 credential 的唯一标识"""
        return hashlib.sha256(credential.tobytes()).hexdigest()
    
    def check_and_mark(self, credential: np.ndarray) -> bool:
        """检查 credential 是否已用，如果未用则标记
        
        返回:
            True: 未被使用（检查通过）
            False: 已被使用（replay 检测到）
        """
        cred_hash = self.compute_credential_hash(credential)
        
        if cred_hash in self.used_credentials:
            return False  # Replay detected
        
        self.used_credentials.add(cred_hash)
        return True  # Fresh credential
    
    def reset(self):
        """重置（仅用于测试）"""
        if isinstance(self.used_credentials, set):
            self.used_credentials.clear()
```

### 接口定义

#### 数据结构

```python
from enum import Enum
from dataclasses import dataclass

class ReasonCode(Enum):
    """拒绝原因码"""
    SUCCESS = 0
    LWE_VERIFICATION_FAILED = 1
    REPLAY_DETECTED = 2
    INVALID_SHAPE = 3
    NON_FINITE = 4
    WRONG_DTYPE = 5
    DIMENSION_MISMATCH = 6

@dataclass
class VerificationEvidence:
    """验证证据（无授权能力）"""
    lwe_verified: bool          # LWE 验证是否通过
    replay_detected: bool       # 是否检测到 replay
    error_norm: float          # LWE 误差范数
    reason: ReasonCode         # 原因码
    credential_hash: str       # Credential 哈希（用于审计）

@dataclass
class AuthorizationDecision:
    """授权决策（由协调器提交）"""
    allow: bool                # 是否允许访问深层
    gate_signal: float        # 门控信号（训练时软值，推理时 {0, 1}）
    evidence: VerificationEvidence  # 关联的证据
```

#### 模块接口

**LWEVerifier**：
```python
class LWEVerifier(nn.Module):
    def __init__(self, A: np.ndarray, b: np.ndarray, params: LWEParams,
                 replay_detector: ReplayDetector):
        ...
    
    def forward(self, credential: Tensor) -> VerificationEvidence:
        """验证 credential，返回结构化证据"""
        ...
```

**AuthorizationCoordinator**：
```python
class AuthorizationCoordinator(nn.Module):
    def __init__(self, params: LWEParams, temperature: float = 5.0):
        ...
    
    def forward(self, evidence: VerificationEvidence) -> AuthorizationDecision:
        """根据证据做出授权决策"""
        ...
```

**FeatureGate**：
```python
class FeatureGate(nn.Module):
    def forward(self, shallow_features: Tensor, decision: AuthorizationDecision) -> Tensor:
        """应用门控到特征"""
        # gated_features = shallow_features * decision.gate_signal
        ...
```

**GateLayer**（组合）：
```python
class GateLayer(nn.Module):
    def __init__(self, A: np.ndarray, b: np.ndarray, params: LWEParams,
                 temperature: float = 5.0, replay_detector: ReplayDetector = None):
        super().__init__()
        if replay_detector is None:
            replay_detector = ReplayDetector(storage_type='memory')
        
        self.verifier = LWEVerifier(A, b, params, replay_detector)
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
        """
        # 1. 验证器：产生证据
        evidence = self.verifier(credential)
        
        # 2. 协调器：做出授权决策
        decision = self.coordinator(evidence)
        
        # 3. 特征门控：应用决策
        gated_features = self.feature_gate(shallow_features, decision)
        
        return gated_features, decision
```

### 实现步骤

**Step 1**：创建 `src/can/v2/layers/replay_detector.py`（约 80 行）
- 实现 `ReplayDetector` 类
- 支持内存存储（Phase 1.2）
- 预留 Bloom Filter 接口（Phase 3-4）

**Step 2**：创建 `src/can/v2/layers/gate_layer.py`（约 300 行）
- 实现 `ReasonCode`, `VerificationEvidence`, `AuthorizationDecision`
- 实现 `LWEVerifier`（验证器）
- 实现 `AuthorizationCoordinator`（协调器）
- 实现 `FeatureGate`（特征门控）
- 实现 `GateLayer`（组合层）

**Step 3**：创建 `tests/v2/test_replay_detector.py`（约 100 行）
- 测试 Replay 检测
- 测试 credential 哈希计算

**Step 4**：创建 `tests/v2/test_gate_layer.py`（约 350 行）
- 功能测试（valid/invalid，训练/推理）
- Replay 测试（重复使用 credential）
- 差分测试（LWE 验证逻辑）
- 授权边界测试（验证器不做决策，协调器唯一决策）
- 特征门控测试

**Step 5**：运行测试
```bash
pytest tests/v2/test_replay_detector.py tests/v2/test_gate_layer.py -v --cov=src/can/v2/layers
```

**Step 6**：更新文档
- `PROJECT_WORKLOG.md`：Phase 1.2 完成
- `SECURITY.md`：更新 replay 防御说明
- `docs/DESIGN_PROPOSALS.md`：状态改为 `[IMPLEMENTED]`

### 测试要求
[测试用例列表和覆盖率目标]

### 性能指标
[量化的性能目标]

### 风险和限制
[已知风险和缓解措施]
```
