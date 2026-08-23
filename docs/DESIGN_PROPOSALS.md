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

## Phase 1.2: Neural Gate Layer [REVISION-2]

**状态**：[REVISION-2]（修订中，待批准）  
**提出时间**：2026-08-23  
**修订时间**：2026-08-23（基于 Codex 第二轮审阅，采用纯 LWE 验证）  
**依赖**：Phase 1.1 LWE 密码原语（已完成）

**修订原因**（基于 Codex 第二轮审阅的 5 个 P1 问题）：
1. ❌ MLP 融合破坏 fail-closed（可绕过 LWE 验证）
2. ❌ 不再是"LWE 验证的神经编译"（差分测试无法证明 forward()）
3. ❌ Replay 范围说明冲突
4. ❌ 未遵循"验证证据 → 协调器授权"边界
5. ❌ 与 PROJECT_WORKLOG.md 不一致

**用户提出的数据流**：
```
credential → 确定性验证 → crypto_valid
shallow_features → 特征处理
crypto_valid → 硬授权边界
shallow_features * crypto_valid → 深层网络
```

### 设计目标

实现纯密码学验证的 Gate Layer，**LWE 验证作为硬授权边界，不可绕过**：

1. **确定性验证**：credential → LWE 验证 → crypto_valid ∈ {0, 1}
2. **训练模式**：软化验证（sigmoid，可微分）
3. **推理模式**：硬判定（error < threshold，fail-closed）
4. **差分测试**：`forward()` 与 `V_ref()` 的端到端一致性

**明确限制和安全披露**：
- ⚠️ **Toy LWE 安全性**：无模运算，m>n，threshold=48 宽松。可通过最小二乘伪造。**仅用于神经编译演示，不具有密码学安全性**。
- Gate Layer 产生硬授权判定 `crypto_valid`，深层执行由 Phase 1.3 Gated ResNet 控制。
- 当前使用静态 credential，不防御 replay 攻击（需 Phase 3-4 添加 challenge-response）。

### 核心架构

#### 数据流

```
Input: (image, credential)
    ↓
[Shallow Layers] → shallow_features [B, C, H, W]
    ↓
[Gate Layer] ← credential [B, n]
    ↓ LWE 验证（确定性）
    ↓
crypto_valid [B] ∈ {0, 1}  ← 硬授权边界
    ↓
[条件路由]（Phase 1.3 实现）
    if crypto_valid == 1:
        deep_features = layer4(layer3(shallow_features))
    else:
        # fail-closed：深层零调用
        public_output = public_head(shallow_features)
```

#### 关键原则

1. **Gate Layer 不处理特征**：`shallow_features` 作为输入参数但不参与 LWE 验证（为 Phase 1.3 预留接口）
2. **LWE 验证是唯一判定依据**：`crypto_valid = (error_norm < threshold)`
3. **训练时软化，推理时硬判定**：
   - 训练：`sigmoid((threshold - error_norm) / temperature)` 可微分
   - 推理：`(error_norm < threshold).float()` 硬判定
4. **差分测试端到端**：对任意 credential，`forward()` 推理结果 == `V_ref()`

### 接口定义

#### 输入（严格验证，fail-closed on invalid）

**credential**：`Tensor[B, n]` 或 `np.ndarray[B, n]` 或 `[n]`（单样本）
- 类型：float32/float64（自动转换为 float32）
- 约束：有限值，n == LWEParams.n
- 单样本 `[n]` 自动广播为 `[B, n]`

**shallow_features**：`Tensor[B, C, H, W]`（可选，Phase 1.2 不使用）
- 作用：Batch 一致性检查，预留 Phase 1.3 接口

**拒绝的输入**：
- 非有限值（NaN, Inf）
- 布尔、整数、复数类型
- 错误形状（> 2D，维度不匹配）
- 类型混淆（字典、列表、字符串）

#### 输出

**crypto_valid**：`Tensor[B]`
- 训练模式：`∈ [0, 1]`（软判定，可微分）
- 推理模式：`∈ {0, 1}`（硬判定，fail-closed）
- 语义：LWE 验证结果，1=通过，0=拒绝

#### 异常

**ValueError**：输入验证失败
- 调用方（Phase 1.3 Gated ResNet）必须捕获并路由到公开 head（fail-closed）

### 实现步骤

**Step 1**：创建 `src/can/v2/layers/gate_layer.py`（约 180 行）
- 实现 `GateLayer` 类
- 严格输入验证（`_validate_credential`）
- LWE 验证逻辑（`_compute_lwe_error_norm`）
- `forward()`, `verify()`

**Step 2**：创建 `tests/v2/test_gate_layer.py`（约 250 行）
- 功能测试（valid/invalid，训练/推理）
- 差分测试（端到端）
- 输入规范化测试
- 构造期验证测试

**Step 3**：运行测试
```bash
pytest tests/v2/test_gate_layer.py -v --cov=src/can/v2/layers --cov-report=term-missing
```

**Step 4**：更新文档
- `PROJECT_WORKLOG.md`：Phase 1.2 完成，记录测试结果
- `docs/DESIGN_PROPOSALS.md`：状态改为 `[IMPLEMENTED]`

### 测试要求

**测试状态标记**：`[ ]` 未实现，`[x]` 已实现并通过

#### 功能测试
- [ ] `test_valid_credential_training_mode`: valid credential → crypto_valid > 0.5
- [ ] `test_valid_credential_eval_mode`: valid credential → crypto_valid = 1.0
- [ ] `test_invalid_credential_training_mode`: invalid credential → crypto_valid < 0.5
- [ ] `test_invalid_credential_eval_mode`: invalid credential → crypto_valid = 0.0

#### 差分测试（端到端）
- [ ] `test_forward_matches_V_ref_eval_mode`: 推理模式下，`forward()` 与 `V_ref()` 100% 一致（100 次随机测试）
- [ ] `test_verify_matches_V_ref`: `verify()` 与 `V_ref()` 100% 一致（100 次随机测试）
- [ ] `test_end_to_end_invariant`: 对任意 credential，如果 `V_ref=0`，则推理 `crypto_valid=0`

#### 输入规范化与异常处理
- [ ] `test_single_sample_broadcast`: `[n]` → `[1, n]` 正确
- [ ] `test_batch_processing`: `[B, n]` → `[B]` 正确
- [ ] `test_reject_nan`: NaN → ValueError
- [ ] `test_reject_inf`: Inf → ValueError
- [ ] `test_reject_wrong_dimension`: n != params.n → ValueError
- [ ] `test_reject_wrong_dtype`: int/bool/complex → ValueError
- [ ] `test_reject_wrong_shape`: 3D/0D → ValueError
- [ ] `test_verify_exception_returns_false`: 异常时 `verify()` 返回 False

#### 构造期验证
- [ ] `test_constructor_validates_A_shape`: A 不是 2D → TypeError
- [ ] `test_constructor_validates_b_shape`: b 不是 1D → TypeError
- [ ] `test_constructor_validates_A_b_consistency`: A.shape[0] != b.shape[0] → ValueError
- [ ] `test_constructor_validates_finite`: A 或 b 包含 NaN/Inf → ValueError

#### 训练/推理模式
- [ ] `test_training_mode_is_soft`: 训练模式输出连续值
- [ ] `test_eval_mode_is_hard`: 推理模式输出 {0, 1}
- [ ] `test_mode_switch`: `train()` / `eval()` 切换正确

**目标覆盖率**：≥ 95%

### 性能指标

| 指标 | 目标 | 说明 |
|------|------|------|
| Valid credential → crypto_valid (训练) | > 0.5 | 软判定 |
| Valid credential → crypto_valid (推理) | = 1.0 | 硬判定 |
| Invalid credential → crypto_valid (训练) | < 0.5 | 软判定 |
| Invalid credential → crypto_valid (推理) | = 0.0 | 硬判定 |
| 端到端一致性（推理模式） | 100% | `forward()` == `V_ref()` |
| `verify()` 一致性 | 100% | `verify()` == `V_ref()` |
| 测试覆盖率 | ≥ 95% | 行覆盖 |

### 风险和限制

#### 1. Toy LWE 伪造风险（明确披露）

**风险**：无模运算，m>n，threshold=48 宽松，可通过最小二乘伪造。

**实验验证**（Phase 1.2 实现后测试）：
```python
A, secret, b = generate_keypair(params)
x_fake = np.linalg.lstsq(A, b, rcond=None)[0]
error_fake = np.linalg.norm(b - A @ x_fake)
assert error_fake < params.error_threshold  # 伪造成功
```

**缓解措施**：
- **Phase 1-2**：明确标注"仅用于神经编译演示"
- **Phase 3-4（可选升级）**：整数模运算 + 更大参数 + 更紧阈值

#### 2. Replay 攻击（已知限制）

**风险**：静态 credential 可重放。

**当前状态**：
- SECURITY.md 将 replay 列为攻击者能力
- AGENTS.md 要求 replay → 零受保护副作用
- **因此这是一个阻塞问题，需要解决**

**解决方案（Phase 3-4）**：
- Challenge-response：服务器发送随机 challenge，客户端用 credential 签名
- Session binding：credential 绑定会话 ID
- Time-based：credential 包含时间戳，过期失效

**Phase 1.2 记录**："已知限制：静态 credential，不防御 replay。需 Phase 3-4 添加 challenge-response。"

#### 3. Fail-closed 范围限定

**Phase 1.2 范围**：Gate Layer 产生 `crypto_valid` 硬授权判定。

**Phase 1.3 范围**：Gated ResNet 根据 `crypto_valid` 控制深层执行，通过 forward hook 验证深层零调用。

#### 4. 软硬路由语义一致性

**风险**：训练时 sigmoid 软化，推理时硬阈值，边界附近可能不一致。

**缓解措施**：
- Temperature = 5.0（较小，减少差异）
- 训练后期 temperature annealing
- 差分测试覆盖边界情况

### 与 Codex 审阅的对照

#### 解决的 P1 问题

1. ✅ **MLP 不能绕过 LWE 验证**：无 MLP，`crypto_valid = (error_norm < threshold)`
2. ✅ **仍是"LWE 验证的神经编译"**：`forward()` 直接执行 LWE 验证，差分测试有效
3. ⚠️ **Replay 问题**：明确标记为"已知限制，需 Phase 3-4 解决"
4. ✅ **遵循授权边界**：验证逻辑清晰，`crypto_valid` 是硬授权判定
5. ✅ **与 PROJECT_WORKLOG.md 一致**：`error < threshold`（推理），`sigmoid((threshold-error)/T)`（训练）

#### 解决的 P2 问题

6. ✅ **输入验证完整**：credential 严格验证，构造期验证 A, b
7. ✅ **批量路由可执行**：Phase 1.3 示例代码展示逐样本路由
8. ✅ **测试目标明确**：不依赖随机 MLP，直接测试 LWE 验证
9. ✅ **升级表述准确**：明确"toy profile 不等于安全认证协议"

### 总结

**Revision 2 的核心变更**：
1. **放弃 MLP 特征融合**：Gate Layer 只做 LWE 验证
2. **采用用户提出的数据流**：credential → crypto_valid → 硬授权边界
3. **差分测试端到端**：`forward()` 推理结果 == `V_ref()`
4. **明确 Replay 限制**：标记为"需 Phase 3-4 解决"
5. **完整输入验证**：fail-closed on invalid

**与 Revision 1 的对比**：
- Revision 1：credential + shallow_features → MLP fusion → gate_signal
- Revision 2：credential → LWE 验证 → crypto_valid（无 MLP）

**优势**：
- ✅ 解决 Codex 的所有 P1 和 P2 问题
- ✅ 架构清晰，易于验证
- ✅ 与 PROJECT_WORKLOG.md 一致
- ✅ 差分测试能证明端到端正确性

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
