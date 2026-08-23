# 项目工作日志

## 当前研究阶段

**阶段**: V2 - Gate Layer 在计算图中间架构  
**状态**: Phase 1.1 完成 - LWE 密码原语实现  
**最后更新**: 2026-08-21

---

## 研究目标

实现模型内生安全：在神经网络中间嵌入基于 LWE 的认证神经元层（Gate Layer），该层融合浅层特征与密码 credential 信息，根据验证结果控制深层神经元的激活。

**密码方案变更**：从 Module-SIS 改为 LWE (Learning With Errors)，理由：
- LWE 更适合神经网络编译（线性运算 + 噪声注入）
- 验证逻辑更简单（误差范数阈值判断）
- 文献支持更充分（Shamir et al. 使用 LWE 而非 Module-SIS）

### 核心架构

```
Input (image, credential)
    ↓
[Shallow Layers] ──→ shallow_features
    ↓
[Gate Layer] ←──── (shallow_features, credential)
    ↓
  gate_signal (0 or 1)
    ↓
[Deep Layers] ←──── (shallow_features * gate_signal)
    ↓
[Protected Head] ──→ fine-grained output
    
如果 gate_signal = 0:
[Public Head] ←──── shallow_features
    ↓
  coarse output (弱化能力)
```

### 关键设计决策

1. **Gate Layer 位置**：在浅层特征提取后、深层特征提取前
2. **密码方案**：LWE (Learning With Errors，toy profile)
   - 参数：n=128, m=256, σ=1.0, threshold=48.0
   - 验证逻辑：L2 误差范数 < threshold
3. **路由机制**：
   - 训练时：软路由（可微分，使用 sigmoid）
   - 推理时：硬路由（真正不执行深层，gate_signal ∈ {0, 1}）
4. **能力分级**：
   - Valid credential：**10-class fine-grained classification**（深层 + protected head）
   - Invalid credential：**2-class coarse classification**（公开 head，通过知识蒸馏训练）
   - 粗粒度标签：CIFAR-10 → CIFAR-2（animal vs vehicle）
5. **Baseline 模型**：
   - **Phase 1（原型验证）**：ResNet-18 on CIFAR-10
     - 10 类 → 2 类（animal vs vehicle）
     - 目的：快速验证 Gate Layer 架构可行性
   - **Phase 2（能力分级）**：ResNet-18 on CIFAR-100
     - 100 类 → 20 类（超类）或 10 类（更粗）
     - 目的：展示显著的能力差距
   - **Phase 3（可选）**：ResNet-50 on ImageNet 子集
     - 1000 类 → 100 类或 10 类
     - 目的：验证在大规模任务上的可行性

---

## 实施路线图

### Phase 1: 基础架构搭建 [CURRENT]

**目标**：实现 Gate Layer 在计算图中间的基本架构

#### 1.1 LWE 密码原语 [COMPLETED]

**文件**：`src/can/v2/crypto/lwe.py`

实现内容：
- `LWEParams`：参数类（n=128, m=256, σ=1.0, threshold=48.0）
- `generate_keypair()`：生成 (A, secret, b) where b = As + e
- `verify(secret, A, b, params)`：验证 ||b - As|| < threshold
- `compute_error_norm()`：计算 L2 误差范数
- `V_ref(credential, A, b, params) → {0, 1}`：参考验证器

实现细节：
- 使用 NumPy 实现高效矩阵运算
- 支持批量验证（向量化计算）
- 误差分布清晰：valid ~16, invalid ~900, threshold=48
- 防御性异常处理（维度不匹配返回 False 而非崩溃）

测试覆盖率：**100%** (55/55 statements)

**测试文件**：`tests/v2/test_lwe.py`

测试结果（38 个测试全部通过）：
- ✅ 正向：valid credential → verify = True
- ✅ 负向：invalid credential → verify = False
- ✅ 边界值：threshold 边界正确
- ✅ 差分测试：`verify()` 与 `V_ref()` 一致
- ✅ 批量处理：多个 credential 同时验证
- ✅ 误差分布：valid 误差 << threshold < invalid 误差
- ✅ 稳定性：100 次随机 invalid credential 测试，假阳性率 < 5%（实测 0）
- ✅ 异常处理：维度不匹配、无效类型正确处理

**决策文档**：`docs/V2_LWE_IMPLEMENTATION.md`

关键决策：
1. **参数选择**：n=128, m=256（安全性与效率平衡）
2. **误差阈值**：threshold=48（基于经验误差分布 3σ）
3. **数值类型**：float32（GPU 兼容）
4. **验证逻辑**：L2 范数阈值判断（简单高效）

**性能**：
- 单次验证：~0.1ms（NumPy CPU）
- 批量验证：向量化加速
- 内存占用：~130KB per keypair

**已清理代码**：
- ❌ 删除 `src/can/v2/crypto/module_sis.py`
- ❌ 删除 `tests/v2/test_module_sis.py`
- ✅ 更新 `src/can/v2/crypto/__init__.py` 仅导出 LWE 接口

#### 1.2 Neural Gate Layer [NEXT]

**文件**：`src/can/v2/layers/gate_layer.py`

实现内容：
- `GateLayer(nn.Module)`：
  - 输入：`(shallow_features, credential)`
  - 内部：特征融合 + LWE 验证的神经编译
  - 输出：`gate_signal ∈ [0, 1]`（训练）或 `{0, 1}`（推理）
- 训练模式：软路由（sigmoid，可微分）
- 推理模式：硬路由（threshold = 0.5）

关键约束：
- `gate_signal` 必须是可微分的（训练时）
- 推理时必须是离散的（真正的 fail-closed）
- 验证逻辑必须与 `V_ref` 差分测试通过

**测试文件**：`tests/v2/test_gate_layer.py`

测试要求：
- Valid credential → gate_signal > 0.7（训练模式）
- Valid credential → gate_signal = 1.0（推理模式）
- Invalid credential → gate_signal < 0.3（训练模式）
- Invalid credential → gate_signal = 0.0（推理模式）
- 差分测试：`GateLayer.verify() == V_ref()`
- 形状测试：batch 处理正确性

#### 1.3 Gated ResNet-18 [TODO]

**文件**：`src/can/v2/models/gated_resnet.py`

实现内容：
- `GatedResNet18(nn.Module)`：
  - 浅层：layer1 + layer2（到第一个下采样）
  - Gate Layer：在 layer2 之后
  - 深层：layer3 + layer4 + fc
  - 公开 head：从 shallow_features 直接到 coarse classifier
  - 保护 head：从 deep_features 到 fine-grained classifier

路由逻辑：
```python
def forward(self, x, credential):
    shallow_feat = self.layer2(self.layer1(x))
    gate_signal = self.gate_layer(shallow_feat, credential)
    
    if self.training:
        # 软路由（可微分）
        deep_feat = self.layer4(self.layer3(shallow_feat * gate_signal))
        protected_out = self.protected_head(deep_feat)
        public_out = self.public_head(shallow_feat)
        return protected_out, public_out, gate_signal
    else:
        # 硬路由（真正不执行深层）
        if gate_signal > 0.5:
            deep_feat = self.layer4(self.layer3(shallow_feat))
            return self.protected_head(deep_feat), None, 1.0
        else:
            return None, self.public_head(shallow_feat), 0.0
```

**测试文件**：`tests/v2/test_gated_resnet.py`

测试要求：
- Valid credential → 深层执行，输出 fine-grained
- Invalid credential → 深层不执行，输出 coarse
- 形状正确性
- Forward/backward pass 无异常

---

### Phase 2: 训练流程 [NEXT]

**目标**：训练 Gated ResNet-18，使其具有能力分级

#### 2.1 Public head 知识蒸馏 [TODO]

**文件**：`src/can/v2/training/distillation.py`

实现内容：
- 使用预训练 ResNet-18（full model）作为教师
- 训练 public head（从 shallow_features）学习粗粒度分类
- 损失函数：KL 散度 + CE loss
- 目标：coarse accuracy 达到合理水平（如 60-70% on CIFAR-10）

#### 2.2 联合训练 [TODO]

**文件**：`src/can/v2/training/train_gated.py`

训练目标：
- Protected path：fine-grained accuracy ≈ baseline（valid credential）
- Public path：coarse accuracy ≈ 蒸馏目标（invalid credential）
- Gate Layer：正确路由（valid → 1, invalid → 0）

损失函数：
```
L_total = L_protected + λ_public * L_public + λ_gate * L_gate
```

其中：
- `L_protected`：CrossEntropy（protected_out, fine_labels）
- `L_public`：CrossEntropy（public_out, coarse_labels）+ KL（distillation）
- `L_gate`：BCE（gate_signal, is_valid_credential）

**配置文件**：`configs/v2/train_gated_resnet18_cifar10.yaml`

训练超参数：
- Epochs: 100
- Batch size: 128
- Optimizer: SGD（lr=0.1, momentum=0.9, weight_decay=5e-4）
- λ_public: 0.5
- λ_gate: 1.0

**测试文件**：`tests/v2/test_training.py`

测试要求：
- 损失函数计算正确
- 训练循环无异常
- Checkpoint 保存/加载正确

---

### Phase 3: 评估实验（CIFAR-10）[LATER]

**目标**：验证 Gate Layer 的功能正确性和能力分级（快速原型验证）

**数据集**：CIFAR-10（10 类 → 2 类）

#### 3.1 功能正确性实验

**文件**：`src/can/v2/experiments/functional_test.py`

实验内容：
1. **Fail-closed 验证**：
   - Invalid credential → 深层调用计数 = 0
   - 使用 forward hook 统计 layer3/layer4 的实际执行次数

2. **差分测试**：
   - Valid credential 输出 vs 深层 direct 输出
   - 逐 token 比较 logits 差异

3. **Gate signal 分布**：
   - Valid credential → gate_signal 均值和方差
   - Invalid credential → gate_signal 均值和方差

#### 3.2 能力分级实验

**文件**：`src/can/v2/experiments/capability_tiering.py`

实验内容：
1. **Protected accuracy**（valid credential）：
   - Fine-grained classification accuracy
   - 与 baseline ResNet-18 比较

2. **Public accuracy**（invalid credential）：
   - Coarse classification accuracy
   - 与蒸馏目标比较

3. **Logits 等价性**：
   - Valid credential logits vs 深层 direct logits
   - 计算 L2 距离、余弦相似度

#### 3.3 性能实验

**文件**：`src/can/v2/experiments/performance.py`

实验内容：
- Latency：valid vs invalid credential
- GPU 内存占用
- 吞吐量（samples/sec）

**Baseline 比较**：
- External verifier + full model（验证器在模型外部）
- 记录：verifier latency, model latency, total latency

---

### Phase 4: 扩展实验（CIFAR-100）[FUTURE]

**目标**：展示显著的能力分级效果

**数据集**：CIFAR-100（100 类 → 20 类超类）

**为什么需要 CIFAR-100？**
- 能力差距更显著（100 类 vs 20 类，而非 10 类 vs 2 类）
- CIFAR-100 自带 20 个超类标签，无需手动映射
- 更能说服审稿人能力分级是有意义的

#### 4.1 Gated ResNet-18 on CIFAR-100
**文件**：`src/can/v2/models/gated_resnet_cifar100.py`

实现内容：
- Protected path：100 类细粒度分类
  - 示例：具体狗的品种（Chihuahua, German Shepherd...）
- Public path：20 类超类分类
  - 示例：动物大类（哺乳动物、鱼类、花卉...）
- 能力差距预期：
  - Protected accuracy ~75%（100 类）
  - Public accuracy ~55%（20 类）

**CIFAR-100 超类示例**：
```python
超类映射（20 个超类 → 100 个细粒度类）：
- aquatic_mammals: [beaver, dolphin, otter, seal, whale]
- fish: [aquarium_fish, flatfish, ray, shark, trout]
- flowers: [orchid, poppy, rose, sunflower, tulip]
- food_containers: [bottle, bowl, can, cup, plate]
- fruit_and_vegetables: [apple, mushroom, orange, pear, sweet_pepper]
- household_electrical_devices: [clock, keyboard, lamp, telephone, television]
- household_furniture: [bed, chair, couch, table, wardrobe]
- insects: [bee, beetle, butterfly, caterpillar, cockroach]
- large_carnivores: [bear, leopard, lion, tiger, wolf]
- large_man-made_outdoor_things: [bridge, castle, house, road, skyscraper]
- ... (共 20 个超类)
```

#### 4.2 能力分级对比实验
**文件**：`src/can/v2/experiments/capability_comparison.py`

实验内容：
- 对比 CIFAR-10 和 CIFAR-100 的能力差距：
  ```
  CIFAR-10:  10 类 (92%) → 2 类 (65%)  差距 = 27%
  CIFAR-100: 100 类 (75%) → 20 类 (55%) 差距 = 20%（但类别数差距更大）
  ```
- 分析：类别数越多，能力分级越显著
- 可视化：混淆矩阵对比（100 类 vs 20 类）

#### 4.3 多级能力分级（可选）
**文件**：`src/can/v2/models/multi_level_gated_resnet.py`

实现内容：
- 多个 Gate Layer（串联）
- Protected path：100 类（需要高级 credential）
- Intermediate path：20 类超类（需要中等 credential）
- Public path：10 类粗粒度（无 credential）
- 展示多级 Gate Layer 的可能性

**架构示意**：
```
Input (image, credential_high, credential_mid)
    ↓
[Shallow Layers]
    ↓
[Gate Layer 1] ← credential_mid
    ↓ (if pass)
[Mid Layers]
    ↓
[Gate Layer 2] ← credential_high
    ↓ (if pass)
[Deep Layers] → 100 类
    ↓ (if fail at Gate 2)
[Mid Head] → 20 类
    ↓ (if fail at Gate 1)
[Public Head] → 10 类
```

---

### Phase 5: 大规模验证（ImageNet）[OPTIONAL]

**目标**：验证在大规模任务上的可行性

**数据集**：ImageNet（1000 类 → 100 类或 10 类）

**为什么需要 ImageNet？**
- 能力差距极其显著（1000 → 100 或 1000 → 10）
- 更接近真实应用场景（高分辨率图像，复杂场景）
- 论文说服力最强（顶会审稿人期望的规模）

**挑战**：
- 训练成本高（需要多 GPU，数天训练）
- 需要更大的模型（ResNet-50 或 ResNet-101）
- 粗粒度标签需要手动定义（ImageNet 无自带超类）
- 数据集大（1.28M 训练图像，224×224）

**收益评估**：
- ✅ 极强说服力（顶会论文的标准）
- ✅ 真实应用场景
- ❌ 研究原型阶段可能过早（先验证 CIFAR-100）

**建议**：
- 先完成 CIFAR-10 原型验证
- 再完成 CIFAR-100 能力分级展示
- 如果两者都成功，再考虑 ImageNet（论文投稿前）

---

## 当前状态

### 已完成
- [x] 项目文档初始化（AGENTS.md, SECURITY.md, README.md, PROJECT_WORKLOG.md）
- [x] Git 仓库初始化
- [x] **Phase 1.1: LWE 密码原语实现**（2026-08-21）
  - [x] `src/can/v2/crypto/lwe.py` 实现完成
  - [x] `tests/v2/test_lwe.py` 38 个测试全部通过
  - [x] 测试覆盖率 100%
  - [x] 决策文档 `docs/V2_LWE_IMPLEMENTATION.md` 完成
  - [x] 清理 Module-SIS 相关代码

### 进行中
- [ ] 无

### 下一步（唯一下一步）

**实现 Neural Gate Layer**

具体任务：
1. 创建 `src/can/v2/layers/gate_layer.py`：
   - 实现 `GateLayer(nn.Module)` 类
   - 集成 LWE 验证逻辑（使用 `V_ref`）
   - 实现软路由（训练模式）和硬路由（推理模式）
   - 支持 batch 处理

2. 创建 `tests/v2/test_gate_layer.py`：
   - 测试 valid/invalid credential 路由行为
   - 测试训练/推理模式切换
   - 差分测试：`GateLayer.verify()` vs `V_ref()`
   - 形状和 batch 测试

3. 运行测试：`pytest tests/v2/test_gate_layer.py -v`

4. 更新本工作日志

**注意**：Gate Layer 的关键在于如何将 LWE 验证（NumPy）编译为可微分的神经计算（PyTorch）。可能需要：
- 将 A, b 转换为 `nn.Parameter`（冻结，不更新）
- 将矩阵乘法和范数计算用 PyTorch ops 实现
- 训练时使用 soft threshold（sigmoid），推理时使用 hard threshold

---

## 开放问题

1. **Coarse labels 如何定义？**
   - CIFAR-10 → CIFAR-2（animal vs vehicle）
   - 映射规则：
     ```
     vehicle (0): {airplane, automobile, ship, truck}  # 类别 0,1,8,9
     animal (1):  {bird, cat, deer, dog, frog, horse}  # 类别 2,3,4,5,6,7
     ```
   - ✅ **已决定**：使用语义聚类（animal/vehicle）
   - **CIFAR-100**：使用自带的 20 个超类（无需手动映射）

2. **Gate Layer 的神经编译细节？**
   - 如何将 LWE 验证嵌入可微分计算？
   - **关键问题**：
     - A, b 如何存储？（nn.Parameter 冻结 or nn.Buffer？）
     - 验证逻辑如何微分？（训练时需要软 threshold）
     - 如何确保推理时的精确验证？（硬 threshold，与 V_ref 一致）
   - **初步方案**：
     - A, b 存储为 `nn.Buffer`（不参与梯度更新）
     - 训练时：`gate = sigmoid((threshold - error) / temperature)`（软化）
     - 推理时：`gate = (error < threshold).float()`（精确）

3. **训练收敛性？**
   - 软路由和硬路由的切换是否影响性能？
   - 需要在 Phase 2 中实验验证

4. **数据集选择策略？** [NEW]
   - **Phase 1（原型验证）**：CIFAR-10（10→2 类）
     - 优势：快速迭代，2 小时训练
     - 劣势：能力差距不够显著
   - **Phase 2（能力分级）**：CIFAR-100（100→20 类）
     - 优势：能力差距显著，有自带超类
     - 劣势：训练时间稍长（1 天）
   - **Phase 3（可选）**：ImageNet（1000→100 类）
     - 优势：最强说服力
     - 劣势：成本高，需要多 GPU

5. **多级能力分级的必要性？**
   - 是否需要展示 3 级能力分级（100→20→10 类）？
   - 还是 2 级分级（100→20 类）已经足够？
   - 取决于 Phase 2 的实验结果和审稿人反馈

---

## 风险与限制

### 当前阶段风险

1. **LWE 神经编译可行性** [NEW RISK]：
   - LWE 验证是非线性操作（范数 + 阈值判断）
   - 训练时的软化版本（sigmoid）是否能保持语义一致性？
   - 推理时的硬阈值是否会导致梯度消失问题？
   - **缓解措施**：使用 temperature 参数控制软化程度，逐步退火

2. **训练收敛**：软路由 → 硬路由切换可能导致性能下降

3. **能力分级效果**：Public head 能力是否真的弱于 protected head
   - **CIFAR-10 风险**：10→2 类的差距可能不够显著（都很容易）
   - **缓解措施**：Phase 2 使用 CIFAR-100（100→20 类）

4. **PyTorch 与 NumPy 一致性** [NEW RISK]：
   - Gate Layer 使用 PyTorch 实现 LWE 验证
   - 必须与 NumPy 版本的 `V_ref` 差分测试通过
   - 浮点精度、广播规则可能导致细微差异

5. **数据集规模权衡** [NEW RISK]：
   - CIFAR-10：快但能力差距小
   - CIFAR-100：平衡点（推荐）
   - ImageNet：成本高，可能在原型阶段过早
   - **策略**：渐进式实验（10→100→1000）

### 明确的非目标（当前阶段）

- **白盒攻击防御**：当前阶段不考虑
- **TEE 部署**：后续阶段
- **密码学安全归约**：Toy profile 不保证（LWE 参数 n=128 过小）
- **生产部署**：研究原型阶段
- **ImageNet 实验**：Phase 1-2 不考虑（Phase 3 可选）

---

## 实验结果

### LWE 密码原语实现 [COMPLETED 2026-08-21]

**测试状态**：✅ 38/38 通过  
**覆盖率**：✅ 100% (55/55 statements)  
**差分测试**：✅ `verify()` 与 `V_ref()` 完全一致  
**性能**：
- 单次验证：~0.1ms (NumPy CPU)
- 内存占用：~130KB per keypair

**误差分布验证**（`test_false_positive_rate`：100 次随机 invalid credential，scale=5.0）：
- Valid credential 误差：均值 ~16，远小于 threshold=48
- Invalid credential 误差：均值 ~900，远大于 threshold=48
- 假阳性率：断言 < 5%，实测 0
- 假阴性率：0%

**未测量项**（避免与 README 旧表述混淆）：
- 大样本（≥1000 次）统计的假阳性率置信区间
- 单次验证 latency 的基准测试（~0.1ms 为估算，非 benchmark 实测）

**参数配置**（toy profile，非生产）：
- n=128, m=256, σ=1.0, threshold=48.0
- 数值类型：float32（GPU 兼容）

---

### Gate Layer 实现（待完成）
- 测试状态：未运行
- 差分测试：未运行
- 软/硬路由切换：未验证

---

### Gated ResNet-18（待完成）

#### CIFAR-10 实验（Phase 1 原型验证）
- 训练状态：未开始
- Protected accuracy：未测量（目标 ≥90%，10 类）
- Public accuracy：未测量（目标 ~65%，2 类）
- Logits 等价性：未验证
- Latency：未测量

#### CIFAR-100 实验（Phase 2 能力分级展示）
- 训练状态：未开始
- Protected accuracy：未测量（目标 ≥72%，100 类）
- Public accuracy：未测量（目标 ~55%，20 类超类）
- 能力差距：未测量（预期 100→20 比 10→2 更显著）
- 与 CIFAR-10 对比：未完成

#### ImageNet 实验（Phase 3 可选）
- 状态：暂不考虑（成本过高，原型阶段不需要）
- 建议：等 CIFAR-100 实验成功后再决定

---

## 参考文献

- Shamir et al., "How to Securely Implement Cryptography in Deep Neural Networks"
- Regev, "On lattices, learning with errors, random linear codes, and cryptography" (LWE 原始论文)
- Knowledge Distillation: Hinton et al., "Distilling the Knowledge in a Neural Network"

---

## 提交历史

### Checkpoint: LWE 密码原语实现 [已就绪]

**Status**: Ready to commit

**Files to commit**:
- `src/can/v2/crypto/lwe.py`
- `src/can/v2/crypto/__init__.py`
- `tests/v2/test_lwe.py`
- `docs/V2_LWE_IMPLEMENTATION.md`
- `PROJECT_WORKLOG.md` (updated)

**Files deleted**:
- `src/can/v2/crypto/module_sis.py`
- `tests/v2/test_module_sis.py`

**Commit message**:
```
feat: Implement LWE cryptographic primitive (Phase 1.1)

- Add LWE implementation (n=128, m=256, toy profile)
- 38 tests, 100% coverage, all passing
- Error distribution: valid ~16, invalid ~900, threshold=48
- Remove Module-SIS (replaced by LWE)
- Add implementation decision doc

Next: Neural Gate Layer implementation
```

**Next**: Implement Neural Gate Layer

---

### Checkpoint: 初始项目搭建 [已完成 2026-08-21]

**Commit message**:
```
Initial project setup: V2 Gate Layer architecture

- Add project governance docs (AGENTS.md, SECURITY.md)
- Add research roadmap (PROJECT_WORKLOG.md)
- Define V2 architecture: Gate Layer in computation graph
- Roadmap: LWE → Gate Layer → Gated ResNet-18 → Training → Evaluation

Scope: Research prototype, white-box defense out of scope
```

---

## 备注

- 本文档是唯一的动态工作日志，每次实现后必须更新
- 所有测试结果（pass/fail/skip）必须记录在此
- 所有设计决策和风险必须记录在此
- 保持"唯一下一步"明确且可执行

**数据集选择策略**：
- **Phase 1**：CIFAR-10（快速原型验证，10→2 类）
- **Phase 2**：CIFAR-100（能力分级展示，100→20 类，推荐用于论文）
- **Phase 3**：ImageNet（可选，大规模验证，1000→100 类，成本高）

**为什么选择这个顺序**：
1. CIFAR-10 快速验证架构可行性（1-2 天训练）
2. CIFAR-100 展示显著能力差距（自带超类，无需手动映射）
3. ImageNet 仅在前两者成功后考虑（论文投稿前）

**能力差距对比**：
- CIFAR-10：10 类(92%) → 2 类(65%)，差距 27%，但绝对类别数少
- CIFAR-100：100 类(75%) → 20 类(55%)，差距 20%，但类别数差距 5 倍，更有说服力
- ImageNet：1000 类(76%) → 100 类(50%)，差距最大，但训练成本高
