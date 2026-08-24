# 项目工作日志

## 当前研究阶段

**阶段**: V2 - Gate Layer 在计算图中间架构  
**状态**: Phase 2 修复完成 - 等待 Claude 复验
**最后更新**: 2026-08-24

---

## 研究目标

实现模型内生安全：在神经网络中间嵌入基于 LWE 的认证神经元层（Gate Layer），该层融合浅层特征与密码 credential 信息，根据验证结果控制深层神经元的激活。

**密码方案变更**：从 Module-SIS 改为 LWE (Learning With Errors)，理由：
- LWE 更适合神经网络编译（线性运算 + 噪声注入）
- 验证逻辑更简单（误差范数阈值判断）
- 实现复杂度更低（无需多项式环运算）

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
   - 训练时：软路由（可微分，`gate_signal = sigmoid((threshold-error)/T)`）
   - 推理时：硬路由（真正不执行深层，gate_signal ∈ {0, 1}）
   - **已实现**：Phase 1.2 Gate Layer 产生 gate_signal 并应用到 shallow features
   - **已实现**：Phase 1.3 Gated ResNet 根据 gate_signal 控制深层实际执行
4. **能力分级**：
   - Valid credential：**10-class fine-grained classification**（深层 + protected head）
   - Invalid credential：**2-class coarse classification**（公开 head，通过知识蒸馏训练）
   - 粗粒度标签：CIFAR-10 → CIFAR-2（animal vs vehicle）
5. **Baseline 模型**：
   - **Phase 1-2（架构与训练原型）**：ResNet-18 on CIFAR-10
     - 10 类 → 2 类（animal vs vehicle）
     - 目的：快速验证 Gate Layer 架构可行性
   - **Phase 4（能力分级扩展）**：ResNet-18 on CIFAR-100
     - 100 类 → 20 类（超类）或 10 类（更粗）
     - 目的：展示显著的能力差距
   - **Phase 5（可选）**：ResNet-50 on ImageNet 子集
     - 1000 类 → 100 类或 10 类
     - 目的：验证在大规模任务上的可行性

---

## 实施路线图

### Phase 1: 基础架构搭建 [COMPLETED]

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

#### 1.2 Neural Gate Layer [COMPLETED]

**文件**：`src/can/v2/layers/gate_layer.py`

设计内容：
- `LWEVerifier`：将 credential 转换为批量、结构化的 `VerificationEvidence`
- `AuthorizationCoordinator`：唯一生成批量 `AuthorizationDecision`
- `FeatureGate`：执行 `shallow_features * gate_signal`，不让图像特征参与认证判定
- `GateLayer(nn.Module)`：组合上述组件，对外返回 `(gated_features, decision)`
- 训练模式：软路由（sigmoid，可微分）
- 推理模式：硬路由（`error_norm < error_threshold`）
- Phase 1-2 使用静态 credential，不实现 replay 防御；replay 留到后续研究阶段

**完成时间**：2026-08-23

**实现内容**：
- `LWEVerifier`：无副作用的 LWE 验证器
- `AuthorizationCoordinator`：唯一授权决策点
- `FeatureGate`：将 gate_signal 应用到 shallow features
- `GateLayer`：组合层，支持 batch、device 转移和 autograd

**关键特性**：
- Tensor-based 数据结构，支持 batch、GPU device 契约和 autograd
- 无状态设计，可重复调用
- 可微分但不可训练：A、b 冻结，梯度回传到 shallow_features
- Fail-closed 输入验证：非法 credential 产生 `gate_signal = 0.0`
- 训练时使用软门控，推理时使用硬判定

**测试结果**：
- 测试通过率：43/43（100%）
- LWE 验证：5 个测试（差分、边界、无状态）
- 训练/推理模式：3 个测试（软门控、硬判定、一致性）
- Batch 处理：5 个测试（单样本、mixed batch、一致性）
- 输入验证：15 个测试（NaN/Inf、dtype、shape、device）
- 梯度传播：3 个测试（A/b 冻结、反向传播、无参数）
- 特征门控：5 个测试（allow/deny、shape 保持、batch）
- 授权边界：7 个测试（组件职责分离、类型验证）
- 完整 `tests/v2`：81/81 通过

**代码规模**：
- 实现：`src/can/v2/layers/gate_layer.py`（476 行）
- 测试：`tests/v2/test_gate_layer.py`（487 行）

**安全声明**：
- Toy LWE 参数（默认 n=128），无生产级密码学安全保证，可被最小二乘伪造
- Phase 1-2 不防御 replay 攻击
- 当前结果仅验证“LWE 验证可以编译为神经网络”的技术可行性

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

#### 1.3 Gated ResNet-18 [COMPLETED - CLAUDE ACCEPTED]

**文件**：`src/can/v2/models/gated_resnet.py`

设计内容：
- 使用 CIFAR ResNet-18 stem：3x3 stride=1，不使用 maxpool
- Gate Layer 位于 layer2 之后，输入 shallow_features [B,128,16,16]
- 训练模式返回两个完整 batch logits，不在模型内计算任务损失
- 训练模式仅让 valid 子批进入深层，避免 invalid 样本污染深层 BatchNorm；protected logits 的 invalid 行为零占位
- 推理模式按 allow mask 向量化拆分 batch，只让 valid 子批进入 layer3/layer4
- 推理输出携带 logits 对应的原 batch indices，不用 None 或压缩后丢失位置

规范接口与路由伪代码以 `docs/DESIGN_PROPOSALS.md` 的 Phase 1.3 Revision 1 为准。训练输出使用完整 batch 的 `TrainingOutput`，推理输出使用携带原 batch indices 的 `InferenceOutput`；空路由使用稳定二维空 Tensor，不使用 `None`。

**测试文件**：`tests/v2/test_gated_resnet.py`

测试要求：
- Valid credential → 深层执行，输出 fine-grained
- Invalid credential → 深层不执行，输出 coarse
- 形状正确性
- Forward/backward pass 无异常

**Codex 开发侧实现结果（2026-08-23）**：
- `BasicBlock` 与 CIFAR ResNet-18 `[2,2,2,2]` stage 已实现
- `TrainingOutput` 返回完整 batch logits；invalid protected 行为与浅层计算图相连的零占位
- `InferenceOutput` 返回稀疏 logits 和递增的原 batch indices
- 全 invalid batch 不调用 layer3、layer4 或 protected head
- mixed batch 在训练和推理时均只把 valid 子批送入深层，避免污染深层 BatchNorm
- 新增测试：25/25 通过；模型模块行覆盖率 99%
- 完整 `tests/v2`：106/106 通过
- Black、isort、`py_compile`、`git diff --check`：通过
- 环境：Python 3.11.8、PyTorch CPU；CUDA/GPU 路径未实测
- Claude 已于 2026-08-23 完成独立验收

---

### Phase 2: 训练流程 [IMPLEMENTED - OFFLINE VERIFIED]

**目标**：训练 Gated ResNet-18，使其具有能力分级

#### 2.1-2.4 Training Pipeline Revision 1 [IMPLEMENTED - OFFLINE VERIFIED]

规范来源：`docs/DESIGN_PROPOSALS.md` Phase 2 Revision 1。

设计内容：
- Phase 2.1：CIFAR-10/CIFAR-2 数据、固定 split 和 V_ref rejection-sampling credential
- Phase 2.2：masked protected CE + public coarse CE + 冻结 teacher KD
- Phase 2.3：Stage A protected baseline → Stage B public distillation → Stage C joint fine-tuning
- Phase 2.4：严格 YAML 配置、CLI、原子 checkpoint 和确定性恢复
- Teacher 固定为 Stage A best checkpoint 的冻结副本；Stage B/C 通过路径和 SHA-256 绑定，缺失或不一致时 fail fast
- 默认 epoch 上限：Stage A/B/C = 20/60/20，并分别使用 validation 指标 early stopping
- 验证严格使用 `InferenceOutput.protected_indices` / `public_indices` 对齐标签
- 单元测试不得联网；当前环境缺少 torchvision，实现前必须安装兼容版本并记录
- 已实现：`data.py`、`loss.py`、`metrics.py`、`trainer.py`、默认 YAML 和配置入口
- 当前环境：torchvision 未安装；离线 fake dataset 测试不依赖 torchvision

**实现与验证结果（2026-08-24）**：
- `tests/v2/test_training.py`：40/40 通过
- 完整 `tests/v2/`：146/146 通过
- 配置 dry-run：严格 schema、重复 key、未知字段和设备校验通过
- CPU offline smoke（`smoke_size=16`、`batch_size=4`）：A/B/C 各 1 epoch，loss 分别为 4.5726/0.3525/3.2549，三阶段 checkpoint、teacher 链接和 Stage C 约束路径通过
- 非法 smoke（`smoke_size=16`、`batch_size=128`）现在明确报错，不再以 `loss=None` 静默成功或写出空训练 checkpoint
- smoke 使用 synthetic CIFAR-like 数据；不代表 CIFAR-10 准确率或训练收敛结果
- 真实 CIFAR-10 训练尚未执行，原因是当前环境没有 torchvision 且不允许隐式联网下载
- Phase 2 training 模块覆盖率：约 86%（Trainer 85%、data 86%、loss 80%、metrics 88%；剩余主要为少量异常分支）
- 完整 V2 行覆盖率：90%，达到设计目标
- 新增 fake torchvision 适配测试：覆盖 transform 构造和 CIFAR-10 Dataset 初始化，不触发网络下载
- 验收修复：显式 keypair RNG、LWE/split metadata、Stage C fail-fast、valid>=2 约束、CLI `--resume`
- 空训练修复：脚本 smoke 参数预检、DataLoader 构建后检查和 trainer epoch 样本检查形成三层 fail-fast
- resume smoke 从 Stage C epoch 1 恢复到 epoch 2 成功

损失函数：
```
L_total = alpha * L_protected_masked
        + beta_ce * L_public_ce
        + beta_kd * T^2 * L_public_kd
```

Gate Layer 当前无可训练参数，不添加 `L_gate` 或 gate regularization。

**配置文件**：`configs/v2/train_gated_resnet18_cifar10.yaml`

默认训练超参数：
- Stage A/B/C epoch 上限：20/60/20；patience：5/10/5
- Batch size: 128
- Optimizer: SGD（lr=0.1；joint lr=0.01；momentum=0.9；weight_decay=5e-4）
- KD temperature: 4.0
- Stage C protected baseline 最大允许下降：0.03（绝对 accuracy）

**测试文件**：`tests/v2/test_training.py`

测试要求：
- 损失函数计算正确
- 训练态使用 `TrainingOutput.decision.allow` 对 protected logits/labels 做相同 mask
- 评估态使用 `InferenceOutput.protected_indices` / `public_indices` 对齐 logits、labels 和指标，禁止假设稀疏 logits 仍按完整 batch 排列
- 全 invalid batch 不对空 protected 目标调用 CrossEntropy，而是返回与图相连的零 loss
- 训练循环无异常
- Checkpoint 保存/加载正确

**当前限制**：真实 CIFAR-10、多种子、GPU 和正式指标仍待执行；Phase 2 training 子模块约 86%，完整 V2 行覆盖率已达到设计目标 90%。

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
- [x] **Phase 1.2: Neural Gate Layer 实现**（2026-08-23）
  - [x] `src/can/v2/layers/gate_layer.py` 实现完成
  - [x] `tests/v2/test_gate_layer.py` 43 个测试全部通过
  - [x] Gate Layer 行覆盖率 99%
  - [x] 完整 `tests/v2` 81 个测试全部通过
  - [x] black、isort 和差分测试通过
- [x] **Phase 1.3: Gated ResNet-18 Codex 开发实现**（2026-08-23）
  - [x] CIFAR ResNet-18、Gate Layer 和双 head 集成完成
  - [x] `TrainingOutput` / `InferenceOutput` 契约实现完成
  - [x] 全 invalid、mixed batch、fail-closed 和梯度测试完成
  - [x] 新增测试 25/25，模型覆盖率 99%，完整 V2 测试 106/106
  - [x] Claude 独立验收
- [x] **Phase 2.1-2.4: Training Pipeline Revision 1 实现**（2026-08-24）
  - [x] CIFAR-10/CIFAR-2 数据接口、固定 split 工具和 V_ref credential rejection sampling
  - [x] masked protected CE、public CE、冻结 teacher KD 和 all-invalid 图连接零 loss
  - [x] Stage A/B/C trainer、稀疏 indices 指标对齐、early stopping、Stage C protected 约束
  - [x] 严格 YAML/CLI、显式下载开关、原子 checkpoint 和 RNG 恢复
  - [x] 离线训练测试 40/40、完整 V2 测试 146/146、Phase 2 training 覆盖率约 86%、完整 V2 覆盖率 90%
  - [x] CPU 三阶段 smoke 通过；空 DataLoader 和不合法 smoke 配置已 fail fast

### 进行中
- [ ] **Phase 2 修复后的 Claude 复验与真实实验准备**

### 下一步（唯一下一步）

**先由 Claude 复验 resume、teacher/LWE/split 约束和覆盖率，再安装兼容 torchvision 运行真实 CIFAR-10 三阶段训练。**

审核重点：
1. torchvision 前置依赖、离线单元测试与数据划分
2. rejection sampling credential 生成和确定性 RNG
3. masked protected loss、public CE + KD 及三阶段训练策略
4. `InferenceOutput` indices 指标对齐、checkpoint 和多种子评估

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
   - **Phase 1-3（架构、训练与 CIFAR-10 评估原型）**：CIFAR-10（10→2 类）
     - 优势：快速迭代；正式耗时以当前硬件 smoke benchmark 为准
     - 劣势：能力差距不够显著
   - **Phase 4（能力分级扩展）**：CIFAR-100（100→20 类）
     - 优势：能力差距显著，有自带超类
     - 劣势：训练时间需要在 Phase 4 资源评估后确定
   - **Phase 5（可选）**：ImageNet（1000→100 类）
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
   - **缓解措施**：Phase 2 先建立 CIFAR-10 训练基线，Phase 4 再使用 CIFAR-100（100→20 类）验证更明显的能力分级

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
- **ImageNet 实验**：Phase 1-4 不考虑（Phase 5 可选）

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

### Gate Layer 实现 [COMPLETED 2026-08-23]

**开发侧环境**：
- Python 3.11.8
- PyTorch 2.13.0+cpu
- pytest 9.0.2
- CPU-only；本阶段未运行 GPU 测试

**开发侧测试结果（2026-08-23）**：
- Gate Layer：43/43 通过
- Gate Layer 行覆盖率：99%（190 statements，2 missed）
- 完整 `tests/v2`：81/81 通过
- 差分测试：推理 `decision.allow` / `gate_signal` 与 `V_ref()` 一致
- Mixed batch：valid、invalid、NaN/Inf 逐样本处理通过
- 训练/推理：软门控、硬门控和模式切换通过
- 梯度：`gated_features` 可向 `shallow_features` 回传梯度
- 格式：black 与 isort (`--profile black`) 检查通过

**覆盖率命令**：
```bash
pytest tests/v2/test_gate_layer.py -v --cov=src/can/v2/layers --cov-config=.coveragerc --cov-report=term-missing
```

**残余验证缺口**：
- CUDA/GPU device 路径尚未实测（当前环境仅 CPU）

---

### Gated ResNet-18 实验路线（与当前方案同步）

#### CIFAR-10 实验（Phase 2-3 原型验证）
- 训练状态：Phase 2 训练代码已实现；真实数据训练尚未运行
- Protected/Public accuracy：待 Phase 2 smoke test 和正式训练测量
- Logits 等价性与 latency：Phase 3 评估实验测量
- 选择规则：只使用 validation 指标选择 checkpoint，冻结后再评估官方 test set

#### CIFAR-100 实验（Phase 4 能力分级扩展）
- 状态：尚未开始，依赖 CIFAR-10 Phase 2-3 验证完成
- 任务：100 类 protected → 20 类超类 public
- 目标：验证比 CIFAR-10 更明显的能力分级；具体准确率不预先宣称

#### ImageNet 实验（Phase 5 可选）
- 状态：当前路线不执行
- 触发条件：CIFAR-100 结果和资源评估完成后重新审议

---

## 参考文献

- Shamir et al., "How to Securely Implement Cryptography in Deep Neural Networks"
- Regev, "On lattices, learning with errors, random linear codes, and cryptography" (LWE 原始论文)
- Knowledge Distillation: Hinton et al., "Distilling the Knowledge in a Neural Network"

---

## 提交历史

### Checkpoint: Neural Gate Layer Revision 5 [已就绪]

**Status**: Ready to commit

**Files to commit**:
- `.coveragerc`
- `src/can/v2/layers/gate_layer.py`
- `src/can/v2/layers/__init__.py`
- `tests/v2/test_gate_layer.py`
- `docs/DESIGN_PROPOSALS.md`
- `PROJECT_WORKLOG.md`

**验证结果**：
- `pytest tests/v2/test_gate_layer.py -q --cov=src/can/v2/layers --cov-config=.coveragerc --cov-report=term-missing`：43 passed，99% coverage
- `pytest tests/v2/ -q`：81 passed
- black、isort、`py_compile` 和 `git diff --check`：通过

**建议 commit message**：
```text
feat: implement neural gate layer

- compile toy LWE verification into batched PyTorch operations
- add structured evidence, authorization coordination, and feature gating
- support soft training gates and fail-closed hard inference gates
- add 43 gate-layer tests with 99% coverage
```

**历史记录**：该 checkpoint 生成时下一步为 Phase 1.3；当前路线已进入 Phase 2 训练流程。

---

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
- **Phase 1-2**：CIFAR-10（架构与训练原型，10→2 类）
- **Phase 4**：CIFAR-100（能力分级扩展，100→20 类，推荐用于论文）
- **Phase 5**：ImageNet（可选，大规模验证，1000→100 类，成本高）

**为什么选择这个顺序**：
1. CIFAR-10 快速验证架构可行性（1-2 天训练）
2. CIFAR-100 展示显著能力差距（自带超类，无需手动映射）
3. ImageNet 仅在前两者成功后考虑（论文投稿前）

**能力差距对比**：
- CIFAR-10：10 类(92%) → 2 类(65%)，差距 27%，但绝对类别数少
- CIFAR-100：100 类(75%) → 20 类(55%)，差距 20%，但类别数差距 5 倍，更有说服力
- ImageNet：1000 类(76%) → 100 类(50%)，差距最大，但训练成本高
