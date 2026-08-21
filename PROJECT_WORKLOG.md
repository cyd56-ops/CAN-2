# PROJECT_WORKLOG

## Current research phase

**Phase**: V2 - Gate Layer 在计算图中间架构  
**Status**: 初始化  
**Last updated**: 2026-08-21

---

## Research objective

实现模型内生安全：在神经网络中间嵌入基于 Module-SIS 的认证神经元层（Gate Layer），该层融合浅层特征与密码 credential 信息，根据验证结果控制深层神经元的激活。

### Core architecture

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

### Key design decisions

1. **Gate Layer 位置**：在浅层特征提取后、深层特征提取前
2. **密码方案**：Module-SIS（SIS over module lattice，toy profile）
3. **路由机制**：
   - 训练时：软路由（可微分，使用 sigmoid）
   - 推理时：硬路由（真正不执行深层，gate_signal ∈ {0, 1}）
4. **能力分级**：
   - Valid credential：fine-grained classification（深层 + protected head）
   - Invalid credential：coarse classification（公开 head，通过知识蒸馏训练）
5. **Baseline 模型**：ResNet-18 on CIFAR-10（后续扩展到 ImageNet 子集）

---

## Implementation roadmap

### Phase 1: 基础架构搭建 [CURRENT]

**目标**：实现 Gate Layer 在计算图中间的基本架构

#### 1.1 Module-SIS 密码原语 [TODO]

**文件**：`src/can/v2/crypto/module_sis.py`

实现内容：
- `ModuleSISParams`：参数类（n, q, d, σ, bound）
- `generate_keypair()`：生成 (A, secret_key)
- `sign(message, secret_key, A)`：生成签名 z
- `verify(message, signature, A, bound)`：验证 Az ≈ H(m) mod q
- `V_ref(credential) → {0, 1}`：参考验证器

测试要求：
- 正向：valid signature → verify = True
- 负向：篡改 message/signature → verify = False
- 边界值：bound 边界测试
- 差分测试：与 toy profile 规范一致

**测试文件**：`tests/v2/test_module_sis.py`

#### 1.2 Neural Gate Layer [TODO]

**文件**：`src/can/v2/layers/gate_layer.py`

实现内容：
- `GateLayer(nn.Module)`：
  - 输入：`(shallow_features, credential)`
  - 内部：特征融合 + Module-SIS 验证的神经编译
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

### Phase 3: 评估实验 [LATER]

**目标**：验证 Gate Layer 的功能正确性和能力分级

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

## Current status

### Completed
- [x] 项目文档初始化（AGENTS.md, SECURITY.md, README.md, PROJECT_WORKLOG.md）
- [x] Git 仓库初始化

### In progress
- [ ] 无

### Next step (唯一下一步)

**创建 V2 目录结构并实现 Module-SIS 密码原语**

具体任务：
1. 创建目录结构：
   ```
   src/can/v2/
   ├── __init__.py
   ├── crypto/
   │   ├── __init__.py
   │   └── module_sis.py
   ├── layers/
   │   ├── __init__.py
   │   └── gate_layer.py
   ├── models/
   │   ├── __init__.py
   │   └── gated_resnet.py
   ├── training/
   │   ├── __init__.py
   │   ├── distillation.py
   │   └── train_gated.py
   └── experiments/
       ├── __init__.py
       ├── functional_test.py
       ├── capability_tiering.py
       └── performance.py
   
   tests/v2/
   ├── __init__.py
   ├── test_module_sis.py
   ├── test_gate_layer.py
   ├── test_gated_resnet.py
   └── test_training.py
   
   configs/v2/
   └── train_gated_resnet18_cifar10.yaml
   ```

2. 实现 `src/can/v2/crypto/module_sis.py`（参考 Phase 1.1）
3. 实现 `tests/v2/test_module_sis.py`
4. 运行测试：`pytest tests/v2/test_module_sis.py -v`
5. 更新本工作日志

---

## Open questions

1. **Coarse labels 如何定义？**
   - CIFAR-10 → CIFAR-2（animal vs vehicle）？
   - 还是随机聚类？
   - 决定：使用语义聚类（animal/vehicle）

2. **Gate Layer 的神经编译细节？**
   - 如何将 Module-SIS 验证嵌入可微分计算？
   - 决定：使用可学习的 embedding + 近似验证逻辑（训练时），精确验证（推理时）

3. **训练收敛性？**
   - 软路由和硬路由的切换是否影响性能？
   - 需要在 Phase 2 中实验验证

---

## Risks and limitations

### Current phase risks

1. **Module-SIS 神经编译可行性**：未验证密码验证逻辑是否能有效编译为神经层
2. **训练收敛**：软路由 → 硬路由切换可能导致性能下降
3. **能力分级效果**：Public head 能力是否真的弱于 protected head

### Explicit non-goals (current phase)

- **白盒攻击防御**：当前阶段不考虑
- **TEE 部署**：后续阶段
- **密码学安全归约**：Toy profile 不保证
- **生产部署**：研究原型阶段

---

## Experimental results

### Module-SIS 实现（待完成）
- 测试状态：未运行
- 差分测试：未运行
- 性能：未测量

### Gate Layer 实现（待完成）
- 测试状态：未运行
- 差分测试：未运行
- 软/硬路由切换：未验证

### Gated ResNet-18（待完成）
- 训练状态：未开始
- Protected accuracy：未测量
- Public accuracy：未测量
- Logits 等价性：未验证
- Latency：未测量

---

## Bibliography

- Shamir et al., "How to Securely Implement Cryptography in Deep Neural Networks"
- Module-SIS: "Lattice-Based Signatures via Module-SIS"
- Knowledge Distillation: Hinton et al., "Distilling the Knowledge in a Neural Network"

---

## Commit history

### Checkpoint: Initial project setup [READY]

**Status**: Ready to commit

**Files to commit**:
- `AGENTS.md`
- `SECURITY.md`
- `README.md`
- `PROJECT_WORKLOG.md`
- `.gitignore`

**Commit message**:
```
Initial project setup: V2 Gate Layer architecture

- Add project governance docs (AGENTS.md, SECURITY.md)
- Add research roadmap (PROJECT_WORKLOG.md)
- Define V2 architecture: Gate Layer in computation graph
- Roadmap: Module-SIS → Gate Layer → Gated ResNet-18 → Training → Evaluation

Scope: Research prototype, white-box defense out of scope
```

**Next**: Implement Module-SIS primitive

---

## Notes

- 本文档是唯一的动态工作日志，每次实现后必须更新
- 所有测试结果（pass/fail/skip）必须记录在此
- 所有设计决策和风险必须记录在此
- 保持"唯一下一步"明确且可执行
