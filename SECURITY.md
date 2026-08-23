# Security

## Status and scope

本项目是防御性科研原型。本文档描述研究范围内的信任模型、安全不变量和明确不保证的性质，不构成生产安全保证。

研究对象：`CAN: Cryptographic Authentication Neural Gate Layer`。

当前研究重点：实现 Gate Layer 在计算图中间的架构，验证密码验证逻辑可以嵌入神经网络。**白盒攻击防御不在当前阶段范围内**（后续通过 TEE 或服务器端部署解决）。

所有 toy、实验性、单机、小参数或有限测试结论都必须明确标注，不得包装成生产访问控制或跨设备保证。

## Threat model

### Current phase scope (V2)

当前阶段重点验证"Gate Layer 在计算图中间"的可行性，攻击者模型限定为：

**攻击者可以：**
- 提交任意格式的 (image, credential) 输入
- 尝试伪造、篡改、重放 credential
- 观察模型输出（分类结果、置信度）
- 测量推理延迟

**攻击者不能（当前阶段不防御）：**
- 读取模型权重（白盒攻击）
- 修改模型代码或推理图
- 直接调用深层模块
- 侧信道攻击（时序、功耗等）

### Trusted components

- 模型权重和推理代码（部署在可信环境）
- LWE 验证参数和算法实现
- 训练流程和 checkpoint
- Gate Layer 的神经编译正确性

## Required security invariants (current phase)

- 不可信 credential 输入先规范化解析，再进入 Gate Layer。
- Gate Layer 在推理模式下实现 fail-closed：invalid credential 时深层神经元零调用。
- credential 验证失败、格式错误、篡改时默认路由到公开 head（弱化能力）。
- 训练时使用软路由（可微分），推理时使用硬路由（真正不执行深层）。
- LWE 神经编译的正确性通过差分测试验证（GateLayer.verify() == V_ref()）。

## Explicitly unsupported guarantees (current phase)

除非后续研究阶段另行冻结并有对应证据，当前阶段不保证：

- **白盒攻击防御**：攻击者读取、修改、删层、剪枝、微调或替换模型/推理代码后的安全性；
- **TEE/安全启动**：可信执行环境、远程证明、宿主机控制或完整侧信道防护；
- **密码学安全归约**：toy LWE profile (n=128) 不等同生产级参数或已证明困难性假设；
- **生产部署安全**：有限随机测试不替代全域 soundness、形式证明或密码学安全归约；
- **跨设备/分布式安全**：单机实验结果不外推到生产、其他硬件、其他数据分布或任意部署方式。

**论文主张限定**：
- 证明 Gate Layer 可以在计算图中间实现（架构可行性）
- 证明密码验证逻辑可以编译为神经层（技术可行性）
- 证明 fail-closed 路由可以在推理时生效（功能正确性）
- **不声称**相对外部验证器有安全优势（当前重点是实现设想，而非安全性比较）

## Security testing requirements (current phase)

测试至少覆盖：
- **输入验证**：合法 credential、格式错误、篡改、边界值
- **Gate Layer 行为**：
  - Valid credential → gate_signal 高（> 0.7）
  - Invalid credential → gate_signal 低（< 0.3）
  - 差分测试：GateLayer.verify() == V_ref()（LWE 验证正确性）
- **Fail-closed 验证**：
  - Invalid credential → 深层调用计数 = 0
  - Invalid credential → 仅输出公开能力（coarse classification）
- **能力分级**：
  - Valid → fine-grained accuracy ≈ 深层 baseline
  - Invalid → coarse accuracy ≈ 公开模型
- **Logits 等价性**：valid credential 输出与深层 direct 输出一致

每个新增功能都必须在对应测试文件中标注其测试覆盖的安全性质。
