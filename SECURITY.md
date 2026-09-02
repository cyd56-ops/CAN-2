# Security

## Status and scope

本项目是防御性科研原型。本文档描述研究范围内的信任模型、安全不变量和明确不保证的性质，不构成生产安全保证。

研究对象：`CAN: Cryptographic Authentication Neural Gate Layer`。

当前研究重点：实现 Gate Layer 在计算图中间的架构，验证固定的 toy LWE-inspired
关系判定可以编译为神经网络运算，并在可信部署边界内控制能力路由。
**白盒攻击防御不在当前阶段范围内**。

所有 toy、实验性、单机、小参数或有限测试结论都必须明确标注，不得包装成生产访问控制或跨设备保证。

## Threat model

### TM-API（当前正向保证适用的模型）

攻击者可以无限次提交任意格式的 `(image, credential)`，并观察服务返回的能力结果；
攻击者不持有模型权重，不能修改进程内存、计算图或直接调用内部模块。模型权重、推理代码、
协调器和部署入口属于可信组件。

`TM-API` 的边界是服务层 response envelope，而不是原始 PyTorch `InferenceOutput`。
原始模型输出中的 `decision`、`error_norm`、`reason_code`、`verified`、`gate_signal` 和
路由 indices 只允许测试 evaluator 使用，不得直接暴露给调用方。Phase 3.6 的 envelope
尚未实现，当前 C-003/C-006 的 API 语义仍只在模型层成立。

### TM-WB（明确不主张抗性的模型）

攻击者持有 checkpoint 与运行时，可以插入 hook、修改张量或直接调用内部方法。当前实现对此
不提供任何安全保证：credential 只影响控制流，不使 protected 权重失效。攻击者可以直接调用
受保护内部路径，或进行常数规模运行时篡改绕过 Gate；不得将该结论简化为“单次赋值”，也不提供
针对第三方模型的可迁移绕过 PoC。

### TM-NA

与攻击者无关的实现正确性和工程属性，例如 PyTorch 与参考验证器的差分一致性、索引对齐和
确定性测试。

### Trusted components

- 模型权重和推理代码（部署在可信环境）
- LWE 验证参数和算法实现
- 训练流程和 checkpoint
- Gate Layer 的神经编译正确性

## Required security invariants (current phase)

- 不可信 credential 输入先规范化解析，再进入固定 toy LWE-inspired 关系验证门。
- Gate Layer 在推理模式下实现 fail-closed：invalid credential 时 `layer3`、`layer4` 和
  protected head 零调用。
- 当前 CIFAR-10 模型对 invalid credential 路由到 2 类 public head；这表示能力受限，
  不表示验证失败具有密码学授权语义。服务层的固定 envelope 尚待 Phase 3.6 实现。
- 训练时使用软路由（可微分），推理时使用硬路由（真正不执行深层）。
- toy LWE-inspired 关系编译的正确性通过差分测试验证（模型判定与 `V_ref()` 一致）。

## Explicitly unsupported guarantees (current phase)

除非后续研究阶段另行冻结并有对应证据，当前阶段不保证：

- **白盒攻击防御**：攻击者读取、修改、删层、剪枝、微调或替换模型/推理代码后的安全性；
- **TEE/安全启动**：可信执行环境、远程证明、宿主机控制或完整侧信道防护；
- **密码学安全归约**：toy LWE-inspired profile (n=128) 不等同生产级参数或已证明困难性假设，
  可被最小二乘伪造；
- **Replay 攻击防御**：当前使用静态 credential，不维护状态，不检测重放；未来若研究
  Challenge-Response 或 nonce，必须作为独立协议重新评审，不能视为当前路线承诺；
- **签名不可伪造性、身份认证和 access-control soundness**：当前关系不提供这些性质；
- **生产部署安全**：有限随机测试不替代全域 soundness、形式证明或密码学安全归约；
- **跨设备/分布式安全**：单机实验结果不外推到生产、其他硬件、其他数据分布或任意部署方式。

**论文主张限定**：
- 证明 Gate Layer 可以在计算图中间实现（架构可行性）
- 证明 toy LWE-inspired 关系判定可以编译为神经层（技术可行性）
- 证明 fail-closed 路由可以在推理时生效（功能正确性）
- **不声称**相对外部验证器有安全优势，也不声称模型本身具有白盒安全性。

## Security testing requirements (current phase)

测试至少覆盖：
- **输入验证**：合法 credential、格式错误、篡改、边界值
- **Gate Layer 行为**：
  - Valid credential → gate_signal 高（> 0.7）
  - Invalid credential → gate_signal 低（< 0.3）
  - 差分测试：模型判定 == `V_ref()`（toy 关系编译正确性）
- **Fail-closed 验证**：
  - Invalid credential → 深层调用计数 = 0
  - Invalid credential → 仅输出公开能力（coarse classification）
- **能力分级**：
  - Valid → fine-grained accuracy ≈ 深层 baseline
  - Invalid → coarse accuracy ≈ 公开模型
- **Logits 等价性**：valid credential 输出与深层 direct 输出一致
- **服务层脱敏**：Phase 3.6 实现后，envelope 不含验证量、连续距离、reason code、内部特征或路由 indices，
  且 valid/invalid 返回结构逐样本同构。

FAR/FRR 只表示固定 toy 采样分布下的实现正确性观测，不是密码学安全指标。
训练权重在 `A`、`b`、阈值、credential generator、规范化、dtype 和设备均冻结时不改变 Gate 判定；
跨 Stage 测量仅用于配置回归检查。

每个新增功能都必须在对应测试文件中标注其测试覆盖的安全性质。
