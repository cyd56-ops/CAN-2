# Security

## Status and scope

本项目是防御性科研原型。本文档描述研究范围内的信任模型、安全不变量和明确不保证的性质，不构成生产安全保证。

研究对象：`CAN: Cryptographic Authentication Neural Gate Layer`。

当前研究重点：实现 Gate Layer 在计算图中间的架构，验证固定的 toy LWE-inspired
关系判定可以编译为神经网络运算，并在可信部署边界内控制能力路由。
**白盒攻击防御不在当前阶段范围内**。

截至 2026-09-10，Phase 3/3.6 的 CIFAR 实现已验收；Revision 2 的 G0/P0/P1 v1.2
设计已由用户确认完成审阅，尚未实现或运行。以下新路径条款是实施要求，不是已获得的安全证据。
接口和验收矩阵见 [实现设计包](docs/GATE_PRETRAINED_G0_P0_P1_IMPLEMENTATION_PLAN.md)，
动态状态以 [工作日志](PROJECT_WORKLOG.md) 为准。

所有 toy、实验性、单机、小参数或有限测试结论都必须明确标注，不得包装成生产访问控制或跨设备保证。

## Threat model

### TM-API（当前正向保证适用的模型）

攻击者可以无限次提交任意格式的业务输入与 credential（CIFAR 为 image，新语言路径为 prompt/input IDs），并观察服务返回的能力结果；
攻击者不持有模型权重，不能修改进程内存、计算图或直接调用内部模块。模型权重、推理代码、
协调器和部署入口属于可信组件。

`TM-API` 的边界是服务层 response envelope，而不是原始 PyTorch `InferenceOutput`。
原始模型输出中的 `decision`、`error_norm`、`reason_code`、`verified`、`gate_signal` 和
路由 indices 只允许受信内部组件和测试 evaluator 使用，不得直接暴露给调用方。Phase 3.6
的 CIFAR envelope 已实现并通过验收，C-003/C-006 的服务语义限定在该可信进程内入口。
固定 10 槽 probabilities 和 capability level 是预期可观察输出，不主张隐藏类别结构或模型探测面。
该入口不是 HTTP/gRPC 协议；预训练语言路径的 wire schema 尚未设计/实现，内部
`RoutedGeneration`、raw logits、cache 和 evidence 不得直接晋升为外部响应。

### TM-WB（明确不主张抗性的模型）

攻击者持有 checkpoint 与运行时，可以插入 hook、修改张量或直接调用内部方法。当前实现对此
不提供任何安全保证：credential 只影响控制流，不使 protected 权重失效。攻击者可以直接调用
受保护内部路径，或进行常数规模运行时篡改绕过 Gate；不得将该结论简化为“单次赋值”，也不提供
针对第三方模型的可迁移绕过 PoC。

新路径的私有 object seal、不可变授权 tuple、policy/config/request 绑定只用于可信进程内
来源检查及防止接口误用。它们不抵抗反射、内存修改或运行时控制，不能据此主张权重保密或白盒抗性。

### TM-NA

与攻击者无关的实现正确性和工程属性，例如 PyTorch 与参考验证器的差分一致性、索引对齐和
确定性测试。

### Trusted components

- 模型权重和推理代码（部署在可信环境）
- 规范解析器、冻结关系参数、verifier、唯一协调器与 policy
- dispatcher、host plugin、请求/cache 生命周期管理，以及固定的依赖/后端
- 训练流程和 checkpoint
- Gate Layer 的神经编译正确性

## Required security invariants (current phase)

### 已有 CIFAR 路径

- 不可信 credential 输入先规范化解析，再进入固定 toy LWE-inspired 关系验证门。
- Gate Layer 在推理模式下实现 fail-closed：invalid credential 时 `layer3`、`layer4` 和
  protected head 零调用。
- 当前 CIFAR-10 模型对 invalid credential 路由到 2 类 public head；这表示能力受限，
  不表示验证失败具有密码学授权语义。Phase 3.6 的固定 envelope 已交付。
- CIFAR/T0 系列的历史协议为训练软路由、推理硬路由（真正不执行深层）。
- toy LWE-inspired 关系编译的正确性通过差分测试验证（模型判定与 `V_ref()` 一致）。

### Revision 2 固定认证路径（待实现）

verifier 只产 evidence，只依赖规范 credential 和冻结关系配置，不读取 token、hidden 或 logits；
唯一协调器提交 route，dispatcher 负责实际执行。固定认证不随训练/推理模式改变。
合法 hidden 恒等通过，原宿主权重在 P1 全冻结。采用独立且由服务端绑定的策略：

| 策略 | 关系通过 | 规范但关系失败 | 格式或数值错误 |
|---|---|---|---|
| P1 `p1-protected-or-deny-v1` | PROTECTED | DENY | DENY/请求级失败 |
| P2 `p2-capability-routing-v1`（未来） | PROTECTED | PUBLIC | DENY/请求级失败 |

未知 dtype/rank/维度、batch 不匹配或空 batch 使请求整批失败；形状合法时可定位的 NaN/Inf
逐行拒绝，不能进入关系矩阵运算。有限输入的计算溢出也逐行 DENY，不能映射 PUBLIC。
规范域和 `< threshold` 边界以 `toy-real-fp32-v1` 为准；关系失败不等于攻击被密码学识别。
P1 DENY 不调用 suffix、原 final norm/lm_head 或 public 分支，不建立 public/suffix cache。
P2 的 PUBLIC 是预配置业务路径，学习式生成 ACCESS-DENIED 也不等于执行层 DENY。

G 首次 prefill 在输入预检后先执行 prefix，再在图中间验证并提交授权；request ID 可先生成，
它本身不授予权限。结构错误可在 prefix 前拒绝，规范关系失败的 prefix 成本据实记录。
同构外部对照 E 可提前拒绝并省去 prefix，不要求 G/E 的 DENY prefix 调用数相同。
两者共用验证算法、policy 和 dispatcher，不预设模型内位置带来额外安全或性能优势。

生成只提交一次 route；增量步在任何 prefix/suffix 计算或 cache 更新前核查全部活动请求的
既有 route、请求身份、模型/配置/policy/cut、cache shape/位置/有效长度和生命周期。
不以 cache 自述身份替代受信请求状态；外部不能提交 route、policy 或 cache。绑定失败记为
`cache_state_validation_failed` / `decode_preflight`，本步 prefix/suffix/head 零调用，
此前真实调用累计保留。拒绝或已结束请求不能继续，DENY 的 prefix 临时状态及时释放。

首轮非流式 batch 是结果提交单元。正常逐行 DENY 可与合法结果共存；意外 parser/verifier/
coordinator/执行异常及增量状态校验失败使整批终止、清理状态，不返回 partial、不自动重试，
也不切换 PUBLIC/DENY 伪装成功。已经执行的计算无法回滚，内部记录真实阶段与计数，对外错误脱敏。

## Explicitly unsupported guarantees (current phase)

除非后续研究阶段另行冻结并有对应证据，当前阶段不保证：

- **白盒攻击防御**：攻击者读取、修改、删层、剪枝、微调或替换模型/推理代码后的安全性；
- **TEE/安全启动**：可信执行环境、远程证明、宿主机控制或完整侧信道防护；
- **密码学安全归约**：toy LWE-inspired profile (n=128) 不等同生产级参数或已证明困难性假设，
  可被最小二乘伪造；
- **Replay 攻击防御**：历史 toy/静态 credential 尚无防重放证据。后续路线必须参考 NCS 的 stateful authorization 思路，设计并评估 Challenge-Response 或 nonce/计数器、request binding、一次性消费、撤销和并发原子性；请求/cache 生命周期检查不能替代 credential 新鲜性验证。在该里程碑完成前不得宣称已具备防重放性质；
- **签名不可伪造性、身份认证和 access-control soundness**：当前关系不提供这些性质；
- **生产部署安全**：有限随机测试不替代全域 soundness、形式证明或密码学安全归约；
- **跨设备/分布式安全**：单机实验结果不外推到生产、其他硬件、其他数据分布或任意部署方式。

**论文主张限定**：
- 证明 Gate Layer 可以在计算图中间实现（架构可行性）
- 证明 toy LWE-inspired 关系判定可以编译为神经层（技术可行性）
- 证明 fail-closed 路由可以在推理时生效（功能正确性）
- **不声称**相对外部验证器有安全优势，也不声称模型本身具有白盒安全性。

## Security testing requirements (current phase)

已有 CIFAR 回归至少覆盖：
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
- **服务层脱敏**：Phase 3.6 已实现的 envelope 不含验证量、连续距离、reason code、内部特征或路由 indices，
  且 valid/invalid 返回结构逐样本同构。

FAR/FRR 只表示固定 toy 采样分布下的实现正确性观测，不是密码学安全指标。
训练权重在 `A`、`b`、阈值、credential generator、规范化、dtype 和设备均冻结时不改变 Gate 判定；
跨 Stage 测量仅用于配置回归检查。

每个新增功能都必须在对应测试文件中标注其测试覆盖的安全性质。

G0/P1 新测试依据实现包 §5–§7：认证 reference 判决逐行一致且不受 hidden 或 `train()/eval()`
影响；seal/来源/完整请求顺序绑定；实际模块调用及参与样本身份；mixed 索引和 cache 隔离；
在成功 prefill 后注入错误 cache，验证失败步零调用且历史计数保留；整批异常不输出部分结果。
H 校准及 H/S、S/G、G/E 按冻结配置比较全部有效位置 logits、greedy tokens 和停止位置。
业务 logits 工程容差不能用于认证判决；校准失败仅否定该配置，不代表所有宿主均失败。
P2 公共效用与 protected 能力保持分别验收，不能以蒸馏或有限越界测试保证能力隔离或知识保密。
