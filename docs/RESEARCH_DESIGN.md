# Research Design

## 1. Research question

### Primary question

<用一句可证伪的问题描述研究目标。>

### Hypotheses

- H1：<主要假设>
- H2：<对照或替代假设>

### Contribution boundary

<明确本项目相对已有方法新增什么，哪些内容只是工程组合或复现。>

## 2. Scope and terminology

### In scope

- <研究对象>
- <形式化关系、算法或模型>
- <实验和可复现环境>

### Out of scope

- <不研究的能力>
- <不支持的安全主张>

### Terminology

| Term | Definition | Forbidden interpretation |
| --- | --- | --- |
| `<term>` | <精确定义> | <不能声称什么> |

## 3. System boundary

```text
untrusted input
-> canonical parser
-> deterministic reference/verifier
-> evidence-only result
-> single coordinator
-> protected model/tool side effect
```

说明每个边界的可信性、输入输出、失败语义和可观测结果。验证器不直接产生授权；只有协调器提交最终决定。

## 4. Formal obligations

- 输入域和 canonical encoding：<定义>
- reference relation/oracle：<定义>
- completeness：<需要证明或明确不主张>
- soundness-preservation：<需要证明或明确限定>
- 数值误差/量化预算：<定义>
- replay、tamper、route confusion 和 failure side effects：<验收条件>

有限随机测试不能替代全域证明、穷举、形式方法或密码学归约；每项结论必须写明证据类型。

## 5. Experimental protocol

| Item | Frozen decision |
| --- | --- |
| Dataset/input | <来源、摘要、许可和切分> |
| Model/system | <拓扑、版本和参数> |
| Environment | <OS、Python、依赖、硬件> |
| Seeds/repeats | <确定性设置> |
| Metrics | <准确率、等价性、延迟、调用计数等> |
| Artifacts | <生成位置、摘要、忽略规则和保留期限> |
| Stop conditions | <失败或异常时停止条件> |

实验结果必须与代码版本、配置、数据摘要和环境 tuple 绑定，不把未执行的结果写成已验收结论。

## 6. Research stages

| Stage | Objective | Exit criteria | Status |
| --- | --- | --- | --- |
| S0 | 精确定义和最小 oracle/reference | 输入域、关系和负向测试闭合 | pending |
| S1 | 最小可验证实现 | 单元、差分和安全测试通过 | pending |
| S2 | 受控实验/系统组合 | 预注册指标和失败矩阵通过 | pending |
| S3 | 论文证据和限制 | 主张、证据、残余风险一致 | pending |

后续阶段不得覆盖前序路线；跨阶段只复用无协议语义的通用 helper，并保持入口和接受集合隔离。

## 7. Claim and evidence ledger

### 威胁模型标签

每条主张必须标注其成立的威胁模型，禁止跨模型引用。

| 标签 | 攻击者能力 | 说明 |
| --- | --- | --- |
| `TM-API` | 持有 API 访问权，可任意构造模型输入与 credential；**不持有权重**，不能修改进程内存或计算图 | Phase 1-3 以及 Phase 5 服务入口的黑盒威胁模型 |
| `TM-REP` | 仅在受信评估环境中取得冻结 checkpoint 的指定中间表示样本；不能修改权重、运行时或直接调用 protected 路径 | Phase 5 表示泄漏探针的实验模型；不是对外部署接口 |
| `TM-CP` | 取得公开分发的 checkpoint 文件，并在预注册的离线数据、步骤和计算预算内训练恢复模型；不获得训练密钥、服务端运行时或内部调用权限 | Phase 5 checkpoint 恢复实验；若可插 hook、改运行时或直接调用内部路径则升级为 `TM-WB` |
| `TM-WB` | 持有 checkpoint 与运行时，可插 hook、改张量、直接调用内部方法 | **当前不主张任何抗性** |
| `TM-NA` | 与攻击者无关的实现正确性或工程属性 | 不承担安全语义 |

### 证据类型标签

`proof`=形式证明或结构性论证；`unit`=单元/差分测试；`exp-val`=validation split 实验；`exp-test`=官方 test split 实验；`analytic`=解析计算；`none`=尚无证据。

### 主张台账

| Claim ID | Claim | 威胁模型 | Required evidence | 证据类型 | Current status |
| --- | --- | --- | --- | --- | --- |
| C-001 | toy LWE 验证可编译为批量可微 PyTorch 运算，推理态判定与 NumPy `V_ref` 逐样本一致 | `TM-NA` | `tests/v2/test_lwe.py` 差分测试 + `tests/v2/test_gate_layer.py` 推理态 `allow`/`gate_signal` 一致性 | unit | **satisfied**（38/38 + 43/43，CPU；CUDA 路径未实测） |
| C-002 | 该 toy 参数下 valid 与 invalid 的误差范数分布可由固定阈值分离 | `TM-NA` | 误差分布测量 + 假阳性率采样 | unit | **partial**：valid ~16 / invalid ~900 / threshold=48，100 次采样 FP=0；**缺** ≥1000 次采样与置信区间 |
| C-003 | 推理态 invalid credential 不执行 layer3 / layer4 / protected head（fail-closed 控制流） | `TM-API` | forward hook 深层调用计数 = 0，覆盖全 invalid 与 mixed batch | unit | **satisfied（可信服务入口）**：模型层 hook 测试证明 invalid 深层零调用；`InferenceService` 只通过该模型入口执行真实 credential，并由 C-013 的脱敏边界约束返回 |
| C-004 | 稀疏输出的 `protected_indices` / `public_indices` 与原 batch 位置严格对齐 | `TM-NA` | 索引对齐测试 + 训练/评估指标对齐测试 | unit | **satisfied**（146/146 全量 V2 通过） |
| C-005 | **在 A/b、error_threshold、credential generator、输入规范化、dtype 与设备配置均冻结的前提下**，训练模型权重不改变 Gate 判定，因此 FAR/FRR 跨 Stage 不变 | `TM-API` | `A`/`b` 以 buffer 注册且不在 optimizer 中；验证链对图像特征零依赖；Stage A/B/C 实测 FAR/FRR 一致 | proof + none | **partial**：在上述冻结前提下结构性论证成立（`register_buffer`，`residual = b - credential @ A.T` 不含特征项）；**缺** 跨 stage 实测。跨 stage FAR/FRR 的用途是**配置回归检查**（检测 dtype / 设备 / generator 被意外改动），**不是**学习稳定性的证据 |
| C-006 | 训练后 valid credential 获得 10 类细粒度能力，invalid credential 仅获得 2 类粗粒度能力 | `TM-API` | 官方 test split 上的 protected / public accuracy，3 seed mean ± std | exp-val + unit | **satisfied（可信服务入口）**：三个 Stage C best checkpoint 的官方 test split protected/public accuracy 为 `0.89157 ± 0.01366` / `0.96427 ± 0.00141`；服务测试验证 valid/public capability namespace、全 valid、全 invalid 与 mixed batch |
| C-007 | 与"外部验证器 + 完整模型"相比，门控的推理开销可接受 | `TM-NA` | latency / 吞吐 / 显存对照测量 | none | **pending**（Phase 3） |
| C-008 | `capability_gap_fine` 的无授权基线是解析随机猜测值，**不是任意攻击者的能力上界** | `TM-API` | 解析计算 + 字段标注 `is_analytic: true` | analytic | **satisfied**（已在 `DESIGN_PROPOSALS.md` 冻结命名与标注） |
| C-009 | **反向主张**：`TM-WB` 下攻击者可通过直接调用受保护内部路径、或**常数规模**的运行时篡改绕过控制流，且不承担能力损失 | `TM-WB` | 解析论证（`gated_resnet.py:270` 可直接调用 `_forward_protected`；`gated_resnet.py:303-305` 路由由 `gated_features` 与 `decision.allow` 共同决定）+ 已有测试 `test_valid_logits_match_direct_protected_path` 证明 direct-path 输出与 valid 路径等价 | analytic + unit | **satisfied**（已确立的局限，非待改进项）。**不得写成"单次赋值"**：仅翻转 `decision.allow` 时 `gated_features` 已被 `gate_signal` 清零，送入 protected 路径得到的是常量 logits 而非恢复的语义；绕过至少需同时处理 gate_signal 与 allow，或整体绕开路由 |
| C-010 | **不主张** 抵抗 replay：credential 静态可重用 | `TM-API` | 设计声明 | proof | **declared**（`DESIGN_PROPOSALS.md:85`） |
| C-011 | **不主张** 密码学安全性：toy 参数（n=128）可被最小二乘伪造，无安全归约；原始 `InferenceOutput` 仍向可信 evaluator 提供连续 `error_norm` | `TM-API` | 设计声明 + 原始接口事实 + 服务脱敏测试 | proof + unit | **satisfied（限定泄露面收敛）**：可信服务入口不返回 `error_norm`、reason code 或异常链；原始 evaluator 仍可访问完整证据，且 probabilities 仍构成一般模型探测面，不主张密码学安全或消除全部侧信息 |
| C-013 | 服务层 response envelope 不向调用方泄露额外验证证据或内部路由证据 | `TM-API` | envelope 实现 + 脱敏测试：剥离 `decision`（含 `gate_signal`、`evidence.error_norm`、`reason_code`、`verified`、`indices`）；每样本一条记录，字段集合与固定概率 shape 完全一致；`capability_level`、分类 probabilities 及其固定 10 槽位属于预期公开的能力/架构可观察性，不视为额外 credential 验证证据 | unit | **satisfied（可信进程内适配入口）**：30 项 Phase 3.6 专项测试、service 98% 行覆盖率；正常与错误返回均不暴露内部证据或异常链。明确接受 10 槽位结构可观察性，不主张隐藏类别规模、模型探测面、同进程旁路或网络 wire schema 安全 |
| C-014 | protected 路径能力显著高于同构无 Gate ResNet-18 baseline（能力分级未以牺牲绝对性能换取） | `TM-API` | 独立训练的 no-Gate 同构 baseline + test split 对照 | none | **pending（未来消融实验）**：仓库中**不存在**独立训练的 no-Gate baseline，需一次完整训练。**与 Phase 2 配置中的 `Stage C protected baseline 最大允许下降 0.03` 无关** —— 后者的 baseline 指 Stage A protected accuracy，是已实现的 fail-fast 约束 |
| C-012 | 能力差距在类别数更多的任务上更显著 | `TM-API` | CIFAR-100（100→20）与 CIFAR-10（10→2）对照实验 | none | **pending**（Phase 4；当前不预先宣称具体数值） |
| C-015 | Phase 5 Transformer 的 Gate 判决只依赖规范化 credential 与冻结的 LWE 公共参数；prompt、token 和 hidden state 不改变接受集合 | `TM-NA` | reference verifier 差分测试、hidden-state/文本扰动不变性测试、跨 Stage FAR/FRR 配置回归 | unit + analytic | **pending（Phase 5）**：设计已要求 hidden state 只能被 gate signal 门控，尚无实现证据；不得用 CIFAR 的 C-005 代替 |
| C-016 | Phase 5 的 valid credential 获得达到绝对下限的 protected 能力，invalid credential 获得达到门槛的 public 能力并对 private query 稳定拒答或限定公开范围 | `TM-API` | T-pretrain go/no-go；3 seed test 的 protected/public 规范化答案 exact match、token accuracy、private refusal 与 public-scope compliance；服务 schema 测试 | exp-val + exp-test + unit | **pending（Phase 5）**：validation 门槛须在 test 前冻结；direct-reference 等价性不能替代 protected utility 绝对下限 |
| C-017 | mixed 自回归生成中每条序列只提交一次 route，invalid 序列对 protected blocks 为 zero-call，稀疏索引与各分支 KV-cache 无串扰 | `TM-NA` | routed/reference 逐 token 对照、KV-cache 长度与索引检查、全 valid/invalid/mixed 和不同停止长度测试 | unit | **pending（Phase 5）** |
| C-018 | 给定指定表示层、训练样本数、probe 类别和实体切分，shared prefix 对 private scope 的可探测性可由方向无关的 `max(AUC, 1-AUC)` 量化 | `TM-REP` | 预注册 probe、实体隔离 train/test、随机标签与多数类基线、3 seed AUC/置信区间 | exp-test | **pending（Phase 5）**：AUC 是泄漏测量，不等于攻击成功率或 TM-API 保证 |
| C-019 | 在预注册 checkpoint、离线数据、优化步骤和计算预算下，public 能力对 protected 能力的恢复曲线可复现 | `TM-CP` | 每个预算点的恢复率、计算/数据预算、baseline、3 seed 区间与失败运行 | exp-test | **pending（Phase 5）**：只描述受限恢复实验，不声称 checkpoint 机密性或 TM-WB 抗性 |

> **术语约束**：本项目的 Gate Layer 应表述为**固定的 toy LWE-inspired 关系验证门** ——
> 它判定调用方是否持有满足 `‖b − As‖ < τ` 的向量。**不得**称为"密码学验证门"或
> "密码学访问控制"：当前关系不提供签名不可伪造性、身份认证或 access-control soundness
> （见 C-011）。与 GateBreaker 的 "gate" 的区别应表述为机制事实 ——
> 前者由**固定关系判定**，后者由**输入内容**决定 —— 而非强弱对比。

### 台账使用规则

1. 论文中每条陈述必须映射到一个 Claim ID，并携带其威胁模型标签。
2. `pending` / `partial` 的主张不得以已验收语气书写。
3. 新增主张必须同时写入所需证据与威胁模型，禁止先写结论后补证据。
4. C-009 是**已确立的局限**而非缺陷待办：若未来引入权重级绑定，须新增独立 Claim 并重新界定其威胁模型，不得改写 C-009。
5. `TM-API` 的保证以**服务层 envelope**（C-013）为实现边界，而非以原始 `InferenceOutput` 为边界。
   evaluator 作为测试仪器可访问 `decision`；服务层不可。引用 C-003 / C-006 时必须说明这一层次差异。
6. "baseline" 一词在本项目有两个互不相关的含义，禁止混用：
   `stage_a_reference`（Stage A protected accuracy，Phase 2 已实现的 fail-fast 约束）
   与 `no_gate_ablation`（独立训练的无 Gate 同构模型，C-014，尚不存在）。
7. Phase 5 的 C-015 至 C-019 均为新路线的 `pending` 主张；不得用 CIFAR 的 C-005/C-006 或设计文档本身将其标记为 satisfied。
8. `TM-CP` 只适用于预注册、受限的离线 checkpoint 恢复实验；一旦允许修改运行时、插 hook 或直接执行 protected 路径，必须按 `TM-WB` 报告并明确当前不主张抗性。

## 8. Related work and paper positioning

<记录检索范围、来源、差异和不能声称的内容。相关工作材料与私有论文不得未经许可提交到仓库。>
