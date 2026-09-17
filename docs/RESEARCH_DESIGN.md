# Research Design

当前设计依据：[G0/P0/P1 实现前设计包 v1.2](GATE_PRETRAINED_G0_P0_P1_IMPLEMENTATION_PLAN.md)
及 [Revision 2 路线](GATE_PRETRAINED_ROADMAP_REV2_20260909.md)。截至 2026-09-10，用户已确认
v1.2 完成审阅，G0/P0/P1 尚未实施；审核通过不等于获得实验证据。动态状态以
[PROJECT_WORKLOG.md](../PROJECT_WORKLOG.md) 为准，旧 C-001 至 C-019 保留其历史协议范围。

## 1. Research question

### Primary question

在固定预训练宿主、输入、后端和策略下，图中间的固定 credential Gate 能否保持合法请求的原模型输出，并对 DENY 请求实现逐样本 protected suffix 零调用？通过后，再研究共享 prefix 的公共读出能否满足预注册效用和范围约束。

### Hypotheses

- H1（待验证）：合法 hidden 恒等传递且权重冻结时，切分及 Gate 插入可满足预登记 logits/token 等价判据，拒绝行不执行 protected 分支。
- H2（待比较）：同一 verifier、policy 和 dispatcher 的外部位置 E 可以提供相同的路由执行约束；模型内位置 G 的成本须由配对测量确定，不能预设优势。

### Contribution boundary

首轮贡献范围是固定关系判定在冻结宿主内的插入正确性、实际执行路由与强对照测量。标准 Transformer、缓存、摘要和协调器机制本身不作为新密码学贡献；公共效用、完整认证安全和白盒防御不能从插入成功推出。G1/I1 的关系内核研究单独登记和验证。

## 2. Scope and terminology

### In scope

- G0 固定 hard verifier、唯一协调器、受信 route/dispatcher 与离线 tiny-host。
- P0 原生 tokenizer/冻结预训练宿主预检；P1 H/S/G/E 无训练比较；P2/P3 后续公共读出与同构评估。
- toy-real-fp32-v1 兼容关系、严格输入域、失败语义和按 execution config 绑定的可复现实验。

### Out of scope

- 本轮不延长旧从零训练主线，不实施 Teacher–Student、MoE、工具扩展或语言 wire schema。
- 当前旧 toy credential 尚无 credential replay 防护证据；后续认证协议必须参考 NCS 的 stateful authorization，纳入新鲜性、请求绑定、一次性消费、撤销与并发状态。仍不主张签名不可伪造性、身份认证、知识机密性、TM-WB 抗性或通用生产安全。

### Terminology

| Term | Definition | Forbidden interpretation |
| --- | --- | --- |
| 固定关系 verifier | 从规范 credential 与冻结参数产生 evidence | 不读取业务 hidden，不直接授予权限；toy 关系不等于数字签名 |
| 已提交 route | 唯一协调器绑定请求、policy、配置后产生的进程内结果 | 私有 seal 不构成白盒防伪；request ID 本身不是权限 |
| DENY / PUBLIC | DENY 不调用业务 suffix/head；PUBLIC 是 P2 显式公共路径 | 不能把学习式拒答、执行异常 fallback 或低泄漏混称为 DENY |
| H / S / G / E | 原宿主 / 恒等切分 / 图内 Gate / 同构外部 verifier | 不把任意外部完整模型对照视为同构，不预设 G 更安全或更快 |

## 3. System boundary

```text
首次 G：业务输入/credential → 规范预检 → prefix → 图中间 verifier → evidence → 唯一协调器 → route → dispatcher
合法 prefix hidden ───────────────────────────────────────────────────────────────────────→ suffix
增量 G：已有请求/route/cache 预检 → prefix → 图中间 route 检查 → dispatcher/suffix
```

verifier 的判决输入仅为 credential；图中箭头表示时序，不表示它读取 prefix hidden。
外部输入不可信；规范解析、verifier、协调器、dispatcher、host plugin 和请求/cache 管理属于可信进程。
首次请求身份先建立，Gate 仍在 prefix 后授权；增量步检查首次已提交 route，不重复验签。
结构错误整批失败，可定位的格式/数值错误逐行 DENY；意外运行/状态校验失败使非流式整批终止。
增量状态预检失败步零调用，执行中途异常则据实记录本步及历史调用；内部诊断不直接作为外部响应。详细边界见 [SECURITY.md](../SECURITY.md)。

## 4. Formal obligations

- 输入域：首轮仅 FP32 Tensor[B,n]，不隐式广播/类型转换；空 batch 和形状错误整批拒绝，NaN/Inf 行排除出矩阵运算。
- reference：冻结 `norm(Ac-b) < threshold`，等号失败，计算溢出数值拒绝；新整数/编码关系由 G1 单独设计。
- completeness/判定保持：目标是冻结规范域、参数、dtype/device 下与 reference 逐行一致；有限差分不证明全域等价，也不证明攻击者不能构造满足关系的 credential。
- 业务数值：采用实现包 §7.5 的预登记工程容差与 token/停止精确判据；门槛没有目标宿主实测或文献推导保证，不能按 H/S 误差放大；认证判决不使用业务容差。
- 状态与副作用：来源/策略/请求/cache 混淆必须拒绝；增量预检在本步计算前完成，失败终止整批并保留历史计数。后续协议还须以 nonce/计数器、request binding、一次性消费、撤销和并发原子性验收 credential 新鲜性；旧状态拒绝不能单独充当防重放证据。

有限随机测试不能替代全域证明、穷举、形式方法或密码学归约；每项结论必须写明证据类型。

## 5. Experimental protocol

| Item | 已审核规则及运行前待冻结项 |
| --- | --- |
| Dataset/input | P0 24 条合成 context fixture，三类各8条，源文本/答案/模板/hash 在首次候选输出前审核；calibration 与 P1 输入独立；实际文件待实现 |
| Model/system | 首轮仅 Qwen2.5-0.5B-Instruct 候选，revision/许可证/权重/tokenizer 待核验；H/S/G/E；cut 和实际配置待预检锁定 |
| Environment | CPU tiny fixture 起步，目标 RTX A4000；Python/依赖/后端/精度/确定性和预算按实际配置记录，旧 T2 环境不能代入 |
| Seeds/repeats | fixture/配对顺序 seed 20260903；H 固定配置5次校准，G/E各5次预热及20轮配对；属于有限重复性/性能测量，不是训练 seeds |
| Metrics | P0 格式 strict EM≥7/8、单跳 normalized EM≥7/8、两步≥5/8；P1 allclose、tokens/停止、真实逐行调用和缓存隔离；G/E配对时延/显存 |
| Artifacts | protocol/config/code/模型/输入/环境摘要绑定；manifest独立可信摘要；新目录保存成功及失败记录，不提交秘密、raw credential 或大型诊断/权重 |
| Stop conditions | hash/schema错误 invalid_run；已执行硬门失败 failed；缺配置 partial；全部通过 passed。校准失败阻止该配置进入P1；预算预登记，禁止删除失败配置或现场放宽门槛 |

实验结果必须与代码版本、配置、数据摘要和环境 tuple 绑定，不把未执行的结果写成已验收结论。

## 6. Research stages

| Stage | Objective | Exit criteria | Status |
| --- | --- | --- | --- |
| G0 | 固定 hard 授权与 tiny-host | 输入/判决/路由/缓存负向测试及覆盖率满足实现包 | 设计已审阅，待实现 |
| P0 | 原宿主能力、依赖与数值预检 | 选型和独立H校准通过、预算/manifest锁定 | planned |
| P1 | G0/P0后无训练插入 | H/S、S/G、G/E、zero-call与cache矩阵通过 | planned |
| P2/P3 | 公共读出与效用/越界/性能对照 | 独立协议及三seed公共训练证据；P2后重跑P1 | planned |
| G1-a/b、I1 | 独立关系内核及替换集成 | G1-a先审阅；I1依赖G1-b与P1，不依赖P3 | planned |

后续阶段不得覆盖前序路线；跨阶段只复用无协议语义的通用 helper，并保持入口和接受集合隔离。

## 7. Claim and evidence ledger

### 威胁模型标签

每条主张必须标注其成立的威胁模型，禁止跨模型引用。

| 标签 | 攻击者能力 | 说明 |
| --- | --- | --- |
| `TM-API` | 持有 API 访问权，可任意构造模型输入与 credential；**不持有权重**，不能修改进程内存或计算图 | 已有 CIFAR 服务及新路线拟验证的可信入口；预训练语言 wire schema 未交付 |
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
| C-010 | 后续认证协议在状态化授权下提供 credential 新鲜性和防重放性质 | `TM-API` | nonce/计数器、request binding、一次性消费、撤销、并发原子性与重放测试；协议安全分析 | none | **pending（后续认证协议）** |
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

### Revision 2 独立主张（C-020 至 C-026）

以下“证据类型”只标出需要取得的证据；当前均无新路径运行证据，不能引用旧 CIFAR、T1/T2、
D0 测试或设计审核将其升级。效用和性能均限定预登记配置及数据范围。

| Claim ID | 待验证主张 | 威胁模型 | 所需证据及类型 | 当前状态 |
|---|---|---|---|---|
| C-020 | G0 固定关系判决仅依赖规范 credential 与冻结配置，hidden/prompt及父模块train/eval不改变接受集合 | TM-NA | reference差分、模式/hidden扰动、类型/边界/溢出测试；unit + analytic | pending（G0，尚未实现） |
| C-021 | 可信入口仅经唯一协调器提交授权，dispatcher核对seal、policy、config与完整有序request IDs | TM-API | 跨协调器/旧请求/换序/未知route/修改源evidence负向测试，来源核验前protected零调用；unit | pending（G0；不主张白盒防伪） |
| C-022 | 固定宿主的无训练H/S切分与合法S/G插入满足预登记logits、greedy token及停止一致性 | TM-NA | 权重摘要、恒等hidden测试、独立H校准及P1输入的逐配置全元素差分；unit + exp-val | pending（P1；有限集合输出保持） |
| C-023 | P1 DENY在可信调用边界下对每个protected block、原norm/head逐样本零调用，mixed仅允许合法行进入 | TM-API | 实际module hook与参与request IDs、错行同大小故障注入、独立reference及格式失败测试；unit + exp-val | pending（P1；prefix可能已执行） |
| C-024 | 每序列只提交一次route，mixed增量cache身份/位置/生命周期隔离，状态失败步零调用且保留历史累计 | TM-NA | 无KV/KV、重排/提前EOS、跨请求/配置/层/长度错误、成功prefill后状态注入及整批无partial测试；unit + exp-val | pending（G0/P1；非credential防重放） |
| C-025 | 同一verifier/policy/dispatcher下，G/E合法输出和route及suffix/head调用一致，成本差可在固定配置下配对测量 | TM-NA | H/S与S/G通过后G/E等价检查；各5次预热、20配对轮、独立无hook性能运行、端到端/组件/峰值显存；unit + exp-val | pending（P1；不预设更快或更安全） |
| C-026 | 冻结宿主的共享prefix公共读出可在预登记任务上达到绝对效用门槛，并保持protected能力及原权重 | TM-API | 独立P2/P3协议、validation冻结门槛、3seed效用/越界/能力保持、权重摘要及重跑P1；unit + exp-val + exp-test | pending（P2/P3；不由蒸馏保证，不主张知识保密） |

新主张状态转移：设计审阅后仍为 pending；未覆盖该主张所需范围时标 partial，CPU/tiny 证据
不能满足真实宿主 P1 主张。每条主张自身要求的实现、配置/宿主矩阵及实验通过并由 Claude
验收后，才可标限定范围的 satisfied；G0 的 CPU 证据可以独立登记，不须等待 P2/P3。失败保留原配置、
原因与结果，不以新配置覆盖失败。C-026 的失败不撤销 C-022 的旧配置证据。
G1/I1 后续需新增独立主张，有限数值差分不能替代规范域证明或完整认证安全。

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
7. C-015 至 C-019 限定旧 Phase 5/T1/T2 协议，本轮保留历史台账状态；其中实现描述可能落后于代码，须另行核对旧测试/实验后更新，不将其当作新宿主证据。Revision 2 使用 C-020 至 C-026，不能用旧结果或设计本身晋升状态。
8. `TM-CP` 只适用于预注册、受限的离线 checkpoint 恢复实验；一旦允许修改运行时、插 hook 或直接执行 protected 路径，必须按 `TM-WB` 报告并明确当前不主张抗性。
9. 新 TM-API 主张先限定受信进程入口和实际执行约束；预训练语言外部响应需独立设计并验收，不能继承 CIFAR C-013 的10槽envelope或声称网络服务已实现。

## 8. Related work and paper positioning

<记录检索范围、来源、差异和不能声称的内容。相关工作材料与私有论文不得未经许可提交到仓库。>
