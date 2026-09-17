# CAN 认证 Expert-MoE 后续路线设计

**日期**：2026-09-10
**状态**：PROPOSED / 待审阅；本文件不表示代码已实现、协议已选定或实验已开始。
**依据**：[GATE_PRETRAINED_ROADMAP_REV2](GATE_PRETRAINED_ROADMAP_REV2_20260909.md)、Claude 对认证 Expert 问题的评估、AES Expert 论文（IACR 2026/411）和 NCS 论文（arXiv:2607.15596）。

## 1. 研究目标

研究能否将 credential 驱动的固定认证关系接入 MoE，使未认证请求只能执行公共专家，认证请求只能在其授权 scope 对应的专家集合中路由，同时保持主任务输出和执行边界可验证。

目标不是让普通神经 Router 猜测认证结果，也不是把 toy LWE 包装成生产签名。论文主张应拆成：

1. 认证关系能否作为可替换组件接入 MoE；
2. 已提交的认证 scope 能否约束专家可执行集合；
3. 受约束 Router 是否保持业务效用且不调用未授权专家。

## 2. 与两篇论文的综合判断

### 2.1 可借鉴之处

AES Expert 的 Algorithm 1 使用 `M ← (Pro_router > 0.8) ∨ (X = <ENC_PAYLOAD>)`，并在第一阶段监督 Router 识别特殊 token；这说明其安全触发有确定性兜底，并非完全依赖语义学习。该论文还提供专家模块接口、训练期代理与推理期真实实现替换、主任务效用与路由联合评估。其 SoftXOR/SoftLUT/GF-conv 是 AES 函数代理，不能直接作为认证判决方案。

NCS 提供更直接的授权边界参考：神经规划负责草案，确定性控制器负责签名验证、状态绑定、逐步释放和工具执行；stream head 签名加 hash chain 的“1+N”验证可用于自回归请求的性能设计。

### 2.2 必须保留的分歧

| 问题 | AES Expert 范式 | 本项目认证 Expert 范式 |
|---|---|---|
| 输出 | 加密后的 hidden/token 表示 | evidence 和授权候选，不直接产生业务 hidden |
| 是否需要可微 | 需要，便于端到端学习加密路由 | 推理认证不应依赖可微近似；训练代理仅作可选消融 |
| 路由来源 | Router 学习识别敏感 token | verifier 固定判定 credential；task Router 只能在 allowed_mask 内选专家 |
| 失败代价 | 敏感片段漏加密 | 错误放行导致权限提升，代价更高 |
| 安全边界 | 真实 AES 替换后仍需验证 | coordinator seal、scope、请求/cache 绑定和 protected zero-call 必须独立验证 |

因此“认证 Expert 与其他 Expert 一样”只能在 MoE 接口和调度抽象层成立；在权限语义上它必须是受约束的特殊 Expert。若把它完全当成可训练普通 Expert，不能承诺认证安全。

## 3. 目标架构

```text
prompt/tokens + structured credential
              |
       shared embedding/prefix
              |
      +-------+--------+
      |                |
  task features   Authentication Expert
                       |
                 fixed verifier
                       |
                    Evidence
                       |
              Trusted Coordinator
                       |
                Committed RouteContext
                       |
          allowed_mask + policy + cache binding
                       |
                constrained top-1 Router
             +---------+----------+
             |                    |
          PUBLIC E0          PROTECTED E1...Ek
```

推荐的权限语义：

```text
P1: relation accepted -> PROTECTED; relation failure/format error -> DENY
P2: relation accepted -> PROTECTED; canonical relation failure -> PUBLIC; format error -> DENY
P3 (optional explicit protected entry): authentication failure -> DENY; no fallback
```

P1 仍只验证 PROTECTED/DENY；P2 才引入按关系失败映射的 PUBLIC。P3 是独立的显式 protected-entry 策略。认证 Expert 不读取业务 hidden 来决定接受，task Router 不能创建或扩大 `allowed_mask`。

## 4. 可插拔接口设计

```python
evidence = auth_expert.verify(canonical_credential, request_context)
route = coordinator.commit(evidence, trusted_policy)
expert_id = constrained_router.select(hidden, route.allowed_mask)
output = dispatcher.execute(expert_id, route)
```

可插拔契约包括：credential canonical encoding、Evidence schema、policy/protocol ID、coordinator seal、allowed mask、request/cache identity、失败语义和调用台账。候选后端可以依次为 toy relation、标准签名 reference verifier、精确神经关系内核。

每次替换后必须重新完成 reference 差分、边界、scope、zero-call、缓存和 P1 direct-reference 验收。“接口可替换”不等于“任意神经权重安全等价”。

## 5. 端到端训练方案

### 5.1 默认方案：固定认证，训练任务 Router

先冻结认证 Expert、Coordinator、scope registry 和业务专家，仅训练 task Router 或公共读出。Router 的监督目标是“在已授权集合内选择任务专家”，而不是学习 allow/deny。

```text
training: fixed verifier -> allowed_mask -> differentiable masked router
inference: reference verifier -> committed mask -> hard top-1 dispatch
```

训练损失可写为：

```text
L = L_task + λ_route L_masked_router + λ_balance L_load
```

`L_masked_router` 只对 `allowed_mask` 内候选计算；越界专家的 logit 必须在 dispatch 前硬屏蔽。认证结果不从任务 loss 反向更新。

### 5.2 可选方案：代理替换消融

可训练一个 differentiable verifier proxy 帮助 Router 学习接口，但最终推理必须替换为 reference verifier，并在规范域逐输入比较：

```text
V_proxy(c) == V_ref(c)
```

代理不提供安全依据。若替换后主任务崩溃，应记录为 proxy-to-reference 不一致，而不是放宽真实 verifier。

### 5.3 不采用的方案

不采用 `credential -> neural Router -> protected expert` 的纯学习授权。它只能学习训练分布分类，可能接受相似非法 credential、依赖 prompt/hidden state 或在分布外输入上错误放行；AES Expert 的特殊 token 兜底也说明安全触发不应完全交给语义 Router。

## 6. 验签/认证算法路线

### A0：当前兼容性基线

继续使用 toy 实数残差关系，只验证接线、scope、route 和 zero-call。明确其可被最小二乘结构满足，不称为数字签名或密码学访问控制。

### A1：标准 reference verifier

优先采用经过审阅的标准签名作为真实认证后端：Ed25519 用于工程基线，ML-DSA 用于后量子签名基线。credential 至少绑定：`version`、`key_id`、`subject`、`scope`、`model_id`、`policy_id`、`request_digest`、有效期、预留的 `nonce` 和签名值。A1 的 canonical schema 从一开始固定 nonce 字段；nonce 的新鲜性、一次性消费、撤销和并发状态在 M5 启用并验收，字段存在但未启用时不得宣称防重放。

### A2：格关系研究

若研究神经化格关系，先冻结规范整数域、reference、允许算子和安全目标，再实现 Module-SIS/Module-LWE 候选的精确内核。区分“关系判定正确”与“签名不可伪造”；不继承 Kyber 或 ML-DSA 安全等级。

### A3：自回归性能

不要每个 token 重做非对称验签。可采用一次 stream-head 验证，后续 token 复用已提交 route，并以 hash-chain 或完整 cache/request 绑定检查序列连续性。此机制参考 NCS，但需要在本项目中单独定义状态、撤销和失败语义。

## 7. 分阶段工作路线

| 阶段 | 工作 | 依赖 | 验收 |
|---|---|---|---|
| M0 | 将 G0 接口抽象为 AuthExpert/Coordinator/Dispatcher contract；不改业务模型 | 当前 G0 验收 | CPU contract、类型/manifest、旧 G0 回归 |
| M1 | 认证 Expert 单组件接入 tiny MoE，固定 `E0 + E1`，无学习授权 | M0、P1 | valid→E1、public→E0、deny→zero-call；route 提交通道封闭，外部输入不能创建或扩大 `CommittedRoute` |
| M2 | 多 protected experts、scope registry、`allowed_mask` 和 constrained top-1 Router | M1 | 越界专家 zero-call、scope/route confusion、负载和延迟 |
| M3 | Router 端到端任务训练；认证固定，训练只优化 masked selection | M2、公共/受保护任务数据 | protected/public utility、router 在 mask 内选择；相对无 mask Router 上界的效用差不超过训练前冻结阈值 |
| M4 | 标准 verifier 替换与性能方案 | M1、G1-a/b、I1（两轨汇合） | 同一 MoE 上 reference verifier 与 toy 对照、P1 等价和成本 |
| M5 | stateful stream/hash-chain 与工具调用协议 | M4、独立安全协议设计 | 每步绑定、篡改拒绝、重放拒绝、撤销生效、并发双消费拒绝、无未授权副作用；不自动继承 NCS 结论 |

M0/M1 属于 Revision 2 的后续扩展，不能替代当前 G0/P0/P1。建议先完成 G0/P1，再启动 M0；M1/M2 可用 A0 做架构验证。G1-a → G1-b → I1 与 M0–M3 并行；M4 在 M1 与 I1 就绪后即可启动，不依赖 M2/M3 完成。

### M5 状态化 replay 设计约束（参考 NCS）

M5 不能把已有 route/cache 生命周期检查当作 credential 防重放。建议采用“stream-head 授权 + 服务端状态”模型：

1. 首次请求由标准 verifier 验证包含 `nonce`、`request_digest`、`scope`、`model_id`、`policy_id`、有效期和签名的 canonical credential；协调器以原子事务将 nonce 标记为 `unused → consumed`，只生成一个已提交 stream route。
2. 每个后续增量步骤携带单调 `counter` 或前序摘要 `prev_hash`，并绑定请求身份、cache 身份、模型/策略版本和输入片段摘要。状态机只接受期望的下一步，重复、跳步、重排、跨请求复用和过期步骤均 fail-closed。
3. 撤销表在每次新 stream 建立及工具副作用前检查；撤销或状态不一致时不得调用 protected expert/tool。并发请求必须使用 compare-and-swap/唯一约束，证明同一 nonce 或 counter 至多成功一次。
4. 对照实验至少包括：同一 credential 二次提交、并发双提交、跨请求/跨模型复用、counter 重排、hash-chain 篡改、撤销竞态和服务重启后的持久化恢复；报告 replay accept rate、duplicate-consume rate、false reject 和状态恢复时间。

该设计沿用 NCS 的逐步绑定、hash-chain 和 fail-closed 思路，但协议字段、状态机、撤销语义和安全论证必须由本项目独立定义；在 M5 验收前，只能说“历史实现尚无防重放证据”。

## 8. 评估矩阵

必须包含：

- 原始 dense/MoE 模型；
- 无认证的同构 Router；
- 外部 verifier + 相同专家、prefix、Router 和 dispatcher；
- 认证 Expert/in-graph verifier + 相同业务组件；
- 同一 MoE、同一冻结主干、同一数据/预算、无认证 mask 约束的 Router 训练（效用上界）；
- 固定 mask 与学习 mask 的消融；
- 代理 verifier 与 reference verifier 替换对照。

指标分四组：

1. **认证正确性**：reference 差分、边界错误接受/拒绝、scope 绑定、格式错误；
2. **执行隔离**：未授权 expert zero-call、每专家实际 forward 计数、失败无 fallback、异常计数真实；
3. **业务效用**：public/protected EM/F1 或任务成功率、主任务输出保持、Router 负载和延迟；
4. **泄漏与绕过**：TM-API 预算化恢复、TM-REP probe、TM-CP 恢复；TM-WB 继续明确不主张。

认证失败的 ASR 不是唯一指标。应分别报告 false accept、false reject、unauthorized expert call rate 和 task utility，因为错误放行与错误拒绝代价不对称。

## 9. 关键风险与边界

- 把 AES Expert 的可微代理直接当认证判决，会引入连续近似和梯度伪造面；
- 把认证 Expert 输出的 allow token 交给 Router 自由解释，会形成权限提交旁路；
- 只验证 stream head 而不绑定后续步骤，会允许重排、追加或 scope 混淆；
- public 专家回答了 protected 问题不等于 protected 专家被执行，也不自动等于知识泄漏；
- shared prefix 可能已经编码受保护能力，zero-call 不等于信息论隔离；
- 普通软件中白盒攻击者可直接调用 protected expert 或修改 route，当前不主张抗性；
- replay 防护是后续必做设计项：nonce、一次性消费、撤销、并发原子性和 hash-chain 需在状态化授权中定义并验收；当前已有 route/cache 生命周期检查不能替代 credential 新鲜性验证；
- A0 toy 关系只能支撑架构可行性，不支撑密码学安全结论。

## 10. 与当前 Revision 2 的关系和建议决策

建议批准“作为后续 M 轨道的设计方案”，不改变当前唯一下一步。实施顺序保持：

```text
架构轨：G0/P0/P1 -> M0/M1 -> M2 -> M3
关系轨：G1-a -> G1-b -> I1
verifier 替换汇合：M1 + I1 -> M4
状态化授权：M4 -> M5 stream/hash-chain 与工具调用
```

若论文主张是“模型内认证专家控制 MoE 能力访问”，M1/M2 是最小必要证据；若只完成 A0 toy 接线，论文应定位为架构/执行隔离原型。AES Expert 支持“可微代理和专家化接口”的动机，NCS 支持“确定性授权和逐步执行边界”的方法依据；两者均不能替代本项目自己的 MoE 对照与安全证据。

相对 NCS 的候选差异化主张是“授权判定与模型内部能力可达性在计算图和专家调度接口上绑定”。该主张必须由“外部 verifier + 相同业务组件”与“in-graph 认证 Expert + 相同业务组件”的强对照支持，比较绕过面、调用隔离、状态绑定和成本；若两者在预登记指标上没有可归因差异，应将贡献定位降为工程集成，而不能宣称额外安全增益。

## 11. 已确定边界与待审阅分歧

“模型内生安全”在本项目中固定限定为可信部署下计算图中的能力可达性与执行约束，不扩展为白盒安全；该项不是开放分歧。

剩余分歧：

1. 是否将认证 Expert 在论文中称为 specialized expert 而不是普通 expert；
2. M1 是否先采用 A0 toy 关系，还是直接使用 Ed25519 reference；
3. M3 是否增加 proxy verifier 作为消融；
4. M5 的状态化 stream/hash-chain 是否纳入首篇论文，或在后续独立里程碑完成；两种安排都不改变 replay 防护是后续认证协议必做项。
