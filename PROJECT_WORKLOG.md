# 项目工作日志

## 当前研究阶段

**阶段**: V2 - Gate Layer 在计算图中间架构  
**状态**: R3、G0、M0、M1a tiny-MoE contract、M2 多专家 scope contract、G1-a reference、G1-b CPU verifier、I1、P0-MoE 本地实现与正式 fixture 均已通过 Claude contract 验收；真实 registry 待服务器元数据核验。
**最后更新**: 2026-09-18（P0-MoE fixture 审阅通过）

**当前唯一下一步**：在服务器执行 P0-A 元数据阶段，解析候选官方 resolved revision、许可证、remote-code 和文件摘要，生成正式 `candidate_registry.json` 并交 Claude 审阅。在 registry 审阅通过前不下载模型、不启动 P0-B/C/D。

**2026-09-18 P0-MoE fixture 与本地实现验收 checkpoint**：Claude 已审阅并通过 `fixture_v1.json`、严格 fixture/registry schema、fake-host 结构门、artifact runner、负向测试矩阵和 branch coverage 证据。已确认本阶段不包含真实模型 registry、权重下载或 P1 AuthExpert/Coordinator；当前转入服务器 P0-A，仅允许先获取并冻结官方元数据。

**2026-09-17 P0-MoE fixture 冻结 checkpoint（待 Claude 审阅）**：新增 `experiments/p0_moe_host_v1/fixture_v1.json` 及生成脚本 `scripts/generate_p0_moe_fixture.py`。fixture canonical bytes SHA-256 为 `233bf2a2517182510515bbf7d1a33a4c37fd06b9a8da55139fbb137652547391`，包含 `format_copy`、`single_hop`、`two_hop` 各 8 条，后两组均标记 `fact_source=stated_in_prompt` 并含 rationale；项目 loader 校验通过（24 条）。fixture 仅使用 prompt 内显式公开事实，不包含 credential、私有事实或正式 test。候选 registry 暂不创建：真实 resolved commit、许可证、remote-code 审计和文件摘要必须在服务器 P0-A 从官方元数据核验后冻结。

**2026-09-17 P0-MoE 本地实现完成 checkpoint（待 Claude contract 验收）**：新增 `src/can/v2/pretrained_moe_p0/`、`scripts/preflight_moe_host.py` 和 `tests/v2/test_pretrained_moe_p0.py`。实现严格 registry/fixture schema（含 C3 unknown variant、fact_source/rationale）、固定 profile/候选顺序、fake-host 七道结构门、hook 只读记录、artifact 原子写入和首个通过者/no_suitable_host 状态机；本地 fake-host 负向矩阵覆盖 dense、无 shared、post-dispatch mask、无逐行 mask、无 expert counter、top-k 漂移、batch 重排、KV 未绑定和 state-dict 变化。P0 专项 **41 passed**；P0 核心包 statement coverage **99%**、branch coverage **约 98%**（artifacts、registry、runner、host_adapter 均 100%，fixture 97%）；全量 `tests/v2/` **905 passed in 46.50s**，仅 1 个既有 PyTorch sparse warning；compileall、`git diff --check` 通过。本阶段未创建正式 registry/fixture，未下载或加载 Qwen/DeepSeek/Granite，未访问服务器，未实现 P1 AuthExpert/Coordinator。

**2026-09-17 P0-MoE 实现启动 checkpoint**：R3.17 已通过 Claude 复审。当前实现范围固定为 `src/can/v2/pretrained_moe_p0/` 的 registry、fixture、供应链元数据类型、只读 fake-host adapter、结构探查契约、artifact 原子写入和候选选择状态机，以及 `scripts/preflight_moe_host.py` 与本地负向测试。不会在本阶段下载 Qwen/DeepSeek/Granite 权重、访问服务器、修改 I1 verifier 或实现 P1 AuthExpert/Coordinator。C3 的 MoE 变体和 C2 remote-code 结论仍必须在真实 P0-A 中从官方 resolved revision 核验。

**2026-09-17 P0-MoE 方案复审修订 checkpoint（待 Claude 复审）**：采纳审阅意见，补充 C3 `expected_moe_variant=unknown_requires_verification` 与 BF16→NF4 仅限显存失败、从 P0-A 重新开始的原子 profile 切换；明确 remote-code 审计必须提交 Python 文件路径/SHA-256、危险调用扫描、审阅人/时间/结论，并在离线只读独立进程中加载；`single_hop`/`two_hop` fixture 强制 `fact_source=stated_in_prompt` 与 rationale，禁止依赖预训练知识；P0 仅记录 instrumentation hook 点和 P1 预期用途，P1 AuthExpert/Coordinator 独立实现，不复用 P0 hook 业务逻辑。当前未创建 registry/fixture、未下载模型、未启动服务器。

**2026-09-17 P0-MoE 详细方案 checkpoint（待 Claude 审阅）**：在唯一权威设计文档新增 R3.17，固定四道预检门：供应链/许可证/remote-code、24 条公开能力 fixture、MoE 结构可插入性、A4000 资源与确定性。候选 registry 预登记 C1 `Qwen/Qwen1.5-MoE-A2.7B-Chat`、C2 `deepseek-ai/deepseek-moe-16b-chat`、C3 `ibm-granite/granite-3.1-1b-a400m-instruct`，按固定顺序选择首个全门通过者；候选公开元数据在当前环境无法联网核验，因此方案明确要求 P0-A 在任何权重下载/forward 前解析不可变 revision，并核验许可证、代码和文件摘要。方案固定三组各 8 条 fixture 与 `7/8`、`7/8`、`5/8` 门槛，定义 native shared/routed、mask-before-dispatch、真实 expert zero-call、mixed-batch/KV 可控等结构硬门，设定 snapshot/墙钟/显存预算、artifact schema、覆盖率、失败码与 `no_suitable_host` 停止条件。P0 只做服务器推理预检，不训练、不修改 I1 verifier、不读取正式 test；本轮仅修改设计文档和工作日志。

**2026-09-17 I1 Claude contract 验收 checkpoint**：Claude 已完成对 I1 protocol 抽象、canonical int64 adapter、独立 evidence 完整性、三层对照、P1/P2 runner、负向路径、coverage 和 `run-20260917-05` 交付物的复审并确认通过。I1 的结论限定为 CPU tiny/experimental 模整数 verifier 与 M2 AuthExpert/Coordinator 的接口及路由契约成立；不提供真实预训练 MoE、GPU backend、训练效用、签名不可伪造、抗重放或白盒安全证据。下一阶段必须先形成 P0-MoE 详细方案并审阅，不直接进入服务器下载或实验。

**2026-09-17 I1 开发侧实现完成 checkpoint（待 Claude contract 验收）**：新增 `src/can/v2/modint_verifier_m2_i1/`、`scripts/run_i1_full.py` 和 `tests/v2/test_modint_verifier_m2_i1.py`；新增公开、runtime-checkable 的 `M2VerifierProtocol`，并使 `M2AuthExpert` 仅依赖该协议。`G1BVerifierAdapter` 仅接受原生 CPU contiguous `torch.int64[B,n]` canonical credential，使用独立 evidence seal/validator，固定映射 `RELATION_WITHIN_BOUND -> SUCCESS`、`RELATION_OUT_OF_BOUND -> RELATION_FAILED`，未知 reason fail-closed；manifest/loader 绑定 G1-a/G1-b/M2 摘要、整数范围和 FP32 无损诊断上界，配对 fixture 不保存 raw valid credential。A0-only M2 回归 **74 passed**；I1 专项 **37 passed**；G1-a/G1-b/M2/I1 联合回归 **133 passed**；全量 `tests/v2/` **864 passed in 42.20s**，仅 1 个既有 PyTorch sparse invariant warning。I1 package statement coverage **95.51%**、branch coverage **91.74%**；compileall、black、isort 和 `git diff --check` 均通过。正式 CPU 交付物为 `results/i1-g1b-m2-integration-v1/{p1-protected-or-deny-v1,p2-capability-routing-v1}/run-20260917-05/`：两种 policy 均为 `status=complete`、`exit_code=0`、`case_count=10`，`mathematical_difference_count=0`、`adapter_difference_count=0`、`route_difference_count=0`、`unauthorized_routed_calls=0`、determinism difference `=0`。P1 manifest SHA-256 为 `5e0b46f959226c73ec33450339fdcaa9a0acb407df2bb465ae17a3e0d3ba51e2`，coverage artifact SHA-256 为 `645e11af3ab697bfa9a5f128d30520acc7866fffbeea09cf0ed4ace0529d12c0`。历史 `run-01` 至 `run-04` 按不可覆盖规则保留。本实现仍限定 CPU tiny/experimental contract，不实现 GPU/E、真实宿主、训练、抗重放、签名不可伪造或白盒安全。

**2026-09-17 I1 实现启动 checkpoint**：Claude 已通过 R3.16 复审，用户指定 Codex 开始实现。范围固定为公开 M2 verifier protocol、原生 canonical int64 `G1BVerifierAdapter`、独立 evidence seal/validator、G1-a/G1-b/adapter 三层对照、配对抽象 relation fixture、P1/P2 integration artifacts，以及 route/scope/zero-call/KV/并发/异常回归；不训练、不修改 G1-b canonical 接受集合、不实现 GPU/E 或真实宿主。当前唯一下一步：先完成 protocol 抽象并运行原 A0 的完整 M2 专项回归。

**2026-09-17 I1 protocol checkpoint**：新增公开、runtime-checkable 的 `M2VerifierProtocol`，`M2AuthExpert` 不再绑定具体 `FixedRelationVerifier` 类；原 A0 直接满足 `__call__`/`validate_evidence` 契约，无额外包装层。仅注入原 A0 的 M2 专项回归 **74 passed**（1 个既有 sparse invariant warning），证明接口抽象未改变既有 M2 行为。当前唯一下一步：实现只接受 canonical int64 的 `G1BVerifierAdapter` 及 adapter-only 负向测试。

**2026-09-17 I1 方案第二轮审阅修订 checkpoint（待 Claude 复审）**：进一步冻结实施顺序和 schema：公开 verifier protocol 必须先仅使用原 A0 跑完整 M2 专项并证明零行为变化，之后才允许编写 G1-b 集成；adapter 在接入 AuthExpert 前须通过独立 canonical input、seal、unknown reason、digest 和私有 helper 隔离测试。FP32 兼容诊断按 centered residual 全域上界 `q//2` 而非 `tau` 校验，manifest 登记并由 loader 复算 `max_abs_residual_upper_bound`、`fp32_exact_integer_limit=2**24` 和 `fp32_lossless_verified`。配对 fixture 新增 expected reason、人工 rationale 和机器可校验 `boundary_role`；增量 decode 明确从受信 session record 读取首次提交 route，完成 preflight 后复用 route 而不重新验签。当前唯一下一步仍为交 Claude 复审，复审通过前不编码。

**2026-09-17 I1 方案首轮审阅修订 checkpoint（待 Claude 复审）**：根据接口实码核查，明确 `ModIntNeuralVerifier` 不能仅因同为 `nn.Module` 就直接替换 `FixedRelationVerifier`；I1 必须引入公开 `M2VerifierProtocol`，由 A0 和 `G1BVerifierAdapter` 共同实现 `forward()`/`validate_evidence()` 契约。方案明确只接受原生 canonical `torch.int64[B,n]` credential，禁止用 `scale_factor`、round、截断或隐式 cast 把旧 FP32 credential 量化为 G1-b 输入；adapter 使用独立 evidence seal/validator，不依赖旧 verifier 私有 helper。A0 与 G1-b 改为三层对照：G1-a/G1-b 数学一致、G1-b/adapter 字段一致、配对抽象 relation class 下 A0-M2/G1b-I1 路由语义一致；不再要求不同数学关系对同一数值 credential 有相同接受集合。当前唯一下一步为交 Claude 复审，复审通过前不编码。

**2026-09-16 I1 详细方案 checkpoint（首轮审阅前版本）**：在 `docs/DESIGN_PROPOSALS.md` 新增 R3.16，定义仅替换 M2 A0 verifier 的适配层、G1-a/G1-b/M2 manifest 与 digest 绑定、固定 evidence/reason/policy 映射、scope/route/zero-call/KV/并发/异常不变量、双 policy 结果目录、fixture/ledger/summary schema、reference 差分、coverage/确定性硬门槛和停止条件。I1 保持 M2 路由与执行 contract，不训练、不引入学习 Router、不实现 GPU/E 或真实宿主。

**2026-09-16 G1-b CPU 实现 checkpoint（待 Claude contract 验收）**：新增 `src/can/v2/modint_verifier_g1b/`，实现 `IntegerBounds` 全域上界证明、CPU `torch.int64` canonical `mod q`/centered lift、checked tensor batch、无参数 `ModIntNeuralVerifier`、严格 backend manifest 和 H/G artifact runner；新增 `tests/v2/test_modint_verifier_g1b.py`。专项 **12 passed**，G1-b package branch coverage **99%（64 branches，1 partial）**、statement **约 99%**；全量 `tests/v2/` **826 passed in 51.86s**，仅 1 个既有 PyTorch sparse invariant warning；compileall、`git diff --check` 通过。`scripts/run_g1b_cpu.py` 已生成成功交付物 `results/g1b-modint-v1/run-20260916-02/`，`h_g_difference_count=0`；首次索引 oracle 错误留下的 `run-20260916-01` 按不可覆盖规则保留。当前实现仅 CPU int64 tiny fixture，不实现 E/GPU、不接入 I1、不训练；当前唯一下一步：交 Claude contract 验收。

**2026-09-16 G1-b CPU 实现启动 checkpoint**：用户确认 R3.15 已验收，开始实现固定 CPU `torch.int64` backend。实现范围为 `integer_bounds` 解析上界、canonical `mod q`、centered lift、checked dot、离散 evidence、无参数 `nn.Module` 和 G1-a reference 对照；不实现 GPU/E、定点 backend、I1 接入或训练。当前唯一下一步：完成 package、专项测试和 coverage。

**2026-09-16 G1-b 方案审阅修订 checkpoint（待 Claude 复审）**：采纳 Claude 对溢出、canonical mod、分阶段 H/S/G、coverage 和 GPU 可行性的建议。R3.15 现在要求 G1-b runner 从 `q,n` 解析登记 `max_product_abs`、`max_accumulator_abs`、`required_signed_bits` 和公式版本，先证明 `int64` 全域安全再启动 tensor kernel，禁止以有限 vectors 最大值替代解析上界；明确 `canonical_mod_q` 必须覆盖负值/负整倍数并在溢出检查后执行；H/S/G 先在 G1-a 小 fixture 完成全量差分，再扩展批准的大配置；CPU core coverage 与设备适配分层统计；GPU/E 后端降为 CPU contract 通过后的可选 feasibility smoke test，int32/定点必须新建 backend revision。G1-a 已验收 manifest 不回写修改，G1-b manifest 以 `parameter_sha256` 绑定新增 `integer_bounds`。本轮未实现代码、未训练、未启动 GPU；当前唯一下一步：交修订后的 R3.15 Claude 复审。

**2026-09-16 G1-b 详细方案 checkpoint（待 Claude 审阅）**：新增 R3.15，明确 G1-b 只把已验收 G1-a reference 编译为固定 PyTorch 整数 verifier：无可训练参数、无 autograd、无业务 hidden 输入，先 CPU `torch.int64` 后可选 GPU/定点后端；固定 `mod q`、centered lift、checked dot、linf/tau 离散判定、H/S/G/E 四层对照、overflow fail-closed、manifest/backend provenance、成本测量、coverage/determinism 门槛和 I1 入口。G1-b 参数必须复制已批准 G1-a manifest；扩大 q/n/m 或切换 backend 必须新 execution config。方案阶段不实现代码、不训练、不下载模型、不启动 GPU；当前唯一下一步：交 R3.15 Claude 审阅。

**2026-09-16 G1-a runner/测试入口修复 checkpoint**：新增仓库级 `pytest.ini`，将 `src` 注册为 pytest `pythonpath`，直接执行 `python -m pytest tests/v2/` 不再依赖外部 `PYTHONPATH`。artifact runner 通过 `git status --porcelain --untracked-files=all` 真实计算 `provenance.dirty`，无法确认时 fail-closed；runner 标准输出/错误流在支持时切换 UTF-8，降低 Windows 控制台乱码。新增 dirty 状态模拟测试。G1-a 专项 **10 passed**；全量 `tests/v2/` **814 passed in 60.52s**，仅 1 个既有 PyTorch sparse invariant warning；compileall、`git diff --check` 通过。当前唯一下一步仍为交 Claude contract 验收。

**2026-09-15 G1-a reference 实现 checkpoint（待 Claude contract 验收）**：新增 `src/can/v2/modint_verifier_g1a/`，实现不可变 `G1AConfig/G1AParameters/G1aEvidence` 类型、确定性 `master_seed -> label` 派生、公开 A/b 生成与摘要、canonical mod、奇数 q centered lift、纯 Python int reference、逐行接受集合和稳定 reason code、严格 manifest 校验，以及不写入 raw secret/credential 的 artifact runner。新增 `tests/v2/test_modint_verifier_g1a.py`，专项 **9 passed**；`scripts/run_g1a_reference.py` 生成 `results/g1a-modint-v1/run-20260915-01/`（A.npy、b.npy、vectors.json、manifest.json、summary.json），manifest 原始 bytes SHA-256 可复核，runner 拒绝覆盖既有目录。修正 provenance 校验以接受 40/64 位 Git commit。全量 `tests/v2/` 回归 **813 passed in 41.30s**，仅有 1 个既有 PyTorch sparse invariant warning；compileall、`git diff --check` 通过。当前实现仅为 CPU reference/tiny fixture，不实现 G1-b 神经核、不提供密码学安全结论、不训练、不启动 GPU；当前唯一下一步：交 Claude contract 验收。

**2026-09-15 G1-a 详细方案修订 checkpoint（待 Claude 复审）**：根据 Claude 审阅补齐公开矩阵 `A`/向量 `b` 的 canonical bytes 或确定性 PRNG 来源、`master_seed -> seed_A` 派生字段、参数摘要复现规则；明确 `centered_q(q//2)=+q//2` 与 `centered_q(q//2+1)=-(q//2)` 的边界 vectors，以及 `z<0` 的 canonical mod 交叉实现测试；新增不可变 `G1aEvidence` 字段/类型/长度/摘要 schema、稳定 reason code 和 evidence 不得直接授权的约束；明确 Python 任意精度 reference 与 G1-b 固定宽度实现的溢出检查分工，并要求乘加前转换为 Python `int`；将认证整数/定点流与业务 hidden/autograd 图的分离定义为行为约束，不提前锁死 `torch.int32/int64`。同时优化 `tau` 单位、参数未立即冻结和独立模约简的表述。本轮仍只修改设计文档和工作日志，未实现 G1-b、未训练、未下载模型或启动 GPU；当前唯一下一步：将修订后的 R3.14 交 Claude 复审。

**2026-09-15 G1-a 详细方案 checkpoint（待 Claude 审阅）**：在 `docs/DESIGN_PROPOSALS.md` 新增 R3.14，明确 G1-a 只冻结模整数 verifier 的 canonical domain、候选 `(q,n,m)` 选择依据、整数关系 `r=centered_q(A*c-b mod q)`、阈值接受集合、纯 Python/NumPy reference 步骤与边界 vectors、evidence/Coordinator 边界、神经实现允许算子与位宽/溢出约束、有限域 soundness/completeness 目标、安全性不声明、CPU 成本评估、manifest/fixture/测试矩阵、coverage/确定性门槛和 G1-b 入口门槛。候选 `q=3329,n=256,m=512,tau` 未在审阅前冻结；本阶段不实现 G1-b、不训练、不下载模型、不启动 GPU。M2 最新 `run-20260915-04` 与 73 项专项、804 项全量测试证据按此前验收记录保留。当前唯一下一步：将 R3.14 交 Claude 审阅。

**2026-09-15 M2 Claude 验收 checkpoint**：增加显式 scope DAG/环检查、assignment registry seal 消费校验、Router finite/device 校验和 Dispatcher 对每行 allowed expert 的二次越权检查；修复自定义 expert 缺省 E0 时的初始化错误。严格校验 ExpertSpec 九字段、scope_grants 三字段、fixture/cases、summary 的 `cases/determinism/coverage/provenance` 嵌套结构。runner 计算真实 expert state-dict SHA-256、assignment SHA-256、git provenance，并执行重复 forward determinism 检查；最新 `run-20260915-04` 的 P1/P2 交付物已 loader 复核，`difference_count=0`。专项测试 **73 passed**；statement coverage **100%**、branch coverage **91.85%（169/184）**；全量 `tests/v2` **804 passed in 38.24s**；compileall、git diff --check 通过（专项含 1 个 PyTorch sparse invariant warning，不影响结果）。旧 `run-20260914-01` 保留为历史不完整产物，不覆盖、不删除。本实现仍限定 CPU float32 tiny contract，不提供真实 verifier、安全性或专家能力结论。Claude contract 验收已完成，当前转入 G1-a 详细方案审阅。

**2026-09-14 M2 实现 checkpoint（待 Claude contract 验收）**：新增 `src/can/v2/auth_expert_moe_m2/`，实现冻结 Shared E0 + Routed E1/E2/E3、固定 scope lattice 与受信 assignment registry、A0 evidence 绑定、P1/P2 route coordinator、确定性 constrained top-1 Router、原始索引稀疏 dispatch、routed zero-call 与调用台账，以及 manifest/fixture/summary 校验工具。新增 `tests/v2/test_auth_expert_moe_m2.py`（18 passed）和 `scripts/run_m2_full.py`，已生成 `results/m2-multi-expert-v1/{p1-protected-or-deny-v1,p2-capability-routing-v1}/run-20260914-01/02/03/` CPU fixture 交付物，且最新 run 已通过 manifest/fixture/ledger/summary loader 复核。专项测试 18 passed；全量 `tests/v2` 749 passed in 43.77s；compileall、git diff --check 通过。专项 statement coverage 99%，branch coverage 91%（达到设计门槛 `>=95%/>=90%`）。本实现仅为 CPU float32 tiny contract，不提供真实 verifier、安全性或专家能力结论。当前唯一下一步：将 M2 负向路径、branch coverage 和正式 CPU 交付物交 Claude contract 验收。

**2026-09-14 M2 首轮审阅修订 checkpoint（待 Claude 复审）**：在 R3.13 新增 M2.2a/b、M2.4a、M2.5a：固定 `scopes` 精确 JSON schema、拓扑声明/DFS 或 Kahn 环检测、ancestor closure 与单调超集规则；明确 A0 不携带 scope，不构造虚假的 evidence-to-scope 映射，而由受信 fixture registry 签发 `M2ScopeAssignment` 并与 evidence/request/policy/config 绑定；Router 仅对 `[E1,E2,E3]` 使用 `[B,3]` mask，E0 不参与 routed 选择，全 false 行显式返回 None，零 logits 按最小 allowed 索引确定性选择；明确 top-k=1 为每行最多一个 routed expert，并要求 E2/E3 胜出测试。另冻结 scope_grants schema、assignment 摘要语义、核心错误码和 `run-YYYYMMDD-NN`/双 policy 原子预检规则。本轮只修改方案与工作日志，未实现 M2、未运行训练或 GPU。当前唯一下一步：将修订后的 R3.13 交 Claude 复审；复审通过且用户指定实现者后再编码。

**2026-09-14 M2 详细方案 checkpoint（待 Claude 审阅）**：在 `docs/DESIGN_PROPOSALS.md` 新增 R3.13，固定 Shared E0 + Protected Routed E1/E2/E3、`standard ⊂ advanced ⊂ privileged` scope lattice、A0 evidence 与受信 scope grant 绑定、P1/P2 独立 policy、M2 execution/protocol/结果目录、模块接口、manifest/fixture/ledger/summary schema、18 类测试条件、coverage/确定性/zero-call/索引/异常门槛、CPU-only 实施顺序和停止条件。M2 不实现代码、不训练、不进入 G1；当前唯一下一步：将 R3.13 交 Claude 审阅，审阅通过且用户指定实现者后再开始 M2 实现。

**2026-09-14 M1a Claude 验收完成 checkpoint**：Claude 已验收 M1a 核心模块、完整 runner、P1/P2 结果交付物、负向路径测试和带 branch coverage 的报告。M1a 专项 **37 passed**，全量 `tests/v2` **731 passed**；statement coverage **97%**、branch coverage **93%**。本地已生成 `results/m1a-tiny-moe-v1/{p1-protected-or-deny-v1,p2-capability-routing-v1}/run-*`，包含独立 manifest、fixture、reference、call ledger 和 summary。M1a 结论限定为 CPU float32 tiny fixture、固定 A0、单 protected E1 的执行隔离与路由组合；不提供真实 MoE、模整数 verifier、训练能力或密码学安全结论。当前唯一下一步：在 `docs/DESIGN_PROPOSALS.md` R3 下编写 M2 多 routed experts 与 scope lattice 详细实现方案并交 Claude 审阅；审阅通过且用户指定实现者前不实现 M2。

**2026-09-14 M1a 实现启动 checkpoint**：用户确认 Claude 已审阅通过 R3.12 M1a 详细方案并要求开始实现。当前新增 `src/can/v2/auth_expert_moe/` tiny-MoE contract package，范围固定为 Shared E0 + Protected Routed E1、P1/P2 独立 policy、A0 fixture、严格 manifest/fixture/ledger/summary schema；不实现 G1 模整数 verifier、不引入真实宿主或训练。实现完成后须运行 M1a 专项和 `tests/v2` 回归、覆盖率及格式检查，再交 Claude contract 验收。当前唯一下一步：完成 M1a package 与对应专项测试。

**2026-09-14 M1a 实现 checkpoint（待 Claude contract 验收）**：完成 `src/can/v2/auth_expert_moe/`：冻结 Shared E0/Protected Routed E1 FFN、独立 M1AAuthExpert/M1ARouteCoordinator（P1/P2）、固定 constrained Router、只能收窄的 scope registry、稀疏 MoE dispatch、原始 batch 索引与 routed zero-call、L2 normalize（`eps=1e-12`）、严格 manifest/fixture/ledger/summary 校验工具。新增 `tests/v2/test_auth_expert_moe_m1a.py` 共 **35 passed**；M1a package 定向覆盖为 statement **97.27%**、branch **93.24%**（超过 `>=95%/>=90%` 门槛）。全量 `tests/v2` 回归为 **729 passed in 47.83s**；Black、isort、compileall、`git diff --check` 均通过。实现限制：CPU float32 tiny fixture、单 protected E1、固定 A0 verifier，不提供 G1 模整数安全性、真实 MoE、训练、密码学不可伪造或知识保密结论；尚未生成正式结果目录/manifest fixture 产物。当前唯一下一步：将本 checkpoint 交 Claude 做 M1a contract 验收，验收前不进入 M2。

**2026-09-14 M1a runner/交付物 checkpoint（待 Claude contract 验收）**：新增 `scripts/run_m1a_full.py`，以显式 run ID 生成 P1/P2 独立结果目录，每套包含 `manifest.json`、`fixture.json`、`inputs.npy`、`reference_outputs.npy`、`call_ledger.json`、`summary.json`；runner 拒绝覆盖既有 run，并记录 artifact SHA-256、Git/Python/PyTorch/NumPy/device provenance。已生成 `run-20260914-01/02/03` 三组本地结果并通过 manifest/fixture loader 复核；结果为 CPU float32 tiny fixture，不提交 raw credential。补充 artifacts/router 负向测试后专项为 **37 passed**；带 `--branch` 的正式报告 `coverage_m1a.json`：statement **97%**、branch **93%**（总覆盖统计 96%/228 branches，均超过门槛）；全量 `tests/v2` 为 **731 passed in 33.47s**；Black、isort、compileall、`git diff --check` 均通过。当前唯一下一步：将 M1a runner、交付物 schema 与测试交 Claude contract 验收，验收前不进入 M2。

**2026-09-14 M1a 详细方案 checkpoint（待 Claude 审阅）**：在 `docs/DESIGN_PROPOSALS.md` 新增并补齐 R3.12，明确 `Shared E0 + Protected Routed E1` tiny-MoE 的独立 execution/protocol/manifest、固定结果目录 `results/m1a-tiny-moe-v1/{p1-protected-or-deny-v1,p2-capability-routing-v1}/`、M0 基线 commit `7c2434e`、manifest 字段与 SHA-256 语义、seed/input/reference fixture 格式、错误 code、调用台账和 summary schema、P1/P2 失败分类、固定 CPU fixture、模块接口、shared 恒执行、routed zero-call、原始索引、数值/确定性门槛、能力矩阵和停止条件。M1a 仅使用 A0 作为 contract fixture，不把 A0 当作正式安全 verifier；模整数主线仍从 M2 后的 G1-a/G1-b/I1 开始，Ed25519 保留为后续可选对照。本轮仅修改设计文档，未实现 M1a、未创建 freeze、未下载模型或启动服务器。当前唯一下一步：将 R3.12 交 Claude 审阅；审阅通过并由用户指定实现者后才开始代码实现。

**2026-09-14 M0 Git 收尾 checkpoint**：已将 Claude 验收通过的 M0/G0 实现、测试及路线文档提交并推送到 `origin/master`。commit：`7c2434ed5590cef71acd0a9b595da4fb6778e89b`（`feat: complete M0 authenticated expert contract`）。提交包含 22 个文件；其他用户/历史改动未纳入。M0 验收证据保持为专项 `78 passed`、全量 `tests/v2` `694 passed`、statement coverage `96.62%`、branch coverage `91.91%`。当前唯一下一步：在 `docs/DESIGN_PROPOSALS.md` R3.11 下编写 M1a tiny-MoE 详细实现方案并交 Claude 审阅；方案审阅通过且用户指定实现者前，不实现 M1a、不创建 freeze、不下载真实模型或启动服务器。

**2026-09-13 verifier 主路线调整 checkpoint**：根据用户选择，将后续主线改为“模整数神经 verifier”，Ed25519 降为后续可选标准 reference。M1a/M2 仍先使用已验收的 A0 作为 tiny-MoE contract fixture，仅验证 Shared/Routed、scope、zero-call 和 constrained dispatch；A0 不承担正式安全结论。M2 之后必须依次完成 G1-a（规范域/接受集合/reference）→ G1-b（模乘加、约简、centered lift、边界与后端一致性）→ I1（接入 M2 AuthExpert/Coordinator），通过后才进入 P0/P1-MoE 和 M3。M4a Ed25519 不再阻塞主路线，M5 状态化授权在主 verifier 稳定后执行。当前唯一下一步仍为提交并推送已验收的 M0 checkpoint，之后编写 M1a 详细方案；本次仅同步路线文档，未实现 G1、未下载模型或启动服务器。

**2026-09-13 M0 contract Claude 验收完成 checkpoint**：Claude 已验收 M0 contract 及补充的并发隔离、KV 中途失败和预检回滚测试。M0 的开发侧最终证据保持为专项 `78 passed`、全量 `tests/v2` `694 passed`、statement coverage `96.62%`、branch coverage `91.91%`，Black/isort/compileall/`git diff --check` 均通过。M0 结论边界不变：仅支持单机 CPU、TinyDecoder/TinyKV、单 protected E1 与 toy A0，不提供真实 MoE、GPU、签名不可伪造性或白盒安全结论。当前唯一下一步：先提交并推送已验收的 M0 checkpoint；同步完成后再依据 `docs/DESIGN_PROPOSALS.md` R3.11 编写 M1a tiny-MoE 详细实现方案并交 Claude 审阅。在方案审阅通过前不实现 M1a、不创建 freeze、不下载真实模型或启动服务器。

**2026-09-13 M0 Claude 建议补充 checkpoint（待 Claude contract 验收）**：采纳并落实两项非语义变更建议：并发隔离测试现在显式断言并发拒绝后原 session 仍为 ACTIVE、KV 句柄仍登记且 protected 调用计数未改变；新增 KV suffix 中途异常及预检首行通过/次行失败的注入测试，验证 session 进入 FAILED、旧/新句柄全部失效且不发生部分提交。M0 专项测试更新为 **78 passed**，全量 `tests/v2/` 更新为 **694 passed in 32.42s**；覆盖率重新测量仍为 statement **96.62%**、branch **91.91%**。`docs/DESIGN_PROPOSALS.md` M0-T08/T09 已同步上述验收口径。当前唯一下一步仍为将更新后的 M0 checkpoint 交 Claude contract 验收；验收前不进入 M1a。

**2026-09-13 M0 最终验证 checkpoint（待 Claude contract 验收）**：重新执行最终版本验证：`python -m pytest tests/v2/test_auth_expert_m0.py -q` 为 **76 passed**；`python -m pytest tests/v2/ -q` 为 **692 passed in 37.49s**；针对 `src/can/v2/auth_expert` 的 `coverage run --branch` 全量回归为 statement **96.62%**、branch **91.91%**（超过 `>=95%/>=90%`）；Black、`isort --profile black`、compileall 与 `git diff --check` 通过。M0 固定范围仍为单机 CPU、TinyDecoder/TinyKV、单 protected E1、toy A0；不提供签名不可伪造性、多专家能力结论、真实宿主或 GPU 结论。`coverage_m0.json` 为本地生成验证产物，不纳入提交。交 Claude 的 M0 相关文件为 `src/can/v2/auth_expert/`、其依赖的 `src/can/v2/pretrained_gate/`、`tests/v2/test_auth_expert_m0.py`、`tests/v2/test_pretrained_gate.py`、`tests/v2/test_pretrained_gate_contracts.py`、`docs/DESIGN_PROPOSALS.md` 与 `PROJECT_WORKLOG.md`；工作树中其他既有改动不属于本 checkpoint。当前唯一下一步：将上述 M0 实现与准确文件清单交 Claude contract 验收；验收前不创建 freeze、不进入 M1a、不下载模型或启动服务器。

**2026-09-13 M0 实现收尾 checkpoint（待 Claude 验收）**：完成 `src/can/v2/auth_expert/` 的固定 M0 contract：A0 evidence 绑定、唯一 ScopeCoordinator route、E1-only ScopeRegistry/Dispatcher、原始索引与 zero-call、TinyDecoder/TinyKV host bridge、session close/FAILED 原子清理、KV preflight/句柄替换和严格 manifest 摘要/字段校验。修复 KV decode 在追加 token 前快照旧输入/mask 的作用域问题。专项测试 `tests/v2/test_auth_expert_m0.py` 为 **73 passed**；全量 `tests/v2/` 为 **689 passed in 75.03s**。对 `src/can/v2/auth_expert` 的 coverage（`coverage run --branch`，全量 tests）为 statement **96.5%**、branch **92.2%**，超过 R3.10 的 `>=95%/>=90%` 门槛；Black、isort、compileall 与 `git diff --check` 均通过。新增边界测试覆盖 T01–T12 的类型/来源/生命周期/KV/manifest/并发场景；仍保留 TinyKV 仅 CPU fixture、M0 单 protected 槽、toy A0 不提供签名不可伪造性等限制。当前唯一下一步：将本 checkpoint（准确文件清单见 git status）交 Claude 做 contract 验收；验收前不创建 freeze、不开启真实宿主或 GPU 实验。

**2026-09-13 M0 文档状态同步**：将 `docs/DESIGN_PROPOSALS.md` R3.10 从“方案/代码尚未实现”修订为“实现完成、待 Claude contract 验收”，并把文件职责表与实际聚合测试文件 `tests/v2/test_auth_expert_m0.py` 对齐。该修订只纠正文档状态和测试入口，不扩大 M0 范围，也不改变 M0 的单槽、CPU/TinyKV、toy A0 限制。当前唯一下一步仍为 Claude contract 验收；验收通过前不进入 M1a、不创建 freeze、不下载模型。

**2026-09-13 M1a/M2 能力分级方向与 M0 审阅补充 checkpoint**：根据用户提出的 DeepSeek-MoE 风格组织，已在 `docs/DESIGN_PROPOSALS.md` R3.11 记录后续方向：Shared General Experts 始终执行，Authenticated Routed Experts 仅在可信 route/allowed_mask 内选择；未授权路径为 shared-only，授权路径为 `shared + alpha * normalized(routed)`。专家的业务强弱由结构、权重、数据和冻结能力矩阵验收，认证只控制可达性，不从 expert ID 或 verifier 结果推导能力等级。M1a 固定 Shared E0 + Protected Routed E1，M2 扩展多 routed experts 与显式 scope lattice；必须做 shared-only 能力上限和未授权泄漏评估，不能把 routed zero-call 等同知识保密。另根据 Claude 建议在 M0.6 加入 cache 句柄失效、授权清理与 FAILED 状态转换的原子性要求，并在 M0-T08 加入并发调用隔离测试。以上均为 M0 待审阅后的后续设计约束，未修改代码、未运行测试、未创建 freeze、未下载模型或启动服务器；当前唯一下一步仍是将 `m0-contract-plan-v1` 连同本次补充交 Claude 审阅。

**2026-09-13 M0 实现启动 checkpoint**：用户确认 Claude 已审阅并要求开始实现 M0。当前按 `m0-contract-plan-v1` 实现新 `src/can/v2/auth_expert/` package，先复用 G0 `FixedRelationVerifier`、`TinyDecoderHost`/`TinyKVDecoderHost`、`CacheRegistry` 和 `CallLedger`，不改变旧接口或接受集合。实现范围从契约类型、A0/AuthExpert、ScopeCoordinator/Registry、固定 E1 Dispatcher 到 session/host bridge/manifest；每阶段以 M0 专项测试和 G0 回归为门槛。新增代码尚未完成，测试与覆盖率待运行，本轮暂不下载模型、启动服务器或创建 freeze。

**2026-09-13 M0 初始实现 checkpoint（待 Claude 验收）**：新增 `src/can/v2/auth_expert/` 的 types、authentication、scope、dispatch、host_bridge、runtime、manifest 与 package 导出；实现固定 A0 evidence、唯一 route 提交、E1 scope、原始 batch 索引、selection/mask 权限复核、session close/失败清理、无 KV prefill/decode 和严格 `m0-contract-v1` manifest 摘要/嵌套字段校验。新增 `tests/v2/test_auth_expert_m0.py` 覆盖 mixed/deny zero-call、scope 收窄、mask 副本、错误 selection、关闭 session、manifest 摘要与未知字段，并验证 TinyKV cut 前后 bridge 输出与完整 host 一致。专项测试 `6 passed`，编译通过，完整 `tests/v2` 回归 `622 passed in 30.86s`。当前明确限制：M0 runtime 对 `cache_mode="kv"` 返回 `kv_bridge_not_implemented`，尚未宣称跨 decode 步 session KV 增量完成；cache registry 原子替换、T01–T12 完整矩阵、覆盖率与并发测试仍是后续实现任务。工作日志唯一下一步改为完成跨步 KV 生命周期/并发专项并交 Claude 验收，未下载模型、未启动服务器、未创建 freeze。
**2026-09-13 M0 KV/并发补充 checkpoint（待 Claude 验收）**：`M0TinyHostBridge` 已复用 TinyKV 的 `forward_full/forward_step` 完成 cut 前 prefix 与 protected suffix 的真实分段；`M0Session` 对 `cache_mode="kv"` 建立 per-request `CacheRegistry` 句柄，续步前全批 `preflight_kv_batch`，成功后原子完成新句柄登记、旧句柄失效和 session 状态更新，失败则清理整个 registry。修复了 KV 续步预检中旧 `input_ids`/`attention_mask` 快照作用域错误：快照现在在 decode append 前建立，避免 `NameError` 并确保预检使用旧 cache 长度。新增非法 decode、空 batch、超长 prefill、evidence/route/scope/selection 来源和类型错误、KV mixed/right-padding prefill、cache 元数据篡改、组件绑定错误、manifest 字段缺失/重复/非法类型/非有限值和 KV 张量损坏的 fail-closed 测试。M0 专项测试 `33 passed`，完整 `tests/v2` 回归通过，compileall 与 `git diff --check` 通过。专项 coverage 为 statement 83%、branch 约 75%，仍低于设计门槛 `>=95%/>=90%`，原因是 manifest 其余严格字段分支及 T01–T12 大量边界分支尚未覆盖；TinyKV 仍禁止右 padding 后重新激活 padding 行。M0 尚未完成验收。当前唯一下一步是继续补全 cache fault/生命周期边界和剩余 coverage，再交 Claude 验收。

**2026-09-11 replay 口径修订 checkpoint**：根据 Claude 对认证 Expert-MoE 路线的审阅意见，统一修订 `PROJECT_WORKLOG.md`、`docs/DESIGN_PROPOSALS.md`、`docs/GATE_PRETRAINED_G0_P0_P1_IMPLEMENTATION_PLAN.md` 及既有安全规则中的 replay 表述。当前口径为：历史 toy/static credential 尚无防重放证据；后续认证协议必须参考 NCS 的 stateful authorization，定义 nonce/计数器、request binding、一次性消费、撤销、并发原子性及必要的 stream/hash-chain，并以独立测试和安全论证验收。保留历史阶段使用可重复 credential 的事实，不将其表述为当前路线永久非目标，也不把 route/cache 生命周期检查等同于 credential 防重放。此次仅修改文档，未修改代码、未运行模型或 GPU 实验、未提交或推送。

**2026-09-10 认证 Expert-MoE 路线提案**：新增 `docs/AUTH_EXPERT_MOE_ROADMAP_20260910.md`，综合 AES Expert 与 NCS 论文及 Claude/Codex 评估。该提案将认证 Expert 定义为 MoE 接口上的特殊可插拔组件：固定 verifier 产生 evidence，协调器提交 route，task Router 只能在 `allowed_mask` 内选择；端到端训练只优化受约束任务路由，代理 verifier 仅作消融。M0–M5 均为后续 PLANNED，不改变当前 G0/P0/P1 唯一下一步；本轮未修改代码、未运行实验、未选择最终签名方案。

**当前目标**：完成 M2 多 Expert/scope 详细方案的 Claude 审阅。R3.13 已写入 `docs/DESIGN_PROPOSALS.md`，审阅通过且用户指定实现者前不实现 M2、不实现 G1、不创建 freeze、不下载真实模型或启动服务器。

**2026-09-11 M0 contract 详细方案 checkpoint（待 Claude 审阅）**：在 `docs/DESIGN_PROPOSALS.md` R3.10 新增 `m0-contract-plan-v1`。方案将 G0 的固定 verifier 作为 A0 后端，新增 AuthExpert、ScopeCoordinator、ScopeRegistry、M0Dispatcher、M0Session、manifest 与 tiny host bridge 的独立职责；固定 M0 为 E0 禁用目录 + E1 单 protected 槽和 P1 PROTECTED/DENY，不提前实现 P2、多 Expert、学习 Router 或真实宿主。已写明 evidence/route/view 对象身份与上下文绑定、allowed-mask 只能收窄、原始 batch 索引、prefill/decode/KV 时序、异常与 zero-call、严格 `m0-contract-v1` manifest、T01–T12 测试矩阵、覆盖率/性能测量和停止条件。特别记录现有 `TinyKVDecoderHost` 不能直接作为 cut 后增量入口，M0 需复用其 block 算子实现 bridge，不复制注意力数学。此次未修改代码、未运行测试/GPU、未下载模型、未创建 freeze；`git diff --check` 已核对通过。

**2026-09-11 Unified Roadmap R3 设计整理 checkpoint（Claude 已验收）**：
`docs/DESIGN_PROPOSALS.md` 已明确为项目后续工作的唯一权威设计文档；工作日志只负责动态事实和唯一下一步，其他 G0/P0/P1、Revision 2 与认证 Expert-MoE 专题文件保留为历史提案/审阅材料。R3 将后续主线重排为 G0 验收 → M0 contract → M1a tiny-MoE → M2 scope/多专家 → P0-MoE/P1-MoE 真实宿主 → M3 Router 训练；Ed25519 M4a → M5 状态化授权形成标准认证轨，G1-a/b → I1/M4b 作为可并行且不阻塞核心结果的神经 verifier 研究轨。方案统一记录 P1/P2 policy 隔离、24 条 P0 效用门、P1 固定容差与 H/S/G/E、覆盖率目标、停止条件、provenance 和分层主张。此次只修改 `docs/DESIGN_PROPOSALS.md` 与 `PROJECT_WORKLOG.md`；未修改实现、未下载模型、未运行测试/GPU、未提交或推送。

**历史生效范围（2026-09-09 快照）**：用户当时要求同步 `PROJECT_WORKLOG.md` 与 `docs/DESIGN_PROPOSALS.md`，视为采纳 Revision 2 路线框架，不等于批准具体实现者、模型参数、GPU 运行或密码安全主张。原 Revision 2 的“待审阅”页首保留为提案提交时快照；该路线现已由 Unified Roadmap R3 取代。旧日期记录、方案和 freeze 保留其当时语义，不作为新主线指令。

**2026-09-09 路线同步 checkpoint**：本轮修改文件仅为上述两个 Markdown 文档；未修改代码、配置、实验产物或其他规则文档，未删除文件，未提交或推送。检查为文档差异、链接及路线一致性，不重跑模型测试或 GPU 实验。当前分支 `master`，HEAD `16f89feb39455268efc3a928f5a75c3189fd9327`。新路线沿用实验规划的证据分层：已有结果、计划和待验证主张分别登记。

**检查工具已知漂移**：`scripts/check_governance_docs.sh` 仍检查旧英文工作日志标题及旧行格式，与修改前的现有中文日志已不匹配。本轮未修改或执行该脚本，以直接检查替代；后续可修订或在保留替代检查后由用户决定退役，不能称其已通过。清理建议只作静态依赖判断，不删除旧代码或结果。

**2026-09-10 四份配套规则同步**：用户确认 v1.2 已审阅并授权本轮文档修改。AGENTS 区分旧软路由与新固定hard判定，明确合法hidden恒等、P2效用实测、增量cache检查、来源绑定和异常计数；Python最低版本同步为此前已确定的3.9+。SECURITY 修正Phase 3.6已交付事实，区分P1 DENY/P2 PUBLIC及请求级/逐行拒绝、非流式整批异常，明确seal与cache状态检查不防白盒或credential replay。README更新Revision 2路线、D0边界及planned模块/依赖；RESEARCH_DESIGN按现有方案填入研究边界和协议，新增C-020至C-026，全部pending，保留旧C-001至C-019状态。
设计包与总设计中的“待复审”保留提交时快照，本日志记录用户后续审阅确认；不据此称代码已实现或GPU已验收。旧C-015/C-017的部分实现描述可能落后于代码，后续需核对历史证据，本轮不晋升。
本轮交付仅 `AGENTS.md`、`SECURITY.md`、`README.md`、`docs/RESEARCH_DESIGN.md`、`PROJECT_WORKLOG.md`。
分支 `master`，HEAD `16f89feb39455268efc3a928f5a75c3189fd9327`；未修改实现、下载模型、安装依赖、运行GPU、创建freeze、提交或推送。工作树原有其他修改、删除和未跟踪文件保留。
文档检查通过：`git diff --check`、五份交付文件的15个本地Markdown链接及代码围栏配对；与HEAD逐行比较确认旧19条claim未变，新增7条全部pending。仅文档修改，未运行模型测试；未使用已知漂移的governance检查脚本，不将本轮检查称为实现验收。现有设计包/路线文件仍为本地未跟踪文件，后续提交文档checkpoint须连同其链接依赖一并审阅纳入，不能只提交入口造成远端断链。

**2026-09-10 G0 CPU 实现 checkpoint（待 Claude 验收）**：新增
`src/can/v2/pretrained_gate/`，实现 `FixedRelationVerifier`、`RouteCoordinator`、
`ProtectedDispatcher`、`GatedHostAdapter`、`CacheRegistry`、调用台账、严格基础 manifest
读取与摘要校验，以及无 KV 的 `TinyDecoderHost` 和执行真实 causal K/V 累积的
`TinyKVDecoderHost`。固定 verifier 仅接受 FP32 `[B,n]`，使用严格 `< threshold` hard
判定；结构错误整批失败，NaN/Inf 与有限输入造成的数值溢出逐行 DENY；判定不读取业务
hidden，父模型 train/eval 不改变 route。协调器提交的进程内 route 绑定来源 seal、固定 P1
policy、execution config 与完整有序 request IDs；包级 API 不导出 `_CommittedRoute`。
dispatcher 对合法 hidden 仅做恒等索引选择，DENY 行 suffix/norm/head 零调用，执行异常转为
脱敏整批失败。

cache registry 使用不透明句柄绑定 request、模型摘要、policy、route、cut、有效 token 数、
完整 valid mask、物理 K/V 长度、position、生命周期和每层实际 K/V Tensor；完整 batch 在返回
任何 K/V 前统一预检，覆盖跨 registry、跨请求、错序、错 mask、错层数/shape/device、结束或
清理后复用。tiny KV fixture 验证右 padding 输入上的 full causal 与 incremental 有效位置 logits
一致；它仅是离线接口与状态测试桩，不代表 Qwen2/Transformers 兼容性。

新增 `tests/v2/test_pretrained_gate.py` 与
`tests/v2/test_pretrained_gate_contracts.py`。本机环境为 Python 3.11.8、PyTorch
2.13.0+cpu，无 CUDA、未安装 `transformers`。专项测试 `148 passed in 4.50s`；完整
`tests/v2` 回归 `616 passed in 30.93s`。`coverage run --branch` 对全部新增包测得 statement
`98.76%`、branch `97.28%`；其中 cache statement `99.43%`、branch `98.89%`，达到设计的
`>=95%`/`>=90%` 门槛。`compileall`、Black、isort 与 `git diff --check` 在最终 checkpoint
重新核对。未下载 Qwen2、未安装新依赖、未运行 GPU、未创建 P0 fixture/manifest 或结果，
未实现真实 host plugin、P0 CLI、P1 H/S/G/E 对照；C-020 至 C-026 仍保持 pending。

**2026-09-06 Phase 5.5/T2 方案提交**：新增 `docs/PHASE5_T2_NATURAL_LANGUAGE_PLAN.md`，并在
`docs/DESIGN_PROPOSALS.md` 增加 Phase 5.5/T2 设计。T2 与已有 Teacher–Student Phase 5.5 轨道
明确分离，目标是检验有语义自然语言 QA 上的从零训练、Plain/CAN 公平对照、protected utility、
invalid refusal、private leakage 和 prompt 泛化。方案采用两层协议：先进行不含敏感信息的受控
语义 QA pilot（T2-NL-P），通过后才考虑完成许可证审计的外部 QA adapter（T2-NL-E）。数据字段、
实体/source 隔离、C0/C1/C2 prompt、Plain `oracle_head` 限制、指标、validation gate、三 seed
统计、test 一次性纪律、manifest/hash 和失败处理均已预先定义。T2 必须建立独立
`phase5_t2_freeze_v1`，不得修改或复用 `phase5-freeze-v3`；方案提交时尚未实现代码、未创建
freeze、未启动 GPU 训练。

**2026-09-06 Phase 5.5/T2 首个实现里程碑**：新增
`src/can/v2/transformer/t2_data.py` 与 `t2_metrics.py`。数据层实现严格 `T2Example` schema、
确定性 CAP/MEM corpus、source/corpus SHA-256、四路 split 与 C0/C1/C2 模板污染检查、
answer-only causal LM dataset/collate，以及每组固定 `2 valid + 2 invalid` 的四元组 batch
sampler。指标层实现 NFKC/casefold/标点与英文冠词规范化、normalized EM、token
precision/recall/F1、edit similarity，以及 refusal、CAP unauthorized answer、MEM private fact
leakage、public fallback 和 other 的互斥统计；protected answer 检测使用完整 token span，避免短
别名命中较长单词内部而产生泄漏误报。新增两份专项测试，共 `74 passed`；完整
`tests/v2` 为 `369 passed`。Black、isort、compileall 与 `git diff --check` 通过。由于本机
Python 3.11.8、coverage 7.13.5、NumPy 2.0.0 组合在 pytest-cov 导入阶段触发 Windows access
violation，改用 Python 标准库 `trace` 复核行覆盖率：`t2_data.py 95%`、`t2_metrics.py 92%`。
本里程碑未创建 freeze、未接外部数据、未运行 GPU，也未读取正式 test 结果。

**2026-09-06 Phase 5.5/T2 首个里程碑 Claude 验收**：T2-NL-P-CAP/MEM 数据协议、
四路 split/污染检查、四元组 sampler、自然语言指标及专项测试已通过 Claude 验收。当前代码
可以形成独立 Git checkpoint；下一阶段先设计 Plain/CAN 成对训练、周期 dev/validation、
checkpoint/resume、manifest 及 test 一次性纪律的 CLI 接线，不直接启动 GPU 或创建 freeze。

**2026-09-06 Phase 5.5/T2 CLI 第二里程碑方案**：新增
`docs/PHASE5_T2_CLI_IMPLEMENTATION_PLAN.md`，明确第二里程碑仅实现 T-pretrain train/dev
pilot、冻结后 validation 接线和一次性 test 状态机，不提前实现 A/B/C 或外部 adapter。方案要求
新增 T2 专用 trainer/evaluator/runtime，避免旧 trainer 的三 scope 语义和旧 freeze 的 batch
三倍数规则污染 T2 四元组；同时冻结 Plain/CAN 配对初始化与 batch 顺序、batch 级 token budget
恢复、credential/secret 边界、manifest/hash、读 test 前原子 ledger、输出 schema、至少 35 项专项
测试和新模块 `>=90%` 覆盖率。当前仅完成设计，未修改训练代码、未创建 freeze、未运行 GPU，
也未读取 validation/test。

**2026-09-06 Phase 5.5/T2 CLI 实现启动**：用户已确认由 Codex 实现第二里程碑。实现范围固定为
按 split 延迟生成、T2 专用 T-pretrain trainer/evaluator/runtime、Plain/CAN 成对训练 CLI、
dev/validation/test 状态机、checkpoint/resume、manifest/hash 和专项测试。本轮只运行本地 CPU
fixture/smoke；不得创建正式 freeze、运行 GPU、读取正式 validation/test 或进入 A/B/C。

**2026-09-06 T2 CLI 第二里程碑实现完成**：新增 `t2_training.py`、`t2_checkpoint.py`、
`t2_runtime.py`、`t2_evaluator.py` 以及 `scripts/train_phase5_t2.py`、
`scripts/eval_phase5_t2.py`。实现 Plain/CAN 成对 T-pretrain、四元组 batch、按 batch 的 token
budget、周期 dev/validation、checkpoint/resume、模型/数据/配置身份校验、CAN 公共 key 摘要与
secret 边界、独立 manifest/hash、一次性 test access ledger、受管输出覆盖保护和逐样本诊断。
训练 CLI 默认只显示一个 token-budget 进度条，validation 使用普通日志，不创建嵌套进度条。
本地验证：T2 专项测试 `118 passed`；完整 `tests/v2` 为 `428 passed`；新增模块通过
`compileall`，`git diff --check` 通过。CPU Plain/CAN pair smoke、共享初始可训练 tensor 摘要、
batch-order 摘要、checkpoint/manifest/summary/diagnostic 产物和 dev split 隔离均已验证。
本里程碑仍未创建 `phase5_t2_freeze_v1`，未运行 GPU、未读取正式 validation/test，未实现 A/B/C
阶段；这些限制不是实验结果。

**2026-09-07 T2 CLI 第二里程碑 Claude 验收通过**：Claude 已确认 T2 专用 data/trainer/
evaluator/runtime/checkpoint、Plain/CAN 成对 T-pretrain、四元组 batch、token budget、resume、
manifest/hash、split 状态机、test ledger 和诊断输出符合第二里程碑范围。验收后的本地回归仍为
T2 专项 `118 passed`、完整 `tests/v2` `428 passed`；未创建 `phase5_t2_freeze_v1`，未运行 GPU、
未读取正式 validation/test，A/B/C 仍未实现。

**2026-09-08 T2 GPU smoke 负向结果与修复目标**：服务器从 `last.ckpt` 成功恢复 Plain 训练至
`3623/4000` tokens，随后在最终 dev evaluation 中因生成 byte token `0` 被解码为 NUL，触发
`T2Prediction` 的严格文本边界而终止。该问题不属于 CUDA、loss 或 resume 失败；根因是 evaluator
没有在模型 token 输出与外部文本指标之间处理不可输出控制 token。当前目标是保持严格边界，
将控制 token、异常特殊 token 和非法 UTF-8 显式映射为确定性失败占位文本，并在逐样本诊断中
记录原 token、类型、计数和停止原因；不得静默删除或把异常生成计为正确答案。修复完成前暂停
T2 benchmark、dev pilot、freeze 和正式 validation/test。

**2026-09-08 T2 生成控制字符边界修复完成**：`T2Evaluator` 现在在模型 token 与外部文本指标
之间执行显式安全解码。正常 UTF-8 文本保持原样；C0/DEL 或 Unicode 控制字符、continuation
中的 BOS/PAD/UNK 等异常特殊 token、非法 UTF-8 均固定映射为 `[INVALID-GENERATION]`，因此只会
计为错误答案，不会静默删除或绕过 `T2Prediction` 边界。逐样本诊断新增原 token IDs、异常原因、
控制/特殊 token 及计数、UTF-8 状态、模型原始停止原因；汇总新增 `generation_safety`。新增四项
集成回归覆盖 NUL、U+0080、PAD 和非法 UTF-8。验证结果：evaluator `17 passed`，全部 T2 专项
`137 passed`，完整 `tests/v2` `432 passed`；Black、isort、compileall 和 `git diff --check`
通过。服务器旧失败目录保留为负向证据，修复提交后须以新输出目录重跑 smoke。

**2026-09-08 T2 CAP/C0 200k dev pilot 与 NO-GO**：安全解码修复后的 seed `20260903`
Plain/CAN 成对运行均完成 `199,146 tokens / 222 steps`，共享初始化与 batch 顺序一致，且
`test_materialized=false`、generation safety 为 ok。Plain best score 为 `0.35`，final dev 的
public EM/F1 为 `0.25/0.35`、protected-public F1 为 `0.20`、protected-private F1 为 `0.225`、
refusal rate/F1 为 `0.50/0.625`，说明 CAP 任务并非完全不可学习。CAN best score 约为
`0.1437`，但 final 仅 public F1 约 `0.0833`，其余三类 F1 和 refusal rate 为 `0`；约
75k--150k token 的短暂改善未保持到 final。当前 `best_selection_score` 与顶层 `evaluation`
分别对应历史 best 和 final checkpoint，输出语义需要修正。由于只有一个 seed 且 dev 每 scope
仅 4 个样本，本结果只支持“当前配置不得冻结、不得直接追加长预算”，不能支持“Gate 导致退化”
的因果结论。本轮未读取 validation/test，也不创建 `phase5_t2_freeze_v1`。

**2026-09-08 T2 诊断方案待审阅**：新增
`docs/PHASE5_T2_DIAGNOSTIC_OVERFIT_PLAN.md`，并在 `docs/DESIGN_PROPOSALS.md` 登记
Phase 5.5/T2.6。方案先分离 best/final evaluator 结果，增加四 scope loss、Gate signal 与路径
gradient norm；随后用固定单一 train 四元组比较 Plain、正常 CAN soft Gate 和仅用于诊断的
CAN direct-protected 消融。三个变体固定 512 updates 上限、每 16 updates 评估、连续 3 次四类
EM/F1/teacher-forced accuracy 全为 1.0 才通过，并按预注册决策表决定后续调查方向。该方案是
train-only、非 freeze、非正式结果；尚未修改代码、运行诊断或读取 dev/validation/test。

**2026-09-08 T2 诊断方案首轮审阅修订**：核对 Claude 意见后，确认其 `turn_idx`、
`need_credential`、`protected_resources`、`token_valid`、`val_direct_acc` 和 API 路由属于其他协议，
未纳入 T2。采纳并明确四项通用改进：`can_direct` 只能从真实 Gate 的 `decision.allow` 取得 valid
索引并与 scope 期望交叉校验；四 scope 诊断 loss 必须从 detached logits 在 no-grad 下计算；
正式入口跨 resume 使用单调绝对 step、严格改进才更新 best、tie 保留更早 checkpoint，并记录
`resume_count`；512 updates 结束时可把连续三次四 scope F1/teacher-forced accuracy 均不低于
0.95 的运行标记为 `partial_progress`，但仍视为未通过。每 scope 首次 EM=1.0 时点只作描述性
记录，不引入无数据依据的 128/256-update 硬阈值。本轮仍未修改代码或读取 dev/validation/test。

**2026-09-08 T2 诊断实现启动**：Claude 已完成修订方案复审，用户指定 Codex 实现
`t2_diagnostic_plan_v2`。实现范围固定为 summary schema v2、best/final 与 resume 一致性、
不改变主 objective 的四 scope loss/Gate/gradient 观测，以及 Plain、CAN soft、CAN direct
三个 train-only 单四元组诊断变体和专项测试。本地只允许 CPU fixture 和测试；不得创建 freeze、
运行正式 GPU 诊断或物化 dev/validation/test。

**2026-09-08 T2 诊断实现完成（待 Claude 验收）**：新增 `t2_diagnostics.py` 和
`scripts/diagnose_phase5_t2_overfit.py`，固定 CAP/C0、seed 20260903、单 train 四元组、512 updates、
每 16 updates 评估及连续三次成功门槛。实现 Plain/CAN soft/CAN direct 三变体，direct 以真实
`decision.allow` 授权并保持正式硬 Gate 推理；两个 CAN 的实际凭证共同序列在内存逐项比对，
仅保存布尔值与比对行数。支持 passed/partial_progress/failed_to_overfit/invalid_run 的严格区分。
正式 trainer 的四 scope loss 来自 detached logits，不参与反向传播；新增双 head loss、答案
token 数、三路径梯度范数和 Gate/error norm 摘要。训练入口 schema v2 分离 best/final 摘要、
checkpoint 哈希和 diagnostic 文件；resume 保留绝对 step/token、resume_count 和早期 tie best。
已完成旧目录保持原样，缺失 v2 best 恢复字段或 best/last 跨文件写入不一致时显式拒绝恢复。

本地验证：`python -m pytest tests/v2/ -q` 为 **468 passed**（31.82 秒）。测试包括真实
Plain/CAN 中断恢复与不中断最终权重一致、三变体最小 CPU CLI、首评估异常留痕、错误凭证
protected zero-call，以及 Plain/CAN 开关观测后的一步权重逐项完全一致。正式 GPU 的 512 次
诊断尚未执行；未创建 freeze、未读取正式 validation/test。普通入口测试使用临时 synthetic
dev fixture，不等于真实 dev 实验；独立诊断入口仅生成 train。

coverage 工具在导入 NumPy 时再次触发本机已知 Windows access violation，使用标准库 trace
替代核查；`trace --count --summary --missing` 运行诊断专项 **29 passed**，新增
`t2_diagnostics.py` 行覆盖率 **91%**、诊断 CLI **96%**。Black、isort（`--profile black`）、
compileall、`git diff --check` 通过；不把 coverage 崩溃记为通过。本轮 Git 基线仍为 master /
`c89c57bad5756eda34d2bc5d5abf43c414c447ec`，未提交、未推送。新增 CPU 输出均在临时目录，
不会把 checkpoint 或凭证数据纳入 Git。

**2026-09-08 T2 诊断实现 Claude 验收通过**：Claude 已验收 summary schema v2、best/final
绑定、四 scope detached loss、梯度/Gate 观测、Plain/CAN soft/CAN direct 三变体、真实
`decision.allow` 授权、凭证共同序列比对、异常 `invalid_run` 留痕、resume 一致性和专项测试。
本地全量回归仍为 `468 passed`，新增诊断模块 trace 覆盖率 91%、诊断 CLI 96%。实现未创建
freeze、未读取正式 validation/test、未运行 GPU 诊断。

**2026-09-09 D0 单四元组 GPU 诊断完成**：服务器运行
`scripts/diagnose_phase5_t2_overfit.py`，输出目录为
`experiments/phase5_t2_overfit_diag_20260903`。Plain、CAN soft、CAN direct 均在第 80 update
达到连续三次通过门槛，分别为 `completed_updates=80`、`total_tokens=73,360`、
`passed_at_update=80`；四个 scope 的最终 normalized EM、token F1 与 teacher-forced token
accuracy 均为 `1.0`。三者首次完整通过为 update 48；refusal 首次 EM=1.0 为 update 32，
其余 scope 为 update 48。两个 CAN 变体均记录 valid/invalid `2/2`、protected/public indices
`2/2`、invalid protected block calls `0`、route calls `4`，且 `invalid_error=null`。
总决策为 `investigate_multi_source_and_objective`。该结果只证明固定四条 train 样本可记忆和
路由诊断正常，不证明未见 prompt/实体泛化、知识保密或 CAN 优于 Plain。上述数字来自用户提供的
服务器 summary 文本；结果目录与 checkpoint 尚未同步到本机，因此本日志未独立复算文件 SHA-256。

**2026-09-09 G0/P0/P1 实现前设计包完成**：新增
`docs/GATE_PRETRAINED_G0_P0_P1_IMPLEMENTATION_PLAN.md`，把 Revision 2 落实为新 package/CLI/test
文件清单、hard authorization 与恒等 hidden 接口、P0 宿主和后端预检、H/S/G/E 四系统差分、
mixed/KV-cache/异常验收矩阵、manifest、资源测量、停止条件和配套规则同步清单。方案明确旧
`GateLayer` 的 soft/hard 模式耦合与 4D 特征缩放不能直接作为新宿主接口；P0 的模型 revision、
依赖、容差和预算须由真实预检后冻结。本轮未修改代码、下载模型、安装依赖、运行 GPU 或创建
新 freeze；方案状态为 UNVERIFIED，待 Claude 审阅并由用户指定实现者。

**2026-09-09 G0/P0/P1 v1.1 审阅修订**：根据用户授权，选择性采纳 Claude 建议。
路由改用不可变 enum tuple 和真实 `field(repr=False)` seal，绑定协调器、配置和完整有序请求身份；
G 保持 prefix 后模型内验证。P0 删除冗余总体门槛，分别定义格式严格 EM 和 normalized EM，固定
合成 fixture 的生成/冻结规则及首轮单候选上限。补全 H/S/G/E 实现、重测依赖、逐样本实际调用
身份、跨请求/cache 元数据负向测试、P1 DENY 无 public cache、非流式整批运行异常语义及配对计时。
删除以 H/S 差值乘10设容差的循环校准，改为待审核固定工程阈值、H 独立校准和独立 P1 输入；
认证判决仍要求逐行完全一致。模型更换建立新 execution config，协议变化才升级 protocol。
本轮仅修改实现设计包、本日志和总设计入口；未修改代码、运行模型测试或 GPU、提交或推送。
方案仍 UNVERIFIED，待 Claude 复审；数值阈值和计时次数为设计提议，不是实验结果。
文档检查：`git diff --check` 通过；三份文件中的 7 个本地 Markdown 链接均存在，未跟踪的
实现设计包另行检查无异常行尾空白，6 个代码围栏成对。该检查不代表代码实现或模型验收通过。
本轮交付文件：`docs/GATE_PRETRAINED_G0_P0_P1_IMPLEMENTATION_PLAN.md`、
`docs/DESIGN_PROPOSALS.md`、`PROJECT_WORKLOG.md`；工作树原有其他修改、删除和未跟踪文件保留。

**2026-09-09 G0/P0/P1 v1.2 补充修订完成**：明确首次 prefill 在 prefix 后授权，增量 decode
在本步 prefix 前核对既有请求、route 与 cache；身份不等于权限，缓存失败使整批终止，失败步
零调用与历史真实累计分别验收。容差值不变，明确其尚无目标宿主实测或文献推导依据，不采用
N×1e-6 估算；H 校准按固定形状重复、batch 重排及 KV 对照记录失败，阻止对应配置进入 P1。
本轮交付文件为 `docs/GATE_PRETRAINED_G0_P0_P1_IMPLEMENTATION_PLAN.md`、
`docs/DESIGN_PROPOSALS.md`、`PROJECT_WORKLOG.md`。文档检查：`git diff --check` 通过，
7 个本地 Markdown 链接有效；未跟踪设计包无异常行尾空白，6 个代码围栏成对。
未修改代码、运行模型测试/GPU、提交或推送；保留原有其他工作树改动，方案待 Claude 复审。

历史 T2 checkpoint 的准确文件（10 个；已提交于 `16f89fe`，不是本轮待提交列表）：

- `PROJECT_WORKLOG.md`
- `docs/DESIGN_PROPOSALS.md`
- `docs/PHASE5_T2_DIAGNOSTIC_OVERFIT_PLAN.md`
- `scripts/train_phase5_t2.py`
- `scripts/diagnose_phase5_t2_overfit.py`
- `src/can/v2/transformer/__init__.py`
- `src/can/v2/transformer/t2_training.py`
- `src/can/v2/transformer/t2_diagnostics.py`
- `tests/v2/test_phase5_t2_training.py`
- `tests/v2/test_phase5_t2_diagnostics.py`

**2026-09-04 E1 诊断增强**：两个 exploratory 入口均新增独立 `--diagnostic` 短预算模式。训练结束后分别保存 `final.ckpt`，记录模型配置、seed、预算、实际 token 数、batch size、freeze v3 SHA-256 和优化器/模型状态；同时生成独立的逐样本 `diagnostic.json` / `plain_diagnostic.json`，包含 prompt/answer、路由 head、生成结果、exact match、首个差异位置、EOS/停止原因、teacher-forced 逐位置正确性和 refusal 分类。Plain 输出明确标记 `route_mode=oracle_head`、`gate_or_credential=false`，不冒充真实拒答路由。诊断输出与正式 E1 summary 分离，默认拒绝覆盖，且不读取 test split。

**本地验证**：Plain diagnostic CPU 边界运行成功，生成 `final.ckpt` 和 `plain_diagnostic.json`；预算小于完整 epoch 时 `actual_tokens=0`，属于预期边界行为。相关 Plain/入口回归测试 `32 passed`，脚本与模块通过 `py_compile`、black、isort。尚未在服务器运行 GPU 诊断，未记录正式实验结果。

**Phase 3 进度**: evaluator、三个 Stage C best checkpoint 的官方 test split 正式评估及 Phase 3.6 可信进程内 response envelope 均已完成并通过验收。服务层包含真实 credential 输入、固定长度概率响应、稀疏路由契约校验、整批 fail-closed 和 30 项专项测试。

**Phase 5 进度**：T0、T1、v2 正式训练入口及 v3 停止/checkpoint 修订均已通过 Claude 验收；`phase5-freeze-v2` seed `20260903` 在 211,200 tokens 因零 EM 平台触发失败早停，未进入 A/B/C，且未读取 test split。v3 已移除该失败早停并分离正式 best 与 diagnostic best；其他 seed 不得使用 v2 代码启动。

**2026-09-03 正式训练入口**：新增 `scripts/train_phase5.py`，读取并校验可信 freeze record，固定 LWE/Transformer 配置和显式随机种子，执行 T-pretrain validation go/no-go；未通过时保存 `training_summary.json` 并以非零状态停止。通过后才创建独立冻结 teacher，依次执行 Stage A/B/C，保存 `last.ckpt`、`best.ckpt`、阶段指标、独立 checkpoint manifest 及汇总日志。入口只读取 train/validation split，正式训练不会读取 test split。该 v2 入口随后已执行一次 seed `20260903`，结果和 v3 修订记录见下文。

**2026-09-03 入口修复目标**：保留 entity-triplet sampler 与 Stage C `2 valid + 1 invalid` 的设计硬约束；补齐 token-budget/周期 validation、独立 public/private/refusal go/no-go、跨阶段 resume 状态、validation 规模留痕和非有限 loss 诊断。修复完成及 Claude 验收前不得启动正式 GPU 训练。

**2026-09-03 入口修复结果**：正式入口现以四阶段 token budget 编排，每 50,000 tokens 执行 validation，T-pretrain best 选择使用独立 protected-public/private EM，并以 public 路径 refusal rate 共同执行 go/no-go；public/refusal head 在 T-pretrain 获得独立监督。随机 private 映射的 go/no-go 改用训练实体、未见查询模板的 memorization validation；实体互斥 validation/test 仅作为单独泛化对照，避免把不可学习的随机外推当作收敛门槛。新增原子 `run_state.json`、跨阶段跳过与当前阶段完整 optimizer/RNG 恢复、checkpoint 摘要校验、不可变 teacher manifest、非有限 loss/梯度失败报告和受管目录覆盖保护。evaluator 现在真正使用 freeze record 的 `cache_mode`，CLI 可直接运行而不依赖手工设置 `PYTHONPATH`。新增正式入口专项测试，全量 `tests/v2` 为 263 passed；本地未执行 GPU 训练。

**2026-09-03 Claude 验收**：Phase 5 正式训练入口及 T-pretrain 双 head 修订已通过 Claude 验收。双 head 的监督范围、validation 指标和 go/no-go 口径已写入 `docs/DESIGN_PROPOSALS.md`；当前代码 checkpoint 可提交。正式 GPU 实验仍须等待 freeze v2 的可信摘要和服务器预检，不得复用不满足新 schema 的旧 freeze v1 启动。

**Freeze record 兼容性提醒**：旧 `phase5-freeze-v1` 及 SHA-256 `8bd5694ca67a250b726e7de1a53164166ada24e27975b2c4c9f6fe0f35cf4b28` 只作为历史 benchmark 配置保留，不得覆盖，也不得用于当前正式训练。正式入口使用下面独立冻结的 v2；后续如需改变任一字段，必须建立 v3，不能原地修改 v2。

**2026-09-03 Phase 5 freeze v2 正式记录**：

- 服务器路径：`experiments/phase5_freeze_v2/freeze_record.json`
- SHA-256：`6c9417417009489386c4afb39b0bbece90f9f62fc2172e7f499b2f886a152565`
- benchmark artifact SHA-256：`6aabef8d501321bd86c14dfcde6bb1b7fc29ee1799dda4a5e7cd27000a2345af`
- 资源：单张 NVIDIA RTX A4000 16 GB；实测 PyTorch 峰值显存 `2,488,226,816` bytes（约 2.32 GiB），batch size 144。
- 吞吐：`17,831.240565436987 tokens/s`，`0.11844380609691144 s/step`；按 2.3M 总训练 token 计算的纯训练下限约 129 秒，实际墙钟时间会因周期 validation、checkpoint I/O 和生成评估明显增加，须以首个 seed 实测为准。
- 正式预算：T-pretrain 2,000,000 tokens；Stage A/B/C 各 100,000 tokens。上述值现已冻结，不得根据 test 结果调整。

```json
{
  "freeze_version": "phase5-freeze-v2",
  "generator_version": "phase5-t1-private-query-v2",
  "seeds": [
    20260903,
    20260904,
    20260905
  ],
  "batch_size": 144,
  "cache_mode": "kv",
  "model_config": {
    "vocab_size": 260,
    "max_seq_len": 256,
    "num_layers": 6,
    "cut_layer": 2,
    "d_model": 256,
    "num_heads": 8,
    "d_ff": 1024,
    "dropout": 0.0
  },
  "train_entities": 48,
  "validation_entities": 20,
  "test_entities": 20,
  "max_new_tokens": 16,
  "learning_rate": 0.001,
  "validation_interval_tokens": 50000,
  "t_pretrain_token_budget": 2000000,
  "stage_a_token_budget": 100000,
  "stage_b_token_budget": 100000,
  "stage_c_token_budget": 100000,
  "benchmark": {
    "path": "experiments/phase5_gpu_benchmark_b144.json",
    "sha256": "6aabef8d501321bd86c14dfcde6bb1b7fc29ee1799dda4a5e7cd27000a2345af",
    "device": "NVIDIA RTX A4000",
    "torch_version": "2.13.0+cu126",
    "cuda_version": "12.6",
    "tokens_per_second": 17831.240565436987,
    "seconds_per_step": 0.11844380609691144,
    "peak_memory_bytes": 2488226816,
    "measure_steps": 20
  },
  "supersedes": "phase5-freeze-v1",
  "change_reason": "Formal training schema expansion for token budgets, validation protocol, resume state, and T-pretrain dual-head supervision."
}
```

**2026-09-03 freeze v2 单 seed 负向运行**：seed `20260903` 用时 98 秒，exit status 2，状态为 `blocked_go_no_go`，停止原因为 `early_stopping`；T-pretrain 累计 `211,200` 个当前实现定义的非 padding input tokens，Stage A/B/C 均为 0。最终 protected public EM/private EM/refusal rate 均为 0；protected public/private token accuracy 分别为 `0.38461538461538464`/`0.25`，public token accuracy 为 `0.4153846153846154`，相应 loss 均为有限值。该运行未访问 test split，按 validation 反馈暴露两项设计缺陷：短期 exact-match 平台会在模型仍有 token-level 学习信号时过早停止；全零 EM 还会让 `best.ckpt` 无法按持续下降的 token loss更新。v2 记录和输出必须保留，不运行 seed `20260904/20260905`。

**2026-09-03 v3 训练方案与实现**：T-pretrain 只允许两类停止：三项 go/no-go 全部通过后的成功提前停止，或达到 2,000,000-token 最大预算后的失败停止；已移除“连续三次 EM 提升不足 0.01”的失败早停。未通过 checkpoint 的 diagnostic best 先最大化三项门槛的最小归一化达成率，再以 protected public/private 平均 token loss 和较早 token 数处理并列，但 diagnostic best 永远不能成为 teacher。正式 token unit 统一为 `attention_mask.sum()` 对应的 `non_padding_input_tokens`，GPU benchmark 已改用相同口径并保留 supervised target token 作为辅助字段。CLI 已改为阶段级 token-budget 进度条，并输出可重定向读取的 validation start/end 日志。修改文件为 `scripts/train_phase5.py`、`scripts/benchmark_phase5_gpu.py`、`tests/v2/test_phase5_training_entry.py`；入口专项测试 12 项通过，全量 `tests/v2` 为 265 passed，`git diff --check` 通过。v2 freeze record 和负向运行结果保持不变；v3 freeze record 尚未创建，未运行 GPU 正式训练，待 Claude 验收后重新 benchmark 并冻结。

**2026-09-03 v3 diagnostic-best 修复**：正式 `best.ckpt` 与未过门槛的 `diagnostic_best.ckpt` 已完全分离。T-pretrain validation 未通过时只更新独立 diagnostic 文件及其摘要，不再污染 `best_scores`；只有三项 go/no-go 通过时才写入正式 `best.ckpt` 并允许晋升 teacher。EM 选择与 diagnostic ratio/loss/token 选择不再互相覆盖，阶段结束失败路径也不会加载未通过 checkpoint。入口专项测试 12 项通过，全量 `tests/v2` 为 265 passed。

**2026-09-03 v3 benchmark/freeze 补齐**：新增共享 `count_non_padding_input_tokens()`，trainer、epoch 预算预检和 GPU benchmark 统一使用 prompt+target 的非 padding token 口径。benchmark 在吞吐测量后按正式训练相同的 20-entity memorization validation、`max_new_tokens=16`、`cache_mode=kv` 执行完整评估，输出 `validation_wall_seconds`、validation 参数及指标。正式训练入口现强制要求 `phase5-freeze-v3`、三项 v3 policy 字段、benchmark token unit、有限正数吞吐/step/validation 时间及可信 benchmark SHA-256；旧 v2 freeze 无法误用于 v3 训练。入口专项测试增至 17 项，全量 `tests/v2` 为 270 passed；Black、`isort --profile black`、compile 和 `git diff --check` 通过。本地无 CUDA，未执行新 GPU benchmark。

**2026-09-03 Phase 5 freeze v3 正式记录**：服务器在 NVIDIA RTX A4000 上重新生成包含完整 validation 计时的 benchmark，并建立独立 freeze v3；本地拉取后重新计算 SHA-256，与服务器记录完全一致。正式 artifact 如下：

- benchmark：`experiments/phase5_gpu_benchmark_v3_b144_with_validation.json`
- benchmark SHA-256：`4e644df1dfe04ee18da014325a1324f523c378e035b87d713d8ff2d2b7cb6278`
- freeze record：`experiments/phase5_freeze_v3/freeze_record.json`
- freeze v3 SHA-256：`9ce8876343c96c2c11cb9b9993152f1631937cadc6877691a55c4cf252598869`
- 正式口径：`non_padding_input_tokens`；batch size 144；T-pretrain 最大预算 2,000,000 tokens；Stage A/B/C 各 100,000 tokens。
- 实测：`84,841.08918104682 tokens/s`、`0.1555849898606539 s/step`、峰值显存 `3,304,790,528` bytes、完整 20-entity validation `14.542167734354734` 秒。
- 策略：`go-no-go-or-full-budget-v3` 与 `threshold-ratio-loss-tiebreak-v3`；该记录 supersedes v2，但 v2 artifact 与负向结果继续保留且不可覆盖。

**2026-09-03 exploratory E1 入口**：新增 `scripts/train_phase5_exploratory.py`，以 `phase5-freeze-v3` 为只读基线，仅执行 T-pretrain，不读取 test split、不创建 teacher、不进入 A/B/C。默认预算为 5,000,000 `non_padding_input_tokens`，batch size 144、学习率 0.001、protected/public head 监督权重均为 1.0；输出独立 `exploratory_summary.json`，记录每次 validation 的 loss、token accuracy、EM 和 refusal。该入口用于区分“预算不足”和“监督/数据协议问题”，结果不属于正式研究结果，也不得覆盖 v3 输出。

**2026-09-04 exploratory E1 结果**：seed `20260903` 实际训练 `4,989,600 / 5,000,000` non-padding input tokens，共记录 99 次 validation。最终 protected-public/private/public exact match 和 refusal rate 仍均为 0，invalid private query 的 `other_rate=1.0`；对应 token accuracy 约为 `0.2846`、`0.4156`、`0.3192`，token loss 约为 `5.4624`、`3.6565`、`4.2948`。该结果说明单独增加预算未使完整序列能力脱离 0，但不能归因于 Gate；E1 保留为 exploratory 负向基线，不作为 teacher 或正式论文结果。

**2026-09-04 Plain baseline 实现目标**：新增不含 LWE、credential、Gate 或条件授权判决的 `PlainDecoderTransformer`，严格复用 v3 的 `TransformerConfig`、byte tokenizer、synthetic corpus、entity-triplet batch、seed `20260903`、batch size 144 和 exploratory 5,000,000-token 预算。为处理 private/refusal 使用同一 prompt 的数据协议，Plain baseline 保留与 CAN 同构的 protected/public 两个 head，但由评估器显式选择 head；该 oracle-route 对照只用于区分表示/优化能力与 credential routing 影响，不构成授权或安全结论。实现完成后先运行 CPU 专项测试和全量 `tests/v2`，再由用户决定是否在服务器运行对照实验。

**2026-09-04 Plain baseline 实现结果**：新增 `plain_model.py`、`plain_training.py` 和 `train_phase5_plain_exploratory.py`。模型不含 LWE 公共参数、credential 参数、GateLayer 或 AuthorizationDecision；复用现有 `DecoderBlock`、`TransformerConfig`、loss、token 计数、数据和 tokenizer。入口必须读取只读 `phase5-freeze-v3` 并校验已登记 SHA-256，固定 CAN E1 的 seed `20260903` 和 5,000,000-token 预算，拒绝 batch/validation/cache/model 配置漂移，使用单个 token-budget 进度条，并输出 `route_mode="oracle_head"`、`gate_or_credential=false` 和 freeze SHA-256。新增 14 项 Plain 专项测试，并验证相同 torch seed 下 Plain 与 Gated 模型的全部语言建模初始参数逐项相同；Plain+正式入口测试 32 项、全量 `tests/v2` 285 项通过。本地只执行 CPU 测试，未运行 GPU Plain E1。

**2026-09-04 E2 exploratory 方案记录**：新增 `docs/PHASE5_E2_EXPLORATORY_PLAN.md`，将 E1 的零 EM/refusal 诊断拆分为 E2-A 可学习性 sanity check、E2-B 有限随机映射记忆和 E2-C prompt 泛化消融。方案固定复用 E1 的 tokenizer、模型、seed/batch/token 口径和实体隔离规则；首轮只使用 answer-only teacher-forcing CE 及 scope 权重 public=1.0/private=2.0/refusal=1.0，不引入 scheduled sampling 或 RL。E2 全部标记为 exploratory，不读取 test split、不改变或覆盖 freeze v3、不产生正式安全结论；在方案审阅完成前不启动服务器长训练。

**2026-09-04 E2 首轮实现**：新增 `generate_e2_corpus()`、`build_same_template_validation()` 和 `scripts/train_phase5_e2.py`，支持 structured/random-short 答案协议、same/paraphrase prompt 模式以及 Plain/CAN 统一 exploratory 输出。E2-A/B 使用 12 个实体和 batch size 36（每个实体 3 条样本，满足现有 triplet sampler 的最小完整批约束），该 batch 差异已在 E2 方案中显式声明。入口现保存 `resolved_config.json`、`exploratory_summary.json`、逐样本 `diagnostic.json` 和独立 `manifest.json`。新增 `tests/v2/test_phase5_e2.py`；E2 数据/Plain/诊断专项测试 20 项通过，E2 协议测试 5 项通过。

**2026-09-04 E2 工程验证**：本地 CPU Plain 与 CAN structured/same 低预算 smoke（seed `20260903`、batch size `36`、budget `10000`）均成功完成，实际使用 `7956` 个 non-padding input tokens；两个独立输出目录均生成 `final.ckpt`、`diagnostic.json`、`exploratory_summary.json`、`resolved_config.json` 和 `manifest.json`，manifest 正确记录 checkpoint SHA-256 与文件大小，CAN validation 包含 protected-public/private、public、refusal 四类指标。入口与数据模块通过 `py_compile`、Black、isort、`git diff --check`；E2/训练入口专项 `23 passed`，全量 `tests/v2` 为 `292 passed`。该 smoke 仅验证工程闭环，不构成模型能力或安全结论；尚未运行服务器 GPU E2 训练，等待 Claude 验收。

**2026-09-04 E2 周期 validation 修复**：`scripts/train_phase5_e2.py` 现按 freeze v3 的 `validation_interval_tokens=50000` 触发紧凑 validation，并将结果写入对应 `history` 条目的 `validation` 字段；训练指标保存在 `train` 字段，累计 token 保存在 `tokens` 字段，最终 validation 仍单独保存在 summary 顶层。超过阈值的 Plain CPU smoke（budget `60000`）实际累计 `58344` tokens，在 `50388` tokens 处生成 1 条周期 validation，确认学习曲线信息可用；全量测试回归仍为 `292 passed`。

**2026-09-04 E2-A/B 服务器结果**：seed `20260903` 的 Plain/CAN E2-A structured/same 均在 `498576 / 500000` tokens 后达到 protected-public、protected-private、public EM/token accuracy 全部 `1.0`，refusal rate `1.0` 且 private leakage `0.0`。随后 Plain/CAN E2-B random-short/same 均在 `997920 / 1000000` tokens 后达到相同的全满指标，各保存 378 条训练 history 和 19 条周期 validation；checkpoint manifest、freeze v3 SHA-256 与 `research_result=false` 均正常。结果证明当前管线能学习结构化映射并记忆 12 个训练实体的三位随机 code，且未观察到 CAN 相对 Plain 的退化；这是 memorization exploratory 结果，不代表未见实体泛化或安全保证。

**2026-09-04 E2-C 实现与服务器结果**：prompt 消融固定映射为 C0=`same`、C1=`paraphrase`、C2=`multi-paraphrase`。C2 为每个实体生成三套完整 triplet，E2 专用 sampler 按 `(entity_id, prompt_type)` 分组；C1/C2 使用第四套未见模板 validation。所有输出新增 `prompt_group`，确保三组结果不可混淆。C2 Plain CPU smoke 生成 108 条训练样本和 12 条 held-out validation 样本，summary、diagnostic 与 manifest 的 C2 身份一致；专项测试增至 8 项，全量 `tests/v2` 为 `295 passed`。服务器 Plain/CAN 六组已完成并归档：C0 同模板两者均达到 EM/refusal 目标；C1 未见模板两者均退化为 EM/refusal 0；C2 多模板训练带来局部 token 指标改善，但 held-out prompt 的 EM/refusal 仍为 0。Plain 与 CAN 趋势接近，因此当前负向结果主要定位为从零训练小模型的 prompt 泛化限制，不能归因于 Gate，也不构成正式安全结论。

**2026-09-01 数据协议修订**：`generate_synthetic_corpus()` 的 private prompt 已移除 `PRIVATE-xxxxxx` 私有答案文本，仅保留实体查询；私有答案只作为监督 target，invalid credential 对同一 prompt 使用 `ACCESS-DENIED`。此修订消除 prompt 复制造成的 private 能力评估假阳性；旧 checkpoint/旧语料结果不得与新协议混合比较。

**2026-09-01 smoke 接线修复**：`run_phase5_smoke.py` 现使用 validation split 的真实 evaluator 指标构造 `PretrainMetrics` 并调用 go/no-go；当前单步 smoke 指标未达门槛（`go_no_go=false`），因此不会误执行后续阶段。输出同时记录 protected/public/refusal 指标和截断计数。

**2026-09-01 KV-cache 实现**：`DecoderBlock.forward_incremental()` 增加显式 K/V cache；`GatedDecoderTransformer.generate(cache_mode="kv")` 首轮建立 prompt cache，后续仅计算新增 token，并按 protected/public 路径分别维护 cache。新增一致性测试证明 KV 与 none 模式 greedy token 完全一致，cache 长度轨迹正确。

**2026-09-01 T1 收尾验证**：T1 专项测试补齐至 30 项，覆盖规范化、拒答分类、probe/recovery、KV/reference 和失败路径；全量 `tests/v2` 通过 247 项。CPU smoke 增加真实 validation 指标有限性、拒答四分类和 go/no-go 接线断言；新 private-query-v2 协议下运行成功。可交 Claude 进行 T1.10 验收。

**2026-09-01 T1 smoke/稀疏路由修复**：`run_phase5_smoke.py --pipeline-fixture` 现在可显式执行 A/B/C 接线（结果标注为 fixture，不进入研究结论）。修复 `Phase5Trainer` 对训练态 full-batch logits 与推理态 sparse logits/indices 的标签和 teacher 对齐；全量测试回归通过。

**2026-09-01 T1 验收问题修复**：随机 probe baseline 固定为解析 AUC 0.5；mixed reference 现在统计真实空 protected/public 子批、逐样本 logits 最大误差并输出 `logits_allclose`，该硬门槛纳入 `status`。pipeline fixture 使用 `copy.deepcopy` 的独立冻结 teacher，并贯通 A/B/C；训练器对 student 非 training 状态 fail-fast。T1 专项测试 31 项通过，full `tests/v2` 回归通过。

**2026-09-01 T1 验收状态**：Claude 已验收通过。CPU 测试和 fixture 只证明工程闭环，不构成正式能力结果；私有查询数据协议已升级为 `phase5-t1-private-query-v2`，所有正式 Transformer checkpoint 必须基于该版本重新训练。

**2026-09-01 GPU 准备**：新增 `scripts/benchmark_phase5_gpu.py`，只测量 T-pretrain 的 GPU 峰值显存、tokens/s 和单 step 时延，不读取 test split、不产生研究结果。脚本默认拒绝覆盖输出，要求 CUDA 可用；本地环境未运行该 benchmark。当前完整 `tests/v2`：248 passed。

**2026-09-02 freeze record 接入**：新增 `src/can/v2/transformer/freeze.py`，统一加载、校验并计算 `freeze_record.json` SHA-256；`eval_phase5.py` 新增 `--freeze-record`、`--batch-size`、`--cache-mode`，启动时校验运行参数与冻结记录并将 freeze 路径及摘要写入结果。服务器正式 freeze record 路径为 `experiments/phase5_freeze_v1/freeze_record.json`，SHA-256 为 `8bd5694ca67a250b726e7de1a53164166ada24e27975b2c4c9f6fe0f35cf4b28`。完整 JSON 内容未同步到本地，未在此处猜测或重写字段。

**2026-09-01 T1 实现记录**：修复 CPU smoke 合成样本上下文长度（`max_seq_len=256`），单步 T-pretrain 成功并生成 smoke checkpoint。新增 `reference.py`，提供 mixed batch 与逐样本 greedy generation 的确定性比较、direct-reference logits 等价性和分叉诊断；KV 模式在未实现时显式标记 `blocked`。新增 `eval_phase5.py`，可从 checkpoint metadata 重建模型并调用 evaluator；manifest 摘要校验、ROC-AUC probe 和恢复率接口已接入。CLI 默认拒绝覆盖结果，缺少可信摘要或 credential 时 fail-safe。全量 `tests/v2`：230 passed。

**2026-08-27 服务器预运行记录**：seed `20260824` 的首次正式命令在尾批 reference-routing
`assert_close` 处中止，未生成结果 JSON。观测到 CUDA float32 最大绝对差约 `3.74e-4`，原固定
`atol=1e-5/rtol=1e-4` 对不同 batch shape 的 cuDNN 数值路径过严。已改为设备感知容差
（CPU `1e-5/1e-4`，CUDA `5e-4/2e-3`），同时增加 argmax 完全一致硬检查与最大误差记录；
修复后允许对该 checkpoint 重跑，原因属于 evaluator 实现修复而非依据 test 指标调参。

---

## 研究目标

在神经网络中间嵌入固定的 toy LWE-inspired 关系验证门（Gate Layer），根据 credential
关系判定控制深层神经元的实际执行，研究**可信部署边界内的模型能力分级控制**。

当前实现不提供签名不可伪造性、身份认证、密码学访问控制 soundness 或白盒抗性，
不得将本研究原型描述为生产密码系统。

**现有关系与后续研究分开**：当前实现是实数域 `||Ac-b|| < threshold` 残差检查，不含 mod q；历史上从 Module-SIS 原型转向该 toy 关系是为了降低实现复杂度，不能推出 LWE 普遍更适合神经编译。G0/P1 保留它作为兼容性基线；G1-a/b 另行冻结规范域、reference 和精确神经构造，完整认证协议另立里程碑。

### 历史架构快照（Revision 2；后续设计已由 R3 取代）

```
tokens → 冻结 embedding / prefix → hidden
                                  |
credential → 规范解析 → 模型内固定 Gate（verifier → evidence → coordinator）
                                  |
                             已提交 route
                   +--------------+-------------+
                   |              |             |
              PROTECTED        PUBLIC          DENY
          原 hidden 恒等通过   public readout   不执行两业务分支
            → 原 suffix       （P2 后可用）
            → 原 norm/head
```

### 关键设计决策

1. **Gate Layer 位置**：在浅层特征提取后、深层特征提取前
2. **当前兼容性基线**：toy LWE-inspired 实数关系
   - 参数：n=128, m=256, σ=1.0, threshold=48.0
   - 验证逻辑：L2 误差范数 < threshold
3. **路由机制**：
   - 现有训练：`decision.allow` 硬选合法样本；sigmoid 仅软缩放其特征，不使离散选路可微
   - 现有推理：硬路由（真正不执行未授权深层，gate_signal ∈ {0, 1}）
   - G0/P1/P2 目标：认证模式与业务 train/eval 解耦，固定硬判定；合法 hidden 恒等通过，原骨干保持 eval；仅公共读出可训练
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
   - **Phase 4（可选兼容性检查）**：ResNet-18 on CIFAR-100
     - 100 类 → 20 类超类，仅允许一个 seed 或短训练 smoke test
     - 目的：检查数据、head、evaluator 和 response schema，不作为能力隔离主结论
   - **Phase 5（历史原型与诊断）**：小型 decoder-only Transformer
      - 同 tokenizer/vocabulary/prompt 的 public early-exit 与 protected full-path
      - 目的：验证计算图内 Gate 的能力分级、语义等价和能力泄漏边界
   - **Revision 2（历史路线，已由 R3 取代）**：冻结预训练 decoder-only 宿主，G0/P0 → P1 → P2/P3；G1-a/b → I1 独立接入
      - P1 独立验收插入正确性；P2 公共效用失败不撤销 P1。候选约 0.5B，型号及 revision 尚未冻结
   - **Phase 6（暂缓扩展）**：MoE、sandbox tool calling、ImageNet 等；须另立研究问题和资源方案，不把规模扩大视为安全证据

---

## 威胁模型与主张边界

### TM-API（当前正向保证适用的模型）

攻击者可无限次提交任意 `(image, credential)` 并观察预期的能力输出，但不持有模型权重，
不能修改进程内存、计算图或直接调用内部模块。模型权重、推理代码、协调器和部署入口可信。

Phase 3 的 `TM-API` 外部边界是已完成的服务层 response envelope，
**不是**原始 PyTorch `InferenceOutput`。原始输出包含 `decision`、连续 `error_norm`、
`reason_code`、`verified`、`gate_signal` 与路由索引，只允许 evaluator 等测试仪器访问。

下列已有模型层结论由 Phase 3.6 可信进程内适配入口承接，不自动扩展为新预训练 adapter 的保证：

- invalid credential 时 `layer3`、`layer4` 和 protected head 零调用；
- invalid 路径只产生 2 类公开能力，valid 路径产生 10 类受保护能力；
- 推理态 `allow` / `gate_signal` 与 NumPy `V_ref` 逐样本一致。

服务层只能声明不泄露**额外的**验证证据、连续距离、reason code、路由索引或内部特征；
public/protected 能力结果本身可能让调用方推断能力等级，不主张路由不可区分性。

### TM-WB（明确不主张抗性的模型）

攻击者持有 checkpoint 与运行时，可插 hook、改张量或直接调用内部方法。当前实现对此不提供保证：
credential 只控制执行路径，不影响 protected 权重本身的可用性。攻击者可直接调用受保护内部路径，
或通过常数规模运行时篡改绕过控制流；已有 direct-path 等价测试支持绕过后无业务能力损失。

不得把该结论写成“单次赋值”或提供可迁移的攻击 PoC。仅翻转 `decision.allow` 时，
`gated_features` 已被 `gate_signal` 清零，不能恢复正常 protected 语义。

### 统一术语与写作规则

- Gate Layer 统一称为**固定的 toy LWE-inspired 关系验证门**。
- 不得称为“密码学验证门”或“密码学访问控制”；当前关系无安全归约且可被最小二乘伪造。
- 历史静态 credential 可重复使用，尚无 replay 防护证据；后续认证协议必须纳入状态化新鲜性与防重放设计。
- FAR/FRR 是当前采样分布下的实现正确性判据，不是密码学安全指标。
- `capability_gap_fine` 的随机猜测基线必须标注 `is_analytic: true`，不是攻击者能力上界。
- 每条论文安全陈述必须绑定 `TM-API`、`TM-WB` 或 `TM-NA`，并映射到 Claim ID。
- GateBreaker 的 gate 是输入驱动的学习式 MoE 路由器；本项目 Gate Layer 是 credential 驱动的固定关系判定器。只作机制区分，不作安全强弱类比。

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
- Phase 1-2 使用静态 credential，未形成 replay 防护证据；后续认证协议纳入状态化 replay 防护

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
- Phase 1-2 的结果不包含 replay 防护证据；后续协议需单独验收防重放性质
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

**Gate 判定跨 Stage 的限定**：在 `A`、`b`、`error_threshold`、credential generator、
输入规范化、dtype 与设备配置均冻结时，训练模型权重不改变 Gate 判定，因为 Gate 无可训练参数，
且验证链不依赖图像特征。跨 Stage FAR/FRR 仍须测量，但用途是检测配置漂移的回归检查，
不是学习稳定性或密码学安全性的证据。

---

### Phase 3: 评估实验（CIFAR-10）[STAGE C EVALUATION COMPLETED]

**目标**：在模型层验证 Gate Layer 的功能正确性、能力分级、路由隔离和运行代价。

**数据集**：CIFAR-10（10 类 → 2 类）

**前置状态**：Phase 2 已在服务器 RTX A4000（16 GB）完成真实 CIFAR-10 三阶段训练，seed `20260824`、`20260825`、`20260826` 均已完成，并分别生成 Stage A/B/C checkpoint 和摘要文件。

**正式评估状态**：三个 Stage C best checkpoint 已在服务器完成官方 test split 评估；本地已核对
单 seed JSON、混淆矩阵、路由质量门和多 seed `mean ± std`。未根据 test split 修改模型、阈值、
checkpoint 或指标定义；仅扩展聚合器以收录单 seed 文件中已冻结的 capability、Gate、mixed-routing
和 latency 指标。

**当前实现**：

- `src/can/v2/experiments/test_evaluator.py`：模型层指标、mixed batch 路由校验和 latency；
- `scripts/eval_cifar10_test.py`：单 checkpoint CLI、summary/manifest/SHA-256 校验与 Stage C 三 seed 聚合；
- `tests/v2/test_test_evaluator.py`：8 项离线专项测试；
- 当前 aggregate 只接受 Stage C，尚不能直接生成 Stage A/B/C 统一聚合报告。

#### 3.0 Stage C 正式结果（2026-08-27）

三个 seed 的完整性、provenance、索引覆盖、reference-routing logits 和预测 indices 质量门均通过；
FAR/FRR、routing mismatch 和 empty subbatch 计数均为 0。

| 指标 | 三 seed mean ± population std |
|---|---:|
| Protected accuracy | `0.89157 ± 0.01366` |
| Public accuracy | `0.96427 ± 0.00141` |
| Public balanced accuracy | `0.96503 ± 0.00220` |
| Protected coarse accuracy | `0.98243 ± 0.00436` |
| Fine capability gap | `0.69157 ± 0.01366` |

Latency（batch size 256，forward + routing）：all-valid `23.7488 ± 0.0982 ms`，all-invalid
`16.0328 ± 0.0801 ms`，mixed `20.1395 ± 0.0545 ms`。相对 all-valid，all-invalid 约快
`32.49%`，mixed 约快 `15.20%`。

均值 Protected accuracy 达到预期 `>= 0.88`；seed `20260826` 为 `0.8784`，低于逐 seed
参考线 `0.0016`，记录为科学结果的轻微偏差，不视为 evaluator 故障。正式结果仍保留旧 checkpoint
的 `partial_legacy_defaults` LWE provenance 与未知 training credential RNG seed 限制。

#### 3.1 功能正确性实验

实验内容：
1. **Fail-closed 验证**：
   - Invalid credential → 深层调用计数 = 0
   - 使用 forward hook 统计 layer3/layer4 的实际执行次数

2. **差分测试**：
   - Valid credential 输出 vs 深层 direct 输出
   - 逐样本比较 logits，验证 mixed batch 与独立 reference route 一致

3. **Gate signal 分布**：
   - Valid credential → gate_signal 均值和方差
   - Invalid credential → gate_signal 均值和方差

#### 3.2 能力分级实验

实验内容：
1. **Protected accuracy**（valid credential）：
   - Fine-grained classification accuracy
   - 以 Stage A protected accuracy 作为 `stage_a_reference`

2. **Public accuracy**（invalid credential）：
   - Coarse classification accuracy
   - 与蒸馏目标比较

3. **Logits 等价性**：
   - Valid credential logits vs 深层 direct logits
   - 计算 L2 距离、余弦相似度

独立训练的无 Gate 同构 ResNet-18 尚不存在，不得作为当前 Phase 3 已实现 baseline。
该项作为未来消融 `no_gate_ablation`，对应 claim C-014。

#### 3.3 性能实验

实验内容：
- Latency：valid vs invalid credential
- GPU 内存占用
- 吞吐量（samples/sec）

**Baseline 比较**：
- External verifier + full model（验证器在模型外部）
- 记录：verifier latency, model latency, total latency

当前 evaluator 尚未实现 external-verifier baseline 的完整测量；完成前 claim C-007 保持 pending。

#### 3.4 Stage A/B/C 对照

拟报告五项：protected accuracy、public accuracy、protected logits 等价性、public/protected
capability gap、实际 public/protected forward 次数。Stage A/B 可用同一单 checkpoint evaluator
分别运行，但当前聚合器只支持三个 Stage C 结果；A/B/C 统一报告需要另行冻结汇总协议或补充评估编排。

不得把 Stage A/B 中间阶段与 Stage C 主结果并列为最终模型，也不得用 validation 与 test 指标直接作差。

#### 3.5 评估纪律

- 使用真实标签、固定 CIFAR-2 映射、确定性 reference model 和严格的 indices/labels 对齐；
- CIFAR 分类具有 ground truth，不引入 LLM judge；
- 能力差距必须与效用、路由完整性和执行代价成对报告；
- 每个预注册 checkpoint 在官方 test split 上正式评估一次，输出记录时间与 checkpoint SHA-256；
- test split 不用于修改 checkpoint、阈值、训练超参数或选择规则。

#### 3.6 服务层 response envelope [COMPLETED - CLAUDE ACCEPTED，claim C-013 satisfied]

在 Stage C 正式 test split 评估之后单独设计和实现，不阻塞模型层 evaluator。当前实现仅为可信进程内适配层：

1. 剥离 `decision`、连续 `error_norm`、reason code、verified、gate signal 和路由 indices；
2. 每个样本返回同构 envelope，字段集合和 shape 不随 valid/invalid 改变；
3. 只允许暴露预期的 public/protected 能力结果，不声称能力等级不可观察；
4. 增加全 valid、全 invalid、mixed batch 的序列化脱敏测试；
5. evaluator 仍可访问原始内部证据，服务层调用方只获得脱敏 envelope。

实现文件：
- `src/can/v2/service/response_envelope.py`
- `src/can/v2/service/inference_service.py`
- `src/can/v2/service/__init__.py`
- `tests/v2/test_response_envelope.py`
- `tests/v2/test_inference_service.py`（合计 30 tests，service 行覆盖率 98%）

---

### Phase 4: CIFAR-100 兼容性检查 [OPTIONAL/DEFERRED]

**定位**：CIFAR-100 不再是当前论文主线的正式能力分级实验。`100 类 protected / 20 类 public`
仍然存在输出空间不同造成的粒度解释，不能直接解决 T 轨道的同词表、同任务格式和能力泄漏问题。
因此不安排三 seed 正式训练、不预先宣称具体准确率，也不把 C-012 写成正向结论。

如 Transformer 资源暂时不可用，或需要在实现 T 轨道前降低工程风险，可执行一次低成本 smoke test：

- 使用 CIFAR-100 官方 fine/coarse labels，检查数据接口和 split hash；
- 复用 `GatedResNet18` 的类别数参数化，检查 100/20 类 head、evaluator 和独立版本化 response schema；
- 最多一个 seed 或短训练，仅记录 shape、路由、zero-call、direct protected 等价性和序列化结果；
- 结果只作为兼容性/工程附录，不用于证明真实能力隔离、密码学安全或 C-012 的主结论。

不得在该 smoke test 中加入多级 Gate、手工新建类别映射、test split 调参或根据结果反复选择 cut。

---

### Phase 5: T 轨道小型 Transformer 能力分级 [HISTORICAL / D0 CLOSEOUT]

本节 5.1–5.4 保留旧原型的设计和分阶段交付记录，后续已完成事项见日期日志；其中“尚未实现”仅指对应 checkpoint 当时状态。当前不追加从零 T-pretrain/A/B/C 正式预算，不沿用旧 byte tokenizer、freeze 或 go/no-go 作为新宿主前置。D0 已完成有界收尾；当前主线见 `docs/DESIGN_PROPOSALS.md` 文末 Unified Roadmap R3。

**研究问题**：在 `TM-API` 可信黑盒部署中，credential 驱动且位于 Transformer 计算图中间的固定
Gate Layer，能否在保持 protected 路径语义的同时形成可复现的 public/protected 能力边界，
并在受控访问预算下限制受保护能力泄漏或恢复？

**范围**：

- 先实现可在本地完整训练或微调的小型 decoder-only Transformer；不以 Qwen/Llama 或 0.6B 模型
  替代最小原型；
- 主数据流固定为 `tokens, credential -> shared prefix -> in-graph Gate Layer -> hard route`；
  外部 verifier 只能作为对照；
- T0 初版默认沿用当前固定的 toy LWE-inspired 关系门；ML-DSA reference verifier 属于独立 S 轨道，
  不因 T 轨道结果获得密码学安全主张；
- 在若干完整 Transformer block 末端候选 cut 中只用 validation 选择；每条生成序列只提交一次
  route，后续 token 不得动态切换权限；
- public 使用 early-exit LM head，protected 使用完整深层和原 LM head；两条路径共享 tokenizer、
  vocabulary、prompt、停止规则和输出 schema；
- 第一版只做 L0 公开能力与 L1 合成私有知识问答。L2 工具调用留作后续扩展，工具只允许 sandbox/mock
  dispatcher，模型生成 intent 不等于授权执行。

#### 5.1 T0 设计冻结（历史步骤，已完成）

在创建代码前必须冻结并审阅：

- 模型层数、宽度、词表、上下文长度、候选 Gate cut 和每请求一次的路由语义；
- 公开/私有/拒答数据生成规则、实体不重叠约束、train/validation/test split、seed、摘要和 hash；
- 未授权 private query 的稳定拒答或公开范围回答目标，不得用随机退化作为安全目标；
- protected direct-reference、public utility、private refusal rate、probe AUC、恢复率、
  zero-call、延迟和吞吐等指标；
- 所有 cut、epoch、checkpoint 和超参数只由 validation 选择，test 只评估一次；
- 独立版本化的 Transformer response schema，不能复用 CIFAR response envelope 的固定 10/2 类槽位。

该设计冻结及实现者选择已经完成；本段保留为历史约束，不再代表当前下一步。

#### 5.2 训练与评估阶段

- **Stage A**：冻结底座，仅训练 early-exit head，使其具备公开任务能力；
- **Stage B**：使用原模型作为冻结 teacher，仅在公开能力分布上蒸馏 early-exit；
- **Stage C**：混合 valid/invalid credential 联合训练；如使用抑制损失，目标必须是冻结的拒答、
  公开范围回答或禁止 tool intent，不能把随机退化当作能力保护；
- 训练后冻结最终 checkpoint，protected 结果与同一 checkpoint 的 direct full-path 比较；
- 分别在 `TM-API`、`TM-REP` 和 `TM-CP` 下进行 API 观察、表示探针和受限恢复实验；`TM-WB` 不主张
  抗性；
- 输出 public utility、protected utility、private refusal rate、表示 probe AUC、恢复率随预算
  曲线、路由调用计数、zero-call、延迟/吞吐和多 seed 区间。

#### 5.3 Phase 5 验收门

- valid credential 只执行 protected route；invalid credential 不执行 protected route；
- protected 输出与同一冻结 checkpoint 的 direct reference 满足预先定义的等价性；
- public 路径在公开任务达到 validation 冻结的 utility 门槛，并在 private query 上满足拒答/公开范围语义；
- test split 只评估一次，失败或容易恢复的结果必须作为负面结果记录，不改写成密码学安全结论；
- 至少完成 P0 对照：同模型 early-exit/full、粒度/容量对照、前缀数据隔离基线；P1 对照按资源补充。

Phase 5 不声称 toy LWE/ML-DSA 不可伪造、白盒不可绕过、checkpoint 机密性或生产访问控制；credential 新鲜性与 replay 防护纳入后续状态化认证协议并须单独验收。

#### 5.4 T0 CPU 最小原型实现 checkpoint [COMPLETED / CLAUDE ACCEPTED]

**完成时间**：2026-09-01
**验收状态**：Claude 已验收（2026-09-01）

- 新增固定 260 项 byte-level tokenizer、实体隔离的 public/private/refusal 合成数据和 entity-triplet mixed sampler；
- 新增 6-block 默认配置的小型 decoder-only Transformer，Gate 判决只依赖 credential，protected 使用门控 hidden state，public early-exit 使用未门控共享表示；
- 推理区分 protected、public 与格式错误 rejected，invalid 路径对 protected blocks 保持 zero-call；
- 新增每序列一次硬路由的确定性 greedy generation，并以 mixed/逐样本结果一致性测试守护索引隔离；当前实现为无 KV-cache 的重计算参考路径，KV-cache 优化及逐步 cache 对照尚未实现；
- 新增 T-pretrain/Stage A/B/C 损失、冻结 teacher、mixed batch 约束、entity sampler、原子 checkpoint、teacher identity、LWE identity 和 Python/NumPy/Torch/CUDA/credential RNG 恢复；
- 新增 `tests/v2/test_phase5_transformer.py`，覆盖 tokenizer、数据、路由、direct full-path、generation、loss、训练 step 和 checkpoint 恢复。

尚未实现：正式 validation/evaluator、规范化 exact-match/refusal 统计、probe AUC、TM-CP 恢复曲线、manifest/CLI、KV-cache、T-pretrain go/no-go 驱动器、GPU smoke 和多 seed 正式训练。当前 checkpoint 只能证明 T0 CPU 架构与基础训练路径可执行，不能支持 C-015 至 C-019 的正向结论。

---

### Phase 5.5-TS: Teacher–Student 公共模型与认证完整模型对照 [DEFERRED]

以下为保留的候选方案，不进入首轮资源计划。共享 embedding/prefix 加公共读出是 P2 主线；独立学生可在有明确比较问题时另行启动，不是 P1/P2 的前置。

Phase 5.5 不新建独立工程，而是在 Phase 5 已冻结的 tokenizer、数据生成协议、Transformer 配置、Gate 语义、response schema、评估器和 manifest 体系上增加一个可归因的 Teacher–Student 对照。它回答的问题是：公共能力是否可以由完整模型蒸馏为独立的小模型，以及 credential 是否只控制完整模型受保护路径的执行。

**模型组**：

- `Teacher T`：Phase 5 T-pretrain 产生并冻结的完整 Transformer，作为 protected direct reference；
- `Public Student S`：独立的小型 Transformer，只使用 public/refusal 分布和冻结 teacher 的公开目标进行蒸馏；
- `CAN(T,S)`：同一入口中的三态组合，PUBLIC 执行 `S`，PROTECTED 执行冻结 `T` 的完整路径，DENY 不执行任一业务路径；
- `CAN-shared-prefix`：Phase 5 原有 early-exit 结构，作为共享前缀基线，不与独立学生模型混写。

**实验边界**：Teacher–Student 结果必须与 Phase 5 shared-prefix 结果分开报告；学生模型不得被称为密码学隔离模型。TM-API、TM-REP、TM-CP 下仍需报告公开输出泄漏、表示探针和有限预算恢复；replay 防护需在后续状态化协议中单独验收，toy LWE 不可伪造性与 TM-WB 仍不主张。

**实施顺序**：

1. 完成 E2-C Claude 验收并运行 Plain/CAN 成对消融，冻结 Phase 5 baseline；
2. 从通过 go/no-go 的 T-pretrain checkpoint 构造只读 Teacher manifest；
3. 训练独立 Public Student，固定 public/refusal 数据、seed、预算和蒸馏温度；
4. 组合 `CAN(T,S)`，验证 valid credential 只进入 Teacher，invalid public 只进入 Student，deny 不执行任何路径；
5. 与 `Plain Teacher`、`Plain Student`、`CAN-shared-prefix` 做配对比较；
6. 完成三 seed validation/test、direct-reference 等价、public utility、private refusal、延迟/吞吐、probe AUC 和恢复曲线后，才评估 Phase 6。

**首轮验收门**：protected 输出与 Teacher direct reference 等价；PUBLIC 不调用 Teacher protected suffix；DENY 对 Student 和 Teacher 均 zero-call；Student 的 public utility 达到 validation 冻结阈值；所有模型、数据、tokenizer、teacher hash 和蒸馏配置进入独立 manifest。Phase 5.5 不得以 Student 的低恢复率推出密码学安全。

---

### Phase 6: 外部有效性扩展 [OPTIONAL]

当前不启动。待 Revision 2 获得足够插入/效用证据后，按独立问题和资源方案重新评估；无需机械等待旧从零训练或独立学生成功：

- MoE 专家池准入、`allowed_mask` 和受约束 top-1 task router；
- sandbox tool calling、外部 benchmark 或更大开源底座；
- ImageNet 等大规模视觉任务（如仍有明确研究问题、数据许可和资源预算）。

Phase 6 不是当前主线，不预设需要 ImageNet，也不把规模扩大本身当作安全证据。

---

### Phase W: 权重级绑定 [SEPARATE RESEARCH TRACK]

Phase W 不是 Phase 4/5/6 的既定增强，也不排入当前主线。若启动，必须先单独评审方案。

候选方向是用独立内容密钥对 protected 权重做可逆加密或掩码。其可主张范围仅能在单独评审和实验
后确定，当前只记录为候选：

- checkpoint at-rest 机密性（TM-CP）；
- 授权提交与内容密钥释放之间的绑定是否可行。

在普通 PyTorch 软件执行模型中，protected 计算要求解掩码权重出现在攻击者可读地址空间，
因此该方案不能建立 `TM-WB` 运行时抗性。攻击者取得一次合法 credential 后仍可 dump 明文权重。
TEE、split inference 或服务端权重驻留会改变可信计算基，不构成模型内生白盒抗性的证据，
也不能由此推出一般性不可能结论。

启动前必须分别解决构造、密钥管理、checkpoint 中间状态、解掩码时机、浮点/BatchNorm 等价性、
GPU 明文窗口、掩码恢复风险以及训练/部署流程重构。

---

## 主张与证据

权威台账位于 `docs/RESEARCH_DESIGN.md` 第 7 节，包含 C-001 至 C-019。下列状态摘录针对 CIFAR；新路线不得凭文档将 C-015 至 C-019 晋升为 satisfied。

- `satisfied`：C-001、C-003、C-004、C-006、C-008、C-009、C-011、C-013；
- `declared`：C-010；
- `partial`：C-002、C-005；
- `pending`：C-007、C-012、C-014；其中 C-012 已降级为 optional，不属于当前主线。

C-003、C-006、C-011 与 C-013 的 satisfied 状态均限定于可信进程内服务入口，不扩展到 `TM-WB`、网络 wire schema 或同进程旁路。
`stage_a_reference` 与尚不存在的 `no_gate_ablation` 禁止混用。

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
- [x] **Phase 2 真实 CIFAR-10 三阶段训练**：三个 seed 均完成 Stage A/B/C checkpoint 与摘要
- [x] **Phase 3 evaluator 实现**：核心模块、CLI、manifest/SHA-256 校验、Stage C 三 seed aggregate、latency 与 8 项离线测试
- [x] **Phase 3 Stage C 正式评估与结果核验**：三个 seed 官方 test split 结果、质量门、混淆矩阵和 mean/std 均已核对
- [x] **Phase 3 aggregate schema v2**：补齐 capability、Gate、mixed-routing、latency 与跨 seed 质量门
- [x] **威胁模型与 claim/evidence 台账**：TM-API/TM-WB/TM-NA 与 C-001 至 C-014 已记录
- [x] **文档同步后的回归测试**（2026-08-27）：设置 `PYTHONPATH=.` 后运行 `pytest tests/v2/ -q`，151 passed
- [x] **aggregate schema v2 回归测试**（2026-08-27）：设置 `PYTHONPATH=.` 后运行 `pytest tests/v2/ -q`，154 passed

### 进行中
- [x] **Phase 4 正式主实验降级为 optional**：仅保留低成本 CIFAR-100 兼容性 smoke test，不作为当前论文主线
- [x] **Phase 5 T0：小型 Transformer 能力分级方案设计审阅与修订**
- [x] **Phase 5 T0：小型 Transformer CPU 最小原型代码实现并通过 Claude 验收**
- [x] **Phase 5 T1：evaluator、CLI、KV-cache 与正式 smoke 准备已完成并通过 Claude 验收**
- [x] **Phase 5 正式训练入口：token budget、双 head、go/no-go、resume 与失败诊断已完成并通过 Claude 验收**
- [x] **Phase 5.5-TS Teacher–Student 路线设计：已记录，当前暂缓，尚未实现**
- [x] **Revision 2 路线框架已纳入工作日志和设计文档**：尚不表示新代码已交付
- [x] **D0 有界收尾**：诊断代码与 CPU 回归已通过 Claude 验收；正式 GPU 运行三变体均在 update 80 通过并已登记
- [x] **G0/P0/P1 实现前设计与配套规则同步**：用户确认 v1.2 已审阅，2026-09-10 四份规则已同步
- [x] **G0 CPU 实现**：Codex 已实现并完成专项、覆盖率与完整回归，Claude 已验收
- [ ] **P0、P1、P2、P3、G1-a/b、I1**：均未执行，真实模型/参数/预算与新 claim 尚未冻结

### 下一步（唯一下一步）

**将 `docs/DESIGN_PROPOSALS.md` R3.10 `m0-contract-plan-v1` 交 Claude 审阅；审阅通过且用户确定实现者前不实现 M0、不下载 Qwen2、不启动服务器 P0/P1。**

D0 已按原 512-update train-only 上限完成并提前通过；不创建 `phase5_t2_freeze_v1`、不自动追加 500k/2M token、不读 dev/validation/test。D0 成功不作为 G0/P0 的正向证据；若后续发现会影响新路径的授权、索引、缓存或测量错误，仍须修复。

### Revision 2 历史路线快照（2026-09-09；已由 R3 取代）

**历史路线依据**：[工作路线修订方案 Revision 2](docs/GATE_PRETRAINED_ROADMAP_REV2_20260909.md)。本节记录 2026-09-09 当时的实施路线；[Revision 1](docs/GATE_PRETRAINED_ROADMAP_REVIEW_20260909.md) 同为历史。未完成工作的现行设计只以 `docs/DESIGN_PROPOSALS.md` 文末 Unified Roadmap R3 为准。

| 里程碑 | 目标及依赖 | 资源/状态 |
|---|---|---|
| D0 | 旧单四元组诊断有界收尾；成败归档，不自动追加预算 | COMPLETED；三变体均在 update 80 通过，结果边界见日期日志 |
| G0 | 固定授权、恒等通过、硬路由、精度和分支接口；用户确认设计已审阅，配套规则已同步 | CPU 实现与本地验证完成，等待 Claude 验收；目标 GPU 补验未执行 |
| P0 | 预训练宿主效用、权重/tokenizer/数据/后端与 P1 容差冻结；可与 G0 准备并行 | SERVER_REQUIRED 推理；PLANNED |
| P1 | G0/P0 通过后，无训练插入、原输出保持与 protected zero-call 独立验收 | SERVER_REQUIRED 推理；PLANNED |
| P2 | P1 通过后，仅训练共享浅层的公共读出；校验原权重并重跑 P1 | SERVER_REQUIRED 训练；PLANNED |
| P3 | P2 最终状态冻结后，效用/越界/性能同构对照；公共训练至少 3 seeds | SERVER_REQUIRED；PLANNED |
| G1-a | 规范域、候选关系、reference、安全目标和算子集；可与 G0/P0 并行 | CPU 分析；PLANNED |
| G1-b | G1-a 审阅后做精确神经内核、正确性论证和后端验证 | CPU/GPU；PLANNED |
| I1 | G1-b 与 P1 通过后替换 relation verifier；P3 不是前置 | CPU/GPU；PLANNED |

资源候选为 RTX A4000 16 GB，新增任务时长和显存尚未测量；P0 smoke 实测后冻结 P1 评估及 P2 训练墙钟预算。旧 T2 吞吐不能直接代入新宿主。

**不变量与范围**：credential 走结构化通道；模型内 Gate 的 verifier 只产 evidence，协调器唯一提交 route，dispatcher 控制实际执行。合法 hidden 恒等通过，原权重固定。P1 拒绝即 DENY；P2/P3 能力模式显式配置为规范但不满足关系的 credential → PUBLIC，格式错误 → DENY。未来显式 protected 入口认证失败 → DENY，使用独立 policy ID。学习式拒答不等于 DENY。

**独立证据**：P1 成功不证明公共效用或 credential 不可伪造；P2 失败不撤销 P1。CAP 同源单跳/多跳任务用宿主 tokenizer 建立新 suite/freeze，不复用旧四元组 trainer；CAP 越界回答不是秘密事实泄漏，MEM 另行研究。同构“外部 verifier + 相同共享模型/dispatcher”是必要强对照，不能把共享收益归因于神经验签位置。

**密码构造范围**：G1 首轮只承诺关系内核，`q=3329` 等仍是候选。必须区分规范域上的证明和有限差分测试；mod q、FP32 局部乘加上界和 I1 接入都不能代替完整认证安全。后续认证协议必须纳入 nonce/计数器、request binding、一次性消费、撤销和并发原子性，并单独验收；不主张 TM-WB 或知识机密性。

**配套文档已同步（2026-09-10）**：`AGENTS.md`、`SECURITY.md`、`README.md`、`docs/RESEARCH_DESIGN.md` 已按 v1.2 更新；新增 C-020 至 C-026 均 pending，旧 freeze 与历史主张保留。规则改动随首个实现 checkpoint 审核；同步不代表G0/P0/P1已实现。

**文档 checkpoint（2026-09-09）**：本次仅修改 `PROJECT_WORKLOG.md`、`docs/DESIGN_PROPOSALS.md`，新增 `docs/GATE_PRETRAINED_ROADMAP_REVIEW_20260909.md`。代码、实验配置及旧 artifact 未修改。文档检查包括 `git diff --check`、新增本地链接与提案状态核对；不运行模型测试和 GPU 任务。未提交或推送。

**评估对照文档（2026-09-09）**：新增 [Claude 评估建议与 Codex 核查意见](docs/CLAUDE_ASSESSMENT_CODEX_REVIEW_20260909.md)，包含建议采纳表、数学与工程判断、下一版方案建议及用户提供的 Claude 原文附录。“采纳”仅表示审阅建议，不改变现行实施规则、freeze 或唯一下一步。本轮交付文件为该新增文档及本日志入口；已有其他文档改动保留。验证范围为 Markdown/链接、原文附录内容一致性和文本差异检查，未运行代码测试、GPU 或凭据恢复实验，未提交推送。

**Revision 2 文档 checkpoint（2026-09-09）**：本轮新增 `docs/GATE_PRETRAINED_ROADMAP_REV2_20260909.md`，并更新本日志的审阅入口。文档包含依赖、接口、模型/数据、验收、资源、停止条件及两轮评估的修订映射；模型候选、参数与安全强度仍未冻结。检查范围为差异格式、新文件文本和本地链接；未运行模型测试或 GPU 任务，未修改代码、配置与已批准规则，未提交推送。

历史 T0/T1 审阅重点（不代表当前下一步）：计算图内 Gate 位置和每请求一次的硬路由、同 tokenizer/vocabulary/prompt/停止规则、
公开与私有/拒答数据生成及实体隔离、Stage A/B/C 训练协议、TM-API/TM-REP/TM-CP 访问条件、
protected direct-reference 等价性、public utility、private refusal rate、probe AUC、恢复曲线、
P0 对照、GPU 显存和最小原型资源预算。

**本次验证（2026-09-01）**：更新 private prompt 数据协议后，CPU smoke（seed `20260901`，T-pretrain 单步）成功；`PYTHONPATH=.` 下 `pytest tests/v2/ -q` 通过 `230 passed`，T1 专项测试 13 项通过；Black、isort、compileall 和 `git diff --check` 均通过。

**测试环境备注**：直接运行 `pytest tests/v2/ -q` 未设置 `PYTHONPATH` 时在收集阶段报
`ModuleNotFoundError: No module named 'src'`；按仓库导入方式设置 `PYTHONPATH=.` 后完整测试通过。

---

## Revision 2 历史开放问题（已由 R3 重排）

1. **G0/P0/P1 如何落到实现？** 设计及配套规则已同步，待指定实施者并核查宿主实际API；不把设计审阅当成代码验收。
2. **选择哪个宿主和 cut？** 核查约 0.5B 候选的许可证、revision、原任务效用、绑定权重、后端和 2–3 个完整 block 边界；用 train/dev 选择，冻结后独立评估。
3. **哪些数值/调用范围可声明？** batch=1/mixed、KV/无 KV 分别冻结容差，报告 logits 和 tokens；授权前拒绝 protected zero-call，授权后异常如实计数。
4. **P2 公共效用如何定义？** CAP 同源配对，原投影截断基线与有限 adapter 候选；公共效用、范围合规、样本规模和训练预算在正式训练前冻结，MEM 不混入 CAP。
5. **G1 的可证明范围是什么？** 先选择一个规范关系及允许算子，证明/差分/后端验证分开；完整认证与白盒方向另行设计，replay 防护作为后续状态化协议要求。
6. **何时扩展？** CIFAR-100、独立学生、MoE 和工具暂缓；以新增问题和资源审阅触发，不强制串接所有旧阶段。

---

## 风险与限制

### Revision 2 历史阶段风险

1. **插入适配风险**：位置编码、mask、mixed 索引和 KV-cache 可能改变完整路径；先做恒等切分对照，再引入 Gate，按配置报告输出保持范围。
2. **公共读出效用风险**：冻结浅层表示未必支持目标服务；预登记 cut/adapter 和预算，失败后停止，不自动解冻原骨干，P1 结论独立保留。
3. **共享权重污染风险**：公共输出投影可能与 embedding 绑定；检查 optimizer 参数集合及训练前后摘要，P2 后重跑 P1。
4. **能力边界风险**：suffix 零调用不保证 public 不会回答专业问题；效用、范围越界和真实秘密泄漏分开度量，不靠蒸馏承诺机密性。
5. **密码与数值风险**：toy 实数关系存在最小二乘结构缺陷；G1 的域、阈值、约简、溢出、非有限输入和后端需完整验证，不能从有限测试或局部 FP32 上界推导认证安全。
6. **资源与证据风险**：宿主/许可证/任务效用和 A4000 预算尚未核验；原生 tokenizer 下新建 suite/freeze，先 smoke 测量，不借旧模型吞吐或小样本阈值充当依据。

### 当前明确不主张的能力

- **TM-WB 白盒抗性**：当前控制流门控可被直接调用内部路径或常数规模运行时篡改绕过；
- **Replay 防御**：历史静态 credential 尚无防重放证据；后续主线必须设计 challenge-response 或 nonce/计数器、request binding、一次性消费、撤销和并发状态，并单独验收；
- **密码学安全性**：toy LWE-inspired 关系无安全归约，可被最小二乘伪造；
- **生产部署安全**：研究原型；Phase 3.6 仅实现可信进程内 response envelope，不包含网络 wire schema、认证、传输安全或部署旁路隔离；
- **TEE/安全启动与侧信道防护**：不在当前主线；
- **新预训练插入与扩展结果**：G0/P0–P3/G1-a/b/I1 尚未实现；MoE、tool calling 和 ImageNet 未开展。已有 Transformer E1/E2/T2 结果仅在对应旧实验范围成立，不得以方案替代新证据。

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

上述假阳性/假阴性只针对有限的 toy credential 采样与固定阈值，不能解释为签名不可伪造性、
身份认证成功率或密码学访问控制安全性。

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
- Phase 1/2 开发侧 CUDA/GPU device 路径未实测；Phase 2 真实训练已在服务器 RTX A4000（16 GB）完成

---

### 已完成视觉路线与后续 Transformer 路线摘要

#### CIFAR-10 实验（Phase 2-3 原型验证）
- 训练状态：Phase 2 真实 CIFAR-10 三阶段训练已完成，三个 seed（20260824/20260825/20260826）均有独立 checkpoint
- 训练结果：各 seed 的 validation 摘要与官方 test split 结果均已保存；多 seed 汇总已核验
- Protected/Public accuracy：官方 test split 分别为 `0.89157 ± 0.01366` / `0.96427 ± 0.00141`
- Logits 等价性与 latency：三个 seed 的 reference-routing allclose 和 prediction indices exact 均通过；latency 已测量
- 选择规则：只使用 validation 指标选择 checkpoint，冻结后再评估官方 test set

#### CIFAR-100 兼容性检查（Phase 4 optional）
- 状态：未开始；不安排三 seed 正式训练，不作为 C-012 主结论
- 可选任务：官方 fine/coarse labels、head 参数化、evaluator、zero-call 和版本化 response schema 的
  一个 seed 或短训练 smoke test
- 结果定位：工程兼容性/附录，不证明真实能力隔离或密码学安全

#### 小型 Transformer 能力分级（历史原型与 D0）
- T0/T1、训练入口及 T2 诊断代码已交付并通过记录中的 Claude 验收；E2-A/B 同模板记忆成功，E2-C 泛化退化，T2 单 seed 200k pilot 当前配置 NO-GO。
- D0 正式 GPU 运行已登记，三变体均在 update 80 提前通过；不自动追加从零训练预算。旧代码、freeze 与负向证据保留，不迁移成新宿主结果。

#### 冻结预训练宿主插入（Revision 2 历史路线）
- 状态：该路线已由 Unified Roadmap R3 取代；本段只保留当时的 G0/P0 → P1 → P2/P3 设计快照。
- G1-a/b 研究精确关系内核，I1 依赖 G1-b 和 P1，不等待 P3；均不默认获得认证安全或白盒抗性。

#### MoE、工具调用、外部 benchmark 与 ImageNet（Phase 6 历史 optional）
- 状态：此处是 Revision 2 时期的历史判断；MoE 后续路线现由 Unified Roadmap R3 单独规定，工具调用和 ImageNet 仍未进入当前主线。
- 历史触发条件：Revision 2 证据足以支撑新问题且资源方案另行审阅；规模扩大本身不是安全证据。

---

## 参考文献

- Shamir et al., "How to Securely Implement Cryptography in Deep Neural Networks"
- Regev, "On lattices, learning with errors, random linear codes, and cryptography" (LWE 原始论文)
- Knowledge Distillation: Hinton et al., "Distilling the Knowledge in a Neural Network"
- Wu et al., "GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs"（本地 preprint；
  在未由官方来源核实录用信息前不固定声称正式发表场次）

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
- 本文档是唯一动态事实源；`PROJECT_WORKLOG_2.md` 仅保留为 2026-08-26 修订提案历史，不再具有当前状态权威性

**数据集与模型策略（Revision 2）**：

- **Phase 1-2**：CIFAR-10（架构与训练原型，10→2 类）
- **Phase 4**：CIFAR-100 兼容性 smoke test（可选，100→20 类，不承担能力隔离主结论）
- **旧 Phase 5/T2**：保留小型 decoder-only 原型、正负结果及 D0 诊断，不追加从零训练主线
- **P0/P1**：选择已有任务能力的冻结预训练宿主，原生 tokenizer，先验证无训练插入
- **P2/P3**：T2-CAP 同源单跳/多跳配对的新版本，公共读出训练与同构对照；MEM 和外部 QA 后续独立评审
- **Phase 5.5-TS/Phase 6**：独立学生、MoE、工具和 ImageNet 暂缓，不作为 P1 前提

**为什么选择这个顺序**：
1. 保留 CIFAR 已验证架构和效用证据；不再靠扩大类别数解决语言能力分级问题。
2. P1 隔离 Gate 插入正确性与语言从零训练收敛；P2 后再测公共效用，不相互替代结论。
3. G1 构造内核独立推进；I1 不等待公共头成功，新增显存和时长一律先测量。

**能力差距对比**：
- CIFAR-10：官方 test protected/public accuracy 为 `0.89157 ± 0.01366` / `0.96427 ± 0.00141`；10 类与 2 类难度不同，不用准确率相减宣称能力隔离强度
- CIFAR-100：仅作为可选 `100 → 20` 兼容性 smoke test，不预设准确率或能力差距结论
- Transformer：使用同 vocabulary、同 prompt 和同输出 schema，重点报告 utility、拒答、泄漏与恢复率
- ImageNet：归入 Phase 6 optional，不预设需要，也不把类别规模当作安全证据
