# Phase 5.5/T2 训练诊断与单四元组过拟合方案

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-08
- Verification Status: UNVERIFIED
- Version Label: t2_diagnostic_plan_v2

## 1. 背景与停止结论

seed `20260903` 的 CAP/C0、Plain/CAN 成对 200k-token dev pilot 已完成。两者使用相同初始化、
相同 batch 顺序和相同预算，且均未物化 test：

| 模型 | 实际 token / step | best score | final dev 主要结果 |
|---|---:|---:|---|
| Plain | 199,146 / 222 | 0.3500 | public EM/F1 0.25/0.35；protected-public F1 0.20；protected-private F1 0.225；refusal rate/F1 0.50/0.625 |
| CAN | 199,146 / 222 | 约 0.1437 | final public F1 约 0.0833；其余三类 F1 为 0；refusal rate 为 0 |

CAN 在约 75k--150k token 曾出现短暂改善，随后回退。因此 `best_selection_score` 与当前
`evaluation` 所表示的 final checkpoint 不是同一个时点。当前证据只来自单 seed，且 dev 每类
只有 4 个样本；它足以判定**当前配置不应冻结**，但不足以把退化因果归因于 Gate。

本方案的目的不是继续扩大预算，而是先回答两个更窄、可证伪的问题：

1. 当前模型、tokenizer、loss 和 optimizer 是否能在一个完整训练四元组上过拟合？
2. 如果 Plain 能而 CAN 不能，差异是否与训练态 soft `gate_signal` 对 protected 表示的缩放有关？

## 2. 假设、变量与边界

### 2.1 假设

- `H-D1`：Plain 在固定单四元组上能达到四个 scope 全部 normalized EM = 1.0。
- `H-D2`：CAN 的正常 soft-gated 训练若失败，而 direct-protected 诊断消融成功，则 soft Gate
  的训练/推理幅度差异是优先调查对象。
- `H-D3`：若 Plain 和两种 CAN 都成功过拟合，则 200k pilot 的主要问题更可能在多 source
  竞争、泛化或 loss 配比，而不是基本容量或训练接线。

这些假设只用于定位后续工程方向。单四元组过拟合不能证明泛化、能力分级、安全性或 Gate 的
因果效果。

### 2.2 固定变量

- suite/prompt：`T2-NL-P-CAP / C0`；
- seed：`20260903`；
- tokenizer、模型结构、初始化、AdamW、learning rate 和 greedy decoding 与 200k pilot 相同；
- 只读取 `train`，从按 `(source_id, prompt_template_id)` 排序后的第一个完整四元组取样；
- batch size 固定为 4，每次更新重复同一组 `public`、`protected_public`、
  `protected_private`、`refusal`；
- 三个变体使用相同可训练 tensor 初始化；两种 CAN 使用相同 LWE keypair 与 credential RNG 序列；
- 最大 512 次更新，每 16 次更新评估一次，连续 3 次满足成功门槛后允许提前停止；
- 输出必须记录四个 sample ID、四元组 SHA-256、实际 token、更新数和所有 identity。

该协议是 `diagnostic_only=true` 的 train-only 实验，不建立 `phase5_t2_freeze_v1`，不读取 dev、
validation 或 test，也不能与 200k dev 指标聚合。

### 2.3 三个变体

| 变体 | protected 训练路径 | public 训练路径 | 推理路径 |
|---|---|---|---|
| `plain` | Plain protected full path | Plain public early exit | Plain oracle head |
| `can_soft` | 当前 soft `gate_signal * prefix` | CAN public early exit | 正常硬 Gate |
| `can_direct` | credential 判决通过后，仅 valid 行绕过幅度缩放进入 protected full path | CAN public early exit | 正常硬 Gate |

`can_direct` 只能存在于独立诊断模块。它仍必须验证真实 credential、reason code 和 allow mask，且
invalid 行不得执行 protected suffix；但它不符合正式训练的 soft-routing 契约，不能进入生产入口、
正式 freeze 或研究主结果。

`can_direct` 不得根据调用方提交的标签直接决定授权。实现必须先用真实 credential 完成 Gate
验证，以协调器提交的 `decision.allow` 产生 protected 子批索引，再断言该索引与严格 T2 schema
从 `scope/credential_class` 推导的期望 mask 完全一致。只有 `decision.allow == true` 的行可以在
该消融中绕过 soft 幅度缩放；格式错误或验证失败的行保持 fail-closed 和 protected zero-call。

## 3. 实现架构与接口

### 3.1 消除 best/final 语义混淆

修改 `scripts/train_phase5_t2.py`，把新运行的 `summary.json` 提升为 schema v2，并明确保存：

```json
{
  "schema_version": 2,
  "resume_count": 0,
  "best_selection_score": 0.0,
  "best_at_tokens": 0,
  "best_at_global_step": 0,
  "best_evaluation": {},
  "final_selection_score": 0.0,
  "final_at_tokens": 0,
  "final_at_global_step": 0,
  "final_evaluation": {},
  "best_is_final": false,
  "best_checkpoint": {"path": "best.ckpt", "sha256": "..."},
  "final_checkpoint": {"path": "last.ckpt", "sha256": "..."}
}
```

`best_evaluation` 和 `final_evaluation` 保存除逐样本 `diagnostics` 外的完整 evaluator 摘要，包括
`text_metrics`、`teacher_forced_by_scope`、`routing` 和 `generation_safety`。逐样本结果分别写入
`best_diagnostic.json` 与 `final_diagnostic.json`。不再使用含义含糊的顶层 `evaluation`；旧 schema
v1 产物保持原样，读取方必须按 `schema_version` 分支，不能原地改写历史结果。

checkpoint 的 progress 同步保存 `resume_count`、`best_at_tokens`、`best_at_global_step` 和
compact `best_evaluation`，保证 resume 后不会丢失最佳时点。最佳模型若来自最终评估，所有
best 字段和 `best.ckpt` 必须在同一事务阶段更新。

正式 T2 训练入口的 resume 语义固定如下：checkpoint progress 是未完成运行的权威恢复状态；
`global_step` 和 `total_tokens` 跨进程恢复后继续单调累计，`resume_count` 每次成功加载并恢复后加
一。新 selection score **严格大于**历史 best 时才同时替换 `best.ckpt`、best 时点和
`best_evaluation`；相等时保留更早的 best。`best_at_global_step` 始终是从实验起点累计的绝对
step，不编码进 checkpoint 文件名，也不按进程重新从 0 计数。固定路径 `best.ckpt` 始终表示当前
绝对最优 checkpoint，manifest SHA-256 用于绑定其实际内容。

实现兼容边界：已完成的旧目录再次 `--resume` 只返回 `already_completed`，不重写历史文件；
未完成的旧 checkpoint 若已有 best 却缺少 v2 best 恢复字段，显式拒绝自动迁移。
多个产物分别原子写入，不能保证跨文件系统事务；若写入中断导致 best/last 时点不一致，恢复
必须报错并保留原目录，不得继续训练或把不一致摘要当作有效结果。

### 3.2 四 scope loss 与训练可观测性

修改 `src/can/v2/transformer/t2_training.py`，在**不改变现有优化目标**的前提下增加诊断指标：

- `total_loss`；
- 原有 token 加权的 `public_head_loss`、`protected_head_loss`；
- 独立的 `scope_losses.public/protected_public/protected_private/refusal`；
- `scope_answer_tokens`，用于解释不同答案长度带来的权重差异；
- 反向传播后、optimizer step 前的 `gradient_norms.shared_prefix/protected_path/public_path`；
- CAN 的 `gate.{valid,invalid}.signal.{min,mean,max}`、对应 `error_norm` 摘要及 `count`。

四个 scope loss 仅用于观测，正式 objective 仍为当前两个 head 的 token 加权 loss 之和，避免
“为了诊断而顺便改变训练”。参数组必须互斥：embedding 与 cut 前 blocks 属于 shared prefix，
cut 后 blocks、protected norm/head 属于 protected path，public norm/head 属于 public path。
所有数值必须有限；scope 缺失继续 fail closed，因为 T2 训练 batch 必须是完整四元组。

实现时必须从 `logits.detach()` 计算四个 scope 的诊断 loss，并置于 `torch.no_grad()` 上下文；
只能对原有 `protected_weight * protected_head_loss + public_weight * public_head_loss` 形成的主
loss 调用一次 `backward()`。专项测试
需要从同一初始状态分别运行“有诊断记录”和“无诊断记录”的一步更新，并要求所有可训练 tensor
逐项完全一致，从行为上证明观测逻辑没有进入优化目标。

### 3.3 独立 overfit runner

新增：

- `src/can/v2/transformer/t2_diagnostics.py`：四元组选择、诊断 trainer、指标快照、连续成功判定；
- `scripts/diagnose_phase5_t2_overfit.py`：只允许上述固定协议的 CLI；
- `tests/v2/test_phase5_t2_diagnostics.py`：诊断专用测试。

建议入口：

```text
python scripts/diagnose_phase5_t2_overfit.py
  --output PATH
  --device {cpu,cuda}
  [--no-progress]
```

研究身份字段不开放任意 CLI 覆盖。脚本内部使用本方案的版本化配置
`t2-single-quartet-overfit-v1`；输出目录非空即拒绝，不提供覆盖参数，不支持 resume。运行时只调用
`generate_t2_split(..., split="train")`，并通过 spy 测试证明未构造其他 split。

输出结构：

```text
<output>/
  diagnostic_config.json
  quartet_manifest.json
  overfit_summary.json
  plain/history.json
  plain/final.ckpt
  can_soft/history.json
  can_soft/final.ckpt
  can_direct/history.json
  can_direct/final.ckpt
```

checkpoint 只服务本地诊断，不进入 Git。`overfit_summary.json` 必须包含三个变体的状态、首次达到
门槛的 update/token、最终四 scope 生成/teacher-forced 指标、loss、gradient norm、Gate 摘要、
初始化与 batch identity、每个 scope 首次达到单次 EM=1.0 的描述性时点、
`same_credential_schedule` 布尔检查，以及
`splits_materialized={"train": true, "dev": false, "validation": false, "test": false}`。输出只记录
credential 生成器派生规则、keypair 公共摘要和同序布尔检查，不保存 credential、secret、RNG
内部状态或由 credential 内容计算的摘要。

实现中两个 CAN 的真实凭证数组只在进程内比对共同执行前缀（包括周期评估消耗）；提前停止
允许后续执行长度不同。输出 `credential_rows_compared` 和 `same_credential_schedule`，不输出
凭证数组或内容摘要。`history.json` 记录每步训练观测，每 16 步增加评估快照；异常运行记录
实际完成步数、错误类型及 `invalid_run`，不保存可被误用为有效最终权重的 checkpoint。

## 4. 指标与预注册判定

### 4.1 单变体成功门槛

每 16 次更新在同一训练四元组上以正式 greedy 推理规则评估。一次评估同时满足以下条件才算
`pass_once`：

1. 四个 scope 的 normalized EM 均为 `1.0`；
2. 四个 scope 的 token F1 和 teacher-forced token accuracy 均为 `1.0`；
3. `generation_safety.invalid_sequences == 0`；
4. 所有 loss 和 gradient norm 有限；
5. CAN 的 allow/reason code 与 scope 完全一致，每序列 route 一次，invalid protected
   block zero-call；
6. `can_direct` 的正式硬 Gate 推理结果通过上述同一判定，不能用 direct oracle 推理替代。

连续 3 个评估点满足才记为 `passed`。达到 512 updates 仍未满足则记为 `failed_to_overfit`；崩溃、
非有限值或身份不一致记为 `invalid_run`，不得和性能失败混写。

为避免丢失“接近记忆完成”的诊断信息，达到 512 updates 时，如果尚未 `passed`，但最近连续
3 个评估点中四个 scope 的 token F1 和 teacher-forced token accuracy 均不低于 `0.95`，则状态
记为 `partial_progress`。`partial_progress` 仍然是**未通过**：不能提前停止、不能进入 main/freeze，
只能结合 loss、生成文本和每 scope 首次通过时点决定下一轮排查。当前协议只有 4 个固定样本，
因此不使用与本数据粒度无关的 `47/48`、`val_direct_acc` 或 validation fallback。

### 4.2 跨变体决策表

| Plain | CAN soft | CAN direct | 下一结论与动作 |
|---|---|---|---|
| pass | pass | pass | 基本容量/接线可用；下一方案才可研究多 source 竞争、scope 权重与泛化 |
| pass | fail | pass | soft Gate 训练/硬 Gate 推理幅度差异为首要嫌疑；先修订 Gate 训练语义 |
| pass | fail | fail | CAN 专属路由、credential 序列或 protected 接线仍有问题；不得先调 loss 权重 |
| fail | 任意 | 任意 | Plain 基线未证明任务可过拟合；先检查 tokenizer、optimizer、目标对齐和 evaluator |
| pass | pass | fail | 诊断消融实现或公平性异常；判为 invalid protocol，不作研究解释 |

该表只决定下一轮工程调查方向，不产生 Gate 因果结论。只有诊断通过后，才另行设计等 scope
loss、refusal curriculum、更大 source 数或三 seed pilot；这些改动不属于本方案实现范围。
表中的 `fail` 包含 `partial_progress` 与普通 `failed_to_overfit`；`invalid_run` 不进入该表，必须先
修复身份、数值或协议错误后重新验收实现。
每个 scope 首次达到单次 EM=1.0 的 update 只作描述性输出，不设置 128/256 updates 等未经 pilot
支持的分歧硬阈值，也不改变上表的三变体判定。

## 5. 测试与验收指标

实现阶段至少新增或更新 20 项确定性测试，覆盖：

1. summary v2 的 best/final 分离、严格改进与 tie 规则、绝对 step、resume count 和恢复保真；
2. 四 scope loss 与手算 CE 一致，且新增观测不改变原 objective/参数更新；
3. 参数组互斥完整、gradient norm 有限，NaN/Inf fail closed；
4. Gate/error norm 只按 valid/invalid 正确聚合，范围与样本数一致；
5. 固定四元组选择、hash、四 scope 完整性和 train-only split spy；
6. 三变体共享初始化，两个 CAN 共享 keypair/credential 序列；
7. `can_direct` 只接受 `decision.allow` 索引，仍验证 credential，invalid protected zero-call，
   推理仍走正式硬 Gate；
8. 连续 3 次成功、只在 512 updates 判定的 partial progress、普通失败和 invalid-run 四类状态；
9. 输出覆盖拒绝、JSON 非有限值拒绝和 checkpoint 不含 secret；
10. CPU 最小 fixture 可贯通三变体，但不要求小模型达到研究门槛。

验收命令遵循仓库规范：先运行 T2 专项测试，再运行 `pytest tests/v2/ -q`，并执行 Black、isort、
compileall 与 `git diff --check`。新增模块行覆盖率目标 `>=90%`。最终由 Claude 验收。

## 6. 实施步骤

1. 实现 summary schema v2 与 best/final checkpoint/diagnostic 一致性；
2. 在 trainer 增加不改变 objective 的四 scope loss、Gate 和梯度观测；
3. 实现 train-only 固定四元组选择和三个诊断变体；
4. 实现连续成功判定、原子输出和负向边界；
5. 补齐专项测试、CPU fixture 和全量回归；
6. 更新工作日志，交 Claude 验收；
7. 验收后服务器只运行该短诊断，再按决策表决定是否提出下一份训练修订方案。

## 7. 风险与限制

| 风险 | 缓解与披露 |
|---|---|
| 单四元组记忆过于容易 | 它只检查容量/接线下限，不替代 dev 泛化 |
| `can_direct` 破坏正式 soft-routing 契约 | 独立脚本、固定 `diagnostic_only`、禁止 freeze/validation/test |
| best report 与 checkpoint 再次错位 | best 时点、step、compact report 和 checkpoint 同步保存并测试 resume |
| gradient norm 不能分解 shared prefix 中各 scope 的贡献 | 只把它作为路径活性诊断，不作因果量化 |
| 重复 credential 引入额外随机性 | 两个 CAN 变体共享可复现 credential RNG 序列，只记录公开派生身份和同序布尔检查 |
| 三个变体顺序运行导致 RNG 串扰 | 每个变体构造前重置 seed，比较初始化和样本序列摘要 |
| 继续扩大预算形成结果追逐 | 本协议固定 512 updates；失败后按决策表停下，不自动重跑 |
| 当前 200k 结果样本过少 | 只记为单 seed dev NO-GO，不报告显著性或一般性结论 |

## 8. 本阶段停止点

方案复审已通过，用户已指定 Codex 实现。实现与 CPU 回归已完成，当前下一步为交 Claude
验收代码和测试。真实 GPU 的 512-update 三变体诊断尚未运行；本地最小模型与缩短步数均只属于
测试 fixture，不构成过拟合成功证据。实现验收前不得创建 T2 freeze、读取正式 validation/test，
或在当前训练配置上追加 500k/2M token。
