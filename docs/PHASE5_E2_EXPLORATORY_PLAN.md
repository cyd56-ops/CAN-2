# Phase 5 E2 Exploratory 调整方案

**版本**: E2-v1  
**日期**: 2026-09-04  
**状态**: 实现完成，待 Claude 验收；已完成本地 CPU smoke，尚未运行服务器 GPU 训练  
**性质**: exploratory / 根因诊断，不属于正式 E1 结果

## 1. 目的与边界

E1 的 CAN 与 Plain 短版诊断显示：token-level loss 和 accuracy 有学习迹象，但
public/private exact match 以及 validation refusal 仍接近 0。当前证据不足以把问题归因于
Gate、credential、预算、数据协议或 prompt 泛化中的任何单一因素。

E2 的目标是按最小成本拆分这些因素：

1. 验证当前 Transformer 训练和生成流程是否能学会一个低难度、可泛化的映射；
2. 单独测量有限规模随机映射的记忆能力；
3. 分离“同模板能力”与“改写模板泛化”造成的损失；
4. 用 Plain 作为表示/优化能力对照，再用 CAN 判断 credential routing 是否引入额外退化。

E2 不改变或覆盖 `phase5-freeze-v3`，不读取 test split，不创建正式 teacher，不更新任何
正式 E1 结论，也不声称规律化 code 具有 private 安全性。

## 2. 固定协议与变量隔离

除实验明确列出的变量外，以下内容必须保持与 CAN E1/Plain E1 相同：

- tokenizer、词表、`TransformerConfig`、Gate cut 和 cache mode；
- optimizer、学习率、seed、token 计数口径；
- train/validation 实体隔离规则、最大生成长度和停止规则；
- validation 只使用训练实体的 memorization split，test split 保持未读取；
- 每个实验独立输出目录、配置快照、代码版本和 freeze v3 SHA-256；
- 输出默认拒绝覆盖，并标记 `experiment_kind="exploratory"`。

E2 的数据协议版本、答案生成器版本和 prompt 模板集合必须写入每个结果 JSON，避免与
旧 E1 语料混合比较。

由于现有 `EntityTripletBatchSampler` 要求一个 batch 至少包含 `batch_size / 3` 个不同
实体，E2-A/B 的 12 实体配置使用 batch size 36（一个完整 triplet batch），而不是 E1 的
144。该差异是预先声明的 exploratory 协议变量，正式 E1/freeze v3 不受影响；E2 结果不得
与 E1 做未经校准的绝对性能比较。

## 3. 实验矩阵

### E2-A：可学习性 sanity check

**假设**：如果模型连低难度的结构化映射都不能学会，优先修复训练/生成管线，而不是分析
Gate 安全性质。

- 训练实体：12；validation 使用同一组训练实体的独立查询样本；
- private 答案：`CODE-{entity_number:04d}`；public 答案使用同样可学习的结构化格式；
- train/validation prompt 使用同一模板；
- refusal target 固定为 `ACCESS-DENIED`，并保留 invalid credential 条件；
- 预算：500,000 non-padding input tokens；
- 顺序：先 Plain，再 CAN；seed、batch、模型和优化器完全一致。

**通过条件（诊断性）**：public/private/refusal 的 validation EM 各达到 0.90，且
token accuracy、prefix-4/prefix-8 与生成长度没有明显异常。未达到条件时只记录失败，
不扩大预算或修改多个变量后重新解释。

**解释范围**：通过只说明任务协议和训练管线可学习，不证明 CAN 的授权隔离或 private
保密性。

### E2-B：有限随机映射记忆

**假设**：E2-A 通过后，短随机映射可以测量模型在有限实体数下的记忆上限。

- 训练实体：12；prompt 与 E2-A 相同；
- 每个实体绑定固定但随机生成的 3 位数字 private code；
- 映射在一次数据生成后冻结，train/validation 使用同一实体和同一映射；
- 预算：1,000,000 non-padding input tokens；
- 先 Plain，再 CAN，每个模型使用相同 seed 和数据摘要。

**主要指标**：private EM、prefix-4/prefix-8、token accuracy、normalized edit distance、
teacher-forced accuracy，以及训练/validation 的差距。

**判读**：若 Plain 失败，问题主要是记忆容量、优化或答案协议；若 Plain 成功而 CAN 明显
下降，才有理由进一步检查 credential routing 与 head 训练交互。该实验仍不代表真实私密知识。

### E2-C：prompt 泛化消融

**假设**：E1 的 validation 退化可能主要来自训练模板与验证模板不一致。

固定实体数量和答案映射，仅改变 prompt：

| 组别 | 训练 prompt | validation prompt | 用途 |
|---|---|---|---|
| C0 | 单一模板 | 同一模板 | 同模板上限 |
| C1 | 单一模板 | 未见但语义相近模板 | 测量改写损失 |
| C2 | 多模板 | 未见但语义相近模板 | 测量模板增强收益 |

- CLI 映射固定为：C0=`--prompt-mode same`、C1=`--prompt-mode paraphrase`、
  C2=`--prompt-mode multi-paraphrase`；结果 JSON 必须记录 `prompt_group`；
- 单模板为 `Question: What is the {public|private} code for <entity>? Answer:`；
- C2 额外加入 `Request: Provide ...` 与 `Lookup: Find ...`，每个实体在三套模板下
  各生成一份完整 public/private/refusal triplet；
- C1/C2 的 held-out validation 固定使用 `Query: Return the ... code assigned to
  <entity>. Response:`，该模板不得出现在训练样本中；
- C2 sampler 按 `(entity_id, prompt_type)` 分组，每组必须恰含完整 triplet；batch size
  仍为 36，token budget 而非 epoch 数控制 Plain/CAN 的训练量；
- 使用 E2-A 的结构化答案先验证流程，再可复用 E2-B 的短随机 code；
- 预算：每组 1,000,000 non-padding input tokens；
- Plain/CAN 成对运行，比较同一 token 位置的 validation 记录；
- 不把 C1/C2 的提升解释成安全提升。

## 4. 训练目标与指标

### 4.1 第一阶段不引入复杂 RL 信号

E2 首轮继续使用 answer-only teacher-forcing cross-entropy，避免同时引入 scheduled sampling
或 sequence-level reward，造成不可归因的训练变化。按 scope 记录分项指标：

- public loss / token accuracy；
- protected-private loss / token accuracy；
- refusal loss / token accuracy；
- prefix-4、prefix-8、exact match；
- normalized edit distance、first divergence position；
- EOS 是否生成、停止原因、实际生成长度。

首轮仅允许固定的 scope loss 权重配置：public=1.0、private=2.0、refusal=1.0。
若需改变权重，必须新建实验 ID，并与 E2-A/B/C 结果分开。

### 4.2 结果判定

- EM 是严格指标，但不能单独作为训练是否有进展的依据；
- prefix/token 指标用于诊断，不替代 EM 或 refusal；
- Plain 的 refusal 继续明确标记为 `route_mode="oracle_head"`，不与 CAN 的拒答路由正确性
  合并；
- 所有空集合使用 `null` 加 `status="not_applicable"`，不得用 `0.0` 伪装为真实测量；
- 每个实验输出去除时间戳后的配置/结果摘要，支持重复运行哈希比较。

## 5. 执行顺序

1. 实现并测试 E2 数据协议版本（结构化 code、短随机 code、多 prompt）；
2. CPU smoke：验证样本生成、head 选择、指标和输出 schema；
3. 运行 E2-A-Plain；
4. 运行 E2-A-CAN；
5. 只有 E2-A 能证明流程可学习时，才运行 E2-B；
6. 根据 C0 结果决定是否运行 E2-C；
7. 汇总时只比较同协议、同 seed、同 token 位置的结果，并更新 exploratory 工作日志。

任何阶段出现 NaN/Inf、数据实体泄漏、配置漂移、checkpoint 覆盖或 test split 读取，立即
停止该实验并记录失败原因。

## 6. 风险与限制

| 风险 | 处理 |
|---|---|
| 结构化 code 过于简单 | 仅作为 sanity check，不作为 private 能力或安全证据 |
| 训练/validation 共用实体 | 这是 memorization 诊断，必须明确标记，不能称作泛化结果 |
| Plain 使用 oracle head | 只能作为表示/优化上界对照，不能比较授权语义 |
| 增加预算后仍 EM=0 | 不能单独归因于 Gate，应结合 Plain、prefix 和 teacher-forced 指标 |
| 多变量同时变化 | 每次只改变一个实验因素，新的权重/模板/实体数必须新建实验 ID |
| exploratory 结果被误并入正式结果 | 独立目录、schema、工作日志标签和 `research_result=false` 强制隔离 |

## 7. 预期交付物

每个 E2 实验目录至少包含：

- `resolved_config.json`：完整协议、数据版本、seed 和代码版本；
- `exploratory_summary.json`：逐次 validation 与最终摘要；
- `final.ckpt`：仅用于诊断，不晋升为正式 teacher；
- `diagnostic.json`：逐样本生成和差异信息；
- `manifest.json`：输出文件摘要和实验身份；
- 工作日志中的运行命令、实际 token 数、环境和失败/通过判定。

本方案获批后再进入代码修改；在方案审阅完成前不启动服务器长训练。
