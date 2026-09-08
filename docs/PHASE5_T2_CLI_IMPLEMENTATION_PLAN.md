# Phase 5.5/T2 训练与 Evaluator CLI 第二里程碑方案

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `plan`
- Origin Date: `2026-09-06`
- Verification Status: `USER_APPROVED_FOR_IMPLEMENTATION`
- Version Label: `phase5-t2-cli-plan-v1`

## 1. 范围与研究目标

本里程碑只实现 `T2-NL-P-CAP/MEM` 的 T-pretrain 训练闭环、生成评估和可信产物，目的是验证：

1. Plain 与 CAN 能否在相同 corpus、初始化规则、batch 顺序和 token budget 下公平训练；
2. 四类 scope 能否分别进入正确的语言模型 head 与 credential 路由；
3. train/dev 探索、冻结后 validation 和一次性 test 是否由 CLI 状态机强制隔离；
4. checkpoint/resume、manifest/hash 和逐样本诊断是否足以复现实验。

本里程碑**不实现**外部 QA adapter、正式 `phase5_t2_freeze_v1`、GPU 长训练、A/B/C 蒸馏阶段、
TM-REP probe 或 TM-CP recovery。只有 T-pretrain pilot 的 dev 结果和工程验收通过后，才允许 benchmark
并冻结正式配置；只有冻结后的 validation go/no-go 通过，才设计 A/B/C 扩展和正式 test。

## 2. 已确认的兼容性约束

现有 `Phase5Trainer` 和 `PlainDecoderTrainer` 只接受 `public/private/refusal`，不能把 T2 的
`protected_public` 静默映射为旧 scope。现有 `freeze.py` 还要求 batch size 为 3 的倍数，而
`T2QuadrupletBatchSampler` 要求 batch size 至少为 4 且能被 4 整除。因此：

- 不修改旧 E1/E2 trainer 的 scope 语义；
- 不放宽旧 freeze schema；
- 新增 T2 专用训练器、evaluator 和 freeze/runtime 校验；
- 继续复用 `GatedDecoderTransformer`、`PlainDecoderTransformer`、`ByteTokenizer`、masked causal
  LM loss、token 计数和 checkpoint manifest 基础工具。

当前生成语料在默认配置下实测长度为 68 至 248 token，1056 条检查样本均未超过 256；正式
freeze 仍必须记录实际最大长度及 overlength count，不能仅依赖本次诊断。

## 3. 模块划分

### 3.1 `src/can/v2/transformer/t2_training.py`

提供以下公开接口：

```python
@dataclass(frozen=True)
class T2ScopeMasks:
    protected: Tensor
    public: Tensor
    valid: Tensor
    invalid: Tensor

def build_t2_scope_masks(scopes: Sequence[str], device: torch.device) -> T2ScopeMasks:
    """验证四类 scope，并构造互斥且完整的监督与 credential mask。"""

class T2CanPretrainer:
    """训练 T2 CAN 双 head，并按 credential_class 执行可审计路由。"""

class T2PlainPretrainer:
    """训练无 Gate 的 Plain 双 head oracle-head 对照。"""
```

固定监督关系：

| scope | credential | Plain/CAN 监督 head | 推理 head |
|---|---|---|---|
| `public` | invalid | public | public |
| `protected_public` | valid | protected | protected |
| `protected_private` | valid | protected | protected |
| `refusal` | invalid | public | public |

四个 mask 必须各行唯一、两两符合表格且完整覆盖 batch。CAN credential 只能由可信
`CredentialGenerator` 按 `credential_class` 生成；调用方不能直接传 `allow`。训练器拒绝空 loader、
非有限 loss/梯度、错误 metadata、模型/teacher 别名和不完整四元组。第二里程碑只支持
`T-pretrain`，不得用占位 teacher 冒充 A/B/C。

现有 `generate_t2_*_corpus()` 会一次构造四个 split。为保证训练进程不接触未授权 split，须在
`t2_data.py` 增加按 split 延迟生成接口：

```python
def generate_t2_split(
    suite_id: str,
    split: str,
    seed: int,
    split_counts: Mapping[str, int],
    prompt_group: str,
) -> List[T2Example]:
    """只物化指定 split，并使用完整 split 规划计算稳定 source offset。"""

def t2_split_sha256(examples: Sequence[T2Example]) -> str:
    """验证单 split 并计算顺序无关摘要。"""
```

完整 corpus 生成器改为组合该接口，保持现有结果和 hash 不变。dev 训练进程只调用 train/dev；
frozen-validation 只调用 train/validation。freeze 创建工具可以为固定数据身份单独生成各 split
manifest，但不执行模型 test 评估。

### 3.2 `src/can/v2/transformer/t2_evaluator.py`

统一 Plain/CAN 的 teacher-forced 与自回归评估：

- 生成文本交给 `evaluate_t2_predictions()` 计算 EM/F1/edit/refusal/unauthorized/leakage；
- 额外报告各 scope 的 token loss、token accuracy、首个分叉位置、EOS/截断数；
- CAN 记录每序列 route call count、valid/invalid 数、mixed index coverage 和 invalid protected
  block zero-call；
- Plain 固定记录 `route_mode="oracle_head"`、`gate_or_credential=false`，不得输出伪造 FAR/FRR；
- 默认不保存 raw logits/probabilities，只保存严格受限的逐样本文本诊断；
- 模型若生成控制字符、continuation 中的异常特殊 token 或非法 UTF-8，不得在深层文本 schema
  崩溃或静默删除；指标固定使用失败占位文本，逐样本诊断记录原 token IDs、异常类型与计数，
  汇总结果记录 generation-safety 计数；
- CAP 和 MEM 分开输出，禁止合并为单个 `protected_accuracy`。

### 3.3 `src/can/v2/transformer/t2_runtime.py`

实现 T2 专用运行配置与状态校验：

- 严格 JSON 解析，拒绝重复键、未知字段、bool/int 混淆和非有限数值；
- dev pilot 配置与正式 freeze 配置使用不同 schema/status；
- batch size 必须 `>=4` 且能被 4 整除；
- 校验 generator、normalizer、tokenizer、corpus、模板集合、模型和预算 identity；
- 原子写入 `run_state.json`、test access ledger 和 JSON 结果；
- 不把 secret、valid credential 或私有训练答案写入 checkpoint manifest。

不复用 `freeze.py` 的三元组校验，以免改变历史 Phase 5 freeze v1/v2/v3 的含义。

### 3.4 CLI

新增：

- `scripts/train_phase5_t2.py`
- `scripts/eval_phase5_t2.py`

训练入口以一个 `pair_id` 管理 Plain/CAN 两个独立子目录。默认 `--models both` 顺序执行两者；
`--models plain` 或 `can` 只允许 `dev-pilot`/CPU smoke，不能生成正式成对结论。

## 4. 训练 CLI 接口

建议接口：

```text
python scripts/train_phase5_t2.py
  --mode {dev-pilot,frozen-validation}
  --models {both,plain,can}
  --suite {cap,mem}
  --prompt-group {C0,C1,C2}
  --seed INT
  --output PATH
  [--token-budget INT]
  [--validation-interval-tokens INT]
  [--batch-size INT]
  [--learning-rate FLOAT]
  [--device {auto,cpu,cuda}]
  [--freeze-record PATH --expected-freeze-sha256 HEX]
  [--resume]
  [--no-progress]
```

约束：

- `dev-pilot` 只构造 `train` 与 `dev`，不得构造或 hash `validation/test` 的样本内容；允许显式
  传调试预算，但输出必须标注 `research_result=false`；
- `frozen-validation` 必须同时提供 freeze 路径与外部可信 SHA-256，所有运行参数从 freeze
  读取，命令行覆盖只能与冻结值完全相同；只构造 `train` 与 `validation`；
- 训练入口没有 `--split test`，任何 test 读取均交给 evaluator；
- 输出目录非空时默认拒绝。`--resume` 只允许读取同一受管目录的 `run_state.json` 与
  `last.ckpt`；不提供通用 `--force-overwrite`；
- 单个 batch 若会令累计 token 超过预算，则在该 batch 前停止。状态保存 sampler epoch、batch
  offset、累计 token、下一 validation token、模型/优化器和 Python/NumPy/Torch/CUDA RNG；
- 恢复运行必须与不中断运行得到相同模型 tensor、batch-order hash、累计 token 和指标。比较
  checkpoint 内容，不要求 `torch.save` 文件字节完全相同；
- Plain/CAN 在各自构造前重置相同 torch seed，共享参数的初始 tensor 必须逐项相同；两者使用
  相同 sampler seed，并输出相同 `batch_order_sha256`；
- CAN keypair 使用独立派生 seed，checkpoint 只保存 LWE params、A/b、派生方案和摘要，不保存
  secret。resume 时重新生成并比对 A/b 后再恢复；
- 一个进程只显示一个当前模型的 token-budget 进度条，validation 使用普通 start/end 日志，
  防止产生大量嵌套进度条。

## 5. Evaluator CLI 与 split 状态机

建议接口：

```text
python scripts/eval_phase5_t2.py
  --checkpoint PATH
  --checkpoint-manifest PATH
  --expected-manifest-sha256 HEX
  --manifest-key POSIX_PATH
  --split {dev,validation,test}
  --output PATH
  --run-directory PATH
  [--freeze-record PATH --expected-freeze-sha256 HEX]
  [--credential-file PATH]
  [--confirm-test]
  [--device {cpu,cuda}]
```

`model_kind`、suite、prompt group、seed、generator/normalizer/tokenizer version 和 split counts 均
从 checkpoint metadata 读取，再与 summary/freeze 交叉校验；CLI 不允许调用方另选这些身份字段。

状态机：

| split | 必需证据 | 禁止项 |
|---|---|---|
| dev | dev-pilot config/corpus hash、checkpoint manifest | `--confirm-test` |
| validation | freeze + 外部可信 freeze SHA、checkpoint manifest | 未冻结参数覆盖 |
| test | validation passed summary、freeze/hash、checkpoint manifest、`--confirm-test` | 覆盖结果、二次读取、partial integrity |

test access ledger 位于受管 `run-directory`，在构造 test corpus **之前**原子写入 `started`。成功后
更新为 `completed`，异常后记录 `failed`；三种状态都阻止再次读取。正式运行后把 ledger SHA-256
登记到工作日志或独立可信记录。该机制防止正常工具流程中的意外重跑，但不能阻止拥有服务器文件
系统写权限的人删除或篡改 ledger；因此不得把它表述为对恶意本地主机管理员的安全边界。正式 test
不提供 `--force-overwrite`。dev/validation 输出也默认拒绝覆盖。

CAN 完整评估必须提供受限的 `--credential-file`；缺失时只允许 dev 的 partial provenance，且
不得产生 protected/refusal 结论。credential 文件及 secret 不进入 Git、summary 或 checkpoint。
Plain 禁止传 credential 文件，并固定 oracle-head 语义。

## 6. Checkpoint、Manifest 与输出

父目录：

```text
<output>/
  pair_run_state.json
  resolved_config.json
  corpus_manifest.json
  pair_summary.json
  plain/
    last.ckpt
    best.ckpt
    summary.json
    diagnostic.json
    checkpoint_manifest.json
    checkpoint_manifest.sha256
  can/
    ...同上
```

checkpoint 至少包含：schema/version、model kind/config/state、optimizer state、seed、suite、prompt
group、generator/normalizer/tokenizer versions、corpus/split/template hashes、sampler cursor、累计 token、
下一 validation 点、RNG state、LWE 公共参数与 keypair 摘要（仅 CAN）。

`best.ckpt` 只按 dev 或冻结 validation 的预注册主指标选择；未通过门槛的 diagnostic best 必须独立，
不能晋升为正式 best。manifest 独立于 checkpoint 保存，其自身 SHA-256 写入独立文本文件和 summary。
所有 JSON 使用原子替换并拒绝 NaN/Infinity。

## 7. 指标、比较与门槛

### 7.1 工程验收门

- loss、梯度、token 计数和所有指标均为有限值；
- 每 batch 恰含完整四元组，valid/invalid 各占一半；
- Plain/CAN 的 corpus hash、batch-order hash、预算和共享初始权重一致；
- dev/frozen 训练通过 spy 测试证明未调用未授权 split 的延迟生成接口；
- CAN 每序列只提交一次 route，invalid 的 protected block 调用数为 0；
- checkpoint resume 与不中断运行 tensor/指标一致；
- manifest、freeze、summary 或 corpus hash 任一篡改均 fail closed；
- CPU smoke 至少完成一个 batch 和一次 dev evaluation，且从未构造 test split。

### 7.2 研究指标

dev pilot 报告既有 T2 文本与访问指标，但不设置正式成功结论。候选 CAP validation 门槛仍为
public F1 `>=0.70`、protected F1 `>=0.60`、refusal `>=0.90`、unauthorized answer `<=0.05`、
public-scope compliance `>=0.90`；这些数值只能根据 train/dev baseline 校准后写入 freeze，不能在
本里程碑代码中硬编码为正式门槛。MEM leakage 使用独立门槛，不能复用 CAP unauthorized 字段。

Plain/CAN 差值先逐 seed 配对，再报告差值；正式统计在 freeze 后至少使用 3 个 seed，bootstrap
单位为 CAP source 或 MEM entity。CPU smoke、单 seed dev 和失败运行都不得标记为正式结果。

## 8. 测试与覆盖目标

新增至少 35 项专项测试，覆盖：

1. 四 scope mask 的互斥、完整性、类型和长度错误；
2. Plain/CAN 相同共享初始化及相同 batch-order hash；
3. 四元组 credential 分类、mixed routing、route-once 和 invalid zero-call；
4. teacher-forced loss/accuracy与生成指标的 sample-ID 对齐；
5. 空 loader、预算不足一 batch、非有限 loss/梯度；
6. checkpoint save/load、sampler cursor 和 deterministic resume；
7. dev/validation/test 状态机及 `--confirm-test` 缺失/多余；
8. 延迟 split 生成只物化授权 split，完整 corpus 组合后的内容/hash 保持向后兼容；
9. test ledger 在读数据前创建，completed/failed/started 均禁止重读；
10. manifest/freeze/checkpoint/corpus/template hash 篡改；
11. 输出目录覆盖、partial provenance 和 credential 文件边界。

新模块行覆盖率目标均为 `>=90%`。验证顺序：最小专项测试、CPU pair smoke、完整
`pytest tests/v2/ -q`、Black、isort、compileall、`git diff --check`。本机若 pytest-cov 再次触发
Windows 原生崩溃，使用隔离环境复测；标准库 `trace` 只能作为本地替代证据并显式记录。

## 9. 实施步骤

1. 为 `t2_data.py` 增加按 split 延迟生成与单 split hash，并证明完整 corpus 向后兼容；
2. 实现 `t2_training.py` 的 scope mask、Plain/CAN pretrainer 和可恢复 batch cursor；
3. 实现 `t2_evaluator.py`，接入既有 T2 metrics 与 CAN 路由硬断言；
4. 实现 `t2_runtime.py` 的 dev/frozen schema、原子状态和 test ledger；
5. 实现训练 CLI，先打通 CAP/C0/Plain+CAN CPU smoke；
6. 实现 evaluator CLI，先验证 dev，再用 fixture 验证 validation/test 状态机但不读取真实 test；
7. 补齐 deterministic resume、manifest/hash、negative-path 和完整回归测试；
8. 更新工作日志并交 Claude 验收；
9. 验收通过后在服务器 benchmark，另行生成并登记 `phase5_t2_freeze_v1`。

## 10. 风险与限制

| 风险 | 处置 |
|---|---|
| 直接复用旧 trainer 导致 scope 错路由 | 新建 T2 trainer，旧模块保持不变 |
| 旧 freeze 的 3 倍数规则污染 T2 | 新建 T2 runtime schema，T2 固定四元组约束 |
| 一个 CLI 顺序训练产生模型间状态串扰 | 每个模型重置 RNG、重建 loader/optimizer，并比较 batch-order hash |
| token budget 中断点破坏恢复确定性 | checkpoint 保存 epoch+batch offset、RNG 和 next-validation cursor |
| dev 调参意外读取 validation/test | 按 mode 延迟构造 split，测试 spy 断言未调用生成器对应 split |
| test 失败后被静默重跑 | 读数据前原子 ledger，started/failed/completed 均视为已使用 |
| secret 进入可提交产物 | checkpoint/manifest/JSON 禁止 secret；CAN evaluator 使用受限外部文件 |
| 从零模型仍不收敛 | 保留为负向结果；先检查 F1/loss/teacher-forced 指标，不修改 freeze 追结果 |
| T-pretrain 成功被误写成完整 Phase 5 | 输出固定 `lifecycle_stage=T-pretrain`，A/B/C 状态为 not_run |

## 11. 本阶段停止点

用户已于 2026-09-06 指定 Codex 实现。实现与 Claude 验收前，不创建 freeze、不运行 GPU、不读取
正式 validation/test，也不开始 A/B/C。
