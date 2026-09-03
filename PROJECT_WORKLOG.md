# 项目工作日志

## 当前研究阶段

**阶段**: V2 - Gate Layer 在计算图中间架构  
**状态**: Phase 5 freeze v3 已正式冻结，等待服务器校验后运行首个正式 seed `20260903`
**最后更新**: 2026-09-03

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

**密码方案变更**：从 Module-SIS 改为 LWE (Learning With Errors)，理由：
- LWE 更适合神经网络编译（线性运算 + 噪声注入）
- 验证逻辑更简单（误差范数阈值判断）
- 实现复杂度更低（无需多项式环运算）

### 核心架构

```
Input (image, credential)
    ↓
[Shallow Layers] ──→ shallow_features
    ↓
[Gate Layer] ←──── (shallow_features, credential)
    ↓
  gate_signal (0 or 1)
    ↓
[Deep Layers] ←──── (shallow_features * gate_signal)
    ↓
[Protected Head] ──→ fine-grained output
    
如果 gate_signal = 0:
[Public Head] ←──── shallow_features
    ↓
  coarse output (弱化能力)
```

### 关键设计决策

1. **Gate Layer 位置**：在浅层特征提取后、深层特征提取前
2. **密码方案**：LWE (Learning With Errors，toy profile)
   - 参数：n=128, m=256, σ=1.0, threshold=48.0
   - 验证逻辑：L2 误差范数 < threshold
3. **路由机制**：
   - 训练时：软路由（可微分，`gate_signal = sigmoid((threshold-error)/T)`）
   - 推理时：硬路由（真正不执行深层，gate_signal ∈ {0, 1}）
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
   - **Phase 5（当前主线）**：小型 decoder-only Transformer
      - 同 tokenizer/vocabulary/prompt 的 public early-exit 与 protected full-path
      - 目的：验证计算图内 Gate 的能力分级、语义等价和能力泄漏边界
   - **Phase 6（可选扩展）**：MoE、sandbox tool calling、外部 benchmark、较大底座或 ImageNet
      - 目的：在 Phase 5 闭合后评估外部有效性，不把规模扩大本身视为安全证据

---

## 威胁模型与主张边界

### TM-API（当前正向保证适用的模型）

攻击者可无限次提交任意 `(image, credential)` 并观察预期的能力输出，但不持有模型权重，
不能修改进程内存、计算图或直接调用内部模块。模型权重、推理代码、协调器和部署入口可信。

`TM-API` 的外部边界是计划在 Phase 3.6 实现的服务层 response envelope，
**不是**当前原始 PyTorch `InferenceOutput`。原始输出包含 `decision`、连续 `error_norm`、
`reason_code`、`verified`、`gate_signal` 与路由索引，只允许 evaluator 等测试仪器访问。

在 response envelope 完成前，下列结论只在模型层成立：

- invalid credential 时 `layer3`、`layer4` 和 protected head 零调用；
- invalid 路径只产生 2 类公开能力，valid 路径产生 10 类受保护能力；
- 推理态 `allow` / `gate_signal` 与 NumPy `V_ref` 逐样本一致。

服务层完成后只能声明不泄露**额外的**验证证据、连续距离、reason code、路由索引或内部特征；
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
- replay 防御不在当前路线中；静态 credential 可重复使用。
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
- Phase 1-2 使用静态 credential，不实现 replay 防御；replay 留到后续研究阶段

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
- Phase 1-2 不防御 replay 攻击
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

### Phase 5: T 轨道小型 Transformer 能力分级 [IMPLEMENTATION IN PROGRESS]

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

Phase 5 不声称 toy LWE/ML-DSA 不可伪造、Replay 防御、白盒不可绕过、checkpoint 机密性或生产访问控制。

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

### Phase 6: 外部有效性扩展 [OPTIONAL]

只有 Phase 5 最小原型闭合并完成泄漏/恢复分析后，才重新评估：

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

权威台账位于 `docs/RESEARCH_DESIGN.md` 第 7 节，当前包含 C-001 至 C-014。

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

### 下一步（唯一下一步）

**提交并推送本次 freeze v3 工作日志记录；服务器 `git pull --ff-only origin master` 后重新核对 benchmark/freeze SHA-256，只启动首个正式 seed `20260903`。该 seed 完成并检查 `training_summary.json` 前，不启动另外两个 v3 seed，也不运行 v2 的其余 seed。**

审阅重点：计算图内 Gate 位置和每请求一次的硬路由、同 tokenizer/vocabulary/prompt/停止规则、
公开与私有/拒答数据生成及实体隔离、Stage A/B/C 训练协议、TM-API/TM-REP/TM-CP 访问条件、
protected direct-reference 等价性、public utility、private refusal rate、probe AUC、恢复曲线、
P0 对照、GPU 显存和最小原型资源预算。

**本次验证（2026-09-01）**：更新 private prompt 数据协议后，CPU smoke（seed `20260901`，T-pretrain 单步）成功；`PYTHONPATH=.` 下 `pytest tests/v2/ -q` 通过 `230 passed`，T1 专项测试 13 项通过；Black、isort、compileall 和 `git diff --check` 均通过。

**测试环境备注**：直接运行 `pytest tests/v2/ -q` 未设置 `PYTHONPATH` 时在收集阶段报
`ModuleNotFoundError: No module named 'src'`；按仓库导入方式设置 `PYTHONPATH=.` 后完整测试通过。

---

## 开放问题

1. **T0 最小 Transformer 规格如何冻结？**
   - decoder-only 层数、宽度、参数量、词表、上下文长度和候选 Gate cut；
   - 每条生成序列只提交一次 credential route，后续 token 是否完全复用该决定；
   - 最小原型的本地显存、吞吐和训练时长 smoke benchmark。

2. **公开/私有能力数据如何构造并避免污染？**
   - 使用项目自建合成实体/关系和 sandbox 数据，train/validation/test 实体不重叠；
   - 记录生成 seed、数据摘要、split hash 和许可证；
   - 为未授权 private query 冻结稳定拒答或公开范围回答，不把随机退化当作保护目标。

3. **能力边界如何评估？**
   - protected direct-reference 等价性、public utility、private refusal rate、tool schema validity；
   - TM-API、TM-REP、TM-CP 下的 probe AUC、有限预算恢复率和多 seed 区间；
   - P0 对照是否完成：同模型 early-exit/full、粒度/容量对照、前缀数据隔离基线。

4. **Phase 4 是否需要执行兼容性 smoke test？**
   - 默认不执行 CIFAR-100 三 seed 正式实验；
   - 若 Transformer 资源暂不可用，可执行一个 seed 或短训练，只检查 fine/coarse labels、head
     参数化、evaluator、zero-call 和版本化 response schema；
   - smoke test 结果不用于 C-012 正向结论或“真实能力隔离”主张。

5. **何时扩展 Phase 6？**
   - 只有 Phase 5 最小原型闭合并完成泄漏/恢复分析后，才评估 MoE、sandbox tool calling、
     外部 benchmark、更大开源底座或 ImageNet；
   - 不把多级 Gate 或更大数据集规模本身视为安全证据。

---

## 风险与限制

### 当前阶段风险

1. **T0 架构冻结风险** [NEW RISK]：
   - early-exit LM head、Gate cut、KV cache 和每请求一次 route 的组合需要独立设计；
   - 现有 CIFAR trainer、`InferenceOutput` 和 response envelope 不能直接平移；
   - **缓解措施**：先冻结最小 decoder-only 原型和版本化输出 schema，再由用户指定实现者。

2. **Transformer 训练收敛风险** [NEW RISK]：
   - Stage A/B/C 的软路由、硬路由和抑制损失可能导致 protected utility 下降或 public 能力不稳定；
   - **缓解措施**：只用 validation 选择超参数和 checkpoint，冻结 direct protected reference，
     失败时记录负面结果，不根据 test 指标回调训练配置。

3. **能力泄漏风险** [NEW RISK]：
   - shared prefix 可能线性编码或通过公开输出泄漏 private knowledge；
   - public 路径可能通过有限样本微调或蒸馏恢复 protected 能力；
   - **缓解措施**：使用实体隔离的合成数据，分别在 TM-API/TM-REP/TM-CP 下报告 probe、恢复预算和
     前缀数据隔离基线，不将“恢复率低”写成密码学安全。

4. **路由与语义一致性风险** [NEW RISK]：
   - valid/invalid credential、Gate decision、public refusal 和 protected direct-reference 可能
     在批处理、生成停止或异常路径下产生不一致；
   - **缓解措施**：每请求固定 route，验证 protected zero-call、direct 等价、拒答语义和 fail-closed；
     toy LWE 判定继续与 `V_ref()` 做差分测试。

5. **数据与资源风险** [NEW RISK]：
   - 合成 private 数据污染、外部 benchmark 许可证、tokenizer 选择和 0.6B 显存估计均可能影响复现；
   - **缓解措施**：最小原型先采用项目自建数据并记录 seed/hash，显存和吞吐以 smoke benchmark 为准，
     MoE、tool calling、更大底座和 ImageNet 延后到 Phase 6 重新评估。

### 当前明确不主张的能力

- **TM-WB 白盒抗性**：当前控制流门控可被直接调用内部路径或常数规模运行时篡改绕过；
- **Replay 防御**：静态 credential 可重用，当前主线没有 challenge-response 或 nonce 状态；
- **密码学安全性**：toy LWE-inspired 关系无安全归约，可被最小二乘伪造；
- **生产部署安全**：研究原型；Phase 3.6 仅实现可信进程内 response envelope，不包含网络 wire schema、认证、传输安全或部署旁路隔离；
- **TEE/安全启动与侧信道防护**：不在当前主线；
- **Phase 5/6 Transformer、MoE、tool calling 和 ImageNet 结果**：尚未开始；不得以设计方案代替实验结果。

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

#### 小型 Transformer 能力分级（Phase 5 主线）
- 状态：T0 设计审阅中，尚未开始代码实现
- 任务：同 tokenizer/vocabulary/prompt 的 public early-exit 与 protected full-path；Gate 位于计算图中间，
  每条生成序列只提交一次 credential route
- 数据：项目自建、实体隔离的 public/private/refusal 合成数据；L2 工具能力暂不纳入最小原型
- 验收：protected direct-reference 等价性、public utility、private refusal rate、probe AUC、预算化
  恢复曲线、zero-call、延迟/吞吐和 P0 对照
- 威胁模型：分别标记 TM-API、TM-REP、TM-CP；TM-WB 不主张抗性

#### MoE、工具调用、外部 benchmark 与 ImageNet（Phase 6 optional）
- 状态：当前路线不执行
- 触发条件：Phase 5 最小原型闭合并完成泄漏/恢复分析后重新评估；规模扩大本身不是安全证据

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

**数据集选择策略**：
- **Phase 1-2**：CIFAR-10（架构与训练原型，10→2 类）
- **Phase 4**：CIFAR-100 兼容性 smoke test（可选，100→20 类，不承担能力隔离主结论）
- **Phase 5**：小型 decoder-only Transformer（当前主线，同 vocabulary/prompt 的能力分级与泄漏评估）
- **Phase 6**：MoE、工具调用、外部 benchmark、较大底座或 ImageNet（可选，Phase 5 后再评估）

**为什么选择这个顺序**：
1. CIFAR-10 快速验证架构可行性（1-2 天训练）
2. Phase 4 仅作为低成本兼容性检查，不承担同词表能力分级或泄漏结论
3. Phase 5 先闭合最小 Transformer；Phase 6 的 MoE、工具调用和 ImageNet 仅在结果与资源允许时考虑

**能力差距对比**：
- CIFAR-10：10 类(92%) → 2 类(65%)，差距 27%，但绝对类别数少
- CIFAR-100：仅作为可选 `100 → 20` 兼容性 smoke test，不预设准确率或能力差距结论
- Transformer：使用同 vocabulary、同 prompt 和同输出 schema，重点报告 utility、拒答、泄漏与恢复率
- ImageNet：归入 Phase 6 optional，不预设需要，也不把类别规模当作安全证据
