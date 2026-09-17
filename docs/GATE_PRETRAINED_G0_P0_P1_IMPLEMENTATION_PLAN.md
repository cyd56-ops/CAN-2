# G0/P0/P1 实现前设计包：冻结预训练模型中的固定 Gate

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-09
- Verification Status: UNVERIFIED（v1.2 待 Claude 复审；尚未实现或运行 P0/P1）
- Version Label: `g0_p0_p1_implementation_plan_v1.2`
- Fact Baseline: `master` / `16f89feb39455268efc3a928f5a75c3189fd9327`

## 1. 目标、研究问题与阶段边界

本设计包把 Revision 2 的概念路线落实为可审阅的文件、类型接口、验收矩阵和资源计划。首轮回答两个问题：

1. 固定 credential verifier 与唯一授权协调器能否作为模型内 Gate，以确定性硬判定控制冻结预训练 decoder-only 模型的真实 suffix 执行？
2. 对合法 credential，重新切分和插入 Gate 后的 protected 路径能否保持原模型输出；对拒绝请求，protected suffix 是否逐样本零调用？

G0 交付与具体宿主无关的认证、授权和调度契约；P0 只做候选宿主、依赖、任务效用和目标后端预检；P1 在 G0/P0 通过后做无训练插入。P2 的 public readout 训练不在本实现包内。

本轮不主张 credential 不可伪造、模型知识机密性、白盒抗性、生产访问控制或相对外部 verifier 的安全优势。历史 toy/static credential 尚未完成 replay 防护验收；后续认证协议必须纳入状态化新鲜性与防重放设计。P1 的证据类型是 `TM-NA` 插入正确性和可信入口下的 `TM-API` 路由行为。

TM-API 假设服务进程、权重和运行时可信，外部只能提交原业务输入和 credential，不能提交 route、policy 或 cache 对象。私有 seal 是受信组件之间的来源检查；不导出 Python 构造器、不可变字段和对象身份检查均不构成 TM-WB 防御。

## 2. 已有代码核查与复用边界

| 现有组件 | 可复用内容 | 不能直接沿用的内容 |
|---|---|---|
| `layers/gate_layer.py` | `VerificationEvidence`、输入拒绝 reason code、toy relation 的 reference 兼容性 | `AuthorizationCoordinator` 随 `module.training` 切换 soft/hard；`FeatureGate` 会缩放合法 hidden；只接受 4D feature |
| `transformer/model.py` | prefix/suffix、mixed routing、逐请求一次判定和 KV-cache 测试经验 | 自研 byte-level Transformer；旧 head；合法 hidden 可能被 sigmoid 缩放；不能作为预训练宿主 adapter |
| `service/` | 外部 envelope 不暴露 evidence、距离、reason code 和 indices 的边界 | CIFAR 固定 10 槽概率 schema；不能复用为语言模型 wire schema |
| `t2_*` | manifest/hash、输出覆盖保护、diagnostic 留痕和 split 状态机的通用模式 | byte tokenizer、四元组 trainer、从零训练预算和旧 freeze |

因此 G0 新增独立的 hard authorization 组件，不修改旧 `GateLayer` 的历史语义。P1 新增预训练宿主 adapter，不把 Hugging Face 模型塞入现有 `GatedDecoderTransformer`。

## 3. 设计结构与数据流

```text
input_ids / attention_mask
          |
          v
冻结 embedding + blocks[0:cut]
          |
    prefix hidden ──────────────────────────────┐
          |（执行时序；不把 hidden 输入 verifier） |
          v                                    |
模型内 Gate ← 规范化 credential                 |
  FixedRelationVerifier → Evidence             |
                      → RouteCoordinator       |
                      → _CommittedRoute        |
                               |               |
                         Dispatcher ←──────────┘
                    /             |             \
             PROTECTED          PUBLIC          DENY
          原 hidden 恒等传递  P2 才接 readout   两业务分支零调用
                 |
          blocks[cut:N] + 原 norm/lm_head
```

credential 使用独立结构化参数，不进入 prompt token。verifier 不读取 token、hidden、logits 或业务标签。协调器是唯一把 evidence 提交为 route 的组件；请求 payload 不能选择 policy、构造 `CommittedRoute` 或把 DENY/PUBLIC 升级为 PROTECTED。

P1 只启用 `p1-protected-or-deny-v1`：接受为 PROTECTED，规范但不满足关系和格式错误都为 DENY。P2 后才增加独立 policy `p2-capability-routing-v1`，将规范关系失败映射为 PUBLIC。

上图描述 G 的首次 prefill：服务端建立请求身份 → 无副作用输入预检/规范解析 → 使用新建临时状态执行 prefix → 模型内 verifier/协调器 → route 检查 → dispatcher。request ID 是请求身份，不授予权限，可以在 Gate 授权前生成。合法路径绑定 prefix/suffix 缓存与已提交 route；DENY 释放 prefix 临时状态。外部 cache 不可作为首次调用输入。结构错误可以在 prefix 前拒绝；规范关系失败在中间 Gate 判定。执行过 prefix 的拒绝必须计入 prefix 成本；“授权前拒绝”约束的是受保护分支零调用，不要求 prefix 一律零调用。

后续增量 decode 的顺序为：核查受信请求仍活动、已提交 route 及全部待用 cache 的绑定/形状/位置 → prefix → 中间 route 检查 → dispatcher/suffix。请求身份和 route 取自首次调用建立的受信状态，不能以 cache 自述字段互相校验。缓存预检必须在本步任何 prefix/suffix 执行和 cache 读取用于计算或更新前完成；每次 suffix 调用前仍核对受信请求状态。这里复用首次授权，不重新验签或提交 route，也不把首次 Gate 移到 prefix 前。无 KV 模式同样检查请求状态，但不虚构 cache 检查。

## 4. 模块、文件和公开接口

### 4.1 计划新增文件

| 文件 | 责任 |
|---|---|
| `src/can/v2/pretrained_gate/types.py` | 不可变 evidence、route、policy、调用计数和 adapter 输出类型 |
| `src/can/v2/pretrained_gate/authorization.py` | 规范解析、固定 hard verifier、协调器和 policy 绑定 |
| `src/can/v2/pretrained_gate/host.py` | 宿主能力描述、block/norm/head 定位、支持性 fail-fast 检查 |
| `src/can/v2/pretrained_gate/adapter.py` | 冻结 prefix/suffix adapter、逐样本 dispatcher 与异常原子性 |
| `src/can/v2/pretrained_gate/cache.py` | 请求级 route/cache 绑定及逐样本 cache 重排 |
| `src/can/v2/pretrained_gate/instrumentation.py` | prefix/suffix/head 调用数、时延和显存观测；仅测试接口 |
| `src/can/v2/pretrained_gate/manifest.py` | 严格 schema、文件/hash、模型 revision、后端和 protocol provenance |
| `src/can/v2/pretrained_gate/__init__.py` | 审核后的最小公开 API |
| `scripts/preflight_pretrained_host.py` | P0 只读预检和基线推理 CLI |
| `scripts/eval_pretrained_gate_p1.py` | P1 H/S/G/E 四系统比较 CLI |
| `tests/v2/test_pretrained_gate_authorization.py` | G0 解析、hard 判定、策略和异常测试 |
| `tests/v2/test_pretrained_gate_adapter.py` | 恒等切分、mixed 路由、zero-call 和异常测试 |
| `tests/v2/test_pretrained_gate_cache.py` | 无 KV/KV、不同长度、停止位置和 cache 绑定测试 |
| `tests/v2/test_pretrained_gate_manifest.py` | P0/P1 schema、hash、未知字段和覆盖保护测试 |
| `tests/v2/test_pretrained_gate_cli.py` | 离线 tiny fixture 的 CLI 契约测试 |

先在 CPU 实现前核查候选 config 与固定版本库的 API，再确定首个 host plugin；若为 Qwen2 系列，计划新增 `hosts/qwen2.py`。P0 GPU 预检进一步确认后端行为。通用 adapter 不散布模型名判断，也不承诺一个接口立即支持所有 decoder-only 模型；实际 API 若推翻契约，则修订接口和相关测试后复审，不能强行适配。旧 CIFAR、T1/T2 入口保持其历史协议。

### 4.2 核心类型契约

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple
from torch import Tensor

class RouteKind(str, Enum):
    """表示协调器提交的业务路径。"""
    PROTECTED = "protected"
    PUBLIC = "public"
    DENY = "deny"

@dataclass(frozen=True)
class _CommittedRoute:
    """保存进程内已提交路由；不提供白盒防伪能力。"""

    routes: Tuple[RouteKind, ...]  # 不存可被原地改写的授权 Tensor
    policy_id: str
    request_ids: Tuple[str, ...]   # 服务端生成的样本身份，顺序固定
    execution_config_id: str       # 绑定模型、cut、后端和 verifier profile
    _coordinator_seal: object = field(repr=False, compare=False)

@dataclass(frozen=True)
class HostSpec:
    """记录经核查的宿主结构与执行配置。"""

    model_type: str
    model_revision: str
    num_hidden_layers: int
    cut_layer: int
    tied_embeddings: bool
    supports_kv_cache: bool
    attention_backend: str
    dtype: str

@dataclass(frozen=True)
class RoutedGeneration:
    """仅供受信 evaluator 使用的生成结果与计数。"""

    token_ids: Tuple[Tuple[int, ...], ...]
    stop_reasons: Tuple[str, ...]
    routes: Tuple[RouteKind, ...]
    route_call_count: Tensor
    protected_call_count: Tensor
```

`FixedRelationVerifier.forward(canonical_credential)` 只返回 evidence；`RouteCoordinator.commit(evidence, trusted_context)` 返回 package-private 的 `_CommittedRoute`。协调器在初始化时创建唯一 `object()` seal，可信配置绑定 policy 和 execution config。dispatcher 执行前检查 seal 的对象身份、配置身份、完整且有序的 request IDs、batch 大小和 route 枚举；只比 request IDs 长度不够。其他协调器的 route、旧请求 route、换序或手工构造的 route 均拒绝。

`frozen=True` 不冻结 Tensor 内容，因此授权值改用 enum tuple；执行用索引 Tensor 由 tuple 派生，不与外部共享可变引用。evidence 中 Tensor 也须检查 dtype、shape、device、原因码与 verified 的一致性，commit 后不得通过修改 evidence 改变既定 route。seal 不序列化，不写 checkpoint 或 manifest；恢复进程须重新提交授权。测试覆盖跨协调器、旧请求、顺序变化、修改源 evidence 和未知 route；这些检查不保护攻击者可控制的 Python 运行时。

adapter 提供受信测试方法 `direct_reference_logits()`，但不导出到服务入口。P1 受信进程内入口仅接收原模型输入和 raw credential；`RoutedGeneration`、raw token/logits、计数、cache 和 evidence 仅供内部 evaluator 使用，不能直接作为外部 wire schema。

### 4.3 Credential 规范域与失败范围

首轮兼容性 profile 为 `toy-real-fp32-v1`，不新增整数/mod-q 关系。受信 Tensor 入口仅接受 `torch.float32 Tensor[B,n]`；单个 credential 必须显式写成 `[1,n]` 且仅用于 B=1，不隐式广播，不把整数/FP64/FP16 自动转换后当作原输入。任何 JSON/wire adapter 后续需另定编码，本轮不把 Python 任意对象反序列化为 credential。

- 类型、rank、维度或 B 对不齐、空 batch 属请求级格式错误，整次调用失败；无合法 batch 时不虚构逐行结果。
- 形状合法时逐行检查有限值；NaN/Inf 行为格式拒绝，其他行独立验证。解析器排除这些行，不能让异常值进入矩阵运算。
- 规范有限行执行冻结关系 `||Ac-b|| < threshold`；阈值等号为关系失败。是否接受由关系决定，不预先假定某个有限数值（如 9999）必定失败。
- 有限输入导致中间运算溢出/非有限结果时，该行数值拒绝，不能标为接受或在未来 P2 映射 PUBLIC。verifier 意外异常则按 §7.4 整批失败。

未来 G1/I1 使用独立整数/编码 profile，不为兼容本 profile 放宽类型或范围。

### 4.4 宿主边界与测试桩

host plugin 的逻辑输入至少包括 input IDs、有效位 mask、位置/已处理 token 状态、cache 与 cut；prefix 的输出必须包含 hidden 和 suffix 继续计算所需的宿主状态，不能仅返回一个 Tensor。公共边界使用 BoolTensor[B,T] 有效位 mask，plugin 按固定库版本生成宿主要求的 causal/additive mask、position IDs、RoPE 和 cache position；不能假定 Qwen 内部 attention 直接使用该 Bool mask。

G0 使用固定 seed 的合法随机 token IDs 和 tiny synthetic host，覆盖无 KV、增量 KV、padding、重排与状态传递；这些 fixture 长期保留。P0/P1 另加真实 tokenizer、chat template 和实际宿主测试。首宿主 schema 精确列出支持的字段；未知后端字段默认拒绝，接口扩展显式升级 schema。tied embeddings 同时检查 config 与加载后的参数/存储共享关系，不能仅相信配置标志。

## 5. G0 验收指标与测试矩阵

| 类别 | 用例 | 硬性判据 |
|---|---|---|
| 规范输入 | 单个/批量合法、规范关系失败 | 与冻结 reference verifier 逐行一致 |
| 格式错误 | 错误 dtype/rank/长度、空 batch、NaN/Inf、未知字段 | 按 §4.3 区分整批结构错误与逐行拒绝；拒绝行 suffix/public 零调用；不暴露内部异常 |
| 模式独立 | verifier/coordinator 在父模型 `train()`/`eval()` 下运行 | route 完全一致且始终为 hard 值；无 soft gate |
| hidden 独立 | 固定 credential，随机改变 token/hidden | evidence 和 route 逐位相同 |
| 恒等通过 | 合法 route 的 hidden | 数值、dtype、device 保持；允许按索引复制/select 或按宿主要求 contiguous，不做幅度缩放或精度转换 |
| 策略约束 | payload 尝试指定 policy/route | 严格拒绝；只有可信配置可绑定 policy ID |
| batch 路由 | 全 valid、全 invalid、mixed、尾批、空子批 fixture | indices 互斥且并集覆盖；原顺序可逆恢复 |
| 异常 | parser/verifier/coordinator/dispatcher 注入异常 | fail closed；授权后执行异常不 fallback、不返回 partial output |
| 精度 | verifier 在 CPU FP32、目标 CUDA 配置 | reference 一致；autocast/TF32 不改变认证算术配置 |
| 梯度 | 合法 protected 业务 backward fixture | Gate 无参数；hidden 梯度等于恒等切分 reference；拒绝行 suffix 无图和零调用 |

G0 单元测试至少覆盖以上 10 类，新增安全核心模块的语句覆盖率目标为 `>=95%`，分支覆盖率目标为 `>=90%`；CLI/宿主胶水目标为 `>=85%`。覆盖率若因本机工具故障无法测量，必须像 T2 一样记录工具失败并用可审查替代，不得声称通过。

## 6. P0 宿主与后端预检

### 6.1 候选与选择纪律

首候选为约 0.5B 的可修改 decoder-only 指令模型；`Qwen2.5-0.5B-Instruct` 只是待核验候选。P0 在联网下载或 GPU 执行前冻结候选顺序，至少记录：

- 官方模型 ID、不可变 revision/commit、许可证和来源 URL；
- config、generation config、tokenizer 和 chat template 的 SHA-256；
- 权重文件列表、每文件 SHA-256、总大小和 safetensors 元数据；
- `transformers`、PyTorch、CUDA、driver、GPU、attention backend 和 dtype；
- embedding/head 是否 tied，block 容器、final norm、lm head、RoPE、mask、position/cache API；
- 是否能在离线模式从已固定 snapshot 重建；未知 remote code 默认拒绝。

依赖必须通过版本化环境文件新增，不能只记录一次 `pip install` 命令。任何 `trust_remote_code=True` 需求都暂停候选，单独审阅其代码与 revision。

### 6.2 P0 基线任务与输出

P0 只使用公开、非 test 的 smoke/dev 输入，验证模型基本指令跟随、短事实抽取和两步组合回答。该阶段不声称能力分级。每个候选输出：

- 原模型 greedy reference 的 prompt token IDs、生成 token IDs、停止原因和规范化文本指标；
- batch=1 和一个小 batch 的 TTFT、每 token latency、吞吐、峰值显存；
- `use_cache=False/True`、目标 dtype 和 attention backend 的可运行性；
- 约 1/2、2/3、3/4 深度的合法 block cut 候选，去重后的实际整数层号；
- P1 所用输入集合、batch 形状、长度分布和预期运行次数。

P0 使用固定的 24 条公开、项目合成、答案在 context 中可得的 `p0-host-capability-v1` fixture：8 条格式遵循/短复制、8 条单跳事实抽取、8 条两步组合。通过条件为：全部生成均有限且可安全解码、固定配置重复生成 token IDs 一致；格式组严格 EM 至少 7/8，单跳 normalized EM 至少 7/8，两步 normalized EM 至少 5/8。三项合取已经蕴含至少 19/24 正确，不再设冗余总体门槛；总体报告名为 `passed_examples / 24`，同时保留分组指标。

严格 EM 对解码后的 continuation 与目标字符串逐字符比较，不去空白、标点或改大小写；EOS 不属于正文，但截断/非法生成计错。normalized EM 定义为 `int(N(prediction) == N(target))`，仍是逐样本 0/1；首轮 N 固定为 Unicode NFKC、casefold、合并连续空白及去除两端空白，不删除标点/冠词或做答案子串匹配。token F1/edit similarity 如记录，只作辅助观测，不替代门槛。

fixture 采用每类固定 8 个模板、固定 seed `20260903` 填入非敏感合成名称/属性，答案由确定性规则计算；保存 generator 版本、seed、完整文本/答案/分组、解码参数和 SHA-256。模板和生成文件须在首次候选输出前完成审核；不能按模型回答挑选样本、修改 chat 指令或阈值。实际模板/文件尚未实现，未生成/核验时 P0 CLI 必须拒绝开始评估。24 条只承担工程选型，不支持正式泛化结论。

首轮候选清单仅包含待核验的 `Qwen2.5-0.5B-Instruct`（上限 1 个，具体 revision 在 P0 前锁定），不要求凑齐两个模型。候选不适合时记录 `no_suitable_host`，该轮不进入 P1，但 G0 独立验收和 G1 研究可以继续。新增候选需另行预登记清单版本并保留先前失败结果；不自动换模型、重训或增加预算。

### 6.3 P0 CLI 契约

```text
python scripts/preflight_pretrained_host.py
  --model-snapshot <local immutable snapshot>
  --model-revision <full commit>
  --input-manifest <P0 JSON>
  --output <new directory>
  --device cuda
  --dtype <audited dtype>
  --attention-backend <audited backend>
```

输出目录默认必须不存在；仅显式 `--force-overwrite` 才允许覆盖开发输出，正式 provenance 目录永不覆盖。缺少 revision、hash 不匹配、未知 schema 字段、模型结构不支持、remote code 未审核或环境不一致均 fail fast。

## 7. P1 差分、路由和缓存验收矩阵

### 7.1 四系统实现与依赖

| 系统 | 固定执行方式 |
|---|---|
| H | 原宿主标准 forward；保存独立参考，保留原 norm/head、位置和生成语义 |
| S | 独立 IdentitySplitAdapter，直接串联同一个 host plugin 的 prefix/suffix/norm/head；没有认证和条件 dispatcher |
| G | prefix 后执行模型内 verifier/协调器；通过来源检查后，用受信 dispatcher 运行合法子批 suffix |
| E | 在模型 forward 外的同进程入口执行同一 verifier/协调器，再进入与 G 相同的 host plugin/dispatcher；未授权行可提前跳过 prefix |

G/E 的 verifier 算法、参数、dtype/device、解析器、policy、route 来源检查和 dispatcher 完全相同；不把标准数字签名替换同时混入位置对照。S/G 共用底层模块调用封装，但 S 无条件串联，不通过对外 `verify=False` 模式构建。H/S 对照负责发现共享 plugin 的适配错误。

E 的提前拒绝节省 prefix 是部署位置的真实差异。分别报告 prefix、suffix、head 调用及完整端到端成本，G/E 只要求 route、合法业务输出、suffix/head 调用一致，不要求 DENY 的 prefix 调用一致，也不预设 E 一定更快。若另加“两者均运行 prefix”的成本消融，使用独立配置 ID，不混入主位置对照。

先完成 §7.5 的原宿主校准及 H/S 验收，才解释 S/G 的合法差分；G/E 在各自合法路径与拒绝规则通过后才形成可比较性能结果。DENY zero-call、格式和 cache 负向测试独立尽早运行，不等待效用或延迟对照。失败日志均保留为诊断，不能晋升为通过证据。

### 7.2 差分与执行覆盖

| 配置维度 | 覆盖值 |
|---|---|
| route | 全 PROTECTED、全 DENY、mixed valid/invalid、mixed 格式错误 |
| batch | 1、固定小 batch、非整齐尾批；空子批仅作边界 fixture |
| generation | teacher-forced logits、greedy 1 token、greedy 多 token、不同 EOS 位置 |
| cache | `use_cache=False`、`use_cache=True`；每样本 cache 长度和层数核对 |
| length | 相同长度、右 padding 不同长度、接近 context 上限 |
| cut | P0 冻结的 2–3 个完整 block 边界 |
| backend | P0 冻结的 dtype/attention backend；CPU tiny fixture 另报 |
| error | parser 错误、授权前错误、suffix 中途错误、OOM fixture/受控异常 |

逐配置验收：

1. `H` 与 `S`：相同前缀的 logits allclose，记录最大绝对/相对误差；greedy token mismatch 为 0，长度和停止位置一致。
2. `S` 与合法 `G`：同样满足 logits/token 判据，合法 hidden 恒等进入原 suffix。
3. `G` 与 `E`：合法业务输出、route 和 suffix/head 计数一致；prefix 差异按 §7.1 单独报告。
4. DENY 行的每个 protected block、final norm 和 lm head 调用数为 0；mixed batch 中只允许 valid 子批调用，逐样本计数与 indices 对齐。
5. 正常规范输入每条序列提交一次 route，包含规范关系失败的 DENY；请求结构错误在提交前退出，次数为 0。逐行格式拒绝由协调器一次性提交 DENY；生成后续 token 不重新验签。提交前异常不伪造成功次数。
6. 授权前拒绝不返回模型生成内容；suffix 执行后异常返回统一错误，不切换 PUBLIC/DENY 伪装为零调用。

计数仪器同时记录模块实际被调用和参与该次调用的原始 request IDs；不能仅由 `route` 推算计数，也不能只凭子批大小等于 valid 数量认定样本正确。对合法子批执行顺序置换、插入/移除 DENY 行、不同 EOS 时点进行逐样本 reference 比对。注入“同样大小但选错行”的故障必须使测试失败。forward hook 的一次调用只适用于单步相应层；生成计数按 prefill/decode 步骤、活动请求和层号分别累计。

### 7.3 KV-cache 绑定与负向协议

首轮 cache 仅由一次内部 generate 生命周期持有，外部不能传入/取回原始 cache。每个样本状态绑定服务端生成的 request ID、execution config、模型权重 revision/hash、policy、route、cut、原始样本身份、实际层号，以及已处理有效 token 数、物理 K/V 长度、有效位 mask、position/cache position。元数据与张量实际 shape 必须一致；内部重排改变局部索引，不改变原始身份。

常规增量步以“已处理前缀 + 本次新 token 块”定义位置，不使用 `cache.length == input_ids.shape[1]-1`。首轮 greedy decode 每次输入一个新 token；prefill 可多 token。padding 使物理长度与有效 token 数可能不同，两者都记录。若生成的 EOS/最后一个 token 未再送入模型，则它不增加已处理 cache 长度；输出长度不等于 cache 长度。

| 负向或等价测试 | 预期 |
|---|---|
| A 的 cache 交给 B（相同 prompt 和不同历史各一例） | 显式拒绝，不静默清空/复用；使用全新 cache 的独立 B 才与 B 无 KV reference 对比 |
| 修改模型版本、policy、cut 或绑定 route | 在复用任何 cache、执行新一步业务层前拒绝；不转 PUBLIC，不重试 |
| 伪造 seq_len、层号、K/V shape、mask 或 cache position | 拒绝；覆盖少一层、少/多一个 token、越界位置 |
| mixed 输入 A-valid / B-DENY / C-valid | suffix 仅含 A/C；B 不建立 public/suffix cache，已计算 prefix 临时状态释放；返回拒绝占位，不返回 cache |
| A/C 子批互换顺序，C 提前 EOS，下一步仅剩 A | 独立 reference 的 token/logits/停止位置一致；层计数和 cache 元数据始终对应真实身份 |
| 试图继续已结束/失败请求或重复使用旧状态 | 拒绝；不能靠相同外部 request 字符串获得旧 route |

P2 的 PUBLIC cache 测试留到独立 policy 实现后；P1 不把 DENY 当 PUBLIC。边界测试可用受信内部注入接口构造错误状态，但此接口不能暴露给服务调用方。

缓存绑定失败记为 `cache_state_validation_failed`，阶段为 `decode_preflight`；这是可能发生在首次授权之后的增量步执行前状态校验失败，不是新的 credential DENY。按 §7.4 终止整批生成、清理状态，不返回部分输出。失败步的 prefix/suffix/head 调用数必须为 0，此前 prefill/decode 的真实调用累计保留。测试须在成功 prefill 后注入错误 cache，分别断言本步零调用与历史累计不被清零；同批全部活动行的预检完成后才开始本步计算。

### 7.4 异常原子性

首轮以一次 batch 调用为结果提交单元，采用非流式输出；正常的逐行 DENY 是业务结果，不是运行异常。

- 请求级结构错误在 prefix 前整批失败。可定位的逐行格式错误按 §4.3 处理，其余合法行可继续。
- parser/verifier/coordinator 的意外异常使整批失败；协调器失败不属于“授权已提交”，不得继续 dispatcher。prefix 如已执行仍据实计数。
- 增量步请求/cache 状态校验失败按 §7.3 在本步计算前整批终止；不因首次授权曾成功而继续执行，也不将历史已执行步骤伪记为零调用。
- suffix/head/cache 更新中途异常（包括受控 OOM fixture）使整批运行失败，不返回先前完成行的部分生成结果；清理本次全部请求/cache，不自动重试或切换路线。
- 记录失败发生阶段和实际完成调用数，不能声称异常回滚了计算；错误对外脱敏，内部 trace 保留定位信息。

这会使一个批次的故障影响同批其他请求，是首轮明确的可用性代价。独立逐请求事务、部分成功或流式输出需要另行定义提交边界；不由实现者自行变更。

### 7.5 独立数值标准与冻结顺序

取消 v1 中从 H/S 差值乘 10 派生门槛的规则，避免将切分错误吸收到容差。首轮采用以下**待审核的固定工程容差**，不是 dtype 的理论保证；每个配置在运行前写入 manifest，不按 H/S/G/E 观测自适应增大：

| 业务 logits dtype | atol | rtol |
|---|---|---|
| FP32 | 1e-5 | 1e-4 |
| FP16 | 1e-3 | 1e-3 |
| BF16 | 1e-2 | 1e-2 |

阈值来源登记为 `proposed_engineering_acceptance_v1`：这是本设计提出、尚未经目标宿主验证的工程验收标准，当前没有对应的文献推导或目标宿主实测依据，不宣称为 PyTorch 官方推荐值、旧实验验证值或已证实的误差余量。矩阵乘法误差取决于数值尺度、维度、累加精度和后端，多层误差不能用“单次 1e-6、N 层 N×1e-6”推导。审核接受这些门槛只确定本轮可接受的偏差，不证明候选必能达到；下面的独立 H 校准检验配置可用性，不据其输出反推阈值。

只有实际预登记的 dtype/backend 才在验收范围内；列出阈值不等于承诺三个后端都可用。验证关系判决要求逐行完全相同，不使用业务 logits 容差。

1. 在任何候选输出前固定独立 calibration 输入与 P1 evaluation 输入及摘要，两者 sample/source 分离；冻结 eval、seed、TF32/autocast/确定性设置、attention 实现、padding、生成长度与上表门槛。
2. P0 calibration 只运行 H，不运行 S/G/E。在固定输入与形状上重复 5 次，覆盖拟声明的 batch/长度/有无 KV；另比较 H 的单样本与重排子批行为。重复运行及相同语义样本的 logits 必须在固定门槛内，greedy token IDs/停止必须一致。输出逐位相等率作为附加观测，不把 allclose 偷换为逐位相等保证。
3. 校准通过才锁定执行配置和 tolerance manifest/hash，随后在独立 P1 输入上运行 H/S、S/G、G/E。H/S 本身也必须满足同一固定门槛，不能用 H/S 的误差反推自己的合格标准。G 的合法子批与 H/S 原 batch 及独立逐样本 reference 均对齐比较。
4. 任何受检配置存在非有限 logits、token mismatch 或超限，记录失败；重复 5 次仅提供有限重复性证据。后端变更建立新 execution config 并重跑校准/P1；修改阈值需独立方案修订、保留原失败记录与新的评估输入，不能原地调大。

比较固定为逐元素 `abs(candidate-reference) <= atol + rtol*abs(reference)`，全部被测有效位置通过才算 allclose；reference 位于等式右侧，禁止 99% 覆盖替代全量检查。max_abs 与描述性 max_rel 用 FP64 计算，后者为 `max(abs(diff)/max(abs(reference),1e-8))`，epsilon 写入结果；不使用 max_rel 推导阈值。teacher-forced 比较相同 token 前缀，padding 位置按同一 mask 排除。greedy token、长度和停止原因始终要求完全一致。

H 校准失败时记录 `host_calibration_failed`，并分别记录 `comparison_kind`（`fixed_shape_repeat`、`batch_reorder`、`kv_vs_no_kv`）及非空 `failure_reasons`（`nonfinite_logits`、`logits_tolerance_exceeded`、`token_mismatch`、`stop_mismatch`）；一项比较可包含多种原因。KV/无 KV 比较使用同一 token 前缀和对齐位置。失败阻止该 execution config 进入 P1，不直接断言整个模型不稳定或所有后端均不适合。可以排查并以新 execution config 更换后端、精度或确定性设置，保留原失败并重跑校准；不自动修改门槛或预算。

“接近阈值”不单独判失败，绝对差超过 atol 也不必然失败，须按上述 atol/rtol 联合判据逐元素判断。H/S、S/G 和 G/E 在同一配置下使用相同门槛；单独的 `contiguous()` 复制不改变元素值，但后续布局相关内核选择可能改变数值路径。超限时先排查实现与后端，不能预先断言所有差异都是代码 bug，也不能以“合法实现”为由豁免。确需调整阈值仍按上文独立方案修订规则处理。

### 7.6 性能测量

G/E 分别预热 5 次，随后每配置测量 20 个配对轮次，其中 10 个 GE、10 个 EG，按固定 seed `20260903` 打乱顺序并保存执行表。每个调用创建新的请求/cache，前后同步 CUDA；同一轮输入、权重、设备状态与后端一致，避免把旧 KV/prefix 复用成速度收益。硬件并发负载或频率异常如实记录。

端到端以主机墙钟计入解析、验证、数据搬运、调度和生成；组件 CUDA 时间另报，不用组件时间和替代端到端时间。合法请求报告 TTFT、后续 token 延迟、吞吐；DENY 报拒绝响应延迟，TTFT 为 null/not_applicable。首轮无真实流式服务，TTFT 为内部首 token 就绪测量，不是网络可见延迟。报告中位数、IQR、配对差值及峰值显存；剥离诊断 hook 的性能运行另作配置，不能沿用其结果证明 zero-call。

## 8. Manifest、输出和 provenance

P0 manifest 至少记录 `schema_version`、`protocol_id`、`execution_config_id`、候选清单版本/上限、模型/权重/tokenizer revision 与摘要、环境、后端、cut 候选、输入 manifest 摘要、生成参数和创建工具版本。P1 结果另外记录 adapter commit、Gate profile、policy ID、credential fixture 摘要、calibration/evaluation/tolerance manifest 摘要、reference 摘要、逐配置判据、调用计数、计时顺序、时延/显存和失败原因。所有未定字段必须在对应运行前补齐，不能以 null 默认跳过校验。

manifest 独立于模型 snapshot 和结果目录保存，其自身 SHA-256 写入工作日志或独立可信文件。禁止从待校验 snapshot 目录自动采用同名摘要作为信任根。raw credential、secret、完整内部 evidence、hidden 和 logits 不写入可提交结果；测试所需 logits 只保存在本地受管诊断目录或汇总为误差统计。

P1 状态优先级为 `invalid_run > failed > partial > passed`：hash/schema/protocol 错误为 invalid_run；任一已执行硬门槛失败为 failed；没有失败但缺少预登记配置结果为 partial；全部预登记配置通过才为 passed。不支持、timeout 或未运行的配置必须列明，不能从分母静默移除。拒绝是正常业务结果；预期故障注入被正确处理可使负向测试通过，不能与意外运行错误混淆。

## 9. 资源测量与停止条件

目标服务器暂按 RTX A4000 16 GB 规划。P0 smoke 首先测原宿主 batch=1、短序列、greedy，无需先实现切分。记录模型加载显存、峰值显存、TTFT、每 token 延迟、吞吐、无 KV/KV 总时长和结果目录大小；cut 从 config 静态确定。

在 P0 实测前不承诺墙钟。实测后冻结：P1 输入数、最大 prompt/new-token 长度、batch 配置、cut 数量、重复次数、单配置 timeout 和总墙钟上限。P1 是确定性差分验证，重复运行用于检测不确定性，不把重复数称作学习 seed。

资源计划表必须枚举实际配置单元，分别计算正确性运行、5 次校准重复、预热和 20 轮配对性能测量，加上加载/结果写入的实测余量；不能将表中维度行数直接当作运行数。首次 P0 smoke 前必须登记其独立 timeout/资源上限；P1 预算缺失时 CLI 拒绝开始。8 小时可作为后续用户选择的成本上限，本方案不把它写成已验证耗时。若预计超预算，在 P1 结果产生前修订范围；缩小覆盖须保留授权与路由、合法等价、DENY/mixed 和 cache 隔离等硬门槛，缺少 KV 等既定范围时只能报告 partial。运行开始后不得删除失败配置或自动延长预算。

停止条件：

- P0：宿主无法离线固定、许可证/remote code 不清、基本任务不达冻结门槛、结构不支持可靠 prefix/suffix 或 A4000 无法运行时停止该候选；清单耗尽报告 no_suitable_host，不自动追加候选。
- G0：reference 不一致、hidden 影响判定、模式改变 route、合法 hidden 非恒等或异常 fail open 时停止，不进入 P1。
- P1：H/S 恒等切分失败时先修 host adapter；S/G 合法差分失败时先修 Gate 接入；DENY suffix 非零调用或 cache/索引串扰为阻塞问题。
- 不以训练、放宽容差、减少失败配置或改测试输入修补 P1；任何协议变更产生新 version 和新结果目录。

## 10. 实施阶段与审核门

1. **方案审核**：Claude 核对本设计的接口授权边界、宿主适配可行性、P1 强对照和验收矩阵。
2. **规则同步**：审核后更新 `AGENTS.md`、`SECURITY.md`、README 和 `RESEARCH_DESIGN.md`；规则改动与实现同一 checkpoint 审核。
3. **G0 CPU 实现**：先核查首候选的固定库版本/API，再实现类型、hard verifier/coordinator、manifest 和离线 tiny-host fixture；运行专项与完整 `tests/v2`。
4. **P0 服务器预检**：下载/固定候选 snapshot，生成环境与模型 manifest，测基本效用和资源；不训练。
5. **P1 adapter 实现**：根据 P0 的真实 API 添加独立 host plugin；先 CPU tiny fixture，再在服务器运行冻结矩阵。
6. **Claude 验收**：检查测试、类型标注、中文 docstring、manifest、输出和实际 GPU 报告；P1 通过后才设计 P2。

方案通过后由用户指定 Claude、Codex 或其他工具实现。当前文件本身不批准下载模型、安装依赖、运行 GPU、创建 freeze 或进入 P2。

P1 通过后固定宿主权重、cut、dtype、backend 和 verifier profile 的 execution config。更换模型或这些参数须新建 config ID 并重跑相关 P0/P1，旧结果保留且不得跨配置继承；授权规则、输入域或测量方法改变时同时升级 protocol ID。P2 效用不足不抹去旧 P1 成功，也不自动授权换模型。

修复实现后的重测依据依赖内容摘要：host plugin、mask/位置/cache 或共享 dispatcher 改动须重跑受影响的 H/S 及全部下游对照；仅 G 专用代码改动且 H/S 的代码、依赖、权重和输入完全不变时，可引用既有 H/S 证据并记录新 commit 的继承理由。zero-call 和 cache 负向回归始终重跑。不能只因文件名没变就复用旧证据。

## 11. 配套规则同步清单

| 文件 | 审核后需要同步的内容 |
|---|---|
| `AGENTS.md` | 将“训练时必须软路由”限定为历史 CIFAR/T0；新增固定认证 hard 判定与合法 hidden 恒等通过；公共读出效用由实验验证，不写成“蒸馏保证” |
| `SECURITY.md` | 修正 Phase 3.6 已实现状态；增加 P1 DENY 与 P2 PUBLIC 策略区分、授权前/后异常、预训练 adapter 仍不抗 TM-WB |
| `README.md` | 更新当前主线、项目结构、依赖管理和 P0/P1 入口；移除过时的 T2 下一步 |
| `docs/RESEARCH_DESIGN.md` | 为 G0/P1 插入正确性、同构 verifier 对照和 P2 效用建立新 claim；旧 C-015 至 C-019 保留历史范围，不提前晋升 |
| `PROJECT_WORKLOG.md` | 每个里程碑登记状态、实测资源、hash、测试和唯一下一步 |

实现时新增的所有公开类/函数遵守项目规则：完整类型标注、中文 docstring，关键授权、索引、cache 和异常路径使用简洁中文注释。依赖文件计划新增为 `requirements-pretrained.txt`，实际版本只能来自 P0 已验证环境，并在提交前固定精确版本；不得先填未经运行的版本号。

## 12. 风险与限制

| 风险 | 处理 |
|---|---|
| Hugging Face 模型内部 API 随版本变化 | 固定 dependency、snapshot revision 和独立 host plugin；未知结构 fail fast |
| tied embedding/head 被公共训练意外更新 | P1 全冻结；P2 前检查参数身份、optimizer 集合和训练后摘要 |
| mixed batch 重排或 KV-cache 串扰 | 逐样本 request/row/cache 绑定、reference 重跑和层级调用计数 |
| 模型图内 verifier 可能无性能优势 | 与外部 verifier + 相同 dispatcher 做强对照，报告实际成本 |
| toy relation 易伪造 | 只作为兼容性 profile；G1/I1 独立研究，不把 P1 晋升为认证安全 |
| checkpoint 与运行时可控时 Gate 可绕过 | 保持 `TM-WB` 不主张；P1 不改变可信计算基础 |
| 公共读出可能仍恢复 protected 能力 | P2/P3 单独测越界、probe 和 recovery；P1 不声称知识隔离 |
| P0 dev 选择污染正式结果 | 候选顺序、阈值和数据身份在 validation/test 前冻结；test 不用于选型 |

## 13. v1.1 复审索引

| 议题 | 本次决定 | 位置 |
|---|---|---|
| seal、检查时机、可变 Tensor | 保留 prefix 后验证；授权 enum tuple、私有 object seal、完整请求/配置绑定 | §1、§3、§4.2 |
| P0 门槛与 EM | 去掉冗余总体门槛；严格格式 EM 与 normalized EM 分开；固定合成来源与候选上限 | §6.2 |
| H/S/G/E | S 独立入口共用 host plugin；G/E 共用 verifier/dispatcher，明确提前拒绝成本 | §7.1、§7.6 |
| cache 和逐样本 zero-call | 绑定错误显式拒绝；DENY 无 public cache；实际身份与层/步计数；区分输出/已处理长度 | §7.2–§7.3 |
| 异常原子性 | 正常逐行拒绝可共存；运行异常整批失败，保留已发生调用，非流式且不自动重试 | §7.4 |
| 数值验收 | 删除 H/S 误差乘10；预登记固定工程门槛，H 独立校准，独立 P1 输入，全元素比较 | §7.5 |
| 宿主与资源 | 静态 API 核查、tiny fixture 保留、参数共享实测、新 config 重跑和运行前预算 | §4.4、§9–§10 |

本修订未接受“把 G 验证移到 prefix 前”“normalized EM 不是 EM”“P1 DENY 建 public cache”“99% 数值覆盖可替代 allclose”或“接口必不返工”的建议。方案仍为待审核设计，表中阈值/次数是预注册提议，不是已经取得的实验结论。

## 14. v1.2 补充复审索引

- §3、§7.3–§7.4：区分首次 prefill 与增量 decode；身份生成不等于授权，缓存预检复用已提交 route；失败步零调用和历史累计分别验收。
- §7.5：明确容差无目标宿主实测/文献依据，保留待审核工程门槛；不采用 N×1e-6 推导，校准按重复性、batch 重排、KV 对照分类报告，失败针对执行配置。
- 本次未修改容差数值，不把“接近阈值”或单独超过 atol 定义为失败；未修改代码或运行校准。
