# Project

- 项目名称：`CAN: Cryptographic Authentication Neural Gate Layer`。
- 研究目标：实现 Gate Layer 在神经网络计算图中间的模型内生安全架构，融合浅层特征与密码 credential 验证信息。
- 项目面向科研复现、审阅和可验证实验，不默认等同于生产系统。
- 根目录 `PROJECT_WORKLOG.md` 是动态事实、当前目标和唯一下一步的来源；不要用旧聊天记录替代它。
- `SECURITY.md` 记录信任模型和明确不保证的性质（当前阶段不解决白盒攻击问题）。
- 本项目只允许防御性实现和验证，不实现攻击性后门、规避检测、未授权访问、凭据窃取或可直接用于攻击第三方系统的功能。

# Session workflow

- 每次开始工作时，先阅读 `AGENTS.md`、根目录 `PROJECT_WORKLOG.md` 以及与当前任务相关的文档。
- 检查 `git status`、`git branch --show-current` 和完整 `HEAD`。如果目录不是 Git 仓库，明确记录，不得虚构分支或 commit。
- 用户改变目标时，先更新 `PROJECT_WORKLOG.md` 的目标、任务状态和唯一下一步，再修改实现。
- 每次工作遵循：检查现状 -> 实现 -> 测试 -> 更新文档 -> checkpoint 总结。
- 代码、配置和可复现的测试结果优先于过时文档；发现差异时记录差异和后续修正文档任务。
- GPU 训练任务需要时在工作日志中明确标注，并记录预期时间和资源需求。

# Engineering rules

- 使用 Python 3.8+、PyTorch 2.0+、标准格式化工具（black, isort）。
- 优先小而可审查的 patch，不重写无关模块，不格式化无关文件，不做与当前目标无关的重构。
- 仅当抽象减少真实复杂度、消除明显重复或匹配已有架构时才新增抽象。
- 结构化数据使用正式解析器或结构化 API，不用脆弱字符串拼接或正则模拟解析。
- 所有公开 API 都有类型标注（typing）和简洁中文 docstring。
- 测试使用显式、确定性的种子（torch.manual_seed, random.seed），并记录影响复现的环境条件。
- 不提交秘密、凭据、私钥、大型 checkpoint、生成输出或大型二进制文件到 Git。
- 严格验证不可信外部输入；未知字段、重复字段、错误长度、非有限数值、非规范编码和类型混淆默认拒绝。
- 默认行为必须 fail closed；任何降级模式都必须显式配置、可审计且默认关闭。

# Core architecture principles

本项目实现"Gate Layer 在计算图中间"的架构：

```python
# 核心数据流
image, credential -> [浅层] -> shallow_features
shallow_features, credential -> [Gate Layer] -> gate_signal
gate_signal -> [条件路由] -> 深层 or 公开head
```

关键约束：
- Gate Layer 必须在计算图中间（不是外部验证器）
- credential 信息必须流经 Gate Layer（不能完全分离）
- 训练时用软路由（可微分），推理时用硬路由（fail-closed）
- 公开能力通过知识蒸馏保证是深层的弱化版

# Authorization and security boundaries

如果项目包含认证或授权，默认流程是：

```text
不可信输入
-> 规范化解析
-> 无副作用的确定性验证
-> 只包含证据、不具有授权能力的结构化结果
-> 唯一协调器提交授权决定
-> 生成已提交的 capability/context
-> 执行受保护副作用
```

- 请求方不能直接提交 `allow`/`deny` 结果，不能创建授权 context 或 capability，也不能选择更弱的验证路线、算法或策略。
- 验证器只产生证据；只有协调器可以提交最终授权。
- replay、tamper、格式错误、权限提升和验证失败必须产生零受保护副作用。
- 拒绝原因、验证 trace 和审计结果必须稳定、结构化、可测试，且不泄露秘密。

# Testing rules

- 每个新增模块都有对应的单元测试（`tests/v2/test_*.py`）。
- 测试覆盖：正向测试（合法输入）、边界值、类型错误、形状不匹配。
- Gate Layer 测试必须验证：
  - valid credential → gate_signal 高
  - invalid credential → gate_signal 低
  - 形状正确性
  - 与参考实现的差分测试（Module-SIS 验证）
- 优先运行与改动直接相关的最小测试，再运行完整测试套件。
- 测试命令：`pytest tests/v2/ -v`（记录到工作日志）。
- 无法运行的测试必须在最终总结和工作日志中说明原因，不得用”应当通过”代替结果。

# Git and worktree safety

- 保留用户已有改动，忽略无关文件，不覆盖、回滚、删除或格式化无关内容。
- 不使用 `git reset --hard`、`git checkout --` 或其他破坏性命令，除非用户明确授权准确目标。
- 每次会话结束前准备 commit-ready checkpoint，展示准确的待提交文件列表。
- 未经明确要求，不推送分支、不创建 PR、不合并分支、不发布版本。
- 多 worktree 项目以主工作树中的 `PROJECT_WORKLOG.md` 为跨分支唯一事实来源。

# Definition of done

1. 请求的行为已经实现。
2. 相关测试通过，新增模块具有单元测试。
3. 代码符合类型标注和 docstring 要求。
4. 工作文档与代码、配置和 Git 状态一致。
5. 实现范围、未实现内容、测试结果和残余风险已经总结。
6. toy、实验性、单机、小样本和非生产限制均有明确标记。
7. 已形成可提交的 checkpoint，并展示准确的待提交文件列表。
