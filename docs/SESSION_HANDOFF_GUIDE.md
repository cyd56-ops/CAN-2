# 项目会话衔接指南

本文档说明如何在新的 Claude 会话（桌面端、网页端或其他客户端）中快速衔接当前项目的工作。

## 快速开始：标准启动流程

在新的 Claude 会话中，发送以下消息即可快速衔接：

```
我想继续 CAN 项目的工作。请：
1. 读取 AGENTS.md 和 PROJECT_WORKLOG.md
2. 检查 git status 和当前分支
3. 告诉我当前进度和下一步任务
```

Claude 会自动：
- 加载项目规则和工作流程（AGENTS.md）
- 了解当前进度和唯一下一步（PROJECT_WORKLOG.md）
- 检查代码状态（git status）
- 总结当前状态并等待你的指令

## 核心文档系统

项目使用以下文档体系来保证跨会话的连续性：

### 1. 项目治理文档（一次性阅读）

| 文档 | 作用 | 更新频率 |
|------|------|---------|
| `AGENTS.md` | 工作流程、协作规则、代码规范 | 低（规则稳定后很少改） |
| `SECURITY.md` | 信任模型、攻击者模型、安全边界 | 低（阶段性更新） |
| `README.md` | 项目概述、快速开始、文档索引 | 中（功能增加时更新） |

### 2. 动态工作文档（每次必读）

| 文档 | 作用 | 更新频率 |
|------|------|---------|
| `PROJECT_WORKLOG.md` | **当前进度、唯一下一步、测试结果** | **高（每次实现后更新）** |
| `docs/DESIGN_PROPOSALS.md` | **所有功能的设计方案** | **高（新功能前添加方案）** |

### 3. 专项决策文档（按需阅读）

| 文档 | 作用 | 何时阅读 |
|------|------|---------|
| `docs/V2_LWE_IMPLEMENTATION.md` | LWE 实现的技术决策 | 修改 LWE 实现时 |
| `docs/VISION_GAP_ANALYSIS_20260821.md` | V2 架构的设计理由 | 了解架构背景时 |

## 项目状态快照（2026-08-23）

### 当前阶段
- **研究阶段**：V2 - Gate Layer 在计算图中间架构
- **当前状态**：Phase 1.1 完成（LWE 密码原语），Phase 1.2 设计方案已就绪
- **唯一下一步**：实现 Neural Gate Layer

### 已完成工作
✅ Phase 1.1: LWE 密码原语
- 代码：`src/can/v2/crypto/lwe.py`
- 测试：38/38 通过，100% 覆盖率
- 提交：commit `1261b18`

### 待办工作
⏳ Phase 1.2: Neural Gate Layer（设计方案已完成）
- 设计方案：`docs/DESIGN_PROPOSALS.md` § Phase 1.2
- 代码：`src/can/v2/layers/gate_layer.py`（待实现）
- 测试：`tests/v2/test_gate_layer.py`（待实现）

📋 Phase 1.3: Gated ResNet-18（待设计）
📋 Phase 2: 训练流程（待设计）
📋 Phase 3: 评估实验（待设计）

### Git 状态
- 分支：`master`
- 最新提交：`1261b18 feat: 实现LWE密码原语 (Phase 1.1)`
- 工作区：clean（无未提交修改）

## 会话衔接检查清单

新会话启动时，Claude 应该完成以下检查：

- [ ] 读取 `AGENTS.md`（工作流程和规范）
- [ ] 读取 `PROJECT_WORKLOG.md`（当前进度和下一步）
- [ ] 读取 `docs/DESIGN_PROPOSALS.md`（当前设计方案）
- [ ] 检查 `git status` 和 `git branch --show-current`
- [ ] 确认工作目录是 `E:\CAN`
- [ ] 总结当前状态：
  - 当前阶段和进度
  - 唯一下一步是什么
  - 是否有未提交的修改
  - 是否有待审阅的设计方案

## 常见会话启动场景

### 场景 1：继续实现工作

**用户**：
```
继续实现 Gate Layer
```

**Claude 应该**：
1. 读取 `docs/DESIGN_PROPOSALS.md` 中的 Phase 1.2 设计方案
2. 询问用户："方案已审阅通过吗？由谁实现（Claude/Codex/其他）？"
3. 如果由 Claude 实现：按方案实现代码
4. 实现完成后：运行测试、更新文档、准备 checkpoint

### 场景 2：审阅设计方案

**用户**：
```
审阅 Gate Layer 的设计方案
```

**Claude 应该**：
1. 读取 `docs/DESIGN_PROPOSALS.md` § Phase 1.2
2. 总结设计方案的关键点
3. 等待用户反馈（批准、修改、讨论）
4. 如果需要修改：直接编辑 `docs/DESIGN_PROPOSALS.md`

### 场景 3：验收实现结果

**用户**：
```
验收 Gate Layer 的实现
```

**Claude 应该**：
1. 读取实现代码（`src/can/v2/layers/gate_layer.py`）
2. 运行测试套件（`pytest tests/v2/test_gate_layer.py -v --cov`）
3. 检查代码规范（类型标注、中文注释、docstring）
4. 对照设计方案验证实现完整性
5. 生成验收报告
6. 更新 `PROJECT_WORKLOG.md`
7. 准备 commit checkpoint

### 场景 4：讨论技术问题

**用户**：
```
LWE 的 threshold 参数应该如何选择？
```

**Claude 应该**：
1. **不立即修改文档或代码**（遵循 AGENTS.md § 问题讨论阶段）
2. 回答技术问题，给出建议
3. 如果用户明确要求修改：询问是否修改相关文档
4. 修改后更新 `docs/DESIGN_PROPOSALS.md` 或 `docs/V2_LWE_IMPLEMENTATION.md`

## 项目记忆系统

项目使用 **文档驱动** 而非 **对话记忆** 来保证连续性：

### ✅ 正确做法
- 所有设计决策写入 `docs/DESIGN_PROPOSALS.md`
- 所有进度和状态写入 `PROJECT_WORKLOG.md`
- 所有技术决策写入专项文档（如 `docs/V2_LWE_IMPLEMENTATION.md`）
- Git commit message 清晰记录每个 checkpoint

### ❌ 错误做法
- ❌ 依赖 Claude 的对话历史记忆（跨会话会丢失）
- ❌ 口头约定但不写入文档
- ❌ "我们之前讨论过的那个方案"（新会话不知道）
- ❌ 实现代码但不更新工作日志

## 跨客户端使用建议

### Claude 桌面端
- **优势**：本地文件访问快，可以直接修改代码
- **适用**：实现阶段、测试验收、文档更新
- **启动**：打开项目目录，发送衔接消息

### Claude 网页端
- **优势**：访问方便，不需要安装
- **适用**：讨论设计方案、审阅代码、技术咨询
- **启动**：需要手动粘贴关键文档内容（AGENTS.md、PROJECT_WORKLOG.md）

### Claude API / SDK
- **优势**：自动化集成，可以编写脚本
- **适用**：CI/CD 集成、批量测试、自动化验收
- **启动**：通过脚本注入项目上下文

## 最佳实践

### 1. 每次会话开始前

```bash
# 1. 确认工作目录
pwd  # 应该是 E:\CAN

# 2. 检查 git 状态
git status
git log -1 --oneline

# 3. 拉取最新代码（如果多设备协作）
git pull
```

### 2. 给 Claude 的标准启动消息

**方式 A（最简洁）**：
```
继续 CAN 项目工作
```

**方式 B（明确任务）**：
```
继续 CAN 项目工作，下一步是实现 Neural Gate Layer
```

**方式 C（需要回顾）**：
```
我想了解 CAN 项目的当前状态，请读取工作日志并总结
```

### 3. 每次实现完成后

- [ ] 运行测试并记录结果
- [ ] 更新 `PROJECT_WORKLOG.md`（标记完成，更新下一步）
- [ ] 更新 `docs/DESIGN_PROPOSALS.md`（标记方案状态）
- [ ] 提交 git checkpoint
- [ ] 告知 Claude："已完成 Phase X.Y，请总结并准备下一阶段"

## 故障排查

### 问题：Claude 不知道项目进度

**原因**：未读取 `PROJECT_WORKLOG.md`

**解决**：
```
请读取 PROJECT_WORKLOG.md 并告诉我当前进度
```

### 问题：Claude 使用了过时的设计方案

**原因**：未读取最新的 `docs/DESIGN_PROPOSALS.md`

**解决**：
```
请重新读取 docs/DESIGN_PROPOSALS.md 中的 Phase 1.2 方案
```

### 问题：Claude 不遵守工作流程（直接修改代码）

**原因**：未读取 `AGENTS.md`

**解决**：
```
请先读取 AGENTS.md，遵循"先方案后实现"的流程
```

### 问题：文档之间不一致

**原因**：多次修改后未同步更新

**解决**：
```
检查文档一致性：AGENTS.md, SECURITY.md, PROJECT_WORKLOG.md, DESIGN_PROPOSALS.md
```

## 记忆系统规则（来自 AGENTS.md）

根据用户的记忆系统偏好（`C:\Users\DELL\.claude\projects\E--CAN\memory\`）：

- **user**: 用户角色、专业水平、偏好
- **feedback**: 用户对工作方式的反馈和纠正
- **project**: 项目目标、约束、进度（不可从代码推断的信息）
- **reference**: 外部资源（URL、论文、文档）

**当前记忆**：
- `workflow-preferences.md`：先方案后实现，实现者由用户选择，Claude 负责验收

**注意**：记忆系统是 **补充**，不能替代项目文档。核心信息仍然在 Git 仓库的文档中。

## 总结

**最重要的三个文档**：
1. `AGENTS.md` - 工作规则（每个新会话必读）
2. `PROJECT_WORKLOG.md` - 当前状态（每次必读）
3. `docs/DESIGN_PROPOSALS.md` - 设计方案（实现前必读）

**最重要的三条原则**：
1. 文档驱动，不依赖对话记忆
2. 先方案后实现，方案必须写入 `DESIGN_PROPOSALS.md`
3. 每次实现后更新 `PROJECT_WORKLOG.md`

**新会话启动模板**：
```
继续 CAN 项目工作。
当前进度：[如果你知道，否则让 Claude 告诉你]
下一步：[如果你知道，否则让 Claude 告诉你]
```

Claude 会自动读取必要文档并衔接工作。
