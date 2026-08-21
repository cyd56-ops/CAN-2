# CAN: Cryptographic Authentication Neural Gate Layer

这是一个防御性科研原型仓库，研究将密码验证逻辑嵌入神经网络计算图中间的 Gate Layer 架构。

## 研究目标

实现模型内生安全：在神经网络中间嵌入基于 Module-SIS 的认证神经元层（Gate Layer），该层融合浅层特征与密码 credential 信息，根据验证结果控制深层神经元的激活。

**核心设想**：
- 浅层神经元：公开可用，无需认证
- Gate Layer：在计算图中间，融合特征与密码验证信息
- 深层神经元：需要通过 Gate Layer 认证才能访问
- 公开 head：认证失败时输出弱化能力（通过知识蒸馏）

## Repository map

- `AGENTS.md`：长期稳定的工作、工程、测试和架构约束
- `PROJECT_WORKLOG.md`：当前动态事实、实现路线、决定、测试结果和唯一下一步
- `SECURITY.md`：威胁模型、信任边界和明确不保证的性质
- `src/can/v2/`：V2 架构实现（Gate Layer 在计算图中间）
- `tests/v2/`：V2 架构对应的单元测试和集成测试
- `scripts/`：工具脚本

## Development workflow

1. 先读 `AGENTS.md`、`PROJECT_WORKLOG.md` 和任务相关文档。
2. 检查 Git 状态和工作日志中的唯一下一步。
3. 做最小实现，添加相称的单元测试。
4. 运行 `pytest tests/v2/ -v`，把结果写入工作日志。
5. 更新文档，形成 commit-ready checkpoint。

## Scope boundary

当前仓库重点验证"Gate Layer 在计算图中间"的架构可行性和技术可行性。**白盒攻击防御不在当前阶段范围内**。

不要把 toy、实验性、单机或小样本结果描述为生产安全保证或密码学安全归约。
