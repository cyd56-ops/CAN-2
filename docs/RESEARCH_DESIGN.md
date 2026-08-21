# Research Design

## 1. Research question

### Primary question

<用一句可证伪的问题描述研究目标。>

### Hypotheses

- H1：<主要假设>
- H2：<对照或替代假设>

### Contribution boundary

<明确本项目相对已有方法新增什么，哪些内容只是工程组合或复现。>

## 2. Scope and terminology

### In scope

- <研究对象>
- <形式化关系、算法或模型>
- <实验和可复现环境>

### Out of scope

- <不研究的能力>
- <不支持的安全主张>

### Terminology

| Term | Definition | Forbidden interpretation |
| --- | --- | --- |
| `<term>` | <精确定义> | <不能声称什么> |

## 3. System boundary

```text
untrusted input
-> canonical parser
-> deterministic reference/verifier
-> evidence-only result
-> single coordinator
-> protected model/tool side effect
```

说明每个边界的可信性、输入输出、失败语义和可观测结果。验证器不直接产生授权；只有协调器提交最终决定。

## 4. Formal obligations

- 输入域和 canonical encoding：<定义>
- reference relation/oracle：<定义>
- completeness：<需要证明或明确不主张>
- soundness-preservation：<需要证明或明确限定>
- 数值误差/量化预算：<定义>
- replay、tamper、route confusion 和 failure side effects：<验收条件>

有限随机测试不能替代全域证明、穷举、形式方法或密码学归约；每项结论必须写明证据类型。

## 5. Experimental protocol

| Item | Frozen decision |
| --- | --- |
| Dataset/input | <来源、摘要、许可和切分> |
| Model/system | <拓扑、版本和参数> |
| Environment | <OS、Python、依赖、硬件> |
| Seeds/repeats | <确定性设置> |
| Metrics | <准确率、等价性、延迟、调用计数等> |
| Artifacts | <生成位置、摘要、忽略规则和保留期限> |
| Stop conditions | <失败或异常时停止条件> |

实验结果必须与代码版本、配置、数据摘要和环境 tuple 绑定，不把未执行的结果写成已验收结论。

## 6. Research stages

| Stage | Objective | Exit criteria | Status |
| --- | --- | --- | --- |
| S0 | 精确定义和最小 oracle/reference | 输入域、关系和负向测试闭合 | pending |
| S1 | 最小可验证实现 | 单元、差分和安全测试通过 | pending |
| S2 | 受控实验/系统组合 | 预注册指标和失败矩阵通过 | pending |
| S3 | 论文证据和限制 | 主张、证据、残余风险一致 | pending |

后续阶段不得覆盖前序路线；跨阶段只复用无协议语义的通用 helper，并保持入口和接受集合隔离。

## 7. Claim and evidence ledger

| Claim ID | Claim | Required evidence | Current status |
| --- | --- | --- | --- |
| C-001 | <主张> | <证明/测试/实验> | pending |

## 8. Related work and paper positioning

<记录检索范围、来源、差异和不能声称的内容。相关工作材料与私有论文不得未经许可提交到仓库。>
