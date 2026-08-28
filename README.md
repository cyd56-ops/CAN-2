# CAN - Capability Authentication Network

CAN 是一个防御性科研原型，研究将**固定的 toy LWE-inspired 关系验证门**嵌入神经网络中间，
并根据 credential 判定控制模型能力路由。

> 当前实现不是数字签名、身份认证或生产密码学访问控制系统。它不提供密码学安全归约、
> replay 防御或白盒运行时抗性。

## 项目目标

在可信服务部署边界内验证以下能力分级架构：

- valid credential：执行受保护的 10 类细粒度路径；
- invalid credential：不执行 protected 深层路径，只执行 2 类 public 粗粒度路径；
- 推理态采用硬路由，invalid 样本的 `layer3`、`layer4` 和 protected head 不被调用。

核心关系为 `b = A s + e` 的 toy 数值构造，Gate 判定为：

```text
||b - A credential|| < threshold
```

该关系使用浮点运算和固定参数，仅用于验证神经编译、路由隔离和能力分级的实验可行性。

## 当前状态

- Phase 1：toy LWE-inspired 原语、Gate Layer 和 Gated ResNet-18 已完成；
- Phase 2：CIFAR-10 三阶段训练已在 RTX A4000 上完成，3 个 seed 均生成 Stage A/B/C checkpoint；
- Phase 3：test evaluator、单 checkpoint CLI、manifest/SHA-256 校验、Stage C 三 seed 聚合、
  latency 测量和离线测试已完成；
- 当前唯一下一步：审阅并冻结 Phase 4 CIFAR-100 能力分级扩展方案；
- Phase 3.6：可信进程内 response envelope 已实现并通过 Claude 验收；
- Phase 4 CIFAR-100 与 Phase 5 ImageNet 尚未开始。

权威动态状态见 [`PROJECT_WORKLOG.md`](PROJECT_WORKLOG.md)，主张与证据台账见
[`docs/RESEARCH_DESIGN.md`](docs/RESEARCH_DESIGN.md) 第 7 节。

## 安全模型

### TM-API

调用方可无限次提交任意 `image` 和 `credential`，但不能读取权重、修改运行时或直接调用内部模块。
正向保证的边界是服务层 response envelope，不是原始 PyTorch `InferenceOutput`。

当前可验证的模型层性质包括：

- 模型判定与 NumPy `V_ref` 逐样本一致；
- invalid credential 的 protected 深层零调用；
- valid/invalid 路径具有预期的 10 类/2 类能力差异。

Phase 3.6 的可信进程内 response envelope 已实现；外部调用路径只返回固定长度 probabilities、
prediction 和 capability level。原始 `InferenceOutput` 中的 `decision`、`error_norm`、
`reason_code`、`gate_signal` 和路由索引只允许 evaluator 使用，不构成外部 API。

### TM-WB

若攻击者持有 checkpoint 与运行时，则可以直接调用 protected 内部路径或篡改运行时控制流。
当前实现对此不提供抗性，也不提供针对第三方模型的可迁移绕过 PoC。

### 明确限制

- toy 参数无密码学安全归约，可被最小二乘伪造；
- 静态 credential 可重用，不防御 replay；
- 不提供签名不可伪造性、身份认证或 access-control soundness；
- 不提供 TEE、安全启动、侧信道或跨设备安全保证；
- FAR/FRR 是有限采样下的实现正确性观测，不是密码学安全指标；
- 独立无 Gate 同构 baseline 尚不存在，属于未来 `no_gate_ablation` 消融。

## 项目结构

```text
E:/CAN/
├── src/can/v2/
│   ├── crypto/lwe.py                  # toy LWE-inspired 关系原语
│   ├── layers/gate_layer.py           # 验证、协调和特征门控
│   ├── models/gated_resnet.py         # CIFAR-10 双能力路径模型
│   ├── service/                        # Phase 3.6 可信进程内响应适配层
│   └── experiments/test_evaluator.py  # Phase 3 模型层 evaluator
├── scripts/eval_cifar10_test.py       # test split 评估与 Stage C 聚合 CLI
├── tests/v2/                          # 单元、差分和 evaluator 测试
├── configs/v2/                        # 训练配置
├── docs/DESIGN_PROPOSALS.md           # 设计方案与实验协议
├── docs/RESEARCH_DESIGN.md            # 研究问题和 claim/evidence 台账
├── PROJECT_WORKLOG.md                 # 唯一动态工作日志
└── SECURITY.md                        # 信任模型与安全边界
```

## 安装依赖

```bash
pip install numpy pytest torch torchvision pyyaml tqdm
```

真实 CIFAR-10 下载必须显式启用；单元测试使用离线 synthetic fixture，不应隐式联网。

## 运行测试

```bash
# LWE 原语
pytest tests/v2/test_lwe.py -v

# Gate Layer
pytest tests/v2/test_gate_layer.py -v

# 完整 V2 测试
pytest tests/v2/ -v
```

## 运行 Phase 3 evaluator

Phase 3.6 可信进程内服务入口接收真实 credential，不接受 valid/invalid 采样标志：

```python
import torch

from src.can.v2.service import InferenceService

service = InferenceService(model, torch.device("cpu"))
responses = service.infer(images, credentials)  # [B,3,32,32] 与 [B,n]
print(responses[0].capability_level, responses[0].prediction)
```

该入口不返回 `decision`、验证距离、reason code、gate signal 或路由 indices；它不是 HTTP/gRPC
wire schema，也不保证同进程调用者不能绕过入口直接访问模型。

单 checkpoint 评估示例：

```powershell
python scripts/eval_cifar10_test.py `
  --checkpoint checkpoints/v2/cifar10_seed20260824/stage_c/best.ckpt `
  --data-root data/cifar10 `
  --output experiments/cifar10_seed20260824/test_summary_stage_c.json `
  --summary experiments/cifar10_seed20260824/summary.json `
  --device auto --batch-size 256 --mixed-ratio 0.5 `
  --expected-checkpoint-sha256 <CHECKPOINT_SHA256>
```

三个 Stage C 结果聚合：

```powershell
python scripts/eval_cifar10_test.py --aggregate `
  experiments/cifar10_seed20260824/test_summary_stage_c.json `
  experiments/cifar10_seed20260825/test_summary_stage_c.json `
  experiments/cifar10_seed20260826/test_summary_stage_c.json `
  --output experiments/cifar10_multiseed_test_summary.json
```

正式 test 评估前必须冻结代码、配置、checkpoint manifest、数据 split、指标定义和输出路径；
同一 checkpoint 不得因查看 test 指标而重复评估或调参。

## 参考文献

1. Shamir et al., *How to Securely Implement Cryptography in Deep Neural Networks*。
2. Regev, *On lattices, learning with errors, random linear codes, and cryptography*。
3. Hinton et al., *Distilling the Knowledge in a Neural Network*。
4. Wu et al., *GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs*（本地 preprint）。

## 许可证

研究原型，仅供学术研究使用。

**Last Updated**: 2026-08-27
**Status**: Phase 3 pre-run review pending
