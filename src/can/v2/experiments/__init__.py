"""CAN v2 实验模块

包含功能正确性、能力分级和性能评估实验。

计划内容:
- functional_test.py: fail-closed 与差分测试验证
- capability_tiering.py: protected/public 能力分级测量
 - performance.py: latency 与吞吐量测量
"""

from .test_evaluator import TestSplitEvaluator

__all__ = ["TestSplitEvaluator"]
