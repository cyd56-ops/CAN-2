"""CAN V2 的模型组件。"""

from .gated_resnet import (
    BasicBlock,
    GatedResNet18,
    InferenceOutput,
    TrainingOutput,
)

__all__ = ["BasicBlock", "GatedResNet18", "InferenceOutput", "TrainingOutput"]
