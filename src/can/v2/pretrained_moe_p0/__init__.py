"""P0-MoE 真实预训练宿主的离线预检契约。

本包只负责严格解析、结构探查和 fake-host 预检，不加载或修改真实模型。
"""

from .artifacts import ArtifactWriter
from .fixture import load_fixture, validate_fixture
from .host_adapter import FakeHostAdapter, HostAdapter, inspect_host
from .registry import load_registry, validate_registry
from .runner import P0Runner
from .types import P0Error

__all__ = [
    "ArtifactWriter",
    "FakeHostAdapter",
    "HostAdapter",
    "P0Error",
    "P0Runner",
    "inspect_host",
    "load_fixture",
    "load_registry",
    "validate_fixture",
    "validate_registry",
]
