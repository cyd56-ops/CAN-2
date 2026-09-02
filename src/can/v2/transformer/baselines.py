"""Phase 5 T1 的 P0 baseline 合规配置。"""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BaselineConfig:
    """冻结一个 P0 baseline 的数据、tokenizer 和预算 identity。"""

    baseline_type: str
    tokenizer_hash: str
    prompt_hash: str
    split_hash: str
    train_data_scope: str
    token_budget: int
    optimizer_steps: int

    def __post_init__(self) -> None:
        """验证 baseline 类型和不可为空的 identity。"""
        if self.baseline_type not in {
            "early_exit_vs_full",
            "capacity",
            "prefix_isolation",
        }:
            raise ValueError("baseline_type 不受支持")
        if self.train_data_scope not in {"public_only", "public_private"}:
            raise ValueError("train_data_scope 不受支持")
        if self.token_budget <= 0 or self.optimizer_steps <= 0:
            raise ValueError("baseline 预算必须大于 0")
        if not all(
            isinstance(value, str) and value
            for value in (self.tokenizer_hash, self.prompt_hash, self.split_hash)
        ):
            raise ValueError("baseline identity 不能为空")


def validate_baseline_compatibility(
    config: BaselineConfig, main: Mapping[str, object]
) -> None:
    """拒绝 tokenizer、prompt、split 或训练预算与主实验不一致的 baseline。"""

    if not isinstance(config, BaselineConfig) or not isinstance(main, Mapping):
        raise TypeError("config 和 main 必须是合法映射")
    fields = {
        "tokenizer_hash": config.tokenizer_hash,
        "prompt_hash": config.prompt_hash,
        "split_hash": config.split_hash,
        "token_budget": config.token_budget,
        "optimizer_steps": config.optimizer_steps,
    }
    for key, value in fields.items():
        if main.get(key) != value:
            raise ValueError(f"baseline 字段 {key} 与主实验不一致")


__all__ = ["BaselineConfig", "validate_baseline_compatibility"]
