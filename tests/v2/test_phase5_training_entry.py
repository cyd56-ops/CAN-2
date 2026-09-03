"""Phase 5 正式训练入口的冻结配置与状态机专项测试。"""

import json
from pathlib import Path

import pytest
import torch

from scripts.train_phase5 import (
    GENERATOR_VERSION,
    _diagnostic_score,
    _is_better_diagnostic,
    _load_resume_state,
    _loader,
    _new_state,
    _reset_managed_output,
    _score,
    _validate_formal_freeze,
)
from src.can.v2.transformer import (
    ByteTokenizer,
    TransformerConfig,
    count_non_padding_input_tokens,
)
from src.can.v2.transformer.data import generate_synthetic_corpus


def _freeze_record() -> dict:
    """构造满足正式入口约束的最小冻结记录。"""

    return {
        "freeze_version": "phase5-freeze-v3",
        "generator_version": GENERATOR_VERSION,
        "budget_token_unit": "non_padding_input_tokens",
        "t_pretrain_stop_policy": "go-no-go-or-full-budget-v3",
        "t_pretrain_checkpoint_policy": "threshold-ratio-loss-tiebreak-v3",
        "seeds": [20260903],
        "model_config": TransformerConfig().__dict__,
        "train_entities": 48,
        "validation_entities": 20,
        "test_entities": 20,
        "max_new_tokens": 16,
        "validation_interval_tokens": 50_000,
        "t_pretrain_token_budget": 2_000_000,
        "stage_a_token_budget": 100_000,
        "stage_b_token_budget": 100_000,
        "stage_c_token_budget": 100_000,
        "learning_rate": 1e-3,
        "benchmark": {
            "budget_token_unit": "non_padding_input_tokens",
            "tokens_per_second": 1.0,
            "seconds_per_step": 1.0,
            "validation_wall_seconds": 1.0,
            "sha256": "b" * 64,
        },
    }


def test_formal_freeze_accepts_complete_record() -> None:
    """完整且匹配的冻结记录应解析为正式训练配置。"""

    frozen = _validate_formal_freeze(_freeze_record(), 20260903)
    assert frozen["validation_entities"] == 20
    assert frozen["t_pretrain_token_budget"] == 2_000_000
    assert frozen["learning_rate"] == 1e-3


@pytest.mark.parametrize(
    "field,value",
    [
        ("validation_entities", 8),
        ("learning_rate", float("nan")),
        ("model_config", {}),
    ],
)
def test_formal_freeze_rejects_unfit_configuration(field: str, value: object) -> None:
    """过小 validation、非有限学习率和模型漂移必须 fail closed。"""

    record = _freeze_record()
    record[field] = value
    with pytest.raises(ValueError):
        _validate_formal_freeze(record, 20260903)


def test_formal_freeze_rejects_unlisted_seed() -> None:
    """未冻结的随机种子不得启动正式训练。"""

    with pytest.raises(ValueError, match="seed"):
        _validate_formal_freeze(_freeze_record(), 7)


@pytest.mark.parametrize(
    "field,value",
    [
        ("budget_token_unit", "supervised_target_tokens"),
        ("t_pretrain_stop_policy", "legacy-early-stop"),
        ("t_pretrain_checkpoint_policy", "legacy-em-best"),
    ],
)
def test_formal_freeze_rejects_non_v3_policy(field: str, value: str) -> None:
    """正式入口必须拒绝旧 token、停止和 checkpoint 策略。"""

    record = _freeze_record()
    record[field] = value
    with pytest.raises(ValueError, match=field):
        _validate_formal_freeze(record, 20260903)


def test_formal_freeze_requires_validation_wall_time() -> None:
    """v3 benchmark 缺少完整 validation 墙钟时间时不得冻结。"""

    record = _freeze_record()
    del record["benchmark"]["validation_wall_seconds"]
    with pytest.raises(ValueError, match="validation_wall_seconds"):
        _validate_formal_freeze(record, 20260903)


def test_budget_token_counter_uses_full_attention_mask() -> None:
    """trainer 与 benchmark 共用 prompt+target 非 padding token 口径。"""

    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    assert count_non_padding_input_tokens(attention_mask) == 5


def test_loader_is_checked_after_construction() -> None:
    """正式 loader 使用 DataLoader 自身长度验证非空。"""

    examples = generate_synthetic_corpus(3, 2, 2, 2)["train"]
    loader = _loader(examples, ByteTokenizer(), 6, 3, 256)
    assert len(loader) == 1


def test_tpretrain_score_uses_separate_public_private_metrics() -> None:
    """T-pretrain best 分数取 protected public/private 较小值。"""

    metrics = {
        "protected_public": {"exact_match": 0.9},
        "protected_private": {"exact_match": 0.8},
        "public": {"exact_match": 0.1},
        "refusal": {"refusal_rate": 0.0},
    }
    assert _score("T-pretrain", metrics) == 0.8


def test_new_state_records_resume_contract() -> None:
    """运行状态必须记录每阶段 token、历史和 freeze identity。"""

    frozen = _validate_formal_freeze(_freeze_record(), 20260903)
    state = _new_state(20260903, "a" * 64, 144, frozen)
    assert state["current_stage"] == "T-pretrain"
    assert state["completed_stages"] == []
    assert state["stage_tokens"]["C"] == 0
    assert state["freeze_record_sha256"] == "a" * 64
    assert state["diagnostic_best_tokens"]["T-pretrain"] is None


def _validation(public: float, private: float, refusal: float, loss: float) -> dict:
    """构造最小 T-pretrain validation 指标。"""

    return {
        "protected_public": {"exact_match": public, "token_loss": loss},
        "protected_private": {"exact_match": private, "token_loss": loss},
        "refusal": {"refusal_rate": refusal},
    }


def test_diagnostic_score_uses_threshold_ratio_then_loss() -> None:
    """诊断排序先比较三项门槛比例，再比较 protected loss。"""

    better_ratio = _validation(0.8, 0.7, 0.9, 10.0)
    lower_ratio = _validation(0.7, 0.69, 0.9, 0.01)
    assert _diagnostic_score(better_ratio) > _diagnostic_score(lower_ratio)
    equal_ratio_low_loss = _validation(0.8, 0.8, 0.9, 1.0)
    equal_ratio_high_loss = _validation(0.8, 0.8, 0.9, 2.0)
    assert _is_better_diagnostic(equal_ratio_low_loss, 100, equal_ratio_high_loss, 50)


def test_diagnostic_score_token_tiebreak() -> None:
    """比例和 loss 都相同时选择累计 token 更少的 checkpoint。"""

    metrics = _validation(0.5, 0.5, 0.5, 1.0)
    assert _is_better_diagnostic(metrics, 10, metrics, 20)
    assert not _is_better_diagnostic(metrics, 20, metrics, 10)


def test_resume_state_rejects_stage_order_tampering(tmp_path: Path) -> None:
    """恢复状态不得跳过中间阶段或改变冻结身份。"""

    frozen = _validate_formal_freeze(_freeze_record(), 20260903)
    state = _new_state(20260903, "a" * 64, 144, frozen)
    state["completed_stages"] = ["T-pretrain", "B"]
    path = tmp_path / "run_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="阶段"):
        _load_resume_state(path, 20260903, "a" * 64, 144, frozen)


def test_force_overwrite_only_removes_managed_output(tmp_path: Path) -> None:
    """强制覆盖只能清理带本脚本状态标记的实验目录。"""

    unmanaged = tmp_path / "unmanaged"
    unmanaged.mkdir()
    with pytest.raises(ValueError, match="run_state"):
        _reset_managed_output(unmanaged)
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "run_state.json").write_text("{}", encoding="utf-8")
    _reset_managed_output(managed)
    assert not managed.exists()
