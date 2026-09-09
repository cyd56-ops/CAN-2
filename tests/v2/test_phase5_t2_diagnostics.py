"""Phase 5.5/T2 单四元组过拟合诊断专项测试。"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts import diagnose_phase5_t2_overfit as diagnostic_entry
from scripts import train_phase5_t2 as training_entry
from scripts.diagnose_phase5_t2_overfit import _build_variant
from scripts.train_phase5_t2 import _strictly_improves
from scripts.train_phase5_t2 import main as train_t2_main
from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    ByteTokenizer,
    GatedDecoderTransformer,
    T2CanDirectPretrainer,
    T2CausalLMDataset,
    T2Evaluator,
    T2OverfitTracker,
    T2RuntimeConfig,
    TransformerConfig,
    classify_t2_overfit_outcome,
    collate_t2_causal_lm_batch,
    file_sha256,
    generate_t2_cap_corpus,
    generate_t2_split,
    materialize_t2_diagnostic_quartet,
    overfit_snapshot,
    save_t2_diagnostic_checkpoint,
    select_first_t2_quartet,
)


def _config() -> TransformerConfig:
    """返回 CPU 诊断测试使用的最小模型配置。"""

    return TransformerConfig(
        max_seq_len=256,
        num_layers=2,
        cut_layer=1,
        d_model=16,
        num_heads=4,
        d_ff=32,
    )


def _quartet():
    """返回规范排序后的首个 CAP/C0 train 四元组。"""

    rows = generate_t2_cap_corpus(20260903, 2, 1, 1, 1)["train"]
    return select_first_t2_quartet(rows)


def _batch() -> dict:
    """编码固定四元组供 direct trainer 使用。"""

    rows = _quartet()
    dataset = T2CausalLMDataset(rows, ByteTokenizer())
    return collate_t2_causal_lm_batch([dataset[index] for index in range(4)])


def _report(value: float = 1.0, *, can: bool = False, invalid: int = 0) -> dict:
    """构造满足 evaluator 摘要形状的确定性判定 fixture。"""

    scopes = ("public", "protected_public", "protected_private", "refusal")
    return {
        "model_kind": "can" if can else "plain",
        "text_metrics": {
            "by_scope": {
                scope: {"exact_match": value, "token_f1": value} for scope in scopes
            }
        },
        "teacher_forced_by_scope": {
            scope: {"token_accuracy": value} for scope in scopes
        },
        "generation_safety": {"invalid_sequences": invalid},
        "routing": (
            {
                "route_calls": 4,
                "valid": 2,
                "invalid": 2,
                "protected_indices": 2,
                "public_indices": 2,
                "rejected_indices": 0,
                "invalid_protected_block_calls": 0,
            }
            if can
            else None
        ),
    }


def _train_metrics(*, can: bool = False) -> dict:
    """返回有限的最小训练观测。"""

    gate = None
    if can:
        summary = {
            "count": 2,
            "signal": {"min": 0.0, "mean": 0.5, "max": 1.0},
            "error_norm": {"min": 0.0, "mean": 0.5, "max": 1.0},
        }
        gate = {"valid": dict(summary), "invalid": dict(summary)}
    return {
        "loss": 2.0,
        "total_loss": 2.0,
        "public_head_loss": 1.0,
        "protected_head_loss": 1.0,
        "scope_losses": {
            "public": 1.0,
            "protected_public": 1.0,
            "protected_private": 1.0,
            "refusal": 1.0,
        },
        "scope_answer_tokens": {
            "public": 1,
            "protected_public": 1,
            "protected_private": 1,
            "refusal": 1,
        },
        "gradient_norms": {
            "shared_prefix": 0.5,
            "protected_path": 0.5,
            "public_path": 0.5,
        },
        "gate": gate,
    }


def test_select_first_quartet_is_sorted_and_complete() -> None:
    """选择器应固定第一个 source/template 并完整覆盖四 scope。"""

    quartet = _quartet()
    assert [row.scope for row in quartet] == [
        "public",
        "protected_public",
        "protected_private",
        "refusal",
    ]
    assert len({(row.source_id, row.prompt_template_id) for row in quartet}) == 1


def test_select_first_quartet_rejects_non_train() -> None:
    """诊断选择器不得接受 dev/validation/test。"""

    rows = generate_t2_cap_corpus(20260903, 1, 1, 1, 1)["dev"]
    with pytest.raises(ValueError, match="只允许 train"):
        select_first_t2_quartet(rows)


def test_materializer_calls_only_train_split() -> None:
    """split spy 应证明固定入口不会构造 dev/validation/test。"""

    calls = []

    def spy(suite_id, split, seed, counts, *, prompt_group):
        """记录 split 调用后委托真实生成器。"""

        calls.append((suite_id, split, seed, prompt_group))
        return generate_t2_split(
            suite_id,
            split,
            seed,
            counts,
            prompt_group=prompt_group,
        )

    quartet = materialize_t2_diagnostic_quartet(spy)
    assert len(quartet) == 4
    assert calls == [("t2_nl_cap", "train", 20260903, "C0")]


def test_overfit_snapshot_passes_complete_plain_report() -> None:
    """四 scope 全 1 且数值有限时应形成单次成功。"""

    snapshot = overfit_snapshot(_report(), _train_metrics())
    assert snapshot["pass_once"] is True
    assert snapshot["partial_once"] is True


def test_overfit_snapshot_checks_can_routing() -> None:
    """CAN 的路由覆盖或 zero-call 异常必须阻止成功。"""

    report = _report(can=True)
    report["routing"]["invalid_protected_block_calls"] = 1
    with pytest.raises(RuntimeError, match="routing"):
        overfit_snapshot(report, _train_metrics(can=True))


def test_overfit_snapshot_rejects_generation_safety_failure() -> None:
    """异常生成序列不得达到单次通过门槛。"""

    assert overfit_snapshot(_report(invalid=1), _train_metrics())["pass_once"] is False


def test_tracker_requires_three_consecutive_successes() -> None:
    """连续三次通过前 tracker 必须保持 running。"""

    tracker = T2OverfitTracker(max_updates=48)
    snapshot = overfit_snapshot(_report(), _train_metrics())
    assert tracker.observe(16, 100, snapshot) is False
    assert tracker.observe(32, 200, snapshot) is False
    assert tracker.observe(48, 300, snapshot) is True
    assert tracker.status(48) == "passed"
    assert tracker.passed_at_update == 48


def test_tracker_resets_nonconsecutive_streak() -> None:
    """中间失败的评估点必须重置连续成功计数。"""

    tracker = T2OverfitTracker(max_updates=64)
    passed = overfit_snapshot(_report(), _train_metrics())
    failed = overfit_snapshot(_report(0.0), _train_metrics())
    tracker.observe(16, 100, passed)
    tracker.observe(32, 200, failed)
    tracker.observe(48, 300, passed)
    assert tracker.current_streak == 1
    assert tracker.status(48) == "running"


def test_tracker_marks_partial_only_at_budget_end() -> None:
    """连续三次近饱和只在完整预算结束时标记 partial。"""

    tracker = T2OverfitTracker(max_updates=48)
    partial = overfit_snapshot(_report(0.96), _train_metrics())
    for update in (16, 32, 48):
        tracker.observe(update, update * 10, partial)
    assert tracker.status(32) == "running"
    assert tracker.status(48) == "partial_progress"


def test_tracker_marks_failed_and_invalid_separately() -> None:
    """普通性能失败与协议/数值 invalid 必须使用不同状态。"""

    tracker = T2OverfitTracker(max_updates=16)
    tracker.observe(16, 100, overfit_snapshot(_report(0.5), _train_metrics()))
    assert tracker.status(16) == "failed_to_overfit"
    assert tracker.status(16, invalid=True) == "invalid_run"


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (("passed", "passed", "passed"), "investigate_multi_source_and_objective"),
        (("passed", "failed_to_overfit", "passed"), "investigate_soft_gate_semantics"),
        (
            ("passed", "failed_to_overfit", "failed_to_overfit"),
            "investigate_can_routing_and_credential",
        ),
        (
            ("failed_to_overfit", "passed", "passed"),
            "investigate_plain_baseline",
        ),
        (("passed", "passed", "failed_to_overfit"), "invalid_protocol"),
    ],
)
def test_cross_variant_decision_table(statuses, expected: str) -> None:
    """三个变体状态必须严格映射到预注册的下一调查方向。"""

    values = dict(zip(("plain", "can_soft", "can_direct"), statuses))
    assert classify_t2_overfit_outcome(values) == expected


def test_tracker_records_first_scope_em_update() -> None:
    """每 scope 首次 EM=1 的 update 只记录最早时点。"""

    tracker = T2OverfitTracker(max_updates=32)
    snapshot = overfit_snapshot(_report(), _train_metrics())
    tracker.observe(16, 100, snapshot)
    tracker.observe(32, 200, snapshot)
    assert set(tracker.first_scope_em_update.values()) == {16}


def test_tracker_rejects_non_monotonic_updates() -> None:
    """绝对 update 必须严格递增。"""

    tracker = T2OverfitTracker(max_updates=32)
    snapshot = overfit_snapshot(_report(), _train_metrics())
    tracker.observe(16, 100, snapshot)
    with pytest.raises(ValueError, match="严格递增"):
        tracker.observe(16, 200, snapshot)


def test_tracker_rejects_non_finite_snapshot() -> None:
    """NaN/Inf 诊断值必须 fail closed，不能降级为性能失败。"""

    tracker = T2OverfitTracker(max_updates=16)
    snapshot = overfit_snapshot(_report(), _train_metrics())
    snapshot["token_f1"]["public"] = float("nan")
    with pytest.raises(FloatingPointError, match="非有限"):
        tracker.observe(16, 100, snapshot)


def test_can_direct_uses_real_gate_and_reports_zero_call() -> None:
    """direct 消融应真实验签，只在两个 allow 行执行 protected path。"""

    torch.manual_seed(71)
    params = LWEParams(n=32, m=64)
    matrix, secret, vector = generate_keypair(params, np.random.default_rng(72))
    model = GatedDecoderTransformer(matrix, vector, params, _config())
    generator = CredentialGenerator(matrix, secret, vector, params, seed=73)
    trainer = T2CanDirectPretrainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        torch.device("cpu"),
        generator,
    )
    metrics = trainer.train_batch(_batch())
    assert metrics["direct_routing"] == {
        "valid": 2,
        "invalid": 2,
        "protected_rows_executed": 2,
        "invalid_protected_block_calls": 0,
    }
    assert metrics["gate"]["valid"]["count"] == 2


def test_cpu_minimal_fixture_runs_all_three_variants() -> None:
    """最小 CPU fixture 应让三变体完成训练一步与正式评估接线。"""

    quartet = _quartet()
    batch = _batch()
    hashes = set()
    for variant in ("plain", "can_soft", "can_direct"):
        model, trainer, initial_hash, generator = _build_variant(
            variant, _config(), torch.device("cpu"), 20260903
        )
        hashes.add(initial_hash)
        metrics = trainer.train_batch(batch)
        report = T2Evaluator(
            model,
            ByteTokenizer(),
            torch.device("cpu"),
            generator,
            max_new_tokens=1,
            cache_mode="none",
            batch_size=4,
        ).evaluate(quartet)
        assert metrics["global_step"] == 1.0
        assert report["status"] == "ok"
    assert len(hashes) == 1


def test_diagnostic_checkpoint_excludes_secret_and_public_arrays(tmp_path) -> None:
    """诊断 checkpoint 不得保存 secret、credential 或 LWE 公共数组原值。"""

    params = LWEParams(n=16, m=32)
    matrix, _, vector = generate_keypair(params, np.random.default_rng(81))
    model = GatedDecoderTransformer(matrix, vector, params, _config())
    path = tmp_path / "final.ckpt"
    save_t2_diagnostic_checkpoint(path, model, "can_soft", 1, 100)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    serialized_keys = json.dumps(sorted(payload), ensure_ascii=False).casefold()
    assert "secret" not in serialized_keys
    assert "credential" not in serialized_keys
    assert not any(
        name.startswith("gate_layer.verifier")
        for name in payload["model_state_without_lwe_public"]
    )
    assert set(payload["lwe_public_identity"]) == {"A_sha256", "b_sha256"}


def test_diagnostic_checkpoint_refuses_overwrite(tmp_path) -> None:
    """诊断 checkpoint 默认拒绝覆盖已有输出。"""

    model = GatedDecoderTransformer(
        np.zeros((32, 16), dtype=np.float32),
        np.zeros(32, dtype=np.float32),
        LWEParams(n=16, m=32),
        _config(),
    )
    path = tmp_path / "final.ckpt"
    save_t2_diagnostic_checkpoint(path, model, "can_soft", 1, 100)
    with pytest.raises(FileExistsError):
        save_t2_diagnostic_checkpoint(path, model, "can_soft", 2, 200)


def test_training_entry_writes_schema_v2_best_final_binding(tmp_path) -> None:
    """正式训练入口的最小运行应分离 best/final 摘要和诊断文件。"""

    output = tmp_path / "schema-v2"
    assert (
        train_t2_main(
            [
                "--mode",
                "dev-pilot",
                "--models",
                "plain",
                "--suite",
                "cap",
                "--prompt-group",
                "C0",
                "--seed",
                "991",
                "--output",
                str(output),
                "--token-budget",
                "1200",
                "--validation-interval-tokens",
                "100000",
                "--batch-size",
                "4",
                "--max-new-tokens",
                "1",
                "--train-count",
                "1",
                "--dev-count",
                "1",
                "--validation-count",
                "1",
                "--test-count",
                "1",
                "--device",
                "cpu",
                "--smoke-model",
                "--no-progress",
            ]
        )
        == 0
    )
    model_dir = output / "plain"
    summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 2
    assert "evaluation" not in summary
    assert "diagnostics" not in summary["best_evaluation"]
    assert "diagnostics" not in summary["final_evaluation"]
    assert summary["best_checkpoint"]["sha256"] == file_sha256(model_dir / "best.ckpt")
    assert summary["final_checkpoint"]["sha256"] == file_sha256(model_dir / "last.ckpt")
    progress = torch.load(
        model_dir / "last.ckpt", map_location="cpu", weights_only=False
    )["progress"]
    assert progress["resume_count"] == 0
    assert progress["best_at_tokens"] == summary["best_at_tokens"]
    assert progress["best_at_global_step"] == summary["best_at_global_step"]
    assert progress["best_evaluation"] == summary["best_evaluation"]
    assert (model_dir / "best_diagnostic.json").is_file()
    assert (model_dir / "final_diagnostic.json").is_file()


def test_best_selection_requires_strict_improvement_and_keeps_tie() -> None:
    """相同 selection score 必须保留更早 best，不能被 final 覆盖。"""

    assert _strictly_improves(0.5, None) is True
    assert _strictly_improves(0.6, 0.5) is True
    assert _strictly_improves(0.5, 0.5) is False
    assert _strictly_improves(0.4, 0.5) is False


@pytest.mark.parametrize("kind", ["plain", "can"])
def test_interrupted_resume_preserves_absolute_best(
    tmp_path, monkeypatch, kind
) -> None:
    """中断后恢复必须保持绝对 best/tie，并与不中断训练权重一致。"""

    rows = _quartet()
    batch = _batch()
    tokens = int(batch["attention_mask"].sum())
    config = T2RuntimeConfig(
        mode="dev-pilot",
        suite_id="t2_nl_cap",
        prompt_group="C0",
        seed=20260903,
        models=kind,
        split_counts={name: 1 for name in ("train", "dev", "validation", "test")},
        batch_size=4,
        token_budget=3 * tokens,
        validation_interval_tokens=tokens,
        learning_rate=0.001,
        max_new_tokens=1,
        cache_mode="none",
        model_config=_config(),
    )
    monkeypatch.setattr(training_entry, "_selection_score", lambda report: 0.5)
    trainer_class = (
        training_entry.T2PlainPretrainer
        if kind == "plain"
        else training_entry.T2CanPretrainer
    )
    original = trainer_class.train_batch

    def interrupt(trainer, values):
        """在第一步已持久化后模拟第二步前中断。"""
        if trainer.global_step == 1:
            raise RuntimeError("fixture interruption")
        return original(trainer, values)

    monkeypatch.setattr(trainer_class, "train_batch", interrupt)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        training_entry._train_one(
            kind,
            config,
            rows,
            rows,
            tmp_path / "resumed",
            torch.device("cpu"),
            False,
            False,
        )
    monkeypatch.setattr(trainer_class, "train_batch", original)
    resumed = training_entry._train_one(
        kind, config, rows, rows, tmp_path / "resumed", torch.device("cpu"), True, False
    )
    full = training_entry._train_one(
        kind, config, rows, rows, tmp_path / "full", torch.device("cpu"), False, False
    )
    assert resumed["resume_count"] == 1
    assert resumed["best_at_global_step"] == 1
    assert resumed["final_at_global_step"] == 3
    assert resumed["best_at_tokens"] == tokens
    assert resumed["best_is_final"] is False
    assert resumed["final_model_tensor_sha256"] == full["final_model_tensor_sha256"]
    assert resumed["best_evaluation"] == full["best_evaluation"]


def test_cli_cpu_fixture_and_overwrite_guard(tmp_path, monkeypatch) -> None:
    """缩小 CPU fixture 跑过整个 CLI，验证实际输出和 train-only 边界。"""
    monkeypatch.setattr(diagnostic_entry, "TransformerConfig", _config)
    monkeypatch.setattr(
        diagnostic_entry, "T2OverfitTracker", lambda: T2OverfitTracker(max_updates=16)
    )
    output = tmp_path / "diagnostic"
    assert diagnostic_entry.main(["--output", str(output), "--no-progress"]) == 0
    summary = json.loads((output / "overfit_summary.json").read_text(encoding="utf-8"))
    assert summary["same_credential_schedule"] is True
    assert summary["credential_rows_compared"] == 68
    assert summary["shared_initialization"] is True
    assert summary["splits_materialized"] == {
        "train": True,
        "dev": False,
        "validation": False,
        "test": False,
    }
    for kind in ("plain", "can_soft", "can_direct"):
        assert summary["variants"][kind]["completed_updates"] == 16
        assert (output / kind / "final.ckpt").is_file()
    with pytest.raises(FileExistsError):
        diagnostic_entry.main(["--output", str(output), "--no-progress"])


def test_cli_records_invalid_evaluation_without_secondary_crash(
    tmp_path, monkeypatch
) -> None:
    """首次评估损坏必须写 invalid_run；不能因空 snapshots 再次崩溃。"""
    monkeypatch.setattr(diagnostic_entry, "TransformerConfig", _config)
    monkeypatch.setattr(
        diagnostic_entry, "T2OverfitTracker", lambda: T2OverfitTracker(max_updates=16)
    )

    def broken_evaluation(evaluator, rows):
        """模拟 evaluator 首次返回不完整结果。"""
        return {"broken": True}

    monkeypatch.setattr(diagnostic_entry.T2Evaluator, "evaluate", broken_evaluation)
    output = tmp_path / "invalid"
    assert diagnostic_entry.main(["--output", str(output), "--no-progress"]) == 2
    summary = json.loads((output / "overfit_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "invalid_run"
    assert all(row["status"] == "invalid_run" for row in summary["variants"].values())
    assert not (output / "plain" / "final.ckpt").exists()


def test_direct_rejects_wrong_credentials_before_suffix(monkeypatch) -> None:
    """把合法行凭证替换为 invalid 后，任何 protected suffix 都不得执行。"""
    model, trainer, _, generator = _build_variant(
        "can_direct", _config(), torch.device("cpu"), 20260903
    )
    original = generator.generate
    monkeypatch.setattr(generator, "generate", lambda valid: original(False))
    calls = []
    handle = model.blocks[-1].register_forward_pre_hook(
        lambda module, args: calls.append(1)
    )
    try:
        with pytest.raises(RuntimeError, match="Gate allow"):
            trainer.train_batch(_batch())
    finally:
        handle.remove()
    assert calls == []
