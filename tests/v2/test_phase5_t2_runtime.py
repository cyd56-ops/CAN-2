"""Phase 5.5/T2 runtime、checkpoint 与状态机专项测试。"""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from src.can.v2.crypto.lwe import LWEParams
from src.can.v2.transformer import (
    GatedDecoderTransformer,
    PlainDecoderTransformer,
    T2RuntimeConfig,
    TransformerConfig,
    array_sha256,
    atomic_write_json,
    begin_test_access,
    derive_lwe_keypair,
    finish_test_access,
    initialize_run_state,
    load_frozen_runtime,
    load_run_state,
    load_strict_json,
    load_t2_checkpoint,
    model_tensor_sha256,
    runtime_config_from_mapping,
    save_t2_checkpoint,
)


def _model_config() -> TransformerConfig:
    """返回 runtime/checkpoint 单测使用的最小模型。"""

    return TransformerConfig(
        num_layers=2, cut_layer=1, d_model=16, num_heads=4, d_ff=32
    )


def _runtime(mode: str = "dev-pilot", models: str = "both") -> T2RuntimeConfig:
    """构造合法 T2 runtime fixture。"""

    return T2RuntimeConfig(
        mode=mode,
        suite_id="t2_nl_cap",
        prompt_group="C0",
        seed=701,
        models=models,
        split_counts={"train": 2, "dev": 1, "validation": 1, "test": 1},
        batch_size=4,
        token_budget=1000,
        validation_interval_tokens=500,
        learning_rate=1e-3,
        max_new_tokens=1,
        cache_mode="none",
        model_config=_model_config(),
    )


def test_runtime_round_trip_strict_json(tmp_path) -> None:
    """runtime config 应能规范写入并严格读回。"""

    path = tmp_path / "runtime.json"
    atomic_write_json(path, _runtime().to_dict())
    loaded = runtime_config_from_mapping(load_strict_json(path))
    assert loaded == _runtime()


def test_strict_json_rejects_duplicate_key(tmp_path) -> None:
    """重复 JSON 字段不得采用 last-key-wins。"""

    path = tmp_path / "bad.json"
    path.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="重复字段"):
        load_strict_json(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_nonfinite_constant(tmp_path, constant: str) -> None:
    """非标准 JSON 浮点常量必须失败。"""

    path = tmp_path / "bad.json"
    path.write_text(f'{{"value": {constant}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="非有限"):
        load_strict_json(path)


def test_atomic_json_rejects_nan(tmp_path) -> None:
    """结果写出不得包含 NaN。"""

    with pytest.raises(ValueError, match="不含非有限"):
        atomic_write_json(tmp_path / "bad.json", {"value": float("nan")})


@pytest.mark.parametrize("batch_size", [1, 6, True])
def test_runtime_rejects_non_quadruplet_batch(batch_size) -> None:
    """T2 runtime 必须维持四元组 batch 边界。"""

    values = _runtime().to_dict()
    values["batch_size"] = batch_size
    with pytest.raises((TypeError, ValueError)):
        runtime_config_from_mapping(values)


def test_runtime_rejects_unknown_field() -> None:
    """freeze/runtime 未知字段必须 fail closed。"""

    values = _runtime().to_dict()
    values["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        runtime_config_from_mapping(values)


def test_frozen_runtime_requires_external_hash_and_both(tmp_path) -> None:
    """正式 freeze 必须由外部摘要验证且固定 paired models。"""

    path = tmp_path / "freeze.json"
    digest = atomic_write_json(path, _runtime("frozen-validation").to_dict())
    assert load_frozen_runtime(path, digest).mode == "frozen-validation"
    with pytest.raises(ValueError, match="SHA-256 不匹配"):
        load_frozen_runtime(path, "0" * 64)
    atomic_write_json(path, _runtime("frozen-validation", "plain").to_dict())
    with pytest.raises(ValueError, match="成对模型"):
        load_frozen_runtime(
            path,
            atomic_write_json(path, _runtime("frozen-validation", "plain").to_dict()),
        )


def test_run_state_rejects_cross_identity_resume(tmp_path) -> None:
    """resume 不得跨 seed 或跨 corpus 身份。"""

    path = tmp_path / "run_state.json"
    initialize_run_state(path, {"seed": 1})
    assert load_run_state(path, {"seed": 1})["status"] == "running"
    with pytest.raises(ValueError, match="identity"):
        load_run_state(path, {"seed": 2})
    with pytest.raises(FileExistsError):
        initialize_run_state(path, {"seed": 1})


@pytest.mark.parametrize("terminal", ["completed", "failed"])
def test_test_ledger_is_one_shot_for_all_terminal_states(
    tmp_path, terminal: str
) -> None:
    """started/completed/failed 任一状态都必须永久阻止 test 重读。"""

    identity = {"seed": 1, "checkpoint": "a" * 64}
    ledger = begin_test_access(tmp_path, identity)
    finish_test_access(
        ledger, identity, terminal, "failure" if terminal == "failed" else None
    )
    assert load_strict_json(ledger)["status"] == terminal
    with pytest.raises(RuntimeError, match="已被"):
        begin_test_access(tmp_path, identity)


def test_test_ledger_rejects_identity_change(tmp_path) -> None:
    """test ledger 终态更新必须绑定 started 时的身份。"""

    ledger = begin_test_access(tmp_path, {"seed": 1})
    with pytest.raises(RuntimeError, match="身份一致"):
        finish_test_access(ledger, {"seed": 2}, "failed", "x")


def test_plain_checkpoint_round_trip_restores_tensor(tmp_path) -> None:
    """Plain checkpoint 应恢复模型、优化器和进度。"""

    torch.manual_seed(11)
    model = PlainDecoderTransformer(_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    original = model_tensor_sha256(model)
    save_t2_checkpoint(
        tmp_path / "last.ckpt",
        model,
        optimizer,
        {"seed": 1},
        {"epoch": 0, "batch_offset": 1, "total_tokens": 10},
    )
    with torch.no_grad():
        model.public_head.weight.zero_()
    payload = load_t2_checkpoint(
        tmp_path / "last.ckpt", model, optimizer, {"seed": 1}, restore_rng=False
    )
    assert model_tensor_sha256(model) == original
    assert payload["progress"]["total_tokens"] == 10
    assert payload["lwe_public"] is None


def test_can_checkpoint_saves_public_key_without_secret(tmp_path) -> None:
    """CAN checkpoint 必须包含可校验 A/b，但不得包含 secret。"""

    params = LWEParams(n=16, m=32)
    matrix, _, vector = derive_lwe_keypair(17, params)
    model = GatedDecoderTransformer(matrix, vector, params, _model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "can.ckpt"
    save_t2_checkpoint(path, model, optimizer, {"seed": 17}, {"epoch": 0})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "secret" not in payload
    assert "secret" not in payload["metadata"]
    assert payload["lwe_public"]["A_sha256"] == array_sha256(matrix)
    load_t2_checkpoint(path, model, optimizer, {"seed": 17}, restore_rng=False)


def test_checkpoint_rejects_metadata_and_keypair_mismatch(tmp_path) -> None:
    """resume 的 metadata 或派生 keypair 漂移必须失败。"""

    params = LWEParams(n=16, m=32)
    matrix, _, vector = derive_lwe_keypair(21, params)
    model = GatedDecoderTransformer(matrix, vector, params, _model_config())
    path = tmp_path / "can.ckpt"
    save_t2_checkpoint(
        path, model, torch.optim.AdamW(model.parameters()), {"seed": 21}, {"epoch": 0}
    )
    with pytest.raises(ValueError, match="metadata"):
        load_t2_checkpoint(path, model, None, {"seed": 22}, restore_rng=False)
    other_a, _, other_b = derive_lwe_keypair(22, params)
    other = GatedDecoderTransformer(other_a, other_b, params, _model_config())
    with pytest.raises(ValueError, match="keypair"):
        load_t2_checkpoint(path, other, None, {"seed": 21}, restore_rng=False)


def test_checkpoint_metadata_rejects_secret(tmp_path) -> None:
    """任何 secret 命名的 checkpoint metadata 都必须被拒绝。"""

    model = PlainDecoderTransformer(_model_config())
    with pytest.raises(ValueError, match="secret"):
        save_t2_checkpoint(
            tmp_path / "bad.ckpt",
            model,
            torch.optim.AdamW(model.parameters()),
            {"secret_value": "forbidden"},
            {"epoch": 0},
        )


def test_plain_and_can_share_identical_trainable_initialization() -> None:
    """相同 seed 下 Plain/CAN 的全部可训练初始 tensor 必须一致。"""

    torch.manual_seed(99)
    plain = PlainDecoderTransformer(_model_config())
    params = LWEParams(n=16, m=32)
    matrix, _, vector = derive_lwe_keypair(99, params)
    torch.manual_seed(99)
    can = GatedDecoderTransformer(matrix, vector, params, _model_config())
    assert model_tensor_sha256(plain, trainable_only=True) == model_tensor_sha256(
        can, trainable_only=True
    )
