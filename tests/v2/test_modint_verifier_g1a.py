"""G1-a 模整数 reference 的边界、摘要和确定性测试。"""

import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from can.v2.modint_verifier_g1a import (
    G1AConfig,
    G1AError,
    G1AParameters,
    build_manifest,
    canonical_mod,
    centered_lift,
    derive_seed,
    generate_parameters,
    load_g1a_manifest,
    parameter_digest,
    verify_batch,
)
from can.v2.modint_verifier_g1a.artifacts import generate_g1a_artifacts
from can.v2.modint_verifier_g1a import artifacts as artifact_module


def _small_config() -> G1AConfig:
    """创建用于快速测试的奇数模数配置。"""
    return G1AConfig(q=17, n=3, m=4, tau=2, master_seed=7)


def test_centered_lift_boundaries() -> None:
    """验证 q//2 和 q//2+1 的精确边界。"""
    assert centered_lift(8, 17) == 8
    assert centered_lift(9, 17) == -8
    assert centered_lift(0, 17) == 0
    assert centered_lift(16, 17) == -1


def test_canonical_mod_negative_values() -> None:
    """验证负数中间值始终映射到 [0,q)。"""
    assert canonical_mod(-5, 3) == 1
    assert canonical_mod(-18, 17) == 16
    assert all(0 <= canonical_mod(value, 17) < 17 for value in range(-40, 41))


def test_parameter_generation_is_deterministic_and_frozen() -> None:
    """验证 A/b 和 seed 派生在相同配置下逐字节一致。"""
    config = _small_config()
    first, secret, error = generate_parameters(config)
    second, _, _ = generate_parameters(config)
    assert np.array_equal(first.A, second.A)
    assert np.array_equal(first.b, second.b)
    assert derive_seed(config.master_seed, "A") != derive_seed(config.master_seed, "secret")
    assert np.array_equal(first.A @ secret + error, first.A @ secret + error)
    with pytest.raises(ValueError):
        first.A[0, 0] = 0


def test_reference_accepts_generated_secret_and_rejects_tamper() -> None:
    """验证合法 secret 通过，篡改 credential 产生逐行拒绝。"""
    config = _small_config()
    parameters, secret, _ = generate_parameters(config)
    valid = np.asarray([secret], dtype="<i8")
    tampered = valid.copy()
    tampered[0, 0] = (int(tampered[0, 0]) + 1) % config.q
    result = verify_batch(np.concatenate([valid, tampered]), parameters)
    assert result.evidence.accepted[0] is True
    assert result.evidence.accepted[1] is False
    assert result.evidence.reason_code == ("RELATION_WITHIN_BOUND", "RELATION_OUT_OF_BOUND")
    assert result.evidence.batch_indices == (0, 1)
    assert result.evidence.max_abs_residual[0] <= config.tau


def test_reference_rejects_noncanonical_inputs() -> None:
    """验证空 batch、浮点、布尔、负数和错误布局均 fail-closed。"""
    config = _small_config()
    parameters, _, _ = generate_parameters(config)
    invalid_inputs = [
        np.empty((0, config.n), dtype="<i8"),
        np.zeros((1, config.n), dtype="<f8"),
        [[True, 0, 0]],
        [[-1, 0, 0]],
        [[config.q, 0, 0]],
        [[0, 0]],
    ]
    for value in invalid_inputs:
        with pytest.raises(G1AError):
            verify_batch(value, parameters)


def test_reference_preserves_original_indices_and_evidence_schema() -> None:
    """验证自定义原始索引、残差摘要和 evidence 长度约束。"""
    config = _small_config()
    parameters, secret, _ = generate_parameters(config)
    result = verify_batch(np.asarray([secret, secret], dtype="<i8"), parameters, batch_indices=[4, 9])
    evidence = result.evidence
    assert evidence.batch_indices == (4, 9)
    assert len(evidence.accepted) == len(evidence.max_abs_residual) == len(evidence.reason_code) == 2
    assert len(evidence.residual_digest) == 64
    with pytest.raises(G1AError):
        verify_batch(np.asarray([secret, secret], dtype="<i8"), parameters, batch_indices=[9, 4])


def test_manifest_round_trip_and_digest(tmp_path: Path) -> None:
    """验证 manifest 生成、原始 bytes 摘要和篡改拒绝。"""
    config = _small_config()
    parameters, _, _ = generate_parameters(config)
    payload = build_manifest(
        config,
        parameters,
        "a" * 64,
        "b" * 64,
        {"git_commit": "c" * 64, "dirty": False, "python": "3.11", "numpy": "2.0"},
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_g1a_manifest(path, digest)["parameter_sha256"] == parameter_digest(parameters)
    with pytest.raises(G1AError):
        load_g1a_manifest(path, "0" * 64)
    payload["norm"] = "l2"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(G1AError):
        load_g1a_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_parameters_reject_bad_domain() -> None:
    """验证公开参数形状和域错误不能被静默接受。"""
    config = _small_config()
    with pytest.raises(G1AError):
        G1AParameters(np.zeros((config.m, config.n + 1), dtype="<i8"), np.zeros(config.m, dtype="<i8"), config)
    with pytest.raises(G1AError):
        G1AParameters(np.zeros((config.m, config.n), dtype="<i8"), np.full(config.m, config.q, dtype="<i8"), config)


def test_artifact_runner_writes_public_reference_bundle(tmp_path: Path) -> None:
    """验证 runner 生成 manifest、公开参数、vectors 和 summary。"""
    output = tmp_path / "run"
    summary = generate_g1a_artifacts(output)
    assert summary["status"] == "complete"
    assert (output / "A.npy").exists()
    assert (output / "b.npy").exists()
    assert (output / "vectors.json").exists()
    assert (output / "manifest.json").exists()
    assert (output / "summary.json").exists()
    summary_text = (output / "summary.json").read_text(encoding="utf-8").lower()
    assert "credential_sha256" in summary_text
    assert "raw_credential" not in summary_text
    with pytest.raises(FileExistsError):
        generate_g1a_artifacts(output)


def test_artifact_provenance_reports_actual_dirty_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 runner 不再把 dirty 状态硬编码为 True。"""
    monkeypatch.setattr(artifact_module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))
    summary = generate_g1a_artifacts(tmp_path / "run")
    assert summary["provenance"]["dirty"] is False
