"""G1-b CPU int64 verifier 的差分、边界和 fail-closed 测试。"""

from pathlib import Path
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from can.v2.modint_verifier_g1a import G1AConfig, generate_parameters, verify_batch
from can.v2.modint_verifier_g1b import (
    G1BError,
    ModIntNeuralVerifier,
    canonical_mod_q,
    centered_lift_tensor,
    derive_integer_bounds,
    verify_tensor_batch,
)
from can.v2.modint_verifier_g1b.artifacts import generate_g1b_artifacts
from can.v2.modint_verifier_g1b.manifest import load_g1b_manifest
import hashlib
from can.v2.modint_verifier_g1b.manifest import build_g1b_manifest, validate_g1b_manifest
from can.v2.modint_verifier_g1b.types import G1BResult, IntegerBounds
from can.v2.modint_verifier_g1b import artifacts as artifact_module
from can.v2.modint_verifier_g1b import kernel as kernel_module


def _parameters():
    """创建 G1-a 小 fixture 的公开参数。"""
    return generate_parameters(G1AConfig(q=17, n=3, m=4, tau=2, master_seed=7))[0]


def test_bounds_are_analytic_and_int64_safe() -> None:
    """验证上界不是由 vectors 观测值推断，且 tiny fixture 可用 int64。"""
    bounds = derive_integer_bounds(_parameters())
    assert bounds.max_product_abs == 16 * 16
    assert bounds.max_accumulator_abs == 3 * 16 * 16 + 16
    assert bounds.required_signed_bits >= 1
    assert bounds.formula_version == "g1b-bounds-v1"


def test_canonical_mod_and_centered_boundaries() -> None:
    """验证 torch kernel 的负数模和 centered 边界。"""
    values = torch.tensor([[-34, -17, -16, 0, 16, 17]], dtype=torch.int64)
    reduced = canonical_mod_q(values, 17)
    assert reduced.tolist() == [[0, 0, 1, 0, 16, 0]]
    lifted = centered_lift_tensor(torch.tensor([[8, 9]], dtype=torch.int64), 17)
    assert lifted.tolist() == [[8, -8]]


def test_g_backend_matches_g1a_reference_for_mixed_batch() -> None:
    """验证 H/G 的 accepted、reason、residual 和原始索引完全一致。"""
    parameters, secret, _ = generate_parameters(G1AConfig(q=17, n=3, m=4, tau=2, master_seed=7))
    valid = np.asarray(secret, dtype="<i8")
    tampered = valid.copy()
    tampered[0] = (int(tampered[0]) + 1) % parameters.config.q
    batch = np.stack([valid, tampered, valid])
    host = verify_batch(batch.tolist(), parameters, batch_indices=[2, 5, 9])
    graph = verify_tensor_batch(batch, parameters, batch_indices=[2, 5, 9])
    assert graph.evidence == host.evidence
    assert graph.residuals == host.residuals
    assert graph.raw_values == host.raw_values


def test_module_has_no_parameters_or_grad_path() -> None:
    """验证 module 不含可训练参数且不会构建 autograd 图。"""
    parameters, secret, _ = generate_parameters(G1AConfig(q=17, n=3, m=4, tau=2, master_seed=7))
    module = ModIntNeuralVerifier(parameters)
    assert tuple(module.parameters()) == ()
    assert all(not parameter.requires_grad for parameter in module.parameters())
    output = module(torch.from_numpy(np.asarray([secret], dtype="<i8")))
    assert output.evidence.accepted == (True,)
    assert not output.evidence.accepted[0] is None


def test_g_backend_rejects_float_bool_empty_and_wrong_device_inputs() -> None:
    """验证结构、dtype 和 device 错误均 fail-closed。"""
    parameters = _parameters()
    invalid = [
        torch.zeros((0, 3), dtype=torch.int64),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.bool),
        torch.zeros((1, 2), dtype=torch.int64),
        [[0.0, 0.0, 0.0]],
        [[-1, 0, 0]],
        [[17, 0, 0]],
    ]
    for value in invalid:
        with pytest.raises(G1BError):
            verify_tensor_batch(value, parameters)
    if torch.cuda.is_available():
        with pytest.raises(G1BError):
            verify_tensor_batch(torch.zeros((1, 3), dtype=torch.int64, device="cuda"), parameters)


def test_indices_must_be_strictly_increasing() -> None:
    """验证 batch 索引不能被调用方伪造或重排。"""
    with pytest.raises(G1BError):
        verify_tensor_batch(torch.zeros((2, 3), dtype=torch.int64), _parameters(), [3, 1])
    with pytest.raises(G1BError):
        verify_tensor_batch(torch.zeros((2, 3), dtype=torch.int64), _parameters(), [1, 1])


def test_kernel_rejects_noncanonical_mod_inputs() -> None:
    """验证 mod/lift kernel 不接受浮点、非 contiguous 或域外输入。"""
    with pytest.raises(G1BError):
        canonical_mod_q(torch.zeros((1, 1), dtype=torch.float32), 17)
    with pytest.raises(G1BError):
        centered_lift_tensor(torch.tensor([[17]], dtype=torch.int64), 17)
    with pytest.raises(G1BError):
        canonical_mod_q(torch.zeros((2, 2), dtype=torch.int64).transpose(0, 1), 17)


def test_cpu_artifact_runner_and_manifest_are_reproducible(tmp_path: Path) -> None:
    """验证 H/G runner 生成交付物并拒绝 manifest 摘要篡改。"""
    output = tmp_path / "run"
    summary = generate_g1b_artifacts(output)
    assert summary["status"] == "complete"
    assert summary["h_g_difference_count"] == 0
    manifest_path = output / "manifest.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    payload = load_g1b_manifest(manifest_path, digest)
    assert payload["backend_id"] == "g1b-torch-int64-cpu-v1"
    assert payload["integer_bounds"]["formula_version"] == "g1b-bounds-v1"
    with pytest.raises(FileExistsError):
        generate_g1b_artifacts(output)


def test_kernel_negative_and_overflow_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖 kernel 的 q、device、NumPy dtype、转换和 digest 错误分支。"""
    with pytest.raises(G1BError):
        canonical_mod_q(torch.zeros((1, 1), dtype=torch.int64), 0)
    with pytest.raises(G1BError):
        centered_lift_tensor(torch.zeros((1, 1), dtype=torch.int64), 16)
    with pytest.raises(G1BError):
        canonical_mod_q(torch.zeros((1, 1), dtype=torch.int64, device="meta"), 17)
    with pytest.raises(G1BError):
        verify_tensor_batch(np.zeros((1, 3), dtype=np.float64), _parameters())
    with pytest.raises(G1BError):
        verify_tensor_batch([[0, 0, 0]], _parameters(), [0, 0])
    original_tensor = kernel_module.torch.tensor
    monkeypatch.setattr(kernel_module.torch, "tensor", lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("injected")))
    with pytest.raises(G1BError):
        verify_tensor_batch([[0, 0, 0]], _parameters())
    monkeypatch.setattr(kernel_module.torch, "tensor", original_tensor)
    monkeypatch.setattr(kernel_module, "INT64_MAX", 0)
    with pytest.raises(G1BError):
        derive_integer_bounds(_parameters())
    with pytest.raises(G1BError):
        kernel_module._residual_digest(((2**63,),))


def test_manifest_negative_paths(tmp_path: Path) -> None:
    """覆盖 G1-b manifest 的摘要、backend、bounds、provenance 和 JSON 错误。"""
    output = tmp_path / "run"
    generate_g1b_artifacts(output)
    path = output / "manifest.json"
    valid = json.loads(path.read_text(encoding="utf-8"))
    cases = [{}]
    for key, value in (("backend_id", "bad"), ("dtype", "float"), ("device", "cuda")):
        item = dict(valid)
        item[key] = value
        cases.append(item)
    item = dict(valid)
    item["integer_bounds"] = {"q": 1}
    cases.append(item)
    item = dict(valid)
    item["provenance"] = {"dirty": False}
    cases.append(item)
    item = dict(valid)
    item["provenance"] = dict(valid["provenance"])
    item["provenance"]["dirty"] = "yes"
    cases.append(item)
    for payload in cases:
        with pytest.raises(G1BError):
            validate_g1b_manifest(payload)
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(G1BError):
        load_g1b_manifest(path)
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(G1BError):
        load_g1b_manifest(path, "0" * 64)
    with pytest.raises(G1BError):
        build_g1b_manifest("0", derive_integer_bounds(_parameters()), "0" * 64, valid["provenance"])


def test_types_and_artifact_failure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """覆盖不可变结果校验和 runner 的 Git/差分失败分支。"""
    bounds = derive_integer_bounds(_parameters())
    with pytest.raises(G1BError):
        IntegerBounds(1, 1, 1, -1, 1, 1, 1)
    with pytest.raises(G1BError):
        IntegerBounds(1, 1, 1, 1, 1, 1, 1, "")
    result = verify_tensor_batch(torch.zeros((1, 3), dtype=torch.int64), _parameters())
    with pytest.raises(G1BError):
        G1BResult(result.evidence, result.residuals, result.raw_values, bounds, "bad-backend")
    monkeypatch.setattr(artifact_module.subprocess, "check_output", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected")))
    monkeypatch.setattr(artifact_module.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected")))
    assert artifact_module._git_commit() == "0" * 64
    assert artifact_module._git_dirty() is True
    original = artifact_module.verify_tensor_batch
    monkeypatch.setattr(artifact_module, "verify_tensor_batch", lambda *args, **kwargs: replace(original(*args, **kwargs), raw_values=((999,),)))
    with pytest.raises(RuntimeError, match="G1B_H_G_MISMATCH"):
        generate_g1b_artifacts(tmp_path / "failed-run")


def test_manifest_builder_and_module_property_error_paths() -> None:
    """补齐 manifest 构造和 module 参数属性分支。"""
    parameters = _parameters()
    module = ModIntNeuralVerifier(parameters)
    assert module.parameters_ref is parameters
    bounds = derive_integer_bounds(parameters)
    provenance = {"git_commit": "c" * 40, "dirty": False, "python": "3.11", "torch": "2"}
    with pytest.raises(G1BError):
        build_g1b_manifest("0" * 64, bounds, "0" * 64, {"git_commit": "c", "dirty": False, "python": "3.11", "torch": "2"})
    with pytest.raises(G1BError):
        build_g1b_manifest("0" * 64, bounds, "0" * 64, {"git_commit": "c" * 40, "dirty": "no", "python": "3.11", "torch": "2"})
    valid = build_g1b_manifest("0" * 64, bounds, "0" * 64, provenance)
    valid["schema_version"] = 2
    with pytest.raises(G1BError):
        validate_g1b_manifest(valid)
