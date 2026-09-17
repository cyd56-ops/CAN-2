"""G1-b CPU contract 交付物 runner。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..modint_verifier_g1a import G1AConfig, generate_parameters, parameter_digest, verify_batch
from .kernel import derive_integer_bounds, verify_tensor_batch
from .manifest import build_g1b_manifest


def _git_commit() -> str:
    """读取当前 Git commit，失败时返回未知值。"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 64


def _git_dirty() -> bool:
    """读取工作树 dirty 状态，无法确认时 fail-closed。"""
    try:
        result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _sha_bytes(value: bytes) -> str:
    """计算 bytes 的 SHA-256。"""
    return hashlib.sha256(value).hexdigest()


def generate_g1b_artifacts(output_dir: Path, config: Optional[G1AConfig] = None) -> Dict[str, Any]:
    """运行 H/G CPU 对照并写入不含 raw credential 的交付物。"""
    config = config or G1AConfig(q=17, n=3, m=4, tau=2, master_seed=20260916)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    parameters, secret, _ = generate_parameters(config)
    valid = np.asarray([secret], dtype="<i8")
    tampered = valid.copy()
    tampered[0, 0] = (int(tampered[0, 0]) + 1) % config.q
    batch = np.concatenate([valid, tampered, valid], axis=0)
    started = time.perf_counter()
    host = verify_batch(batch.tolist(), parameters, batch_indices=[0, 4, 8])
    graph = verify_tensor_batch(batch, parameters, batch_indices=[0, 4, 8])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    difference_count = int(host.evidence != graph.evidence or host.residuals != graph.residuals or host.raw_values != graph.raw_values)
    np.save(output_dir / "A.npy", parameters.A)
    np.save(output_dir / "b.npy", parameters.b)
    vectors = {
        "schema_version": 1,
        "source_parameter_sha256": parameter_digest(parameters),
        "bounds": derive_integer_bounds(parameters).to_dict(),
        "cases": [{"case_id": "valid", "credential_sha256": _sha_bytes(valid.tobytes()), "accepted": host.evidence.accepted[0]}, {"case_id": "tampered", "credential_sha256": _sha_bytes(tampered.tobytes()), "accepted": host.evidence.accepted[1]}],
        "h_digest": host.evidence.residual_digest,
        "g_digest": graph.evidence.residual_digest,
        "difference_count": difference_count,
    }
    vectors_bytes = json.dumps(vectors, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (output_dir / "vectors.json").write_bytes(vectors_bytes)
    provenance = {"git_commit": _git_commit(), "dirty": _git_dirty(), "python": platform.python_version(), "torch": torch.__version__}
    manifest = build_g1b_manifest(parameter_digest(parameters), derive_integer_bounds(parameters), _sha_bytes(vectors_bytes), provenance)
    manifest_bytes = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    summary = {"schema_version": 1, "status": "complete" if difference_count == 0 else "failed", "exit_code": 0 if difference_count == 0 else 1, "h_g_difference_count": difference_count, "accepted": list(graph.evidence.accepted), "reason_code": list(graph.evidence.reason_code), "latency_ms": elapsed_ms, "integer_bounds": manifest["integer_bounds"], "provenance": provenance}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
    if difference_count:
        raise RuntimeError("G1B_H_G_MISMATCH")
    return summary
