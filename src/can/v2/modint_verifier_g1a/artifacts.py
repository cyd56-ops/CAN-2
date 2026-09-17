"""G1-a reference 交付物生成和摘要工具。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .manifest import build_manifest
from .parameters import generate_parameters, parameter_digest
from .reference import canonical_mod, centered_lift, verify_batch
from .types import G1AConfig


def _sha256_bytes(data: bytes) -> str:
    """计算原始 bytes 的 SHA-256。"""
    return hashlib.sha256(data).hexdigest()


def _git_commit() -> str:
    """读取当前 Git commit；非 Git 环境返回全零占位并由 dirty 标记说明。"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 64


def _git_dirty() -> bool:
    """读取工作树是否存在未提交改动。"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # 无法确认来源状态时 fail-closed，避免把未知状态标成 clean。
        return True
    return bool(result.stdout.strip())


def generate_g1a_artifacts(output_dir: Path, config: Optional[G1AConfig] = None) -> Dict[str, Any]:
    """生成不含 secret/credential 的 G1-a CPU reference 交付物。"""
    config = config or G1AConfig(q=17, n=3, m=4, tau=2, master_seed=20260915)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    parameters, secret, error = generate_parameters(config)
    valid = np.asarray([secret], dtype="<i8")
    tampered = valid.copy()
    tampered[0, 0] = (int(tampered[0, 0]) + 1) % config.q
    started = time.perf_counter()
    valid_result = verify_batch(valid, parameters)
    tampered_result = verify_batch(tampered, parameters)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    np.save(output_dir / "A.npy", parameters.A)
    np.save(output_dir / "b.npy", parameters.b)
    vectors = {
        "schema_version": 1,
        "config": {"q": config.q, "n": config.n, "m": config.m, "tau": config.tau, "norm": config.norm},
        "boundary": {"centered_q_half": centered_lift(config.q // 2, config.q), "centered_q_half_plus_one": centered_lift(config.q // 2 + 1, config.q)},
        "negative_mod": {"input": -5, "q": 3, "output": canonical_mod(-5, 3)},
        "cases": [
            {"case_id": "valid-generated", "credential_sha256": _sha256_bytes(valid.tobytes()), "accepted": valid_result.evidence.accepted[0], "max_abs_residual": valid_result.evidence.max_abs_residual[0]},
            {"case_id": "tampered-credential", "credential_sha256": _sha256_bytes(tampered.tobytes()), "accepted": tampered_result.evidence.accepted[0], "max_abs_residual": tampered_result.evidence.max_abs_residual[0]},
        ],
        "secret_digest": _sha256_bytes(secret.tobytes()),
        "error_digest": _sha256_bytes(error.tobytes()),
    }
    vectors_bytes = json.dumps(vectors, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (output_dir / "vectors.json").write_bytes(vectors_bytes)
    provenance = {"git_commit": _git_commit(), "dirty": _git_dirty(), "python": platform.python_version(), "numpy": np.__version__}
    manifest = build_manifest(config, parameters, _sha256_bytes(vectors_bytes), _sha256_bytes(Path(__file__).read_bytes()), provenance)
    manifest_bytes = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    summary = {"schema_version": 1, "status": "complete", "exit_code": 0, "protocol_id": config.protocol_id, "execution_config_id": config.execution_config_id, "parameter_sha256": parameter_digest(parameters), "vectors_sha256": _sha256_bytes(vectors_bytes), "elapsed_ms": elapsed_ms, "boundary_checks": vectors["boundary"], "cases": vectors["cases"], "provenance": provenance}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
    return summary
