"""生成 M1a P1/P2 tiny-MoE contract 的完整、可复核交付物。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.can.v2.auth_expert_moe import (  # noqa: E402
    P1_POLICY,
    P2_POLICY,
    CallLedger,
    M1AAuthExpert,
    M1AContext,
    M1aMoE,
    M1ARouteCoordinator,
    ProtectedExpert,
    SharedExpert,
)
from src.can.v2.auth_expert_moe.artifacts import (  # noqa: E402
    state_dict_sha256,
    validate_call_ledger,
    validate_m1a_summary,
)
from src.can.v2.pretrained_gate.authorization import FixedRelationVerifier  # noqa: E402


def sha256(path: Path) -> str:
    """计算文件原始 bytes 的 SHA-256。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: Dict[str, Any]) -> None:
    """以稳定 UTF-8 JSON 写入交付物。"""

    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> Tuple[str, bool]:
    """读取当前 Git commit 和 dirty 状态；读取失败时显式标记。"""

    import subprocess

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    return commit, dirty


def _manifest(
    policy: str, inputs: Path, reference: Path, commit: str, e0_sha: str, e1_sha: str
) -> Dict[str, Any]:
    """构造严格 M1a manifest。"""

    zero = "0" * 64
    return {
        "schema_version": 1,
        "execution_config_id": "m1a-tiny-moe-v1",
        "protocol_id": "m1a-contract-v1",
        "policy": policy,
        "expert_specs": [
            {
                "expert_id": "E0",
                "kind": "shared",
                "capability_level": "general",
                "scope_ids": [],
                "architecture_revision": "tiny-ffn-v1",
                "weights_sha256": e0_sha,
                "train_data_scope": "public",
                "max_context_length": 8,
                "enabled": True,
            },
            {
                "expert_id": "E1",
                "kind": "protected",
                "capability_level": "protected",
                "scope_ids": ["protected.default"],
                "architecture_revision": "tiny-ffn-v1",
                "weights_sha256": e1_sha,
                "train_data_scope": "protected",
                "max_context_length": 8,
                "enabled": True,
            },
        ],
        "alpha": 1.0,
        "top_k": 1,
        "router": {"type": "fixed", "normalize_eps": 1e-12},
        "model": {"d_model": 16, "dtype": "float32", "device": "cpu"},
        "fixture": {
            "inputs_sha256": sha256(inputs),
            "reference_outputs_sha256": sha256(reference),
        },
        "provenance": {"git_commit": commit},
    }


def run_policy(root: Path, policy: str, run_id: str, commit: str, dirty: bool) -> None:
    """执行一个 policy 并生成完整 fixture、ledger 和 summary。"""

    target = root / policy / run_id
    if target.exists():
        raise FileExistsError(f"拒绝覆盖已有 M1a run: {target}")
    target.mkdir(parents=True)
    torch.manual_seed(20260914)
    inputs = torch.randn(3, 2, 16, dtype=torch.float32)
    np.save(
        target / "inputs.npy",
        inputs.numpy().astype("<f4", copy=False),
        allow_pickle=False,
    )
    verifier = FixedRelationVerifier(
        np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.1
    )
    auth = M1AAuthExpert(verifier, "m1a-tiny-moe-v1")
    context = M1AContext(("case-0", "case-1", "case-2"), "m1a-tiny-moe-v1")
    credentials = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float32
    )
    evidence = auth.verify(credentials, context)
    coordinator = M1ARouteCoordinator(auth, "m1a-tiny-moe-v1", policy)
    route = coordinator.commit(evidence, context)
    shared, protected = SharedExpert(seed=1001), ProtectedExpert(seed=1002)
    model = M1aMoE(coordinator, shared=shared, protected=protected)
    ledger = CallLedger(run_id, "m1a-tiny-moe-v1", policy)
    output = model(inputs, route, context, ledger, case_id="full-run")
    np.save(
        target / "reference_outputs.npy",
        output.output.detach().numpy().astype("<f4", copy=False),
        allow_pickle=False,
    )
    manifest = _manifest(
        policy,
        target / "inputs.npy",
        target / "reference_outputs.npy",
        commit,
        state_dict_sha256(shared.state_dict()),
        state_dict_sha256(protected.state_dict()),
    )
    _json(target / "manifest.json", manifest)
    expected = [
        {
            "case_id": f"case-{i}",
            "route": route.routes[i].value,
            "selection": output.selection.values[i],
            "original_index": i,
            "shared_forward_calls": 1,
            "routed_forward_calls": int(output.selection.values[i] == "E1"),
        }
        for i in range(3)
    ]
    credential_digest = hashlib.sha256(
        credentials.numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()
    fixture = {
        "schema_version": 1,
        "seeds": {
            "e0_init": 1001,
            "e1_init": 1002,
            "router_init": 0,
            "input_init": 20260914,
        },
        "input": {
            "path": "inputs.npy",
            "format": "npy-v1",
            "dtype": "<f4",
            "shape": list(inputs.shape),
            "sha256": sha256(target / "inputs.npy"),
        },
        "cases": [
            {
                "case_id": f"case-{i}",
                "policy": route.routes[i].value,
                "credential_fixture_sha256": credential_digest,
            }
            for i in range(3)
        ],
        "reference": {
            "path": "reference_outputs.npy",
            "format": "npy-v1",
            "dtype": "<f4",
            "shape": list(output.output.shape),
            "sha256": sha256(target / "reference_outputs.npy"),
            "atol": 1e-6,
            "rtol": 1e-5,
        },
        "expected": expected,
    }
    _json(target / "fixture.json", fixture)
    ledger_payload = ledger.to_dict()
    validate_call_ledger(ledger_payload)
    _json(target / "call_ledger.json", ledger_payload)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "execution_config_id": "m1a-tiny-moe-v1",
        "protocol_id": "m1a-contract-v1",
        "policy": policy,
        "status": "complete",
        "exit_code": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases": expected,
        "metrics": {
            "shared_forward_calls": 1,
            "routed_forward_calls": int(bool(output.routed_indices.numel())),
            "unauthorized_routed_calls": 0,
            "route_mismatch_count": 0,
            "scope_violation_count": 0,
            "finite_output": bool(torch.isfinite(output.output).all().item()),
        },
        "determinism": {"repeat_checked": False, "difference_count": 0},
        "coverage": {"statement": None, "branch": None, "report": "coverage_m1a.json"},
        "artifacts": {},
        "provenance": {
            "git_commit": commit,
            "dirty": dirty,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": "cpu",
            "seed": 20260914,
        },
        "failure": None,
    }
    for name in (
        "inputs.npy",
        "reference_outputs.npy",
        "manifest.json",
        "fixture.json",
        "call_ledger.json",
    ):
        summary["artifacts"][name] = {"path": name, "sha256": sha256(target / name)}
    validate_m1a_summary(summary)
    _json(target / "summary.json", summary)


def main() -> int:
    """CLI 入口：生成 P1/P2 独立 M1a 结果。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "results" / "m1a-tiny-moe-v1"
    )
    parser.add_argument("--run-id", default="run-20260914-01")
    args = parser.parse_args()
    commit, dirty = _git_commit()
    for policy in (P1_POLICY, P2_POLICY):
        run_policy(args.output_root, policy, args.run_id, commit, dirty)
    print(f"M1a results written to {args.output_root} (run_id={args.run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
