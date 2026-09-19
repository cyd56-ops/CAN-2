"""P0-MoE registry、fixture、fake-host 和 artifact 契约测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from can.v2.pretrained_moe_p0 import (
    ArtifactWriter,
    FakeHostAdapter,
    P0Error,
    P0Runner,
)
from can.v2.pretrained_moe_p0 import artifacts as artifact_module
from can.v2.pretrained_moe_p0 import (
    inspect_host,
    validate_fixture,
    validate_registry,
)
from can.v2.pretrained_moe_p0.fixture import load_fixture, normalized_em, strict_em
from can.v2.pretrained_moe_p0.registry import load_registry


def _registry_payload(count: int = 3) -> dict:
    """构造不含真实模型信息的最小 registry。"""
    candidates = []
    for index in range(1, count + 1):
        candidates.append(
            {
                "candidate_id": f"C{index}",
                "repository_id": f"org/model-{index}",
                "attempt_order": index,
                "requested_revision": "main",
                "resolved_commit_sha": ("a" * 40),
                "profile": {
                    "profile_id": "bf16-v1",
                    "dtype": "bf16",
                    "quantization_config": {},
                    "allow_remote_code": False,
                },
                "expected_architecture_family": "moe",
                "expected_moe_variant": "unknown_requires_verification",
                "max_snapshot_bytes": 1024,
                "license_review_status": "approved",
                "metadata_source_sha256": ("b" * 64),
                "p0a_status": "passed",
                "p0a_decision_sha256": ("c" * 64),
                "p0a_failure_codes": [],
            }
        )
    return {
        "schema_version": 1,
        "registry_id": "p0-moe-host-v1",
        "candidates": candidates,
    }


def _case(case_id: str, group: str) -> dict:
    """构造一条公开 fixture 并计算内容摘要。"""
    item = {
        "case_id": case_id,
        "group": group,
        "system_text": "Answer with only the requested text. Do not explain.",
        "user_text": f"The code for item {case_id} is CODE-{case_id}. Return the code.",
        "expected_text": f"CODE-{case_id}",
        "metric": "strict_em" if group == "format_copy" else "normalized_em",
        "max_new_tokens": 24,
    }
    if group != "format_copy":
        item["fact_source"] = "stated_in_prompt"
        item["rationale"] = "答案在 user_text 中逐字给出。"
    body = json.dumps(
        item, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    item["content_sha256"] = hashlib.sha256(body).hexdigest()
    return item


def _fixture_payload() -> dict:
    """构造三组各八条的固定 fixture。"""
    cases = []
    for group in ("format_copy", "single_hop", "two_hop"):
        cases.extend(_case(f"{group}-{index}", group) for index in range(8))
    return {
        "schema_version": 1,
        "fixture_id": "p0-moe-public-fixture-v1",
        "cases": cases,
    }


def test_registry_accepts_fixed_order_and_profile() -> None:
    """验证 registry 正向解析与不可变候选顺序。"""
    candidates = validate_registry(_registry_payload())
    assert [candidate.candidate_id for candidate in candidates] == ["C1", "C2", "C3"]
    assert candidates[0].profile.profile_id == "bf16-v1"


@pytest.mark.parametrize(
    "field",
    ["resolved_commit_sha", "metadata_source_sha256", "p0a_decision_sha256"],
)
def test_registry_rejects_bad_digest_or_revision(field: str) -> None:
    """验证摘要和不可变 revision 失败关闭。"""
    payload = _registry_payload()
    payload["candidates"][0][field] = "bad"
    with pytest.raises(P0Error):
        validate_registry(payload)


def test_registry_rejects_non_contiguous_order_and_unknown_profile() -> None:
    """验证候选顺序与 profile 类型校验。"""
    payload = _registry_payload()
    payload["candidates"][1]["attempt_order"] = 4
    with pytest.raises(P0Error, match="registry_order_invalid"):
        validate_registry(payload)
    payload = _registry_payload()
    payload["candidates"][0]["profile"]["dtype"] = "int4"
    with pytest.raises(P0Error):
        validate_registry(payload)


@pytest.mark.parametrize(
    "mutator,code",
    [
        (lambda p: p.update(schema_version=2), "artifact_schema_mismatch"),
        (lambda p: p.update(candidates=[]), "artifact_schema_mismatch"),
        (
            lambda p: p["candidates"][0].update(repository_id=""),
            "artifact_schema_mismatch",
        ),
        (
            lambda p: p["candidates"][0].update(requested_revision=""),
            "artifact_schema_mismatch",
        ),
        (
            lambda p: p["candidates"][0].update(max_snapshot_bytes=0),
            "artifact_schema_mismatch",
        ),
        (
            lambda p: p["candidates"][0]["profile"].update(profile_id=""),
            "artifact_schema_mismatch",
        ),
        (
            lambda p: p["candidates"][0]["profile"].update(allow_remote_code=1),
            "artifact_schema_mismatch",
        ),
        (
            lambda p: p["candidates"][0].update(license_review_status="unknown"),
            "license_unresolved",
        ),
        (
            lambda p: p["candidates"][0].update(p0a_status="pending"),
            "p0a_decision_invalid",
        ),
        (
            lambda p: p["candidates"][0].update(
                p0a_status="rejected", p0a_failure_codes=[]
            ),
            "p0a_decision_invalid",
        ),
        (
            lambda p: p["candidates"][0].update(
                p0a_status="passed", p0a_failure_codes=["failure"]
            ),
            "p0a_decision_invalid",
        ),
        (
            lambda p: p["candidates"][0].update(candidate_id=""),
            "registry_order_invalid",
        ),
    ],
)
def test_registry_rejects_schema_variants(mutator, code: str) -> None:
    """覆盖 registry 的 fail-closed 字段和类型分支。"""
    payload = _registry_payload()
    mutator(payload)
    with pytest.raises(P0Error, match=code):
        validate_registry(payload)


def test_registry_rejects_duplicate_json_and_load_digest(tmp_path: Path) -> None:
    """覆盖重复字段、文件不存在、摘要漂移和非法 JSON。"""
    path = tmp_path / "registry.json"
    raw = b'{"schema_version":1,"schema_version":1}'
    path.write_bytes(raw)
    with pytest.raises(P0Error, match="artifact_duplicate_field"):
        load_registry(path)
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(P0Error, match="artifact_json_invalid"):
        load_registry(path)
    path.write_text(json.dumps(_registry_payload()), encoding="utf-8")
    with pytest.raises(P0Error, match="snapshot_digest_mismatch"):
        load_registry(path, "a" * 64)
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "missing.json")
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_registry_payload()), encoding="utf-8")
    loaded = load_registry(valid, hashlib.sha256(valid.read_bytes()).hexdigest())
    assert len(loaded) == 3


def test_fixture_accepts_three_groups_and_prompt_fact_metadata() -> None:
    """验证 24 条 fixture 和公开事实标注。"""
    cases = validate_fixture(_fixture_payload())
    assert len(cases) == 24
    assert sum(case.group == "single_hop" for case in cases) == 8
    assert cases[8].fact_source == "stated_in_prompt"


def test_fixture_rejects_missing_rationale_or_tampered_digest() -> None:
    """验证事实来源和内容摘要不能被静默修改。"""
    payload = _fixture_payload()
    del payload["cases"][8]["rationale"]
    with pytest.raises(
        P0Error, match="artifact_schema_mismatch|fixture_fact_source_invalid"
    ):
        validate_fixture(payload)
    payload = _fixture_payload()
    payload["cases"][0]["expected_text"] = "OTHER"
    with pytest.raises(P0Error, match="fixture_digest_mismatch"):
        validate_fixture(payload)


@pytest.mark.parametrize(
    "mutator,code",
    [
        (lambda p: p.update(schema_version=2), "artifact_schema_mismatch"),
        (lambda p: p.update(cases=p["cases"][:1]), "fixture_group_invalid"),
        (lambda p: p["cases"].__setitem__(1, "bad"), "fixture_schema_mismatch"),
        (
            lambda p: (
                p["cases"][0].update(case_id=p["cases"][1]["case_id"]),
                p["cases"][0].update(
                    content_sha256=hashlib.sha256(
                        json.dumps(
                            {
                                key: p["cases"][0][key]
                                for key in sorted(p["cases"][0])
                                if key != "content_sha256"
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                ),
            ),
            "fixture_duplicate_case",
        ),
        (lambda p: p["cases"][0].update(expected_text=""), "fixture_content_invalid"),
        (lambda p: p["cases"][0].update(max_new_tokens=25), "fixture_content_invalid"),
        (
            lambda p: p["cases"][0].update(metric="normalized_em"),
            "fixture_metric_invalid",
        ),
        (
            lambda p: p["cases"][8].update(fact_source="external"),
            "fixture_fact_source_invalid",
        ),
        (lambda p: p["cases"][0].update(group="unknown"), "fixture_group_invalid"),
    ],
)
def test_fixture_rejects_schema_variants(mutator, code: str) -> None:
    """覆盖 fixture 的组、文本、指标和事实来源错误分支。"""
    payload = _fixture_payload()
    mutator(payload)
    with pytest.raises(P0Error, match=code):
        validate_fixture(payload)


def test_fixture_load_digest_and_json_errors(tmp_path: Path) -> None:
    """覆盖 fixture 文件层摘要、JSON 和路径错误。"""
    path = tmp_path / "fixture.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(P0Error, match="artifact_json_invalid"):
        load_fixture(path)
    path.write_text(json.dumps(_fixture_payload()), encoding="utf-8")
    with pytest.raises(P0Error, match="snapshot_digest_mismatch"):
        load_fixture(path, "a" * 64)
    with pytest.raises(FileNotFoundError):
        load_fixture(tmp_path / "missing.json")
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_fixture_payload()), encoding="utf-8")
    assert (
        len(load_fixture(valid, hashlib.sha256(valid.read_bytes()).hexdigest())) == 24
    )


def test_em_normalizers_are_frozen() -> None:
    """验证 strict/normalized EM 的边界规范。"""
    assert strict_em(" Café\n", " Café\r\n")
    assert normalized_em("The Paris, city", "paris city")
    assert not strict_em("Code-A", "code-a")


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("native", True),
        ("dense", False),
        ("no_shared", False),
        ("post_dispatch_mask", False),
        ("no_row_mask", False),
        ("no_counter", False),
        ("topk2", False),
        ("batch_reorder", False),
        ("kv_unbound", False),
        ("state_mutates", False),
    ],
)
def test_fake_host_structure_gates(profile: str, expected: bool) -> None:
    """验证结构门对 dense、越权和 cache 风险 fail-closed。"""
    assert inspect_host(FakeHostAdapter(profile)).passed is expected


def test_fake_host_rejects_unknown_profile_and_dense_has_no_hooks() -> None:
    """覆盖未知 profile 和 dense 无 hook 分支。"""
    with pytest.raises(P0Error, match="host_interface_not_controllable"):
        FakeHostAdapter("unknown").capabilities()
    assert inspect_host(FakeHostAdapter("dense")).hook_points == ()


def test_runner_selects_first_passing_candidate_and_exhausts() -> None:
    """验证固定顺序选择和全部失败的 no_suitable_host。"""
    candidates = validate_registry(_registry_payload())
    result = P0Runner(candidates).run_structure_only(
        {"C1": FakeHostAdapter("dense"), "C2": FakeHostAdapter("native")}
    )
    assert result["status"] == "selected" and result["selected_candidate"] == "C2"
    result = P0Runner(candidates).run_structure_only(
        {candidate.candidate_id: FakeHostAdapter("dense") for candidate in candidates}
    )
    assert result["status"] == "no_suitable_host"
    with pytest.raises(P0Error, match="registry_order_invalid"):
        P0Runner([])
    with pytest.raises(P0Error, match="registry_order_invalid"):
        P0Runner([candidates[1], candidates[2]])
    missing = P0Runner(candidates).run_structure_only({})
    assert missing["status"] == "no_suitable_host"


def test_runner_skips_p0a_rejected_candidate() -> None:
    """验证 P0-A 拒绝候选即使提供 adapter 也不得运行。"""

    payload = _registry_payload()
    payload["candidates"][0].update(
        p0a_status="rejected",
        p0a_failure_codes=["native_shared_expert_missing"],
        license_review_status="rejected",
    )
    candidates = validate_registry(payload)
    result = P0Runner(candidates).run_structure_only(
        {"C1": FakeHostAdapter("native"), "C2": FakeHostAdapter("native")}
    )
    assert result["selected_candidate"] == "C2"
    assert result["attempts"][0] == {
        "candidate_id": "C1",
        "status": "not_run",
        "reason": "p0a_rejected",
        "failure_codes": ["native_shared_expert_missing"],
    }


def test_artifact_writer_is_atomic_and_non_overwriting(tmp_path: Path) -> None:
    """验证 artifact 原子写入和目录不可覆盖。"""
    output = tmp_path / "run"
    writer = ArtifactWriter(output)
    digest = writer.write_json("summary.json", {"status": "complete"})
    assert digest == hashlib.sha256((output / "summary.json").read_bytes()).hexdigest()
    assert writer.write_jsonl("events.jsonl", [{"case_id": "c0"}])
    with pytest.raises(P0Error, match="artifact_exists"):
        ArtifactWriter(output)


def test_artifact_writer_rejects_names_and_cleans_failed_write(
    tmp_path: Path, monkeypatch
) -> None:
    """覆盖 artifact 文件名和 os.replace 失败清理分支。"""
    writer = ArtifactWriter(tmp_path / "run")
    with pytest.raises(P0Error, match="artifact_schema_mismatch"):
        writer.write_json("../bad.json", {})
    with pytest.raises(P0Error, match="artifact_schema_mismatch"):
        writer.write_jsonl("events.json", [])

    def fail_replace(source, target):
        raise OSError("injected")

    monkeypatch.setattr(artifact_module.os, "replace", fail_replace)
    with pytest.raises(P0Error, match="artifact_write_failed"):
        writer.write_json("failed.json", {})
    assert not (writer.output_dir / ".failed.json.tmp").exists()
    with pytest.raises(P0Error, match="artifact_schema_mismatch"):
        ArtifactWriter(tmp_path / "run2").write_json("bad.txt", {})
    with pytest.raises(TypeError):
        ArtifactWriter("not-a-path")
    target = writer.output_dir / "existing.json"
    target.write_bytes(b"old")
    with pytest.raises(P0Error, match="artifact_exists"):
        writer.write_json("existing.json", {})
    temp = writer.output_dir / ".temp.json.tmp"
    temp.write_bytes(b"old")
    with pytest.raises(P0Error, match="artifact_exists"):
        writer.write_json("temp.json", {})
    temp.unlink()

    def fail_write(self, raw):
        raise OSError("injected before temp exists")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    with pytest.raises(P0Error, match="artifact_write_failed"):
        writer.write_json("write-failed.json", {})
