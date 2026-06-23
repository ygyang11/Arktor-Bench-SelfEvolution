from __future__ import annotations

from pathlib import Path

from arktor_bench.grading.validate import validate_static

# Oracle/null (task-validity) is gated by the CLI `arktor-bench validate <pack> --judge`
# (author/nightly, per spec §10/§14) — it needs a real workspace + judge, so it is not a
# unit test.


def test_static_flags_bad_spec(tmp_path: Path) -> None:
    # missing grader.py / criteria mismatch -> load() fails -> not ok
    d = tmp_path / "T_bad"
    d.mkdir()
    (d / "task.md").write_text(
        "---\nid: T_bad\nname: n\nlabels:\n  domain: software_engineering\n"
        "  subdomain: cli\n  model_capability: [code]\n  harness_focus: [instructions]\n"
        "  complexity: linear\nworkspace_files: []\n---\n\n## Prompt\n\nDo.\n\n"
        "## Auto Checks\n\n- {id: c_ok, desc: d, weight: 1}\n")
    r = validate_static(d)
    assert not r.ok and r.issues


def test_static_ok_on_valid_task(tmp_pack: Path) -> None:
    r = validate_static(tmp_pack / "T001_min")
    assert r.ok and not r.issues
