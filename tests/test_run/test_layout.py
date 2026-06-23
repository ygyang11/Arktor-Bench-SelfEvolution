from __future__ import annotations

from pathlib import Path

from arktor_bench.run.layout import iter_cells, read_manifest, write_manifest


def test_iter_cells_walks_three_levels(tmp_path: Path) -> None:
    for h in ("arktor", "codex"):
        for t in ("T1", "T2"):
            for k in ("0", "1"):
                (tmp_path / h / t / k).mkdir(parents=True)
    cells = list(iter_cells(tmp_path))
    assert len(cells) == 8
    c = cells[0]
    assert (c.harness, c.task, c.trial) == ("arktor", "T1", "0")
    assert c.score.name == "score.json" and c.findings.name == "findings.json"


def test_write_manifest_merges_incrementally(tmp_path: Path) -> None:
    write_manifest(tmp_path, "general", ["T1"], 3, ["arktor"])
    write_manifest(tmp_path, "general", ["T2"], 3, ["codex"])
    m = read_manifest(tmp_path)
    assert m.harnesses == ["arktor", "codex"]            # merged + sorted
    assert m.tasks == ["T1", "T2"]
    assert m.trials == 3 and m.pack == "general"
