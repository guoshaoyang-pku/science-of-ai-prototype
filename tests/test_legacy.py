import json
from pathlib import Path

from aiq_kb.legacy import inspect_run


def test_inspect_run_is_bounded_and_read_only(tmp_path: Path) -> None:
    run = tmp_path / "legacy"
    (run / "kb").mkdir(parents=True)
    (run / "epochs" / "e0001").mkdir(parents=True)
    (run / "jobs" / "J1").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"pool_sha256": "x"}))
    (run / "kb" / "kb_0001.json").write_text("{}")
    (run / "jobs" / "J1" / "job.json").write_text("{}")
    before = sorted(p.relative_to(run).as_posix() for p in run.rglob("*"))
    result = inspect_run(run)
    after = sorted(p.relative_to(run).as_posix() for p in run.rglob("*"))
    assert result["read_only"] is True
    assert result["epochs"] == 1
    assert result["kb_snapshots"] == 1
    assert result["jobs"] == 1
    assert before == after
