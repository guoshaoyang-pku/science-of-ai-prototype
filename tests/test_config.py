from pathlib import Path

import pytest

from aiq_kb import KBConfig


def test_defaults_are_outside_architectureiq_data() -> None:
    config = KBConfig.from_env({})
    assert config.data_root.name == "AIQ_KB_DATA"
    assert config.runs_root == config.data_root / "runs"
    assert config.bench_root.name in {"ArchitectureIQ", "aiq_bench_repo"}
    assert config.blog_root is None
    assert config.keys_file == config.bench_root / "data/evals/eval_keys.json"


def test_environment_paths_and_run_name() -> None:
    config = KBConfig.from_env(
        {
            "AIQ_KB_DATA_ROOT": "/tmp/aiq-kb-data",
            "AIQ_BENCH_ROOT": "/tmp/bench",
            "AIQ_KB_BLOG_ROOT": "/tmp/blog",
            "AIQ_KB_KEYS": "/tmp/keys.json",
            "AIQ_KB_POOL": "/tmp/pool/questions.jsonl",
        }
    )
    assert config.run_root("example") == Path("/tmp/aiq-kb-data/runs/example").resolve()
    assert config.require_blog() == Path("/tmp/blog").resolve()
    assert config.pool_file == Path("/tmp/pool/questions.jsonl").resolve()
    for name in ["", ".", "..", "../legacy", "/tmp/outside"]:
        with pytest.raises(ValueError):
            config.run_root(name)


def test_installed_workdir_is_explicit_and_resolution_creates_nothing(tmp_path: Path) -> None:
    workdir = tmp_path / "unused"
    config = KBConfig.from_env({"AIQ_KB_WORKDIR": str(workdir)})
    assert config.data_root == workdir / "AIQ_KB_DATA"
    bundled = Path(__file__).resolve().parents[1] / "external/ArchitectureIQ"
    assert config.bench_root == (bundled if bundled.is_dir() else workdir / "ArchitectureIQ/aiq_bench_repo")
    assert not workdir.exists()
