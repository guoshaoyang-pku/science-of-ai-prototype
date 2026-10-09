"""Filesystem paths for the standalone KB runtime; resolution performs no I/O writes."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _workdir() -> Path:
    checkout = Path(__file__).resolve().parents[2]
    return checkout.parent if (checkout / "pyproject.toml").is_file() else Path.cwd()


def _path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class KBConfig:
    """Filesystem contract shared by KB commands.

    blog_root is optional because solving and curation must work without a
    publisher checkout. keys_file is resolved for discovery only; callers
    still load credentials at runtime and must never copy it into this repo.
    """

    data_root: Path
    bench_root: Path
    blog_root: Path | None
    keys_file: Path
    pool_file: Path

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "KBConfig":
        env = os.environ if environ is None else environ
        workdir = _path(env.get("AIQ_KB_WORKDIR", str(_workdir())))
        data_root = _path(env.get("AIQ_KB_DATA_ROOT", str(workdir / "AIQ_KB_DATA")))
        bundled = Path(__file__).resolve().parents[2] / "external/ArchitectureIQ"
        bench_default = bundled if bundled.is_dir() else workdir / "ArchitectureIQ/aiq_bench_repo"
        bench_root = _path(env.get("AIQ_BENCH_ROOT", str(bench_default)))
        blog_value = env.get("AIQ_KB_BLOG_ROOT", "").strip()
        blog_root = _path(blog_value) if blog_value else None
        keys_file = _path(
            env.get("AIQ_KB_KEYS", str(bench_root / "data" / "evals" / "eval_keys.json"))
        )
        pool_file = _path(env.get(
            "AIQ_KB_POOL",
            str(bench_root / "releases" / "kb_science_pool_v1" / "questions.jsonl"),
        ))
        return cls(
            data_root=data_root,
            bench_root=bench_root,
            blog_root=blog_root,
            keys_file=keys_file,
            pool_file=pool_file,
        )

    def run_root(self, run_name: str) -> Path:
        """Return a run path below the configured data root."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
            raise ValueError("run_name must contain only letters, digits, underscores or hyphens")
        return self.runs_root / run_name

    def require_blog(self) -> Path:
        """Return the publisher checkout or fail with an actionable error."""
        if self.blog_root is None:
            raise RuntimeError("publishing requires AIQ_KB_BLOG_ROOT")
        return self.blog_root

    @property
    def runs_root(self) -> Path:
        return self.data_root / "runs"

    @property
    def lab_root(self) -> Path:
        return self.data_root / "lab"


__all__ = ["KBConfig"]
