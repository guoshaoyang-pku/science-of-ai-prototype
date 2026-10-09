"""Read-only inspection of an existing KB run.

This module deliberately does not import or start the solver loop. It is the
safe bridge for checking legacy artifacts while the old publisher continues to
own its original path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_run(run: str | Path, pool: str | Path | None = None) -> dict[str, Any]:
    """Return bounded metadata and hashes without modifying *run*.

    Reads configuration, directory names, and an optional pool hash. It never
    opens locks, starts workers, or writes a checkpoint.
    """
    root = Path(run).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    config_path = root / "config.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    snapshots = sorted((root / "kb").glob("kb_*.json"))
    epochs = sorted(p for p in (root / "epochs").glob("e*") if p.is_dir())
    jobs = sorted((root / "jobs").glob("J*/job.json"))
    result: dict[str, Any] = {
        "run": str(root),
        "run_name": root.name,
        "config_present": config_path.is_file(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.is_file() else None,
        "epochs": len(epochs),
        "epoch_names": [p.name for p in epochs],
        "kb_snapshots": len(snapshots),
        "latest_kb": str(snapshots[-1]) if snapshots else None,
        "jobs": len(jobs),
        "publisher_descriptor": (root / "publisher_process.json").is_file(),
        "read_only": True,
    }
    if pool is not None:
        pool_path = Path(pool).expanduser().resolve()
        result["pool"] = str(pool_path)
        result["pool_sha256"] = hashlib.sha256(pool_path.read_bytes()).hexdigest()
        result["config_pool_sha256"] = config.get("pool_sha256")
        result["pool_hash_matches_config"] = config.get("pool_sha256") == result["pool_sha256"]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pool", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(inspect_run(args.run, args.pool), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
