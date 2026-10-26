"""Explicit scheduling amendment; frozen model inputs and request budgets stay intact.

Stop the original runner with its STOP file and wait for its lock to release.
Preregister execution-amendment.json, then explicitly remove STOP before running
``python -m aiq_kb.frozen_execution --run-dir DIR``. No scores are read here.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import re

from . import frozen_eval as ev


FIELDS = {"schema_version", "parent_registration_sha256", "frozen_eval_source_sha256",
          "controller_source_sha256", "workers", "reason", "basis", "labels_unopened"}


def run_execution(run_dir, *, transport=None, amendment_filename="execution-amendment.json"):
    run = Path(run_dir).resolve()
    if not re.fullmatch(r"execution-amendment(?:-[A-Za-z0-9_-]+)?\.json", amendment_filename):
        raise ValueError("invalid execution amendment filename")
    stem = Path(amendment_filename).stem
    with ev._run_lock(run):
        if (run / "STOP").exists():
            raise ev.EvaluationStopped("STOP present; the controller never removes it")
        if not (run / "freeze.json").exists():
            raise ValueError("execution amendment requires an existing frozen evaluation")
        registration, jobs, lookups, freeze_hash = ev._contract(run)
        raw = (run / amendment_filename).read_bytes()
        amendment = json.loads(raw)
        if not isinstance(amendment, dict) or set(amendment) != FIELDS:
            raise ValueError("execution amendment fields permit only a scheduling revision")
        if type(amendment["schema_version"]) is not int or amendment["schema_version"] != 1:
            raise ValueError("unsupported execution amendment schema")
        if (amendment["basis"] != "latency_only" or amendment["labels_unopened"] is not True or
            not isinstance(amendment["reason"], str) or not amendment["reason"].strip()):
            raise ValueError("amendment must attest latency-only reasoning and unopened labels")
        if any((run / name).exists() for name in ("summary.json", "scores.jsonl", "per_question.jsonl")):
            raise ValueError("scoring artifacts exist; refuse a labels-unopened amendment")
        hashes = {"parent_registration_sha256": ev.file_hash(run / "registration.json"),
                  "frozen_eval_source_sha256": ev.file_hash(Path(ev.__file__)),
                  "controller_source_sha256": ev.file_hash(Path(__file__))}
        if any(amendment[name] != value for name, value in hashes.items()):
            raise ValueError("execution amendment source or parent registration hash mismatch")
        workers = amendment["workers"]
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ValueError("amended workers must be an integer from 1 to 64")
        frozen = {**hashes, "amendment_sha256": ev.digest(raw)}
        freeze_path = run / f"{stem}.freeze.json"
        if freeze_path.exists() and ev._json(freeze_path) != frozen:
            raise ValueError("frozen execution amendment changed; refuse resume")
        predictions = ev.load_predictions(run)
        if set(predictions) - {job["key"] for job in jobs}:
            raise ValueError("extra prediction files")
        if not freeze_path.exists():
            ev._write(freeze_path, frozen)
            ev._atomic(run / "source" / f"frozen_execution-{stem}.py", Path(__file__).read_bytes())
        runner = ev._Runner(run, registration, jobs, lookups, freeze_hash, transport or ev.http_transport)
        status = {**frozen, "state": "running", "actual_worker_limit": workers,
                  "original_worker_limit": registration["limits"]["workers"],
                  "completed_before": len(predictions), "requests_before": runner.requests}
        status_path = run / f"{stem}.status.json"
        ev._write(status_path, status)
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                try:
                    list(executor.map(runner.solve, jobs))
                except BaseException:
                    runner.stopped.set()  # stop admission before waiting for in-flight calls
                    raise
        except BaseException as error:
            status.update(state="stopped", error_type=type(error).__name__)
            raise
        else:
            status["state"] = "complete"
        finally:
            status.update(completed_after=len(list((run / "predictions").glob("*.json"))),
                          requests_after=runner.requests, registered_jobs=len(jobs))
            ev._write(status_path, status)
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--amendment-filename", default="execution-amendment.json")
    args = parser.parse_args(argv)
    try:
        result = run_execution(args.run_dir, amendment_filename=args.amendment_filename)
    except (ValueError, ev.EvaluationStopped) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
