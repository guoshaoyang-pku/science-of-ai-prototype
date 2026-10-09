#!/usr/bin/env python3
"""Empirical lab for the KB science loop: a measured-outcome database + controlled experiments.

  build  : flatten every locally measured candidate (10-seed GT under aiq_bench_repo/data/datasets) into
           lab_db.jsonl, one row per candidate, EXCLUDING every dataset that belongs to a held-out (val/test)
           group of a given run split (matched by dataset path and by the synthesis seeds quoted in the prompts).
  run    : run a controlled experiment (a JSON job: dataset + list of candidate specs) through the bench's own
           ground-truth pipeline, so new measurements match how the benchmark labels were produced.

Rows are comparable only within one `set_id` (same dataset instance, same total_samples_seen).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

if __package__:
    from .config import KBConfig
else:
    from config import KBConfig

CONFIG = KBConfig.from_env()
ROOT = CONFIG.bench_root.parent
BENCH = CONFIG.bench_root
DATASETS = BENCH / "data/datasets"
POOL = CONFIG.pool_file
LAB = CONFIG.lab_root
SEED_RE = re.compile(r"\b\d{7,}\b")
NAMED_SEED_RE = re.compile(r"\bseed\s*(?:=|:|is)?\s*(\d+)\b", re.I)


def seed_values(value) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if "seed" in str(key).lower():
                found.update(re.findall(r"\b\d+\b", json.dumps(item)))
            elif isinstance(item, (dict, list)):
                found.update(seed_values(item))
    elif isinstance(value, list):
        for item in value:
            found.update(seed_values(item))
    return found


def held_out(split_paths: list[Path], pool: Path = POOL) -> tuple[set[str], set[str]]:
    """Dataset dirs and synthesis seeds of every pool question in a held-out group of any given split."""
    questions = [json.loads(line) for line in Path(pool).read_text().splitlines() if line.strip()]
    by_id = {q["question_id"]: q for q in questions}
    groups, ids = set(), set()
    release_split = Path(pool).parent / "split.json"
    release_map = json.loads(release_split.read_text()).get("group_by_question", {}) if release_split.exists() else {}
    for p in split_paths:
        split = json.loads(p.read_text())
        groups.update(map(str, split.get("eval_groups", [])))
        held = set(split.get("val", [])) | set(split.get("eval", [])) | set(split.get("test", []))
        unknown = held - by_id.keys()
        if unknown:
            raise ValueError(f"held-out question IDs absent from pool: {sorted(unknown)[:5]}")
        ids.update(held)
        mapping = split.get("group_by_question") or release_map
        canonical = {str(mapping[qid]) for qid in held if qid in mapping} | groups
        ids.update(qid for qid, gid in mapping.items() if str(gid) in canonical and qid in by_id)
        groups.update(canonical)
        if not any(key in split for key in ("val", "eval", "test", "eval_groups")):
            raise ValueError(f"split {p} contains no held-out IDs or groups")
    dirs, seeds = set(), set()
    for q in questions:
        if q["question_id"] not in ids and str(q.get("group", "")) not in groups:
            continue
        family = q.get("family") or (q.get("meta") or {}).get("family")
        for obj in (q, q.get("meta") or {}):
            for field in ("group", "dataset_path", "dataset", "dataset_id", "candidate_set"):
                value = obj.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue
                value = value.strip().replace("\\", "/")
                if "datasets/" in value:
                    value = value.split("datasets/", 1)[1]
                value = value.split("/candidates/", 1)[0].strip("/")
                parts = value.split("/")
                if len(parts) >= 2:
                    dirs.add("/".join(parts[:2]))
                elif field == "dataset_id" and family:
                    dirs.add(f"{family}/{value}")
            seeds.update(seed_values(obj))
        text = " ".join(m.get("content", "") for m in q.get("messages", []))
        seeds |= set(SEED_RE.findall(text))
        seeds.update(NAMED_SEED_RE.findall(text))
    return dirs, seeds


def model_row(m: dict) -> dict:
    out = {"model_type": m.get("type")}
    if m.get("type") == "mlp":
        ln = m.get("layer_norm") or []
        out.update(depth=m.get("depth"), width=m.get("width"), residual=m.get("residual"),
                   activation=m.get("activation"), layer_norm=ln, n_ln=sum(bool(x) for x in ln),
                   leaky_slope=m.get("negative_slope"), init=m.get("init") or "default")
    else:
        out.update({k: m.get(k) for k in ("d_model", "num_layers", "num_heads", "d_ff", "hidden_size", "num_layers",
                                          "residual", "embedding_dim") if k in m})
    return out


def build(args) -> None:
    pool = Path(getattr(args, "pool", POOL))
    split_paths = [Path(p) for p in args.split]
    dirs, seeds = held_out(split_paths, pool)
    lab = Path(getattr(args, "lab_dir", LAB))
    lab.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], collections.Counter()
    for spec_path in sorted(DATASETS.glob("*/*/dataset_spec.json")):
        ds_dir = spec_path.parent
        rel = f"{ds_dir.parent.name}/{ds_dir.name}"
        dspec = json.loads(spec_path.read_text())
        if rel in dirs or (seed_values(dspec["params"]) | set(SEED_RE.findall(json.dumps(dspec["params"])))) & seeds:
            skipped["held_out_dataset"] += 1
            continue
        for set_dir in sorted((ds_dir / "candidates").glob("set_*")):
            for cand in sorted(set_dir.glob("c_*")):
                summ = cand / "results/summary.json"
                if not summ.exists():
                    continue
                try:
                    cs = json.loads((cand / "candidate_spec.json").read_text())
                    sm = json.loads(summ.read_text())
                except (OSError, json.JSONDecodeError):
                    skipped["bad_json"] += 1
                    continue
                metric = sm.get("selection_metric") or dspec.get("selection_metric")
                rows.append({
                    "family": dspec["family"], "dataset_id": dspec.get("dataset_id", ds_dir.name),
                    "set_id": f"{ds_dir.name}/{set_dir.name}", "candidate_id": cand.name,
                    "dataset": {k: v for k, v in dspec["params"].items() if k not in ("calibration",)},
                    "budget": cs["budget"], "params": cs.get("trainable_parameter_count"),
                    **model_row(cs["model"]), "model": cs["model"], "optimizer": cs["optimizer"], "loss": cs["loss"],
                    "metric": metric, "mean": sm.get(f"mean_{metric}"), "std": sm.get(f"std_{metric}"),
                    "failed_seeds": sm.get("failed_seeds"), "excluded": sm.get("excluded")})
    rows = [r for r in rows if r["mean"] is not None]
    sizes = collections.Counter(r["set_id"] for r in rows)
    rows = [r for r in rows if sizes[r["set_id"]] >= 2]
    out = lab / "lab_db.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    allowlist = lab / "allowed_datasets.txt"
    allowlist.write_text("\n".join(sorted({f"{r['family']}/{r['dataset_id']}" for r in rows})) + "\n")
    inputs = {"pool": {"path": str(pool.resolve()), "sha256": hashlib.sha256(pool.read_bytes()).hexdigest()},
              "splits": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in split_paths]}
    meta = {"rows": len(rows), "sets": len({r["set_id"] for r in rows}), "datasets": len({r["dataset_id"] for r in rows}),
            "by_family": collections.Counter(r["family"] for r in rows), "skipped": skipped,
            "held_out_dirs": len(dirs), "held_out_seeds": len(seeds), **inputs,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (lab / "lab_db_meta.json").write_text(json.dumps(meta, indent=1))
    (lab / "lab_manifest.json").write_text(json.dumps({"schema_version": 1, **inputs,
        "excluded_dataset_dirs": sorted(dirs), "excluded_prompt_seeds": sorted(seeds),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out, allowlist)}}, indent=2))
    print(json.dumps(meta))


# ------------------------------------------------------------------ controlled experiments

EXECUTABLE_FILES = ("model.py", "optimizer.py", "loss.py", "train.py")
RESULT_FILES = ("candidate_spec.json", "results/summary.json", "results/curves.npz")
EXECUTION_MANIFEST = "results/execution_manifest.json"


def _file_evidence(path: Path) -> dict:
    content = path.read_bytes()
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def _measurement_result(cand_dir: str, summary: dict, cached: bool, capture_sources: bool = False) -> dict:
    directory = Path(cand_dir).resolve()
    process = json.loads((directory / "candidate_spec.json").read_text()).get("process") if capture_sources else None
    diagnostic_files = {}
    if process is not None:
        descriptor = summary.get("process") or {}
        if descriptor.get("request") != process or descriptor.get("recorder_version") != process.get("version"):
            raise ValueError("missing or mismatched process diagnostics; cache reuse refused")
        expected = {f"results/process/seed_{seed['seed']}.{suffix}"
                    for seed in summary.get("seed_results", []) for suffix in ("json", "npz")}
        diagnostic_files = descriptor.get("files") or {}
        if not expected or set(diagnostic_files) != expected:
            raise ValueError("incomplete per-seed process artifacts; cache reuse refused")
        for name, pin in diagnostic_files.items():
            path = (directory / name).resolve()
            path.relative_to(directory / "results/process")
            if _file_evidence(path) != pin:
                raise ValueError("process diagnostic hash mismatch; cache reuse refused")
    metric = summary.get("selection_metric")
    result = {"candidate_dir": cand_dir, "mean": summary.get(f"mean_{metric}"),
              "std": summary.get(f"std_{metric}"), "metric": metric, "cached": cached,
              **{key: summary.get(key) for key in
                 ("failed_seeds", "excluded", "n_seeds", "base_seed", "seed_results")},
              "measurement_files": {}}
    for name in RESULT_FILES:
        path = directory / name
        if path.is_file():
            result["measurement_files"][name] = {"path": str(path), **_file_evidence(path)}
    if process is not None:
        result["process_provenance"] = {"recorder_version": process["version"], "request": process,
                                        "status": "verified", "files": list(diagnostic_files)}
        for name, pin in diagnostic_files.items():
            result["measurement_files"][name] = {"path": str(directory / name), **pin}
    if not capture_sources:
        return result
    result["source_provenance"] = {"status": "unverified",
        "reason": "No execution manifest binds source bytes to this measurement.",
        "omitted": list(EXECUTABLE_FILES)}
    manifest_path = directory / EXECUTION_MANIFEST
    if manifest_path.is_file():
        result["measurement_files"][EXECUTION_MANIFEST] = {"path": str(manifest_path),
                                                             **_file_evidence(manifest_path)}
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema_version") != 1 or manifest.get("execution") != "canonical_snapshot_sync_disabled":
                raise ValueError("unsupported execution manifest")
            relative = Path(manifest["execution_dir"])
            if len(relative.parts) != 2 or relative.parts[0] != "results" or not relative.name.startswith("execution_"):
                raise ValueError("invalid execution snapshot path")
            snapshot = (directory / relative).resolve()
            snapshot.relative_to(directory / "results")
            if set(manifest["sources"]) != set(EXECUTABLE_FILES) or set(manifest["measurements"]) != set(RESULT_FILES) | set(diagnostic_files):
                raise ValueError("incomplete execution manifest")
            for name in (*RESULT_FILES, *diagnostic_files):
                if _file_evidence(directory / name) != manifest["measurements"][name]:
                    raise ValueError(f"measurement hash mismatch: {name}")
            if _file_evidence(snapshot / "candidate_spec.json") != manifest["measurements"]["candidate_spec.json"]:
                raise ValueError("executed candidate spec hash mismatch")
            sources = {}
            for name in EXECUTABLE_FILES:
                path = snapshot / name
                path.resolve().relative_to(snapshot)
                if _file_evidence(path) != manifest["sources"][name]:
                    raise ValueError(f"executed source hash mismatch: {name}")
                sources[f"executed/{name}"] = {"path": str(path), **manifest["sources"][name]}
            result["measurement_files"].update(sources)
            result["source_provenance"] = {"status": "verified",
                "execution": manifest["execution"], "manifest": EXECUTION_MANIFEST,
                "files": list(sources), "omitted": []}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            result["source_provenance"]["reason"] = f"Execution source verification failed: {exc}"
    spec = json.loads((directory / "candidate_spec.json").read_text())
    if "target_transform" in spec:
        transform_source = snapshot / "target_transform_runner.py" if result["source_provenance"]["status"] == "verified" else None
        pin = manifest.get("target_transform_runner") if transform_source is not None else None
        if transform_source is None or not transform_source.is_file() or _file_evidence(transform_source) != pin or summary.get("target_transform") != spec["target_transform"] or not summary.get("seed_results") or any(
            row.get("target_transform", {}).get("runner_source_sha256") != pin["sha256"]
            for row in summary.get("seed_results", [])
        ):
            raise ValueError("target transform source or seed provenance mismatch")
        result["measurement_files"]["executed/target_transform_runner.py"] = {"path": str(transform_source), **pin}
        result["source_provenance"]["files"].append("executed/target_transform_runner.py")
        result["target_transform"] = spec["target_transform"]
    if process is not None and result["source_provenance"]["status"] != "verified":
        raise ValueError("process execution manifest is missing or invalid; cache reuse refused")
    return result


def _gt_worker(payload: tuple[str, str] | tuple[str, str, bool]) -> dict:
    cand_dir, ds_dir = payload[:2]
    capture_sources = len(payload) == 3 and payload[2] is True
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    os.environ["ARCHITECTURE_IQ_TORCH_THREADS"] = "1"
    sys.path.insert(0, str(BENCH / "src"))
    from architecture_iq.candidates.generator import write_candidate
    from architecture_iq.ground_truth.runner import run_ground_truth
    from architecture_iq.profile import load_profile
    from architecture_iq.registry import ensure_registries, get_model_type
    prof = load_profile("v1.5")
    candidate_spec = json.loads((Path(cand_dir) / "candidate_spec.json").read_text())
    threshold = candidate_spec.get("diagnostic_fail_threshold")
    gt_options = {"fail_threshold_override": threshold} if threshold is not None else {}
    if not capture_sources:
        t0 = time.time()
        res = run_ground_truth(Path(cand_dir), prof, Path(ds_dir), **gt_options)
        result = _measurement_result(cand_dir, res, cached=False)
        result["secs"] = round(time.time() - t0, 1)
        return result
    ensure_registries()
    directory = Path(cand_dir).resolve()
    results_dir = directory / "results"
    results_dir.mkdir(exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix="execution_", dir=results_dir))
    spec_bytes = (directory / "candidate_spec.json").read_bytes()
    spec = json.loads(spec_bytes)
    runner_path = BENCH / "src/architecture_iq/ground_truth/runner.py"
    runner_source = runner_path.read_bytes() if "target_transform" in spec else None
    write_candidate(spec, snapshot, get_model_type(spec["model"]["type"]))
    (snapshot / "candidate_spec.json").write_bytes(spec_bytes)
    sources = {name: _file_evidence(snapshot / name) for name in EXECUTABLE_FILES}
    t0 = time.time()
    res = run_ground_truth(snapshot, prof, Path(ds_dir), sync_files=False, **gt_options)
    if runner_source is not None:
        runner_hash = hashlib.sha256(runner_source).hexdigest()
        if runner_path.read_bytes() != runner_source or any(
            seed.get("target_transform", {}).get("runner_source_sha256") != runner_hash
            for seed in res.get("seed_results", [])
        ):
            raise ValueError("target transform runner changed during measurement")
        (snapshot / "target_transform_runner.py").write_bytes(runner_source)
    if (snapshot / "candidate_spec.json").read_bytes() != spec_bytes:
        raise ValueError("executed candidate spec changed during measurement")
    for name in EXECUTABLE_FILES:
        if _file_evidence(snapshot / name) != sources[name]:
            raise ValueError(f"executed source changed during measurement: {name}")
    for name in ("summary.json", "curves.npz"):
        temporary = results_dir / f".{name}.{snapshot.name}.tmp"
        temporary.write_bytes((snapshot / "results" / name).read_bytes())
        temporary.replace(results_dir / name)
    diagnostic_files = (res.get("process") or {}).get("files") or {}
    for name, pin in diagnostic_files.items():
        source = (snapshot / name).resolve()
        source.relative_to(snapshot / "results/process")
        if not re.fullmatch(r"results/process/seed_[0-9]+\.(json|npz)", name) or _file_evidence(source) != pin:
            raise ValueError("invalid runner process artifact")
        target = directory / name
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_name(f".{target.name}.{snapshot.name}.tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(target)
    manifest = {"schema_version": 1, "execution": "canonical_snapshot_sync_disabled",
                "execution_dir": str(snapshot.relative_to(directory)), "sources": sources,
                "measurements": {name: _file_evidence(directory / name) for name in (*RESULT_FILES, *diagnostic_files)}}
    if runner_source is not None:
        manifest["target_transform_runner"] = _file_evidence(snapshot / "target_transform_runner.py")
    temporary = results_dir / f".execution_manifest.{snapshot.name}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(directory / EXECUTION_MANIFEST)
    result = _measurement_result(cand_dir, res, cached=False, capture_sources=True)
    result["secs"] = round(time.time() - t0, 1)
    return result


def validate_allowlist(lab: Path, pool: Path | None = None) -> set[str]:
    lab = Path(lab)
    allowlist = lab / "allowed_datasets.txt"
    if not allowlist.exists() or not (lab / "lab_manifest.json").exists():
        raise ValueError("required release lab allowlist or manifest is missing; rebuild the lab with --pool/--split")
    manifest = json.loads((lab / "lab_manifest.json").read_text())
    for name in ("lab_db.jsonl", "allowed_datasets.txt"):
        path = lab / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"].get(name):
            raise ValueError(f"lab file hash mismatch: {name}")
    if pool is not None and hashlib.sha256(Path(pool).read_bytes()).hexdigest() != manifest["pool"]["sha256"]:
        raise ValueError("lab pool hash mismatch; rebuild for this release")
    for source in manifest["splits"]:
        path = Path(source["path"])
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("lab split hash mismatch; rebuild for this split")
    return set(allowlist.read_text().split())


def run_job(job: dict, workers: int = 12, max_candidates: int = 16, timeout: int = 1500,
            lab_dir: Path | None = None, require_allowlist: bool = False, pool: Path | None = None) -> dict:
    """job = {"dataset": "<family>/<dataset_id>" (from lab_db), "candidates": [{"model": {...}, "optimizer": {...},
    "loss": {...}, "budget": {"training_steps": T, "batch_size": B}}, ...]}.
    Optional capture_executable_sources=true pins future execution source; existing caches need a verified manifest.
    Returns summaries and saved seed/file evidence."""
    capture_sources = job.get("capture_executable_sources", False)
    if not isinstance(capture_sources, bool):
        raise ValueError("capture_executable_sources must be boolean")
    if "process" in job and not capture_sources:
        raise ValueError("process recording requires capture_executable_sources=true")
    if "target_transform" in job and not capture_sources:
        raise ValueError("target transform requires capture_executable_sources=true")
    if "diagnostic_fail_threshold" in job:
        threshold = job["diagnostic_fail_threshold"]
        if "process" not in job or isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("diagnostic_fail_threshold requires process recording and a finite positive threshold")
    if ("process" in job or "target_transform" in job) and set(job) - {"dataset", "candidates", "capture_executable_sources", "process", "target_transform", "diagnostic_fail_threshold"}:
        raise ValueError("unsupported process job control")
    lab = Path(lab_dir or LAB)
    dataset = str(job["dataset"])
    parts = Path(dataset).parts
    if len(parts) != 2 or Path(dataset).is_absolute() or any(p in (".", "..") for p in parts):
        raise ValueError("dataset must be <family>/<dataset_id>")
    ds_dir = DATASETS / dataset
    if not (ds_dir / "dataset_spec.json").exists():
        raise ValueError(f"unknown dataset {dataset}")
    allowlist = lab / "allowed_datasets.txt"
    allowed = validate_allowlist(lab, pool) if require_allowlist else (
        set(allowlist.read_text().split()) if allowlist.exists() else None)
    if allowed is not None and dataset not in allowed:
        raise ValueError(f"dataset {job['dataset']} is not in the lab (held out or unmeasured)")
    sys.path.insert(0, str(BENCH / "src"))
    from architecture_iq.candidates.generator import write_candidate
    from architecture_iq.registry import ensure_registries, get_model_type
    ensure_registries()
    dspec = json.loads((ds_dir / "dataset_spec.json").read_text())
    cands = job["candidates"][:max_candidates]
    exp_root = lab / "experiments"
    payloads = []
    prepared = []
    for c in cands:
        if "process" in job and set(c) - {"model", "optimizer", "loss", "budget"}:
            raise ValueError("unsupported process candidate control")
        spec = {"schema_version": "1.0", "profile": "v1.5", "dataset_id": dspec.get("dataset_id", ds_dir.name),
                "family": dspec["family"], "budget": dict(c["budget"]), "model": copy.deepcopy(c["model"]),
                "optimizer": c["optimizer"], "loss": c.get("loss") or {"loss_id": "mse"},
                "execution": {"device": "cpu"},
                "files": {"model": "model.py", "train": "train.py", "loss": "loss.py", "optimizer": "optimizer.py"}}
        b = spec["budget"]
        b["total_samples_seen"] = int(b["training_steps"]) * int(b["batch_size"])
        if "diagnostic_fail_threshold" in job:
            spec["diagnostic_fail_threshold"] = float(job["diagnostic_fail_threshold"])
        if "target_transform" in job:
            from architecture_iq.ground_truth.runner import validate_target_transform
            spec["target_transform"] = validate_target_transform(job["target_transform"], spec)
        if "process" in job:
            from architecture_iq.candidates.generator import validate_process_request
            spec["process"] = validate_process_request(job["process"], spec)
        h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
        spec["candidate_id"] = f"x_{h}"
        cdir = exp_root / job["dataset"] / spec["candidate_id"]
        prepared.append((spec, cdir))
    for spec, cdir in prepared:
        cdir.mkdir(parents=True, exist_ok=True)
        if not (cdir / "candidate_spec.json").exists():
            write_candidate(spec, cdir, get_model_type(spec["model"]["type"]))
            (cdir / "candidate_spec.json").write_text(json.dumps(spec, indent=2))
        payloads.append((str(cdir), str(ds_dir), capture_sources))
    out = []
    with cf.ProcessPoolExecutor(min(workers, len(payloads) or 1)) as ex:
        futs = {}
        for p in payloads:
            summ = Path(p[0]) / "results/summary.json"
            if summ.exists():
                sm = json.loads(summ.read_text())
                out.append(_measurement_result(p[0], sm, cached=True, capture_sources=capture_sources))
            else:
                futs[ex.submit(_gt_worker, p)] = p
        for f in cf.as_completed(futs, timeout=timeout):
            try:
                out.append(f.result())
            except Exception as exc:  # report per-candidate failures to the caller
                result = _measurement_result(futs[f][0], {}, cached=False, capture_sources=capture_sources) if "process" not in job else {
                    "candidate_dir": futs[f][0], "cached": False, "mean": None, "measurement_files": {},
                    "process_provenance": {"status": "failed", "reason": str(exc)}}
                result["error"] = f"{type(exc).__name__}: {exc}"[:400]
                out.append(result)
    order = {p[0]: i for i, p in enumerate(payloads)}
    out.sort(key=lambda r: order[r["candidate_dir"]])
    for r, c in zip(out, cands):
        r["candidate"] = c
        r.pop("candidate_dir", None)
    return {"dataset": job["dataset"], "family": dspec["family"], "results": out}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", nargs="+", required=True, help="runtime or release split.json files whose val/test are held out")
    b.add_argument("--pool", default=str(POOL))
    b.add_argument("--lab-dir", default=str(LAB))
    r = sub.add_parser("run")
    r.add_argument("job")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--lab-dir", default=str(LAB))
    r.add_argument("--require-allowlist", action="store_true")
    r.add_argument("--pool", default=None, help="verify this release matches the lab manifest")
    a = p.parse_args()
    if a.cmd == "build":
        build(a)
    else:
        print(json.dumps(run_job(json.loads(Path(a.job).read_text()), a.workers, lab_dir=Path(a.lab_dir),
                                 require_allowlist=a.require_allowlist, pool=Path(a.pool) if a.pool else None), indent=1))


if __name__ == "__main__":
    main()
