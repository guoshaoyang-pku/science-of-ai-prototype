"""Prepare a frozen public v1.5 comparison; never calls a model or changes the source run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile

ARCHIVE_SHA256 = "7769b0db31b9e0c12860c84d5cc05c53a45098331d4db372c2c5aec076f8be2f"
MEMBER = "v1.5-main-500/v15-main.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def public_pool(pack):
    """Copy only official public inputs; answers have a separate, grader-owned destination."""
    questions, grader = [], {}
    for meta in pack["questions"]:
        entry = pack["byId"][meta["id"]]
        if entry["profile"] != "v1.5" or entry["evaluation"]["n_seeds"] != 10:
            raise ValueError("unexpected official evaluation profile")
        qid, prompt = entry["id"], entry["detail"]["prompt"]
        if not isinstance(prompt, str) or "<answer>" not in prompt:
            raise ValueError("missing official question or answer-format instruction")
        if qid in grader or entry["reveal"]["correctLetter"] not in "ABCDE"[:entry["numChoices"]]:
            raise ValueError("duplicate question or invalid answer")
        questions.append(dict(qid=qid, group=str(entry["datasetId"]), family=entry["family"],
                              prompt=prompt, num_choices=entry["numChoices"], type=entry["type"]))
        grader[qid] = dict(task="select", answer=entry["reveal"]["correctLetter"],
                           num_choices=entry["numChoices"])
    return questions, grader


def prepare(args):
    from aiq_kb import kb_science_loop as loop

    root = Path(args.data_root).resolve() / "runs" / args.run_name
    if root.exists():
        raise ValueError("new run required; existing outputs are preserved")
    archive, legacy = Path(args.archive).resolve(), Path(args.legacy_run).resolve()
    if sha(archive) != ARCHIVE_SHA256:
        raise ValueError("official release archive checksum mismatch")
    with tarfile.open(archive) as package:
        with package.extractfile(MEMBER) as stream:
            questions, grader = public_pool(json.load(stream))
    if len(questions) != 500:
        raise ValueError("official main set must contain exactly 500 questions")
    root.mkdir(parents=True)
    inputs = {}

    def register(name, path):
        inputs[name] = dict(path=path.relative_to(root).as_posix(), sha256=sha(path))

    def json_input(name, value):
        path = root / (name + ".json")
        write(path, value)
        register(name, path)

    def rows_input(name, rows):
        path = root / (name + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        register(name, path)

    obj = loop.Loop.__new__(loop.Loop)
    obj.run = root
    obj.args = argparse.Namespace(render="brief", science=True, solve_lab=False)
    snapshots, lookups, provenance = {"N": None}, {"N": {}}, {}
    for arm, filename in (("F", "kb/kb_0000.json"), ("L", "kb_live.json")):
        source = legacy / filename
        original = json.loads(source.read_text(encoding="utf-8"))
        json_input("kb_" + arm, original)
        operational = copy.deepcopy(original)
        for rid, report in operational.get("reports", {}).items():
            report_file = legacy / "reports" / rid / (report["repo_commit"] + ".md")
            # Source paths are historical metadata; resolve only the saved immutable version.
            report["path"] = str(report_file)
        kb = loop.KB(operational, root)
        snapshots[arm] = kb
        targets = set(kb.claims) | set(kb.reports) | {"science"}
        for claim in kb.claims.values():
            targets.update(claim.get("tags", []))
        for report in kb.reports.values():
            targets.update(x for x in (report.get("topic"), report.get("job_id")) if x)
        lookups[arm] = {target: obj.open_kb(kb, [target]) for target in sorted(targets)}
        provenance[arm] = dict(source=str(source), source_sha256=sha(source), epoch=original["epoch"],
                               claims=len(kb.claims), reports=len(kb.reports))
    lookup_inputs = {}
    for arm, values in lookups.items():
        name = "lookup_" + arm
        json_input(name, values)
        lookup_inputs[arm] = inputs.pop(name)
    inputs["kb_lookup"] = lookup_inputs
    rows_input("questions", questions)
    json_input("grader", grader)
    jobs = []
    for q in questions:
        public = dict(messages=[dict(role="user", content=q["prompt"])], task="select")
        for arm, kb in snapshots.items():
            messages = obj.solver_messages(public, kb)
            for rep in range(1, 4):
                jobs.append(dict(key=f"{q['qid']}__{arm}__r{rep}", qid=q["qid"], group=q["group"],
                                 family=q["family"], arm=arm, rep=rep, messages=messages))
    rows_input("jobs", jobs)
    historical_ids, historical_groups = set(), set()
    for path in sorted((legacy / "epochs").glob("e*/records.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                historical_ids.add(row["question_id"])
                historical_groups.add(str(row.get("group", "")))
    json_input("exposure", dict(
        classification="public_benchmark_readout_not_fresh_generalization",
        historical_qid_overlap=sorted(q["qid"] for q in questions if q["qid"] in historical_ids),
        historical_group_identifiers_comparable=False,
        known_overlapping_official_groups=sorted({q["group"] for q in questions if q["qid"] in historical_ids}),
        complete_historical_exposure=False, seed_variant_isolation_verified=False,
        note="Historical corpus/lab/seed exposure is incomplete. Group identifier namespaces differ, so zero string matches is not isolation evidence. Public benchmark only; no new scientific KB rule selection."))
    inputs["extra"] = {name: inputs.pop(name) for name in ("kb_F", "kb_L", "exposure")}
    source = Path(loop.__file__).with_name("frozen_eval.py")
    registration = dict(schema_version=1, run_name=args.run_name,
        model=dict(name=args.model, endpoint="https://llmapi.paratera.com/v1/chat/completions",
                   api_key_env="DEEPSEEK_API_KEY", thinking_field="thinking"),
        limits=dict(max_tokens=3000, max_calls_per_solve=3, max_requests=13500, workers=6,
                    max_open_targets=4, max_tool_chars=30000),
        expected=dict(question_count=500, arms=["N", "F", "L"], repeats=3), inputs=inputs,
        source_sha256=sha(source),
        grading=dict(bootstrap_seed=20261026, bootstrap_replicates=10000),
        provenance=dict(archive_sha256=sha(archive), member=MEMBER, kb=provenance,
                        loop_source_sha256=sha(Path(loop.__file__)),
                        scope="Frozen public benchmark; no learning, no discovery, no final sealed test."),
        stopping="Stop at the complete 4500 predictions, any transport/auth failure, STOP file, or a frozen-contract mismatch. No automatic retries.")
    write(root / "registration.json", registration)
    write(root / "config.json", dict(run_name=args.run_name, model=args.model, render="brief", science=True,
                                      solver_backend="api", epochs=0, test_repeats=0, harness="frozen_eval_v1"))
    write(root / "status.json", dict(state="prepared", predictions=0, grader_opened=False))
    print(json.dumps(dict(run=str(root), jobs=len(jobs), knowledge=provenance, provider_requests=0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--legacy-run", required=True)
    p.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
