#!/usr/bin/env python3
"""Live summary of a kb_science_loop run."""
import json, glob, sys
from pathlib import Path

run = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if run is None:
    print("usage: kb_loop_status.py RUN_DIR", file=sys.stderr)
    sys.exit(2)
if not run.exists():
    print(f"run not found: {run}")
    sys.exit(1)

kb_files = sorted(glob.glob(str(run / "kb" / "kb_*.json")))
epochs = sorted([d for d in (run / "epochs").glob("e*") if d.is_dir()]) if (run / "epochs").exists() else []
print(f"run: {run.name}")
print(f"kb snapshots: {len(kb_files)}")
print(f"epochs started: {len(epochs)}")

for ep_dir in epochs:
    solves = list((ep_dir / "solves").glob("*.json")) if (ep_dir / "solves").exists() else []
    if not solves:
        continue
    scores = [json.load(open(p))["score"] for p in solves]
    hyps = sum(1 for p in solves if json.load(open(p)).get("hypothesis"))
    print(f"\n{ep_dir.name}: n={len(solves)} mean={sum(scores)/len(scores):.3f} hypotheses={hyps}")
    trace = ep_dir / "summarizer_trace.json"
    if trace.exists():
        tr = json.load(open(trace))
        print(f"  summarizer turns={len(tr)} secs={sum(t['secs'] for t in tr):.0f}")

print("\n--- eval ---")
for ev_dir in sorted((run / "eval").glob("*")) if (run / "eval").exists() else []:
    p = ev_dir / "_summary.json"
    if p.exists():
        s = json.load(open(p))
        print(f"{ev_dir.name}: n={s['n']} mean={s['mean']:.3f} by_source={s['by_source']}")

print("\n--- commits ---")
commits = run / "commits.jsonl"
if commits.exists():
    ops = {}
    for l in open(commits):
        c = json.loads(l)
        ops[c["op"]] = ops.get(c["op"], 0) + 1
    print(f"total={sum(ops.values())} ops={ops}")
