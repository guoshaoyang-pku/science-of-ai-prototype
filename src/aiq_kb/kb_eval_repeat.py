#!/usr/bin/env python3
"""Re-sample the fixed eval set for a kb_science_loop run to measure KB gain beyond single-sample noise.

Usage: kb_eval_repeat.py --run-name luna_main --provider cctq --model gpt-6-luna --labels nokb kb_0000 kb_0008 --repeats 2
Writes eval/<label>_r<k>/ (fresh samples; original eval/<label>/ is repeat 1) and prints mean ± s.e. per label.
"""
import argparse, json, math, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if __package__:
    from . import kb_science_loop as L
else:
    import kb_science_loop as L

p = argparse.ArgumentParser()
p.add_argument("--run-name", required=True)
p.add_argument("--provider", required=True)
p.add_argument("--model", required=True)
p.add_argument("--labels", nargs="+", required=True)
p.add_argument("--repeats", type=int, default=2)
p.add_argument("--concurrency", type=int, default=2)
a = p.parse_args()

cfg = json.loads((L.OUT_ROOT / a.run_name / "config.json").read_text())
ns = argparse.Namespace(**{**cfg, "provider": a.provider, "model": a.model, "concurrency": a.concurrency})
loop = L.Loop(ns)


def kb_for(label: str):
    if label == "nokb":
        return None
    return L.KB(json.loads((loop.run / "kb" / f"{label}.json").read_text()), loop.run)


report = {}
for label in a.labels:
    kb = kb_for(label)
    for k in range(2, a.repeats + 2):
        lab = f"{label}_r{k}"
        if not loop.eval_done(lab):
            print(f"[eval] {lab}", loop.evaluate(lab, kb), flush=True)
    # per-question mean over repeats, then mean + s.e. over questions
    per_q = {}
    for d in [loop.run / "eval" / label] + [loop.run / "eval" / f"{label}_r{k}" for k in range(2, a.repeats + 2)]:
        for f in d.glob("q_*.json"):
            r = json.loads(f.read_text())
            per_q.setdefault(r["question_id"], []).append(r["score"])
    qs = [statistics.mean(v) for v in per_q.values()]
    se = statistics.stdev(qs) / math.sqrt(len(qs)) if len(qs) > 1 else 0.0
    report[label] = {"n_q": len(qs), "samples": sum(len(v) for v in per_q.values()), "mean": round(statistics.mean(qs), 4),
                     "se": round(se, 4), "per_q": {q: statistics.mean(v) for q, v in per_q.items()}}

base = report.get("nokb")
for label, r in report.items():
    line = f"{label}: mean={r['mean']:.3f} ± {r['se']:.3f} (n_q={r['n_q']}, samples={r['samples']})"
    if base and label != "nokb":
        d = [r["per_q"][q] - base["per_q"][q] for q in r["per_q"] if q in base["per_q"]]
        sd = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
        line += f"  Δ vs nokb = {statistics.mean(d):+.3f} ± {sd:.3f} (paired)"
    print(line)
L.write_json(loop.run / "eval" / "_repeat_report.json", {k: {x: y for x, y in v.items() if x != "per_q"} for k, v in report.items()})
