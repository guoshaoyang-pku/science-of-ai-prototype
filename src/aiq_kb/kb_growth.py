#!/usr/bin/env python3
"""Growth & compression trace of one KB science run: samples and experiments accumulated vs. what the KB compressed
them into (claims, merges, science-field coverage) and what the gate measured, epoch by epoch.

usage: kb_growth.py <run_dir>   -> prints a markdown table and writes <run_dir>/growth.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def main() -> None:
    run = Path(sys.argv[1])
    metrics = read_jsonl(run / "metrics.jsonl")
    commits = read_jsonl(run / "commits.jsonl")
    exps = read_jsonl(run / "lab_experiments.jsonl")
    rows, solves = [], 0
    for m in metrics:
        e = m["epoch"]
        solves += m["n"]
        kb = json.loads((run / "kb" / f"kb_{e:04d}.json").read_text())
        cand = run / "kb_rejected" / f"kb_{e:04d}.json"
        cand_doc = json.loads(cand.read_text()) if cand.exists() else kb
        ops = collections.Counter(c["op"] for c in commits if c["epoch"] == e and c["op"] != "credit")
        res = sorted((run / "epochs" / f"e{e:04d}" / "research").glob("r*.json"))
        soa = m.get("soa") or {}
        g = m.get("gate") or {}
        cl = cand_doc["claims"]
        rows.append({
            "epoch": e, "solves_cum": solves, "train_score": m["mean_score"],
            "exp_rows_cum": sum(1 for x in exps if (x.get("epoch") or 0) <= e),
            "research": len(res), "hypotheses": m.get("hypotheses"),
            "claims_cand": len(cl), "claims_kept": len(kb["claims"]), "ops": dict(ops),
            "mech": round(sum(bool(c.get("mechanism")) for c in cl) / max(1, len(cl)), 2),
            "regime": round(sum(bool(c.get("regime")) for c in cl) / max(1, len(cl)), 2),
            "phase": round(sum(bool(c.get("phase")) for c in cl) / max(1, len(cl)), 2),
            "chars": sum(len(c["text"]) for c in cl),
            "gate_accept": g.get("accept"), "cand_mean": g.get("cand_mean"), "delta_inc": g.get("delta"),
            "se": g.get("se"), "delta_nokb": g.get("delta_vs_nokb"), "se_nokb": g.get("se_vs_nokb"),
            "cells": soa.get("regime_completeness_cells"), "depth": soa.get("refinement_depth_max")})
    (run / "growth.json").write_text(json.dumps(rows, indent=1))
    hdr = ["ep", "solves", "exp rows", "research", "claims(cand/kept)", "ops", "mech/reg/phase", "text chars",
           "gate", "cand", "Δinc±se", "Δnokb±se"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in rows:
        print(f"| {r['epoch']} | {r['solves_cum']} | {r['exp_rows_cum']} | {r['research']} | "
              f"{r['claims_cand']}/{r['claims_kept']} | {' '.join(f'{k}:{v}' for k, v in r['ops'].items())} | "
              f"{r['mech']}/{r['regime']}/{r['phase']} | {r['chars']} | "
              f"{'✓' if r['gate_accept'] else '✗' if r['gate_accept'] is not None else '-'} | {r['cand_mean']} | "
              f"{r['delta_inc']}±{r['se']} | {r['delta_nokb']}±{r['se_nokb']} |")


if __name__ == "__main__":
    main()
