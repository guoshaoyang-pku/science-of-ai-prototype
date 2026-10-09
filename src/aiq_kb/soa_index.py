#!/usr/bin/env python3
"""Science-of-AI index: read-only diagnostics of a KB snapshot (never used for gating or reward).

  mechanism_coverage   share of claims with a mechanism (and among veterans, evidence s+f>=2)
  regime_annotation    share of claims with a structured regime
  regime_completeness  question-mass-weighted share of (family x type) cells, and of optimizer / activation values,
                       covered by >=1 claim whose regime explicitly names them (missing dim = unscoped, not "all")
  fractality           global vs conditional claims; depth of the refinement hierarchy (explicit `refines`, or
                       regime containment: same families, strictly more dims)
  phase_mapped         share of comparative claims carrying a phase boundary
  claim_table          per-claim counterfactual: on eval questions where the solver retrieved the claim, the mean
                       score minus the same question's no-KB mean score (n, delta)
  growth               cumulative research tasks / experiments / commits by op (the growth & compression trace)

usage: soa_index.py <run_dir> [--label kb_0003]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import statistics
from pathlib import Path

if __package__:
    from .config import KBConfig
else:
    from config import KBConfig

CONFIG = KBConfig.from_env()
ROOT = CONFIG.bench_root.parent
POOL = CONFIG.pool_file
COMPARATIVE = re.compile(r"\b(beats?|wins?|loses?|better than|worse than|prefer|outrank|>|<)\b", re.I)
OPTIMIZERS = ["Adam", "AdamW", "RMSprop", "Adagrad", "SGD"]
ACTIVATIONS = ["relu", "leaky_relu", "gelu", "silu", "tanh"]


def _vals(reg: dict, dim: str) -> list[str]:
    v = reg.get(dim)
    if v is None:
        return []
    return [str(x).lower() for x in (v if isinstance(v, list) else [v])]


def _covers(reg: dict, dim: str, value: str) -> bool:
    vals = _vals(reg, dim)
    return any(value.lower() == x or value.lower() in x or x in ("all", "any", "*") for x in vals)


def question_mass(split_path: Path, pool: Path = POOL) -> collections.Counter:
    split = json.loads(split_path.read_text())
    stream = {i for ids in split["stream"].values() for i in ids}
    mass = collections.Counter()
    if not pool.is_file():
        raise FileNotFoundError(f"SoA configured question pool does not exist: {pool}")
    for line in pool.read_text().splitlines():
        q = json.loads(line)
        if q["question_id"] in stream:
            mass[(q["family"], q["type"])] += 1
    return mass


def scored_records(run: Path, label: str) -> dict[str, list[dict]]:
    per = collections.defaultdict(list)
    for d in [run / "eval" / label] + sorted(run.glob(f"eval/{label}_r*")):
        for f in glob.glob(str(d / "q_*.json")):
            r = json.loads(Path(f).read_text())
            per[r["question_id"]].append(r)
    return per


def compute(run: Path, kb_doc: dict, label: str | None = None, pool: Path | None = None) -> dict:
    claims = kb_doc["claims"]
    n = max(1, len(claims))
    vet = [c for c in claims if c["support_count"] + c["failure_count"] >= 2]
    regs = [c.get("regime") or {} for c in claims]
    out: dict = {"n_claims": len(claims), "epoch": kb_doc.get("epoch")}
    out["mechanism_coverage"] = round(sum(bool(c.get("mechanism")) for c in claims) / n, 3)
    out["mechanism_coverage_veteran"] = round(sum(bool(c.get("mechanism")) for c in vet) / max(1, len(vet)), 3)
    out["regime_annotation"] = round(sum(bool(r) for r in regs) / n, 3)

    split = run / "split.json"
    if split.exists():
        if pool is None:
            config = run / "config.json"
            configured = json.loads(config.read_text()).get("pool") if config.exists() else None
            pool = Path(configured) if configured else POOL
        mass = question_mass(split, Path(pool))
        tot = sum(mass.values()) or 1
        cell_cov = sum(m for (fam, typ), m in mass.items()
                       if any(r and _covers(r, "family", fam) and (not r.get("type") or _covers(r, "type", typ))
                              for r in regs))
        out["regime_completeness_cells"] = round(cell_cov / tot, 3)
        out["uncovered_cells"] = sorted(f"{f}/{t}" for (f, t), m in mass.items()
                                        if not any(r and _covers(r, "family", f) and
                                                   (not r.get("type") or _covers(r, "type", t)) for r in regs))
    for dim, values in (("optimizer", OPTIMIZERS), ("activation", ACTIVATIONS)):
        out[f"regime_completeness_{dim}"] = round(sum(any(_covers(r, dim, v) for r in regs) for v in values)
                                                  / len(values), 3)

    # fractality: global = regime with <=1 scoped dim; refinement depth via refines or containment
    ndims = [len([k for k in r if k != "note"]) for r in regs]
    out["global_claims"] = sum(1 for k in ndims if k <= 1)
    out["conditional_claims"] = sum(1 for k in ndims if k >= 2)
    parent = {}
    by_id = {c["id"]: c for c in claims}
    for c in claims:
        if c.get("refines") in by_id:
            parent[c["id"]] = c["refines"]
            continue
        r = c.get("regime") or {}
        fams = set(_vals(r, "family"))
        best = None
        for d in claims:
            rd = d.get("regime") or {}
            if d["id"] == c["id"] or not rd or not fams:
                continue
            dk, ck = set(rd) - {"note"}, set(r) - {"note"}
            if dk < ck and fams <= set(_vals(rd, "family")) | ({"*"} if not rd.get("family") else set()):
                if best is None or len(dk) > len(set(by_id[best].get("regime") or {})):
                    best = d["id"]
        if best:
            parent[c["id"]] = best

    def depth(cid: str, seen=()) -> int:
        p = parent.get(cid)
        return 1 if not p or p in seen else 1 + depth(p, seen + (cid,))

    out["refinement_depth_max"] = max((depth(c["id"]) for c in claims), default=0)
    comp = [c for c in claims if COMPARATIVE.search(c["text"])]
    out["phase_mapped"] = round(sum(bool(c.get("phase")) for c in comp) / max(1, len(comp)), 3)

    # per-claim counterfactual table on val
    if label and (run / "eval" / label).exists() and (run / "eval" / "nokb").exists():
        cand = scored_records(run, label)
        nokb = scored_records(run, "nokb")
        base = {q: statistics.mean(r["score"] for r in rs) for q, rs in nokb.items()}
        table = collections.defaultdict(list)
        for q, rs in cand.items():
            if q not in base:
                continue
            for r in rs:
                for cid in r.get("kb_retrieved") or []:
                    table[cid].append(r["score"] - base[q])
        out["claim_table"] = {cid: {"n": len(v), "delta": round(statistics.mean(v), 3)}
                              for cid, v in sorted(table.items(), key=lambda kv: -len(kv[1]))}
        allr = [r for rs in cand.values() for r in rs]
        out["retrieval_rate"] = round(sum(bool(r.get("kb_queries")) for r in allr) / max(1, len(allr)), 3)
        out["explain_rate"] = round(sum(bool(r.get("kb_explained")) for r in allr) / max(1, len(allr)), 3)
        out["eval_label"] = label

    # growth trace
    ops = collections.Counter(json.loads(x)["op"] for x in (run / "commits.jsonl").read_text().splitlines()
                              if x.strip()) if (run / "commits.jsonl").exists() else {}
    res = sorted(run.glob("epochs/e*/research/r*.json"))
    exps = sum(len(json.loads(p.read_text()).get("experiments") or []) for p in res)
    out["growth"] = {"commits_by_op": dict(ops), "research_tasks": len(res), "experiments": exps,
                     "experiment_candidates": sum(len(e.get("rows") or []) for p in res
                                                  for e in json.loads(p.read_text()).get("experiments") or [])}
    return out


def brief(s: dict, with_table: bool = False) -> str:
    """Summary for the summarizer. The per-claim table is computed on val, so it stays out of curation by default."""
    keys = ["n_claims", "mechanism_coverage", "regime_annotation", "regime_completeness_cells", "regime_completeness_optimizer",
            "regime_completeness_activation", "global_claims", "conditional_claims", "refinement_depth_max",
            "phase_mapped", "retrieval_rate", "explain_rate"]
    line = ", ".join(f"{k}={s[k]}" for k in keys if k in s)
    if s.get("uncovered_cells"):
        line += f"\nuncovered (family/type) cells: {', '.join(s['uncovered_cells'])}"
    if with_table and s.get("claim_table"):
        line += ("\nper-claim val counterfactual (retrieved -> score minus same question's no-KB score; small n is noise): "
                 + ", ".join(f"{c}: {v['delta']:+.2f} (n={v['n']})" for c, v in list(s["claim_table"].items())[:20]))
    return line


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--label")
    p.add_argument("--kb")
    p.add_argument("--pool", type=Path, help="question pool; defaults to run config, then legacy pool")
    a = p.parse_args()
    run = Path(a.run)
    kb = Path(a.kb) if a.kb else sorted((run / "kb").glob("kb_*.json"))[-1]
    s = compute(run, json.loads(kb.read_text()), a.label, a.pool)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
