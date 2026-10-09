#!/usr/bin/env python3
"""Data-driven ceiling diagnostic: how much of the benchmark answer is predictable from the lab alone?

Trains a pairwise gradient-boosting model on lab pairs (same set_id: which candidate reaches the lower 10-seed mean
metric) over hand-built config features (log Δ, optimizer, lr, steps, LN count/placement, residual, depth, width,
activation, model type, family), then answers pool questions by parsing each choice from the prompt and ranking
choices by Borda score of pairwise win probabilities. Graded exactly like the loop (select 1/0, ranking inversions).

usage: lab_predictor.py <split.json> [--set eval|test]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

if __package__:
    from .config import KBConfig
    from .kb_science_loop import POOL, grade
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import KBConfig
    from kb_science_loop import POOL, grade

LAB = KBConfig.from_env().lab_root / "lab_db.jsonl"
FAMS = ["univariate_regression", "multivariate_regression", "bigram_lm", "synthetic_tabular_classification",
        "xor_classification", "spiral_classification"]
OPTS = ["Adam", "AdamW", "RMSprop", "Adagrad", "SGD"]
ACTS = ["relu", "leaky_relu", "gelu", "silu", "tanh"]
MTYPES = ["mlp", "gru_lm", "transformer_lm"]


def delta(opt: str, lr: float, T: int, mom: float) -> float:
    if opt == "Adagrad":
        return 2 * lr * math.sqrt(T)
    if opt == "SGD":
        return lr * T * 0.01 / (1 - (mom or 0.0))
    return lr * T


def feats(c: dict) -> list[float]:
    T, lr = c["steps"], c["lr"]
    ln = c.get("layer_norm") or []
    d = len(ln)
    last_ln = max([i + 1 for i, x in enumerate(ln) if x], default=0)
    return [math.log10(delta(c["opt"], lr, T, c.get("momentum") or 0)), math.log10(lr), math.log2(T),
            math.log2(c.get("batch") or 16), *[float(c["opt"] == o) for o in OPTS], float(c.get("momentum") or 0),
            float(c.get("wd") or 0) * 1e3, float((c.get("betas") or [0.9, 0.999])[1] < 0.99),
            *[float(c["mtype"] == m) for m in MTYPES], float(d), float(sum(map(bool, ln))), float(d - last_ln),
            float(bool(c.get("residual"))), math.log2(c.get("width") or 1), *[float(c.get("act") == a) for a in ACTS],
            math.log2(c.get("d_model") or 1), float(c.get("num_layers") or 0), math.log2(c.get("d_ff") or 1)]


RULES = ["smooth_additive", "piecewise_boundary", "sparse_interaction", "xor", "spiral"]


def expr_feats(e: str) -> list[float]:
    e = e or ""
    depth = cur = 0
    trig_depth = 0
    for m in re.finditer(r"(sin|cos|tanh|exp|log|abs)\(|\(|\)", e):
        tok = m.group(0)
        if tok == ")":
            cur = max(0, cur - 1)
        else:
            cur += 1
            depth = max(depth, cur)
    trig_depth = len(re.findall(r"(sin|cos)\(2\*pi\*(sin|cos|tanh)", e))
    return [float(len(re.findall(r"sin|cos", e))), float(trig_depth), float(depth), float("abs" in e),
            float("**" in e or "^" in e), float(len(e)) / 50]


def dfeats(d: dict) -> list[float]:
    return [float(d.get("input_dim") or 1), float(d.get("n_active") or 0), *[float(d.get("rule") == r) for r in RULES],
            float(d.get("turns") or 0), float(d.get("vocab") or 0), float(d.get("alpha") or 0), *expr_feats(d.get("expr"))]


def lab_dataset(r: dict) -> dict:
    d, fam = r["dataset"], r["family"]
    rule = "spiral" if fam == "spiral_classification" else "xor" if fam == "xor_classification" else d.get("rule_family")
    return {"input_dim": d.get("input_dim"), "n_active": len(d.get("active_features") or []), "rule": rule,
            "turns": d.get("spiral_turns") if fam == "spiral_classification" else 0, "vocab": d.get("vocab_size"),
            "alpha": d.get("alpha"), "expr": d.get("expression")}


def prompt_dataset(q: dict) -> dict:
    t, fam = q["messages"][-1]["content"], q["family"]
    rule = re.search(r"Rule family: `(\w+)`", t)
    act = re.search(r"active coordinates: ([^\n]+)", t)
    dim = re.search(r"Input shape: float32 `\[N, (\d+)\]`", t) or re.search(r"Input dimension: (\d+)", t)
    turns = re.search(r"turns `?([\d.]+)`?", t) if fam == "spiral_classification" else None
    expr = re.search(r"Target expression \(canonical\): `([^`]+)`", t)
    return {"input_dim": int(dim.group(1)) if dim else 1, "n_active": len(re.findall(r"x_\d+", act.group(1))) if act else 0,
            "rule": "spiral" if fam == "spiral_classification" else "xor" if fam == "xor_classification" else (rule.group(1) if rule else None),
            "turns": float(turns.group(1)) if turns else 0, "vocab": num(r"vocab_size: int = (\d+)", t, None, int),
            "alpha": num(r"alpha: float = ([\d.]+)", t), "expr": expr.group(1) if expr else None}


def from_lab(r: dict) -> dict:
    m, o = r["model"], r["optimizer"]
    return {"steps": r["budget"]["training_steps"], "batch": r["budget"]["batch_size"], "lr": o["lr"], "opt": o["type"],
            "momentum": o.get("momentum"), "wd": o.get("weight_decay"), "betas": o.get("betas"), "mtype": m["type"],
            "layer_norm": m.get("layer_norm") if m["type"] == "mlp" else [], "residual": m.get("residual"),
            "width": m.get("width"), "act": m.get("activation"), "d_model": m.get("d_model") or m.get("hidden_size"),
            "num_layers": m.get("num_layers"), "d_ff": m.get("d_ff")}


def num(pat: str, s: str, default=None, cast=float):
    m = re.search(pat, s, re.S)
    return cast(m.group(1)) if m else default


def parse_question(q: dict) -> dict[str, dict] | None:
    t = q["messages"][-1]["content"]
    shared_T = num(r"## Sample budget.*?training_steps:\s*(\d+)", t, None, int)
    shared_B = num(r"## Sample budget.*?batch_size:\s*(\d+)", t, None, int)
    blocks = re.split(r"\n### Choice ([A-E])\s*\n", t)
    out = {}
    for letter, body in zip(blocks[1::2], blocks[2::2]):
        body = body.split("\n## ")[0]
        mt = "gru_lm" if "GRU LM" in body else "transformer_lm" if "transformer LM" in body else "mlp"
        ln = re.search(r"Layer norm per layer:\s*\[([^\]]*)\]", body)
        opt = re.search(r"- Optimizer:\s*(\w+)", body)
        if not opt:
            return None
        betas = re.search(r"Betas:\s*\[([\d.]+),\s*([\d.]+)\]", body)
        out[letter] = {
            "steps": num(r"training_steps:\s*(\d+)", body, shared_T, int), "batch": num(r"batch_size:\s*(\d+)", body, shared_B, int),
            "lr": num(r"Learning rate:\s*([\d.eE+-]+)", body), "opt": opt.group(1),
            "momentum": num(r"Momentum:\s*([\d.]+)", body, 0.0), "wd": num(r"Weight decay:\s*([\d.eE+-]+)", body, 0.0),
            "betas": [float(betas.group(1)), float(betas.group(2))] if betas else None, "mtype": mt,
            "layer_norm": [x.strip() == "True" for x in ln.group(1).split(",")] if ln else [],
            "residual": "Residual connections: True" in body, "width": num(r"Width:\s*(\d+)", body, None, int),
            "act": (re.search(r"Activation:\s*(\w+)", body) or [None, None])[1],
            "d_model": num(r"d_model[^:\n]*:\s*(\d+)", body, None, int), "num_layers": num(r"num_layers:\s*(\d+)", body, None, int),
            "d_ff": num(r"d_ff:\s*(\d+)", body, None, int)}
        if out[letter]["steps"] is None or out[letter]["lr"] is None:
            return None
    return out or None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("split")
    p.add_argument("--set", default="test")
    p.add_argument("--data", action="store_true", help="add dataset descriptors (rule family, dims, expression shape)")
    a = p.parse_args()
    rows = [json.loads(x) for x in LAB.read_text().splitlines()]
    sets: dict[str, list] = {}
    for r in rows:
        sets.setdefault(r["set_id"], []).append(r)
    X, y = [], []
    for rs in sets.values():
        for r1, r2 in itertools.permutations(rs, 2):
            if r1["mean"] == r2["mean"]:
                continue
            fam = [float(r1["family"] == f) for f in FAMS] + (dfeats(lab_dataset(r1)) if a.data else [])
            f1, f2 = feats(from_lab(r1)), feats(from_lab(r2))
            X.append(f1 + f2 + [u - v for u, v in zip(f1, f2)] + fam)
            y.append(int(r1["mean"] < r2["mean"]))
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31).fit(np.array(X), np.array(y))
    split = json.loads(Path(a.split).read_text())
    ids = set(split[a.set])
    scores, by = [], {}
    for line in POOL.read_text().splitlines():
        q = json.loads(line)
        if q["question_id"] not in ids:
            continue
        ch = parse_question(q)
        if not ch:
            print("unparsed", q["question_id"])
            continue
        fam = [float(q["family"] == f) for f in FAMS] + (dfeats(prompt_dataset(q)) if a.data else [])
        borda = {k: 0.0 for k in ch}
        for k1, k2 in itertools.permutations(ch, 2):
            f1, f2 = feats(ch[k1]), feats(ch[k2])
            borda[k1] += clf.predict_proba(np.array([f1 + f2 + [u - v for u, v in zip(f1, f2)] + fam]))[0, 1]
        order = sorted(ch, key=lambda k: -borda[k])
        text = f"<answer>{order[0]}</answer>" if q["task"] == "select" else f"<answer>{'<'.join(order)}</answer>"
        s = grade(q, text)[1]
        scores.append(s)
        by.setdefault(q["source"], []).append(s)
    print(json.dumps({"set": a.set, "data_feats": a.data, "n": len(scores), "mean": round(statistics.mean(scores), 4), "pairs_train": len(y),
                      "by_source": {k: round(statistics.mean(v), 3) for k, v in by.items()}}))


if __name__ == "__main__":
    main()
