#!/usr/bin/env python3
"""Distill a compact seed KB (~20 claims) from the old 294-claim KB with v1.5-500 backtests.

One strong-model agent (Claude Opus via Claude Code, see cc_agent.py) reads the old KB and the v1.5 main-500 questions
(prompts, gold answers, structured choice configs with their measured 10-seed losses), checks candidate rules with
sandboxed Python, backtests the draft seed by actually running the cheap solver (luna) with vs without it, and writes a
seed KB that kb_science_loop.py can start from (--seed-kb).

Leakage control: v1.5 questions whose dataset group is held out (eval/test groups of --split-from) are invisible.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if __package__:
    from .config import KBConfig
    from .cc_agent import run_claude_agent
    from .kb_sandbox import run_python
    from .kb_science_loop import (CLAIM_MAX_CHARS, KB_BLOCK, OUT_ROOT, ROOT, Client, credibility, grade, now,
                                  read_jsonl, write_json, write_jsonl)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import KBConfig
    from cc_agent import run_claude_agent
    from kb_sandbox import run_python
    from kb_science_loop import (CLAIM_MAX_CHARS, KB_BLOCK, OUT_ROOT, ROOT, Client, credibility, grade, now,
                                 read_jsonl, write_json, write_jsonl)

CONFIG = KBConfig.from_env()
V15 = CONFIG.bench_root / "data/v1_review/v15_questions.json"
OLD_KB = Path(os.environ.get("AIQ_KB_OLD_KB", str(CONFIG.data_root / "seed_kb.json")))
FIRST_ID = 1001
CHOICE_KEYS = ["letter", "model_type", "depth", "width", "d_model", "num_layers", "heads", "d_ff", "residual",
               "layer_norm", "activations", "optimizer", "lr", "momentum", "weight_decay", "betas", "loss",
               "batch_size", "training_steps", "total_samples_seen", "params", "mean_test_ce", "mean_test_mse",
               "std_test_ce", "std_test_mse"]

SYSTEM = """You are the SEED DISTILLER for a knowledge base (KB) of empirical laws about small neural-network training
(ArchitectureIQ). A cheap solver model (gpt-6-luna, low reasoning effort) will read your KB, read-only, in its system
prompt while answering questions of the form "which training setup (architecture / optimizer / loss under a fixed
sample budget) gets the lowest held-out loss on this dataset instance". Later questions also include 5-way rankings
(order all choices by loss) and data-flip pairs (same choices, a different dataset makes a different choice win).

The current KB has {n_old} claims accumulated by an earlier agent. It is too large, redundant and mostly backed by a
single observation. Your job: replace it with a compact SEED of about {target} claims (hard limits {lo}-{hi}).

Each seed claim must be:
- a general decision rule the solver can apply: scope (data family, task size/budget, optimizer regime) + the direction
  of the effect + key numbers + the mechanism in one clause;
- backed by the v1.5 questions (you can see their gold answers and every choice's measured 10-seed mean loss);
- non-redundant with the other seed claims (one rule per mechanism; merge special cases into one sharper rule);
- under {max_chars} characters. Put derivations in the `reason` field, not in the claim.
Prefer rules about interactions that decide close calls (budget x depth, lr x optimizer, normalization x residual,
activation x init scale, dataset structure x capacity) over generic platitudes ("Adam beats SGD").

Tools:
- old_kb(ids|contains|sort|limit): read old claims (s/f/credibility from earlier use).
- search_questions / read_question: the visible v1.5 questions ({n_q}).
- python: sandboxed numpy. load_questions() -> list of dicts (question_id, family, type, budget_tier, correct_letter,
  gap, prompt, choices=[structured configs with mean_test_ce / mean_test_mse; activations are only in the prompt's
  code blocks, parse them from `prompt`]); load_old_kb(); load_draft(). Use it to count
  how often a candidate rule picks the gold choice across ALL questions it applies to. This is cheap; do it a lot.
- add_claim / revise_claim / drop_claim / draft_view: edit the draft seed.
- backtest(question_ids | family/type/n): really run the solver with the current draft vs with no KB on v1.5
  questions and report accuracy and flips. Expensive-ish (budget {bt_budget} solves total); use it on the full draft,
  ~40 questions per call, 2-4 times.
- finish(notes): end. Notes = what the seed covers, what you deliberately dropped, and the backtest numbers.

Workflow: (1) skim the old KB by theme and the question mix; (2) python-check 25-40 candidate rules on the structured
data and keep the ones with a clear, wide-scope win rate; (3) write the draft; (4) backtest it, revise or drop the
claims that the flips show to be misleading, backtest again; (5) finish. You have about {calls} tool calls."""


class Distiller:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.dir = OUT_ROOT / a.name
        self.dir.mkdir(parents=True, exist_ok=True)
        split = json.loads((OUT_ROOT / a.split_from / "split.json").read_text())
        held = set(split.get("eval_groups", []))
        qs = json.loads(V15.read_text())["questions"]
        self.q = {}
        for q in qs:
            if f"datasets/{q['family']}/{q['dataset_id']}" in held:
                continue
            self.q[q["question_id"]] = q
        self.hidden = len(qs) - len(self.q)
        self.old = {c["id"]: c for c in json.loads(OLD_KB.read_text())["claims"]}
        self.draft: dict[str, dict] = {}
        self.commits: list[dict] = []
        self.solver = Client(a.solver_provider, a.solver_model)
        self.bt_used = 0
        self.bt_lock = threading.Lock()
        self.vis = self.dir / "visible"
        rows = [{"question_id": q["question_id"], "family": q["family"], "type": q["type"],
                 "budget_tier": q["budget_tier"], "metric": q["metric"], "correct_letter": q["correct_letter"],
                 "gap": float(q["gap"]), "choices": [{k: c.get(k) for k in CHOICE_KEYS} for c in q["choices"]],
                 "prompt": q["prompt"]}
                for q in self.q.values()]
        write_jsonl(self.vis / "questions.jsonl", rows)
        write_json(self.vis / "old_kb.json", {"claims": list(self.old.values())})
        self.sync_draft()

    # ---- state
    def sync_draft(self) -> None:
        write_json(self.vis / "draft.json", {"claims": list(self.draft.values())})

    def commit(self, op: str, **f) -> None:
        self.commits.append({"commit": f"d{len(self.commits) + 1:04d}", "time": now(), "op": op, **f})
        with (self.dir / "commits.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.commits[-1], ensure_ascii=False) + "\n")

    def next_id(self) -> str:
        used = [int(i[1:]) for i in self.draft] + [int(c["claim_id"][1:]) for c in self.commits if c.get("claim_id")]
        return f"K{max(used, default=FIRST_ID - 1) + 1:04d}"

    # ---- tools
    def tools(self) -> list[dict]:
        def fn(name, desc, props, req=()):
            return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
                "type": "object", "properties": props, "required": list(req)}}}
        s, i, arr = {"type": "string"}, {"type": "integer"}, {"type": "array", "items": {"type": "string"}}
        return [
            fn("old_kb", "Read old KB claims (filter by ids / substring; sort credibility|evidence|id).",
               {"ids": arr, "contains": s, "sort": s, "limit": i}),
            fn("search_questions", "List visible v1.5 questions with a one-line summary of each choice.",
               {"family": s, "type": {"type": "string", "enum": ["architecture_only", "optimizer_only", "mixed"]},
                "contains": s, "limit": i}),
            fn("read_question", "Full prompt + gold + structured choices (with measured losses) of one question.",
               {"question_id": s}, ["question_id"]),
            fn("python", "Sandboxed Python 3 + numpy (no network, ~45s). load_questions(), load_old_kb(), load_draft().",
               {"code": s}, ["code"]),
            fn("draft_view", "Show the current draft seed.", {}),
            fn("add_claim", "Add a draft seed claim.", {"text": s, "merged_from_old": arr, "reason": s},
               ["text", "reason"]),
            fn("revise_claim", "Rewrite a draft claim in place.", {"claim_id": s, "text": s, "reason": s},
               ["claim_id", "text", "reason"]),
            fn("drop_claim", "Remove a draft claim.", {"claim_id": s, "reason": s}, ["claim_id", "reason"]),
            fn("backtest", "Run the solver with the current draft vs no KB. Give question_ids, or family/type and n "
               "(<=40, random visible questions, fixed seed per call unless `seed`).",
               {"question_ids": arr, "family": s, "type": s, "n": i, "seed": i}),
            fn("finish", "End distillation.", {"notes": s}, ["notes"]),
        ]

    def summary_line(self, q: dict) -> str:
        def c1(c):
            arch = (f"{c['model_type']} d{c['depth']}w{c['width']}" if c["model_type"] == "mlp" else
                    f"{c['model_type']} L{c['num_layers']}d{c['d_model']}h{c['heads']}")
            return (f"{c['letter']}:{arch} res={c['residual']} ln={c['layer_norm']} act={c['activations']} "
                    f"{c['optimizer']} lr={c['lr']}")
        return (f"{q['question_id']} {q['family']} {q['type']} budget={q['budget_tier']} gold={q['correct_letter']} "
                f"gap={float(q['gap']):.3g} | " + " ; ".join(c1(c) for c in q["choices"]))

    def handler(self, name: str, a: dict) -> str:
        if name == "old_kb":
            cs = list(self.old.values())
            if a.get("ids"):
                cs = [self.old[i] for i in a["ids"] if i in self.old]
            if a.get("contains"):
                cs = [c for c in cs if a["contains"].lower() in c["text"].lower()]
            key = {"evidence": lambda c: -(c["support_count"] + c["failure_count"]), "id": lambda c: c["id"]}.get(
                a.get("sort"), lambda c: -credibility(c["support_count"], c["failure_count"]))
            cs = sorted(cs, key=key)[: int(a.get("limit") or 40)]
            return f"{len(cs)} claims\n" + "\n".join(
                f"[{c['id']}] s={c['support_count']:.1f} f={c['failure_count']:.1f} {c['text']}" for c in cs)
        if name == "search_questions":
            out = [q for q in self.q.values() if (not a.get("family") or a["family"] in q["family"])
                   and (not a.get("type") or q["type"] == a["type"])
                   and (not a.get("contains") or a["contains"].lower() in q["prompt"].lower())]
            lim = int(a.get("limit") or 25)
            return f"{len(out)} matches (showing {min(lim, len(out))})\n" + "\n".join(self.summary_line(q) for q in out[:lim])
        if name == "read_question":
            q = self.q.get(a["question_id"])
            if not q:
                return "not found (or held out)"
            return json.dumps({"question_id": q["question_id"], "family": q["family"], "type": q["type"],
                               "gold": q["correct_letter"], "gap": q["gap"], "prompt": q["prompt"][:12000],
                               "choices": [{k: c.get(k) for k in CHOICE_KEYS} for c in q["choices"]]}, indent=1)
        if name == "python":
            self.sync_draft()
            v = self.vis
            prelude = (f"Q_PATH={str(v / 'questions.jsonl')!r}\nOLD_PATH={str(v / 'old_kb.json')!r}\n"
                       f"DRAFT_PATH={str(v / 'draft.json')!r}\n"
                       "def load_questions():\n    return [json.loads(l) for l in open(Q_PATH) if l.strip()]\n"
                       "def load_old_kb():\n    return json.load(open(OLD_PATH))['claims']\n"
                       "def load_draft():\n    return json.load(open(DRAFT_PATH))['claims']\n")
            return run_python(prelude + a["code"], self.dir / "sandbox", [v])
        if name == "draft_view":
            return f"{len(self.draft)} claims\n" + "\n".join(f"[{c['id']}] ({len(c['text'])} chars, from "
                                                            f"{c['merged_from']}) {c['text']}" for c in self.draft.values())
        if name in ("add_claim", "revise_claim") and len(a["text"].strip()) > CLAIM_MAX_CHARS:
            return f"rejected: {len(a['text'].strip())} chars > {CLAIM_MAX_CHARS}; shorten"
        if name == "add_claim":
            if len(self.draft) >= self.a.hi:
                return f"rejected: draft already has {self.a.hi} claims; merge or drop first"
            cid = self.next_id()
            src = [i for i in a.get("merged_from_old") or [] if i in self.old]
            self.draft[cid] = {"id": cid, "text": a["text"].strip(), "merged_from": src}
            self.commit("add", claim_id=cid, text=a["text"].strip(), merged_from_old=src, reason=a["reason"])
            return f"added {cid} (draft size {len(self.draft)})"
        if name == "revise_claim":
            c = self.draft.get(a["claim_id"])
            if not c:
                return "not found"
            self.commit("revise", claim_id=c["id"], old_text=c["text"], text=a["text"].strip(), reason=a["reason"])
            c["text"] = a["text"].strip()
            return "revised " + c["id"]
        if name == "drop_claim":
            c = self.draft.pop(a["claim_id"], None)
            if not c:
                return "not found"
            self.commit("drop", claim_id=c["id"], text=c["text"], reason=a["reason"])
            return f"dropped {c['id']} (draft size {len(self.draft)})"
        if name == "backtest":
            return self.backtest(a)
        if name == "finish":
            if not self.a.lo <= len(self.draft) <= self.a.hi:
                return f"rejected: draft has {len(self.draft)} claims; need {self.a.lo}-{self.a.hi}"
            self.notes = a["notes"]
            return "ok"
        return f"unknown tool {name}"

    # ---- backtest with the real solver
    def messages(self, q: dict, kb_text: str | None) -> list[dict]:
        sys_txt = "You are solving an ArchitectureIQ benchmark question. Answer in the required format."
        if kb_text is not None:
            sys_txt += "\n\n" + KB_BLOCK.format(claims=kb_text)
        return [{"role": "system", "content": sys_txt}, {"role": "user", "content": q["prompt"]}]

    def solve(self, q: dict, kb_text: str | None, tag_: str) -> dict:
        path = self.dir / "backtest" / tag_ / f"{q['question_id']}.json"
        if path.exists():
            return json.loads(path.read_text())
        res = self.solver.chat(self.messages(q, kb_text), "low", 16000)
        text = res["message"].get("content") or ""
        gq = {"task": "select", "num_choices": len(q["choices"]), "answer": q["correct_letter"]}
        pred, score, _ = grade(gq, text)
        rec = {"question_id": q["question_id"], "pred": pred, "score": score, "text": text[-1500:],
               "usage": res["usage"]}
        write_json(path, rec)
        return rec

    def backtest(self, a: dict) -> str:
        if a.get("question_ids"):
            ids = [i for i in a["question_ids"] if i in self.q][:40]
        else:
            pool = sorted(i for i, q in self.q.items() if (not a.get("family") or a["family"] in q["family"])
                          and (not a.get("type") or q["type"] == a["type"]))
            random.Random(a.get("seed", 7)).shuffle(pool)
            ids = pool[: min(40, int(a.get("n") or 40))]
        if not self.draft:
            return "draft is empty"
        need = 2 * len(ids)
        with self.bt_lock:
            if self.bt_used + need > self.a.bt_budget:
                return f"rejected: backtest budget {self.a.bt_budget} solves, used {self.bt_used}"
            self.bt_used += need
        kb_text = "\n".join(f"[{c['id']}] (credibility 0.50; s=0.0, f=0.0) {c['text']}" for c in
                            sorted(self.draft.values(), key=lambda c: c["id"]))
        ver = f"draft_{len(self.commits):04d}"
        with cf.ThreadPoolExecutor(8) as ex:
            kb_r = list(ex.map(lambda i: self.solve(self.q[i], kb_text, ver), ids))
            no_r = list(ex.map(lambda i: self.solve(self.q[i], None, "nokb"), ids))
        lines, cited = [], collections.Counter()
        for i, k, n in zip(ids, kb_r, no_r):
            c = sorted(set(__import__("re").findall(r"\bK\d{4}\b", k["text"])))
            cited.update(c)
            mark = "+" if k["score"] > n["score"] else "-" if k["score"] < n["score"] else " "
            lines.append(f"{mark} {i} {self.q[i]['family']} {self.q[i]['type']} gold={self.q[i]['correct_letter']} "
                         f"kb={k['pred']} nokb={n['pred']} cited={','.join(c) or '-'}")
        acc_k = sum(r["score"] for r in kb_r) / len(ids)
        acc_n = sum(r["score"] for r in no_r) / len(ids)
        rec = {"version": ver, "n": len(ids), "acc_kb": acc_k, "acc_nokb": acc_n, "ids": ids,
               "draft": list(self.draft.values()), "time": now()}
        with (self.dir / "backtests.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return (f"{ver}: n={len(ids)} acc with draft {acc_k:.3f} vs no KB {acc_n:.3f} "
                f"(+{sum(l[0] == '+' for l in lines)} / -{sum(l[0] == '-' for l in lines)}); "
                f"budget used {self.bt_used}/{self.a.bt_budget}\ncitations: {dict(cited.most_common())}\n"
                + "\n".join(lines))

    # ---- main
    def run(self) -> None:
        a = self.a
        fam = collections.Counter((q["family"], q["type"]) for q in self.q.values())
        user = (f"Visible v1.5 questions: {len(self.q)} ({self.hidden} hidden because their dataset is held out for "
                f"evaluation). Mix by (family, type): {dict(sorted(fam.items()))}.\nOld KB: {len(self.old)} claims "
                f"(ids K0001-K0{len(self.old):03d}); seed claims you write get ids from K{FIRST_ID}.\n"
                "Start by skimming the old KB and the question mix, then follow the workflow.")
        system = SYSTEM.format(n_old=len(self.old), target=a.target, lo=a.lo, hi=a.hi, max_chars=CLAIM_MAX_CHARS,
                               n_q=len(self.q), bt_budget=a.bt_budget, calls=a.max_calls)
        self.notes = ""
        final, trace, info = run_claude_agent(
            system, user, self.tools(), self.handler, model=a.model, effort=a.effort, workdir=self.dir / "agent",
            max_calls=a.max_calls, stop_tool="finish", warn_calls=8, budget_usd=a.budget_usd,
            warn_text="stop exploring; make sure the draft has the right size, then call finish(notes).",
            trace_path=self.dir / "trace.json")
        claims = [{"id": c["id"], "text": c["text"], "support_count": 0.0, "failure_count": 0.0,
                   "credibility": 0.5, "created_epoch": 0, "sources": [], "merged_from": c["merged_from"],
                   "origin": f"seed:{a.name}"} for c in sorted(self.draft.values(), key=lambda c: c["id"])]
        write_json(self.dir / "seed.json", {"schema_version": "architectureiq_kb_v4", "epoch": 0, "claims": claims,
                                            "retired": [], "created_at": now(), "notes": self.notes or final,
                                            "distiller": {"model": a.model, "effort": a.effort}})
        solver_cost = {k: v for k, v in self.solver.usage.items()}
        write_json(self.dir / "report.json", {"claims": len(claims), "agent": info, "solver_usage": solver_cost,
                                              "backtest_solves": self.bt_used, "hidden_questions": self.hidden})
        print(json.dumps({"claims": len(claims), "agent": {k: info[k] for k in ("cost_usd", "calls", "stopped",
                          "secs")}, "solver_usage": solver_cost}, ensure_ascii=False))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="seed_distill_v1")
    p.add_argument("--split-from", default="luna_main", help="run whose eval/test groups stay hidden")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="xhigh")
    p.add_argument("--solver-provider", default="cctq")
    p.add_argument("--solver-model", default="gpt-6-luna")
    p.add_argument("--target", type=int, default=20)
    p.add_argument("--lo", type=int, default=15)
    p.add_argument("--hi", type=int, default=25)
    p.add_argument("--max-calls", type=int, default=90)
    p.add_argument("--bt-budget", type=int, default=480)
    p.add_argument("--budget-usd", type=float, default=40.0)
    return p.parse_args()


if __name__ == "__main__":
    Distiller(parse()).run()
