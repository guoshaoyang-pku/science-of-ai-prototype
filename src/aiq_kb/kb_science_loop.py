#!/usr/bin/env python3
"""Training-free KB + science loop (solver -> discoverer -> summarizer).

Per epoch:
  1. Solve: the model sees only the question and the rendered KB (no tools, no answer). effort=low.
  2. Discover: the same conversation continues with the gold answer revealed, plus tools
     (search/read past questions with answers and model solutions, sandboxed Python for backtests).
     Correct -> effort=low and may skip; wrong -> effort=medium. Optional hypothesis <= 2000 tokens.
     Cited KB claims get additive credit (s += score, f += 1 - score).
  3. Summarize: one agent (effort=xhigh, tools) reads every record of the epoch and curates the KB via
     add / delete / merge / adjust ops, then writes a science digest.
Every KB change is an append-only commit (commits.jsonl). Eval questions are held out by group and
never visible to any agent; history visible to agents only covers earlier epochs (summarizer: <= now).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import random
import re
import ssl
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

if __package__:
    from .config import KBConfig
    from .kb_sandbox import run_python
    from .cc_agent import run_claude_agent
    from .codex_agent import CodexSession
    from .soa_index import brief as soa_brief, compute as soa_compute
    from .kb_lab import model_row, validate_allowlist
    from .kb_jobs import Jobs
    from .kb_jobs import finish_job, process_alive, read_job, update_job, worker_lock
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import KBConfig
    from kb_sandbox import run_python
    from cc_agent import run_claude_agent
    from codex_agent import CodexSession
    from soa_index import brief as soa_brief, compute as soa_compute
    from kb_lab import model_row, validate_allowlist
    from kb_jobs import Jobs
    from kb_jobs import finish_job, process_alive, read_job, update_job, worker_lock

CONFIG = KBConfig.from_env()
ROOT = CONFIG.bench_root.parent
POOL = CONFIG.pool_file
KEYS = CONFIG.keys_file
SEED_KB = Path(os.environ.get("AIQ_KB_SEED_KB", str(CONFIG.data_root / "seed_kb.json"))).resolve()
OUT_ROOT = CONFIG.runs_root
LAB_DIR = CONFIG.lab_root

RANK_CREDIT = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25}
RENDER_TOP, RENDER_PROBATION = 40, 10
CLAIM_MAX_CHARS = 700
TEST_RESERVE_EPOCHS = 24
HYPOTHESIS_MAX_CHARS = 8000
CITE_RE = re.compile(r"\bK\d{4}\b")
TOK_RE = re.compile(r"[a-z0-9]{2,}")


# ------------------------------------------------------------------ prompts (the harness)

KB_BLOCK = """# Knowledge base (read-only)
Claims distilled from earlier training experiments. credibility = (s+1)/(s+f+2), where s/f accumulate
from past answers that relied on the claim. Use claims only when their scope matches this question.

{claims}

When you rely on a claim, mention its ID (e.g. K0016) in your reasoning. Before the final tagged
fields required by the question, write <cited>comma-separated claim IDs you relied on, or none</cited>."""

KB_BLOCK_VARIANTS = {
    # no citation tag at all (attribution off)
    "nocite": KB_BLOCK.split("\n\nWhen you rely on a claim")[0] + "\n\nThe claims are advisory: if your own analysis of "
              "this question disagrees with a claim, trust your analysis.",
    # citation tag AFTER the answer, so it cannot cut the reasoning short
    "cite_after": KB_BLOCK.split("\n\nWhen you rely on a claim")[0] + "\n\nThe claims are advisory: if your own "
                  "analysis of this question disagrees with a claim, trust your analysis. After the final <answer> tag, "
                  "add one line <cited>comma-separated claim IDs you relied on, or none</cited>.",
}

KB_TOOL_NOTE = """# Knowledge base (optional, read-only)
A knowledge base of {n} empirical claims distilled from earlier training experiments is available through the
`kb_search` tool (keyword query; returns the best-matching claims with IDs and credibility). Using it is optional and
the claims are advisory: trust your own analysis of this question when it disagrees. If a claim changed your answer,
add one line <cited>claim IDs</cited> after the final <answer> tag."""

RANKING_HINT = """CRITICAL FORMAT for ranking questions: the answer must list letters from LOWEST loss (BEST choice) on the LEFT to HIGHEST loss (WORST choice) on the RIGHT. Example: if choice C has the lowest held-out loss and A the highest, write `<answer>C<...<A</answer>`. Double-check before outputting: the first letter after `<answer>` must be your top pick."""

DISCOVER_PROMPT = """Grading result
- Gold answer: {gold}
- Your answer: {pred}
- Score: {score:.2f}{rank_note}

You are now the DISCOVERER for this question. The answer is revealed and you have tools:
- search_history / read_record: every question from earlier epochs, with gold answer, the model's solution, comments and hypotheses.
- python: sandboxed Python (numpy, no network). `load_history()` returns earlier records as dicts; `load_kb()` returns the KB claims.
{guidance}

Finish with plain text (no tool call) in this format:
<comment>at most 120 words: what decided this question; which KB claims (IDs) helped or misled</comment>
<hypothesis>optional, at most 2000 tokens: one general, quantitative, falsifiable rule that would have predicted the gold answer, its scope (task, data family, budget, optimizer regime), mechanism, and backtest on history (n checked, hits, misses)</hypothesis>
Omit <hypothesis> if you have nothing general and tested. {skip_note}"""

GUIDE_RIGHT = "Your answer scored full marks. Spend little effort; a hypothesis only if you noticed something new and general."
GUIDE_WRONG = ("Your answer lost points. Find the mechanism you missed, check whether the KB misled you, and backtest any "
               "candidate rule on similar past questions before stating it.")
CORPUS_NOTE = ("\n- python also has load_corpus(): ~490 older solved questions (v1.5) with every choice's structured config "
               "and measured 10-seed test loss; activations are in each item's `prompt`. This is the large backtest set: "
               "check a rule there before trusting 1-2 epoch questions.")
LAB_NOTE = ("\n- python also has load_lab(): ~11k measured candidates (10-seed mean loss) from other dataset instances, "
            "grouped by set_id (one dataset + one budget; compare rows only within a set). Use it to check whether the "
            "effect that decided this question holds across many sets, and in which regime it flips.")
LAB_EXP_NOTE = ("\n- run_experiment ({n} call): a controlled training run on a lab dataset (base = a lab row's "
                "set_id/candidate_id, variants = dotted-key overrides such as {{\"model.layer_norm\": [true, true]}} or "
                "{{\"optimizer.lr\": 0.001}}). Use it to test the causal factor you think decided this question, on a "
                "similar lab setup; results are added to the lab for everyone.")
SOLVE_LAB_NOTE = """# Measured-experiment database (optional)
The `python` tool (numpy, sandboxed) has load_lab(): ~11k candidate training setups measured on OTHER dataset instances
of the same families (10-seed mean of the selection metric; lower is better). Rows are comparable only within one
set_id (same dataset + budget). Fields: family, set_id, budget{training_steps,batch_size}, model_type, depth, width,
residual, activation, layer_norm, n_ln, model, optimizer{type,lr,...}, loss, metric, mean, std. You may use it to check
how similar setups compared; it does not contain this question's dataset."""
CORPUS_RULE = """
Evidence discipline: an epoch has only ~15 questions; python's load_corpus() has ~490 older questions with every
choice's measured loss. Before revising or narrowing a claim because of 1-3 counterexamples this epoch, measure the
claim's hit rate on the corpus for its scope; only change it if the change keeps (or raises) that hit rate. Prefer
adding a narrowly-scoped exception claim over weakening a rule that holds broadly. Preserve supported corrections
and measurements across epochs; a small validation score change does not establish that an individual edit is wrong."""

SUMMARIZER_SYSTEM = """You are the SUMMARIZER: curator of a knowledge base (KB) of empirical laws about neural-network training,
used to predict which training setup (architecture, optimizer, loss, budget) wins on a given dataset instance
(ArchitectureIQ). Future solvers see only a question plus the KB rendered read-only (top {top} claims by credibility
plus the {probation} newest low-evidence claims). Your job at the end of each epoch: read every solver/discoverer
record of the epoch, turn hypotheses into science, and keep the KB compact and non-redundant.

Operations (each one is logged as a commit with your reason):
- add_claim: only general, quantitative, falsifiable rules with explicit scope that survive a backtest on history
  (use python / search_history). Cite source question IDs. At most {max_add} additions per epoch.
- merge_claims: fuse near-duplicates or special cases of one rule into one sharper claim. s and f are summed by default.
  Passing a single ID rewrites that claim in place (counts kept).
- delete_claim: refuted or out-of-scope claims. Claims currently shown to solvers cannot be deleted (merge or sharpen
  them instead). Others need evidence s+f>=5 with credibility<0.4, or >=2 counterexample question IDs from history.
- adjust_counts: discouraged. Changing s alone needs a concrete reason; changing s+f (total evidence) even more so.

Workflow (stick to this):
1. Turn 1-3: load history, scan comments/hypotheses, and identify the 2-4 most important failure modes or new laws.
2. Turn 4-8: backtest only the specific claims you intend to add or merge; do NOT do open-ended exploration.
3. Turn 9+: issue add_claim / merge_claims / delete_claim operations in parallel as soon as each is checked.
4. Last 3 turns: call finish(science=...).

Aim for about {max_chars} characters per short claim: one rule, its scope, and the key numbers. This is a writing
target, not a write limit; preserve necessary conditions and put long derivations in linked reports. You have about {turns} turns total.
Finish by calling finish(science=...): a concise Science-of-AI digest, the strongest current laws (3-8, each one
sentence with quantitative scope and claim IDs), plus one line on what changed this epoch."""

# ---- science mode (--science): claims carry mechanism / regime / phase; solver can kb_explain; summarizer can research
MECH_MAX_CHARS, PHASE_MAX_CHARS = 500, 400
SCIENCE_FIELDS = ("mechanism", "regime", "phase", "refines")
REGIME_DIMS = ("family", "type", "model_type", "optimizer", "lr", "steps", "budget", "delta", "activation", "depth",
               "width", "residual", "n_ln", "loss", "data")

KB_TOOL_NOTE_SCIENCE = KB_TOOL_NOTE.replace(
    "add one line <cited>claim IDs</cited> after the final <answer> tag.",
    "add one line <cited>claim IDs</cited> after the final <answer> tag. `kb_explain(claim_id)` returns a claim's "
    "mechanism (why it holds), its regime (where it was measured) and its phase boundary (where the comparison flips) "
    "- use it when a claim's applicability to this question is unclear.")

SCIENCE_RULE = """
Science mode. Claims may be scoped empirical facts or scientific hypotheses. Besides concise `text` (aim for about
{max_chars} chars), a claim has optional structured fields that solvers can fetch on demand with kb_explain:
- mechanism (aim for about {mech} chars): WHY the regularity holds - the causal story in training-dynamics terms (signal
  propagation, effective step size, curvature, noise, capacity vs optimisation). A mechanism must be LOAD-BEARING:
  it should predict where the rule breaks (and that prediction should match the phase field / data). Pure narrative
  that predicts nothing is decoration - leave the field empty instead.
- regime (JSON object): where the rule was measured; recommended dims are {dims}, and additional relevant dimensions
  are allowed. Values are lists of
  categories or [lo, hi] numeric ranges, e.g. {{"family": ["univariate_regression"], "type": ["architecture_only"],
  "optimizer": ["Adam", "AdamW"], "lr": [1e-4, 3e-3], "steps": [256, 1024]}}. Omit a dim only if the rule truly holds
  across all its values in the data. Free-text caveats go under "note".
- phase (aim for about {phase} chars): for comparative claims (A beats B), the boundary where the winner flips, as a compact
  table or inequality with counts, e.g. "LN-vs-plain: plain wins when Δ<0.01 (9/12); LN wins Δ>=0.03 (41/45)".
- refines: ID of the broader claim this one conditions (builds a multi-scale hierarchy: global law -> regime refinements).
Use annotate_claim to add these fields to existing claims without changing their text or counts.

Evidence discipline (think of each claim as conclusion / scope / evidence / boundary):
- Prefer one-factor-at-a-time paired comparisons (same set, everything else equal) over confounded pooled means.
- Report win rate AND effect size (n, median loss ratio); a direction without magnitude is only a "tendency".
- If two designs/regimes give opposite signs, write separate conditioned claims (or a phase field) - never blend
  them into one averaged rule. Every claim needs its own regime; never extrapolate outside it without saying so.
- Compression is the goal: when several claims are special cases of one mechanism, merge them into a more general
  law whose regime/phase records the conditions, rather than letting the KB grow into a list of facts.

A `research` tool starts a research assistant (another model) for one focused question; it has python over the lab
database (load_lab(): ~11k measured candidates, 10 seeds each, grouped by set_id = one dataset + one budget; compare
rows only within a set) plus the corpus, and can run small controlled training experiments (vary one factor, hold the
rest). Use it to map the phase diagram of a comparative claim or to test a proposed mechanism (e.g. "does LN's
advantage over plain MLPs shrink as lr*T grows, and where does it flip?"). At most {research} research calls per epoch;
ask precise questions. Experiments cannot cover factors absent from the lab - say so in the claim instead of guessing.
Strong-solver gains are measured at macro checkpoints. Curate claims by their evidence, scope and counterexamples;
do not require each edit to improve a small validation set, and do not add claims just to fill fields."""

CONTROL_RULE = """
Solver-utility signal: every training question is also solved once WITHOUT the KB (noKB_score), and you get a
cumulative per-claim table of (KB score - no-KB score) on questions where the solver retrieved the claim. A claim can
be true in the lab yet hurt the solver (it gets applied outside its regime, or its text overrides a better default).
When a claim with decent n is negative, find out why from the records (mis-scoped retrieval? wrong first sentence?
too many caveats?) and fix the TEXT the solver reads - lead with the condition, then the rule - or narrow it. Science
fields are for kb_explain; the first 200 characters of text are what decides the solver's answer."""

COLLECT_RULE = """
PHASE: COLLECT (stamp collecting). This epoch, grow the KB with concrete, scoped empirical facts that would have
changed a solver's answer: what won, in which regime, by how much (counts + effect size). A mechanism is OPTIONAL now
(write one only if the data already pin it down); do not merge or generalise yet unless two claims are literal
duplicates. Prefer facts that correct a SYSTEMATIC solver bias (see the bias table) over facts the solver already
gets right. There is no per-epoch gate: the KB is reviewed macroscopically every few epochs, so add freely but only
what is backed by the lab/corpus/epoch data."""

CONSOLIDATE_RULE = """
PHASE: CONSOLIDATE. The KB has accumulated many scoped facts. This epoch your job is to turn them into science:
1. Cluster the claims by the factor they are about (e.g. effective step size, normalisation, depth, activation x
   target, data properties). For each cluster, look at the facts side by side.
2. Use research calls (you have more this epoch) to run the experiments that would explain a cluster: map the phase
   diagram across the regime boundary, test a candidate mechanism's prediction, replay on other datasets.
3. Replace a cluster of facts by ONE law only when the law (mechanism + phase table) reproduces the facts' numbers;
   keep the concrete thresholds in the text - solvers act on numbers, and over-compressed laws have cost score before.
4. Prioritise the solver's systematic biases (bias table): a law that tells a strong solver exactly where its default
   intuition fails is the most valuable kind of claim."""

RESEARCH_SYSTEM = """You are a RESEARCH ASSISTANT for a science-of-training knowledge base (ArchitectureIQ: predict which
training setup - architecture, optimizer, lr, budget - reaches the lowest held-out loss on small synthetic tasks).
The curator asks you ONE focused question. Answer it with data, not opinion.

Tools:
- python: sandboxed Python (numpy; no torch, no network). load_lab() -> list of measured candidates (fields: family,
  dataset_id, set_id, candidate_id, dataset (generator params), budget{{training_steps,batch_size,total_samples_seen}},
  params, model_type, depth, width, residual, activation, layer_norm, n_ln, model, optimizer{{type,lr,...}}, loss,
  metric, mean (10-seed mean of the metric; lower is better), std). Rows are only comparable within one set_id.
  load_corpus() -> older benchmark questions with structured choices and measured losses. load_kb() -> current claims.
- run_experiment: controlled training (10 seeds, same pipeline as the benchmark labels) on a lab dataset. Give a
  base candidate (set_id/candidate_id from the lab) and a list of variants as dotted-key overrides, e.g.
  {{"model.layer_norm": [true, true, true]}}, {{"optimizer.lr": 0.001}}, {{"budget.training_steps": 1024,
  "budget.batch_size": 32}}. Up to {max_cands} variants per call, {max_exps} calls in total; ~5-60 s each.
  Use it for interventions the observational data cannot settle (confounded or missing cells). Optional
  dataset="<family>/<dataset_id>" replays the same variants on ANOTHER lab dataset of the family: use it to find which
  DATA properties (target expression shape, rule family, active/distractor features, spiral turns, vocab/alpha) flip a
  comparison - data x architecture interactions are what solvers get wrong most often.

Method: (1) quantify the question in the lab (pairwise win rates within sets, conditional on the regime), (2) find
where the effect changes sign (the phase boundary), (3) if useful, run 1-{max_exps} controlled experiments across that
boundary, (4) state a mechanism only if your data support a prediction it makes. Report counts (wins/n) and effect
sizes, the regime actually covered, and what the data cannot tell. You have about {max_calls} tool calls.

Finish with plain text:
<report>at most 350 words: answer, phase table (condition -> win rate n, median loss ratio), mechanism + the
prediction you tested, coverage caveats</report>"""

HARNESS_VERSION = hashlib.sha1((KB_BLOCK + DISCOVER_PROMPT + GUIDE_RIGHT + GUIDE_WRONG + SUMMARIZER_SYSTEM)
                               .encode()).hexdigest()[:10]
SCIENCE_HARNESS = hashlib.sha1((HARNESS_VERSION + KB_TOOL_NOTE_SCIENCE + SCIENCE_RULE + RESEARCH_SYSTEM + CONTROL_RULE)
                               .encode()).hexdigest()[:10]


# ------------------------------------------------------------------ utils

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8") as out:
        out.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def tag(text: str, name: str) -> str | None:
    found = re.findall(rf"<{name}>(.*?)</{name}>", text or "", re.S | re.I)
    return found[-1].strip() if found else None


def credibility(s: float, f: float) -> float:
    return (s + 1.0) / (s + f + 2.0)


# ------------------------------------------------------------------ scoring

def grade(q: dict, text: str) -> tuple[str | None, float, int | None]:
    raw = tag(text, "answer")
    if raw is None:
        return None, 0.0, None
    letters = "ABCDE"[: q["num_choices"]]
    if q["task"] == "select":
        pred = raw.strip().upper()
        return (pred, float(pred == q["answer"]), None) if pred in letters else (None, 0.0, None)
    order = re.findall(r"[A-E]", raw.upper())
    gold = re.findall(r"[A-E]", q["answer"])
    if sorted(order) != sorted(gold):
        return "<".join(order) or None, 0.0, None
    pos = {c: i for i, c in enumerate(order)}
    inv = sum(1 for i in range(len(gold)) for j in range(i + 1, len(gold)) if pos[gold[i]] > pos[gold[j]])
    return "<".join(order), RANK_CREDIT.get(inv, 0.0), inv


# ------------------------------------------------------------------ API client

class Client:
    def __init__(self, provider: str, model: str):
        cfg = json.loads(KEYS.read_text())[provider]
        self.url = cfg["base_url"].rstrip("/") + "/chat/completions"
        self.key, self.model = cfg["api_key"], model
        self.headers = cfg.get("extra_headers", {})
        self.usage = collections.Counter()
        self.lock = threading.Lock()

    def chat(self, messages: list[dict], effort: str, max_tokens: int, tools: list | None = None) -> dict:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "reasoning_effort": effort}
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload).encode()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "User-Agent": "curl/8.7.1", **self.headers}  # cctq sits behind Cloudflare (1010 on urllib UA)
        for attempt in range(10):
            t0 = time.time()
            try:
                req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=1200, context=ssl._create_unverified_context()) as r:
                    doc = json.loads(r.read())
                if not doc.get("choices"):
                    raise RuntimeError(f"no choices: {str(doc)[:300]}")
                usage = doc.get("usage") or {}
                with self.lock:
                    self.usage[f"{effort}_prompt"] += usage.get("prompt_tokens") or 0
                    self.usage[f"{effort}_completion"] += usage.get("completion_tokens") or 0
                    self.usage["calls"] += 1
                msg = doc["choices"][0]["message"]
                return {"message": msg, "usage": usage, "secs": round(time.time() - t0, 1),
                        "finish": doc["choices"][0].get("finish_reason")}
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
                err = f"HTTP {exc.code}: {detail}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError, json.JSONDecodeError) as exc:
                err = repr(exc)[:300]
            time.sleep(min(120, 4 * 2 ** attempt) + 5 * random.random())
        raise RuntimeError(f"API failed after retries: {err}")


def clean_assistant(msg: dict) -> dict:
    out = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def tool_loop(client: Client, messages: list[dict], tools: list, handler, effort: str, max_tokens: int,
              max_turns: int, stop_tool: str | None = None, trace_path: Path | None = None,
              final_nudge: str | None = None, warn_turns: int = 2,
              warn_text: str = "wrap up now.") -> tuple[str, list[dict]]:
    """Run up to max_turns tool turns; if the budget runs out, one extra turn after final_nudge."""
    trace = []

    def step(turn: int) -> tuple[bool, str]:
        res = client.chat(messages, effort, max_tokens, tools)
        msg = res["message"]
        messages.append(clean_assistant(msg))
        calls = msg.get("tool_calls") or []
        entry = {"turn": turn, "content": msg.get("content"), "reasoning": msg.get("reasoning_content"),
                 "tool_calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in calls],
                 "usage": res["usage"], "secs": res["secs"]}
        trace.append(entry)
        stop, left = not calls, max_turns - 1 - turn
        for c in calls:
            name = c["function"]["name"]
            try:
                result = handler(name, json.loads(c["function"]["arguments"] or "{}"))
            except Exception as exc:  # tool errors go back to the model
                result = f"[tool error] {type(exc).__name__}: {exc}"
            if 0 < left <= warn_turns:
                result += f"\n[budget] {left} turn(s) left: {warn_text}"
            entry.setdefault("results", []).append(result[:4000])
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            stop = stop or (name == stop_tool and not result.startswith("rejected"))
        if trace_path:
            write_json(trace_path, trace)
            print(f"[{trace_path.parent.name}] turn {turn}: {[c['name'] for c in entry['tool_calls']]}", flush=True)
        return stop, msg.get("content") or ""

    for turn in range(max_turns):
        stop, text = step(turn)
        if stop:
            return text, trace
    if final_nudge:
        messages.append({"role": "user", "content": final_nudge})
        return step(max_turns)[1], trace
    return "", trace


# ------------------------------------------------------------------ KB with commit log

def science_fields(a: dict) -> tuple[dict, str | None]:
    """Preserve scientific annotations; check object shape without restricting their content."""
    out = {}
    for key in ("mechanism", "phase"):
        v = a.get(key)
        if v is None or v == "":
            continue
        v = str(v).strip()
        out[key] = v
    reg = a.get("regime")
    if reg not in (None, "", {}):
        if isinstance(reg, str):
            try:
                reg = json.loads(reg)
            except json.JSONDecodeError:
                reg = {"note": reg}
        if not isinstance(reg, dict):
            return {}, "rejected: regime must be a JSON object"
        out["regime"] = reg
    if a.get("refines"):
        out["refines"] = str(a["refines"]).strip()
    return out, None


def regime_line(reg: dict | None) -> str:
    if not reg:
        return ""
    parts = []
    for k, v in reg.items():
        if k == "note":
            continue
        parts.append(f"{k}={'-'.join(map(str, v)) if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) else ','.join(map(str, v)) if isinstance(v, list) else v}")
    return "; ".join(parts)


def kb_search(kb: "KB", query: str, k: int = 5, with_regime: bool = False) -> list[dict]:
    """Plain BM25 over claim text (+ regime words in science mode; no credibility mixing)."""
    claims = list(kb.claims.values())
    toks = [TOK_RE.findall((c["text"] + (" " + regime_line(c.get("regime")).replace("_", " ") if with_regime else ""))
                           .lower()) for c in claims]
    q = TOK_RE.findall((query or "").lower())
    if not claims or not q:
        return []
    df = collections.Counter(w for ts in toks for w in set(ts))
    n, avgdl, k1, b = len(toks), sum(map(len, toks)) / len(toks), 1.5, 0.75

    def score(ts):
        tf = collections.Counter(ts)
        return sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1)
                   / (tf[w] + k1 * (1 - b + b * len(ts) / avgdl)) for w in q if tf.get(w))

    ranked = sorted(((score(ts), c) for ts, c in zip(toks, claims)), key=lambda x: (-x[0], x[1]["id"]))
    return [c for sc, c in ranked[:k] if sc > 0]


class KB:
    def __init__(self, doc: dict, run: Path):
        self.claims = {c["id"]: c for c in doc["claims"]}
        self.retired = doc.get("retired", [])
        self.reports = copy.deepcopy(doc.get("reports", {}))
        self.applied_publications = list(doc.get("applied_publications", []))
        self.credit_keys = set(doc.get("credit_keys", []))
        self.publication_id_maps = copy.deepcopy(doc.get("publication_id_maps", {}))
        self.epoch = doc["epoch"]
        self.commits_path = run / "commits.jsonl"
        self.pending: list[dict] = copy.deepcopy(doc.get("pending_commits", []))
        self.publish_commit = None
        self.counter = max([doc.get("commit_counter", 0)] +
                           [int(r["commit"][1:]) for r in read_jsonl(self.commits_path) + self.pending])

    @classmethod
    def seed(cls, run: Path, path: Path = SEED_KB) -> "KB":
        src = json.loads(path.read_text())
        claims = []
        for c in src["claims"]:
            s, f = float(c["support_count"]), float(c["failure_count"])
            claims.append({"id": c["id"], "text": c["text"], "support_count": s, "failure_count": f,
                           "credibility": credibility(s, f), "created_epoch": 0, "sources": [],
                           "merged_from": c.get("merged_from", []),
                           "origin": c.get("origin") or f"seed:{path.name}",
                           **{k: c[k] for k in SCIENCE_FIELDS if c.get(k)}})
        return cls({"schema_version": "architectureiq_kb_v4", "epoch": 0, "claims": claims}, run)

    def to_doc(self) -> dict:
        return {"schema_version": "architectureiq_kb_v4", "epoch": self.epoch, "harness": HARNESS_VERSION,
                "updated_at": now(), "claims": copy.deepcopy(sorted(self.claims.values(), key=lambda c: c["id"])),
                "retired": copy.deepcopy(self.retired), "reports": copy.deepcopy(self.reports),
                "applied_publications": list(self.applied_publications), "credit_keys": sorted(self.credit_keys),
                "publication_id_maps": copy.deepcopy(self.publication_id_maps),
                "commit_counter": self.counter, "pending_commits": copy.deepcopy(self.pending)}

    def next_id(self) -> str:
        nums = [int(i[1:]) for i in list(self.claims) + [r["id"] for r in self.retired] if i[1:].isdigit()]
        return f"K{max(nums, default=0) + 1:04d}"

    def commit(self, op: str, author: dict, **fields) -> dict:
        self.counter += 1
        rec = {"commit": f"c{self.counter:06d}", "epoch": self.epoch, "time": now(), "op": op,
               "author": {**author, "harness": HARNESS_VERSION},
               "base_snapshot": f"kb_{self.epoch - 1:04d}" if self.epoch > 0 else None, **fields}
        self.pending.append(rec)
        return rec

    def checkpoint(self) -> None:
        if self.publish_commit and self.pending:
            self.publish_commit(self.pending[-1], self.to_doc())

    def flush(self) -> None:
        with self.commits_path.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            existing = {c["commit"] for c in read_jsonl(self.commits_path)}
            for rec in self.pending:
                if rec["commit"] not in existing:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.pending = []

    def _set(self, c: dict, s: float, f: float) -> None:
        c["support_count"], c["failure_count"] = round(s, 4), round(f, 4)
        c["credibility"] = credibility(s, f)

    # ---- ops
    def credit(self, deltas: dict[str, tuple[float, float]], author: dict, qid: str) -> None:
        key = f"{self.epoch}:{qid}"
        if key in self.credit_keys:
            return
        applied = {}
        retired = {c["id"]: c for c in self.retired}
        for cid, (ds, df) in deltas.items():
            target, seen = cid, set()
            while target in retired and retired[target].get("merged_into") and target not in seen:
                seen.add(target)
                target = retired[target]["merged_into"]
            c = self.claims.get(target) or retired.get(target)
            if c is not None:
                self._set(c, c["support_count"] + ds, c["failure_count"] + df)
                c["last_evaluated_epoch"] = self.epoch
                old_ds, old_df = applied.get(target, (0, 0))
                applied[target] = [round(ds + old_ds, 4), round(df + old_df, 4)]
        if applied:
            self.commit("credit", author, claim_ids=sorted(applied), payload={"deltas": applied},
                        source_question_ids=[qid])
            self.credit_keys.add(key)

    def add(self, text: str, sources: list[str], reason: str, author: dict, fields: dict | None = None) -> str:
        cid = self.next_id()
        rec = self.commit("add", author, claim_ids=[cid], payload={"text": text, **(fields or {})},
                          source_question_ids=sources, reason=reason)
        self.claims[cid] = {"id": cid, "text": text, "support_count": 0.0, "failure_count": 0.0,
                            "credibility": 0.5, "created_epoch": self.epoch, "sources": sources, "merged_from": [],
                           "origin": rec["commit"], **(fields or {})}
        self.checkpoint()
        return cid

    def annotate(self, cid: str, fields: dict, reason: str, author: dict) -> None:
        c = self.claims[cid]
        self.commit("annotate", author, claim_ids=[cid], payload={"old": {k: c.get(k) for k in fields}, "new": fields},
                    reason=reason)
        c.update(fields)
        self.checkpoint()

    def delete(self, cid: str, reason: str, author: dict) -> None:
        c = self.claims.pop(cid)
        rec = self.commit("delete", author, claim_ids=[cid], reason=reason)
        self.retired.append({**c, "retired_epoch": self.epoch, "retired_by": rec["commit"], "reason": reason})
        self.checkpoint()

    def merge(self, ids: list[str], text: str, reason: str, author: dict, s: float | None, f: float | None,
              fields: dict | None = None) -> str:
        olds = [self.claims[i] for i in ids]
        ss = sum(c["support_count"] for c in olds) if s is None else s
        ff = sum(c["failure_count"] for c in olds) if f is None else f
        sources = sorted({q for c in olds for q in c.get("sources", [])})
        if len(ids) == 1:
            c = olds[0]
            self.commit("revise", author, claim_ids=ids, payload={"old_text": c["text"], "text": text,
                        "counts": [ss, ff], **(fields or {})}, reason=reason)
            c["text"] = text
            c.update(fields or {})
            self._set(c, ss, ff)
            self.checkpoint()
            return c["id"]
        new = self.next_id()
        rec = self.commit("merge", author, claim_ids=ids, new_id=new, payload={"text": text, "counts": [ss, ff],
                          "default_counts": s is None and f is None, **(fields or {})}, source_question_ids=sources,
                          reason=reason)
        for i in ids:
            c = self.claims.pop(i)
            self.retired.append({**c, "retired_epoch": self.epoch, "retired_by": rec["commit"], "merged_into": new})
        claim = {"id": new, "text": text, "created_epoch": self.epoch, "sources": sources, "merged_from": ids,
                 "origin": rec["commit"], **(fields or {})}
        report_ids = sorted({r for c in olds for r in c.get("report_ids", [])})
        if report_ids:
            claim["report_ids"] = report_ids
        for report in self.reports.values():
            if any(i in ids for i in report.get("claim_ids", [])):
                report["claim_ids"] = sorted((set(report["claim_ids"]) - set(ids)) | {new})
        self._set(claim, ss, ff)
        self.claims[new] = claim
        self.checkpoint()
        return new

    def adjust(self, cid: str, s: float, f: float, reason: str, author: dict) -> None:
        c = self.claims[cid]
        old = [c["support_count"], c["failure_count"]]
        self.commit("adjust", author, claim_ids=[cid], payload={"old": old, "new": [s, f],
                    "total_changed": abs((s + f) - sum(old)) > 1e-9}, reason=reason)
        self._set(c, s, f)
        self.checkpoint()

    def rollback(self, best: dict, reason: str, payload: dict) -> None:
        """Restore the claims of an accepted snapshot; claims born after it are retired (IDs never reused)."""
        keep = {c["id"] for c in best["claims"]} | {r["id"] for r in best.get("retired", [])}
        rec = self.commit("rollback", {"role": "gate", "model": "-"}, claim_ids=[], payload=payload, reason=reason)
        dropped = [{**c, "retired_epoch": self.epoch, "retired_by": rec["commit"], "reason": "rolled back"}
                   for c in list(self.claims.values()) + self.retired if c["id"] not in keep]
        self.claims = {c["id"]: dict(c) for c in best["claims"]}
        self.retired = list(best.get("retired", [])) + dropped

    # ---- views
    def rendered(self, top: int = RENDER_TOP, probation: int = RENDER_PROBATION) -> list[dict]:
        ranked = sorted(self.claims.values(), key=lambda c: (-c["credibility"], c["id"]))
        chosen = ranked[:top]
        ids = {c["id"] for c in chosen}
        fresh = sorted((c for c in self.claims.values() if c["id"] not in ids
                        and c["support_count"] + c["failure_count"] < 2),
                       key=lambda c: (-c["created_epoch"], c["id"]))[:probation]
        return sorted(chosen + fresh, key=lambda c: c["id"])

    def render(self, top: int = RENDER_TOP, probation: int = RENDER_PROBATION) -> str:
        return "\n".join(f"[{c['id']}] (credibility {c['credibility']:.2f}; s={c['support_count']:.1f}, "
                         f"f={c['failure_count']:.1f}) {c['text']}" for c in self.rendered(top, probation))

    def render_for(self, query: str, boost: list[str] | None = None, top: int = RENDER_TOP,
                   probation: int = RENDER_PROBATION, backbone: int = 8, mix: float = 0.7) -> str:
        """Per-question retrieval rendering (harness-side; the solver still gets no tools).

        Score = normalized BM25(claim, question + boosted metadata terms) + mix * credibility,
        with the `backbone` highest-credibility claims always kept visible, then the usual
        probation slots. Deterministic: ties broken by credibility then id.
        """
        claims = list(self.claims.values())
        toks = [TOK_RE.findall(c["text"].lower()) for c in claims]
        df: collections.Counter[str] = collections.Counter(w for ts in toks for w in set(ts))
        n = len(toks)
        avgdl = sum(len(t) for t in toks) / max(1, n)
        q = TOK_RE.findall(query.lower()) + [w for b in (boost or []) for w in TOK_RE.findall(b.lower())] * 2
        k1, bb = 1.5, 0.75

        def bm25(ts: list[str]) -> float:
            tf = collections.Counter(ts)
            return sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1)
                       / (tf[w] + k1 * (1 - bb + bb * len(ts) / avgdl)) for w in q if tf.get(w))

        raw = [bm25(ts) for ts in toks]
        mx = max(raw, default=1.0) or 1.0
        ranked = sorted(zip(claims, raw), key=lambda cr: (-(cr[1] / mx + mix * cr[0]["credibility"]),
                                                          -cr[0]["credibility"], cr[0]["id"]))
        chosen = [c for c, _ in ranked[:top]]
        ids = {c["id"] for c in chosen}
        for c in sorted(claims, key=lambda c: (-c["credibility"], c["id"])):
            if len(chosen) >= top:
                break
            if c["id"] not in ids:
                chosen.append(c)
                ids.add(c["id"])
        fresh = sorted((c for c in claims if c["id"] not in ids
                        and c["support_count"] + c["failure_count"] < 2),
                       key=lambda c: (-c["created_epoch"], c["id"]))[:probation]
        return "\n".join(f"[{c['id']}] (credibility {c['credibility']:.2f}; s={c['support_count']:.1f}, "
                         f"f={c['failure_count']:.1f}) {c['text']}" for c in chosen + fresh)

    def index(self) -> str:
        rows = sorted(self.claims.values(), key=lambda c: c["id"])
        return "\n".join(f"{c['id']} cred={c['credibility']:.2f} s={c['support_count']:.1f} f={c['failure_count']:.1f} "
                         f"ep={c['created_epoch']}{' [' + ''.join(k[0].upper() for k in SCIENCE_FIELDS if c.get(k)) + ']' if any(c.get(k) for k in SCIENCE_FIELDS) else ''}"
                         f" | {c['text'][:110]}" for c in rows)


# ------------------------------------------------------------------ history (agent-visible store)

def norm(x: str) -> str:
    return re.sub(r"[\s_\-]+", " ", x.lower()).strip()


class History:
    def __init__(self, rows: list[dict]):
        self.rows = {r["question_id"]: r for r in rows}

    def search(self, args: dict) -> str:
        out = []
        for r in self.rows.values():
            if args.get("source") and r["source"] != args["source"]:
                continue
            if args.get("task") and r["task"] != args["task"]:
                continue
            if args.get("family") and norm(args["family"]) not in norm(r["family"]):
                continue
            if args.get("max_score") is not None and r["score"] > float(args["max_score"]):
                continue
            if args.get("min_score") is not None and r["score"] < float(args["min_score"]):
                continue
            if args.get("contains") and norm(args["contains"]) not in norm(r["question"] + " " + r["explanation"]):
                continue
            out.append(r)
        out.sort(key=lambda r: (-r["epoch"], r["question_id"]))
        limit = int(args.get("limit") or 20)
        lines = [f"{r['question_id']} | ep{r['epoch']} {r['source']} {r['task']} {r['family']} | gold={r['answer']} "
                 f"pred={r['prediction']} score={r['score']:.2f} | {(r.get('comment') or '')[:140]}"
                 for r in out[:limit]]
        return f"{len(out)} matches (showing {len(lines)})\n" + "\n".join(lines) if lines else "0 matches"

    def read(self, args: dict) -> str:
        r = self.rows.get(args.get("question_id", ""))
        if not r:
            return "not found (only earlier-epoch questions are visible)"
        fields = args.get("fields") or ["question", "answer", "prediction", "score", "explanation", "comment", "hypothesis"]
        doc = {k: r.get(k) for k in ["question_id", "epoch", "source", "task", "family", *fields]}
        if isinstance(doc.get("question"), str) and len(doc["question"]) > 14000:
            doc["question"] = doc["question"][:14000] + " ...[truncated]"
        return json.dumps(doc, ensure_ascii=False, indent=1)


def tools_spec(summarizer: bool, corpus: bool = False, kb_tool: bool = False, science: bool = False,
               lab: bool = False, research: bool = False) -> list[dict]:
    def fn(name, desc, props, req=()):
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": props, "required": list(req)}}}
    s, n = {"type": "string"}, {"type": "number"}
    tools = [
        fn("search_history", "List past questions (with gold answers and model scores) matching filters.",
           {"source": {"type": "string", "enum": ["arch170", "ranking_v2", "ranking_v3", "dataflip500"]},
            "task": {"type": "string", "enum": ["select", "ranking"]}, "family": s, "contains": s,
            "min_score": n, "max_score": n, "limit": {"type": "integer"}}),
        fn("read_record", "Read one past question: prompt, gold answer, model solution, comment, hypothesis.",
           {"question_id": s, "fields": {"type": "array", "items": s}}, ["question_id"]),
        fn("python", "Run sandboxed Python 3 (numpy; no network; 30s CPU). load_history() -> list of past records "
           "(question_id, epoch, source, task, family, question, answer, prediction, score, explanation, comment, "
           "hypothesis); load_kb() -> list of KB claims." + (" load_corpus() -> ~490 older questions with structured "
           "choices and measured losses." if corpus else "") + (" load_lab() -> ~11k measured candidates (10-seed mean "
           "loss; compare only within one set_id)." if lab else "") + " Print results.", {"code": s}, ["code"]),
    ]
    if kb_tool and not summarizer:
        tools.insert(0, fn("kb_open", "Open multiple claim IDs, report IDs or science in the frozen KB; returns complete saved reports.",
                           {"targets": {"type": "array", "items": s}}, ["targets"]))
        tools.insert(0, fn("kb_search", "Search the read-only knowledge base of empirical training laws. Returns up to k "
                           "best-matching claims (BM25 over claim text) with id, credibility and evidence.",
                           {"query": s, "k": {"type": "integer"}}, ["query"]))
        if science:
            tools.insert(1, fn("kb_explain", "Explain one KB claim: its mechanism (why), regime (where measured), phase "
                               "boundary (where the comparison flips) and evidence counts.", {"claim_id": s},
                               ["claim_id"]))
    sci = {"mechanism": s, "regime": {"type": "object"}, "phase": s, "refines": s} if science else {}
    if summarizer:
        tools += [
            fn("kb_open", "Read saved ongoing science reports and claim evidence.",
               {"targets": {"type": "array", "items": s}}, ["targets"]),
            fn("kb_view", "Show full KB claims. Filter by ids or substring; sort by credibility|evidence|id.",
               {"ids": {"type": "array", "items": s}, "contains": s, "sort": s, "limit": {"type": "integer"}}),
            fn("add_claim", "Add a new claim (commit).", {"text": s, "source_question_ids": {"type": "array", "items": s},
               "reason": s, **sci}, ["text", "source_question_ids", "reason"]),
            fn("delete_claim", "Delete a refuted or out-of-scope claim (commit). Requires s+f>=5 and credibility<0.4, "
               "or >=2 counterexample question IDs from history. Legacy top-claim rendering protects shown claims; "
               "tool and full-short rendering allow retirement while preserving frozen solver snapshots.",
               {"claim_id": s, "reason": s, "counterexample_question_ids": {"type": "array", "items": s}},
               ["claim_id", "reason"]),
            fn("merge_claims", "Merge claims into one new claim; s,f summed unless overridden (discouraged). "
               "One id = rewrite in place.", {"claim_ids": {"type": "array", "items": s}, "text": s, "reason": s,
               "support": n, "failure": n, **sci}, ["claim_ids", "text", "reason"]),
            fn("adjust_counts", "Set s and f of a claim (discouraged; justify).", {"claim_id": s, "support": n,
               "failure": n, "reason": s}, ["claim_id", "support", "failure", "reason"]),
            fn("finish", "End curation with the Science-of-AI digest.", {"science": s}, ["science"]),
        ]
        if science:
            tools.insert(-1, fn("annotate_claim", "Set mechanism / regime / phase / refines of an existing claim without "
                                "changing its text or counts (commit).", {"claim_id": s, "reason": s, **sci},
                                ["claim_id", "reason"]))
        if research:
            tools.insert(-1, fn("research", "Start a research assistant on ONE focused question (lab database + "
                                "controlled training experiments); returns its report (takes minutes).",
                                {"question": s, "focus": s, "topic": s}, ["question"]))
    if lab and not summarizer and research:
        tools.append(fn("run_experiment", "Controlled training experiment on a lab dataset: base candidate + variants "
                        "(dotted-key overrides). Returns mean/std, actual seed outcomes, failures, exclusion and cache "
                        "status. Measurement file paths and hashes include existing per-seed curves; cached results "
                        "are reused measurements, not independent replications. Source provenance says whether "
                        "exact executed model/optimizer/loss/train bytes are pinned; old caches may omit them. "
                        "Optional process records CPU MLP updates: classification CE/coupled Adam or regression "
                        "MSE/Adam or SGD, with step-zero metrics for regression. Optional target_transform "
                        "centers regression train/test labels using the train mean and adds an offset, keeping inputs "
                        "and splits unchanged; unsupported requests are rejected.",
                        {"base": {"type": "string", "description": "set_id/candidate_id from load_lab()"},
                         "variants": {"type": "array", "items": {"type": "object"}},
                         "dataset": {"type": "string", "description": "optional: run the variants on ANOTHER lab dataset "
                                     "of the same family ('<family>/<dataset_id>' or a dataset_id) to test how the "
                                     "comparison depends on the data"},
                         "process": {"type": "object", "additionalProperties": False,
                             "properties": {"steps": {"type": "array", "items": {"type": "integer"},
                                                       "minItems": 1, "maxItems": 16},
                                             "max_eval_samples": {"type": "integer", "minimum": 1, "maximum": 4096}},
                             "required": ["steps"]},
                         "target_transform": {"type": "object", "additionalProperties": False,
                             "properties": {"center": {"type": "boolean"}, "offset": {"type": "number"}}},
                         "diagnostic_fail_threshold": {"type": "number", "exclusiveMinimum": 0,
                             "description": "optional process-only finite MSE cutoff; record high finite losses instead of discarding them as benchmark exclusions"},
                         "why": s}, ["base", "variants"]))
    return tools


# ------------------------------------------------------------------ loop

class Loop:
    def __init__(self, args: argparse.Namespace):
        for field in ("pool", "lab", "corpus", "seed_kb", "publish_blog"):
            if getattr(args, field, None):
                setattr(args, field, str(Path(getattr(args, field)).resolve()))
        self.args = args
        CONFIG.run_root(args.run_name)
        self.run = OUT_ROOT / args.run_name
        self.run.mkdir(parents=True, exist_ok=True)
        self.client = Client(args.provider, args.model)
        self.client_wrong = (Client(args.discover_wrong_provider or args.provider, args.discover_wrong_model)
                             if args.discover_wrong_model and args.discover_wrong_backend == "api" else self.client)
        self.lock = threading.RLock()
        self.lab_sem = threading.Semaphore(1)
        self.cur_epoch = 0
        self.summ_model = args.summ_model or args.model
        self.cc_cost = 0.0
        self.pool = {q["question_id"]: q for q in read_jsonl(Path(args.pool)) if q["source"] in args.sources}
        release_split = Path(args.pool).parent / "split.json"
        if (release_split.parent / "manifest.json").exists():
            try:
                from .sample_kb_science_release import verify
            except ImportError:
                from sample_kb_science_release import verify
            verify(release_split.parent)
        if release_split.exists():
            groups = json.loads(release_split.read_text()).get("group_by_question", {})
            for qid, q in self.pool.items():
                if qid in groups:
                    q["group"] = groups[qid]
        self.author_solver = {"role": "solver", "model": args.model}
        self.author_summ = {"role": "summarizer", "model": self.summ_model, "backend": args.summ_backend}
        self._split()
        if args.async_agents and args.lab:
            validate_allowlist(Path(args.lab).resolve().parent, Path(args.pool))
        self.jobs = Jobs(self.run, Path(__file__).resolve(), timeout=args.agent_timeout,
                         max_workers=args.job_workers) if args.async_agents else None
        if args.async_agents and args.summ_backend == "api":
            raise ValueError("durable curation requires --summ-backend codex or claude_code")
        self.worker_job = None
        self.worker_path = None
        self.job_base = None
        self.job_sequence = 0
        self.historical_pools = {}

    def _split(self) -> None:
        path = self.run / "split.json"
        pool_hash = hashlib.sha256(Path(self.args.pool).read_bytes()).hexdigest()

        def validate(split):
            if split.get("pool_sha256", pool_hash) != pool_hash:
                raise ValueError("run split belongs to a different question pool")
            partitions = {"train": [i for ids in split["stream"].values() for i in ids],
                          "val": split["eval"], "test": split.get("test", [])}
            assigned, groups = set(), {}
            for name, ids in partitions.items():
                for qid in ids:
                    if qid not in self.pool or qid in assigned:
                        raise ValueError(f"invalid or duplicate split question: {qid}")
                    assigned.add(qid)
                    group = self.pool[qid]["group"]
                    if group in groups and groups[group] != name:
                        raise ValueError(f"group crosses train/val/test: {group}")
                    groups[group] = name
            return {**split, "pool_sha256": pool_hash}

        if path.exists():
            self.split = validate(json.loads(path.read_text()))
            return
        release_split = Path(self.args.pool).parent / "split.json"
        if release_split.exists():
            split = json.loads(release_split.read_text())
            if "stream" in split:
                self.split = validate(split)
                write_json(path, self.split)
                return
            if all(k in split for k in ("train", "val", "test", "group_by_question")):
                for qid, group in split["group_by_question"].items():
                    if qid in self.pool:
                        self.pool[qid]["group"] = group
                rng = random.Random(self.args.seed)
                stream = {}
                for src in self.args.sources:
                    ids = [i for i in split["train"] if i in self.pool and self.pool[i]["source"] == src]
                    rng.shuffle(ids)
                    stream[src] = ids
                self.split = validate({"seed": split.get("seed", self.args.seed), "stream": stream,
                              "eval": [i for i in split["val"] if i in self.pool],
                              "test": [i for i in split["test"] if i in self.pool],
                              "eval_groups": sorted({split["group_by_question"][i] for i in split["val"] + split["test"]})})
                write_json(path, self.split)
                return
        rng = random.Random(self.args.seed)
        by_src = collections.defaultdict(lambda: collections.defaultdict(list))
        for q in self.pool.values():
            by_src[q["source"]][q["group"]].append(q["question_id"])
        eval_ids, test_ids, held_groups, stream = [], [], set(), {}
        for src in self.args.sources:
            groups = sorted(by_src[src])
            rng.shuffle(groups)
            for g in groups[: self.args.eval_per_source]:
                eval_ids.append(sorted(by_src[src][g])[0])
                held_groups.add(g)
            ids = [i for g in groups[self.args.eval_per_source:] for i in by_src[src][g]]
            rng.shuffle(ids)
            stream[src] = ids
        # test groups: the groups whose FIRST appearance in the shuffled stream is latest. With this seed the val split
        # and the stream order are unchanged by --test-per-source (test items are just filtered out), and the test
        # groups never appear within the first TEST_RESERVE_EPOCHS batches, so they are unseen by those epochs.
        for src in self.args.sources:
            first = {}
            for k, i in enumerate(stream[src]):
                first.setdefault(self.pool[i]["group"], k)
            tg = set(sorted(first, key=lambda g: -first[g])[: self.args.test_per_source])
            if tg:
                lim = min(first[g] for g in tg)
                per_epoch = -(-self.args.batch_size // len(self.args.sources))
                if lim < TEST_RESERVE_EPOCHS * per_epoch:
                    raise SystemExit(f"{src}: test groups start at stream pos {lim}; reduce --test-per-source")
            test_ids += [sorted(by_src[src][g])[0] for g in sorted(tg)]
            held_groups |= tg
            stream[src] = [i for i in stream[src] if self.pool[i]["group"] not in tg]
        self.split = validate({"seed": self.args.seed, "eval": sorted(eval_ids), "test": sorted(test_ids),
                      "eval_groups": sorted(held_groups),
                      "stream": stream, "created_at": now()})
        write_json(path, self.split)

    def quota(self, epoch: int) -> list[int]:
        srcs = self.args.sources
        per = [self.args.batch_size // len(srcs)] * len(srcs)
        for k in range(self.args.batch_size - sum(per)):
            per[(epoch + k) % len(srcs)] += 1
        return per

    def batch(self, epoch: int) -> list[str]:
        """Stratified, without replacement across epochs: epoch t only ever sees unseen stream questions."""
        if self.args.async_agents:
            schedule = self.run / "batches.json"
            batches = json.loads(schedule.read_text()) if schedule.exists() else {}
            missing = [e for e in range(1, max(epoch, self.args.epochs) + 1) if str(e) not in batches]
            if not missing:
                return batches.get(str(epoch), [])
            queues = {s: list(self.split["stream"].get(s, [])) for s in self.args.sources}
            used = {i for ids in batches.values() for i in ids}
            for name in self.args.import_history:
                for file in (OUT_ROOT / name / "epochs").glob("e*/records.jsonl"):
                    used.update(r["question_id"] for r in read_jsonl(file))
            queues = {s: [i for i in ids if i not in used] for s, ids in queues.items()}
            for e in missing:
                rows = []
                for _ in range(self.args.batch_size):
                    available = [s for s in self.args.sources if queues[s]]
                    if not available:
                        break
                    src = available[(e + len(rows)) % len(available)]
                    rows.append(queues[src].pop(0))
                batches[str(e)] = rows
            write_json(schedule, batches)
            return batches.get(str(epoch), [])
        out = []
        for i, src in enumerate(self.args.sources):
            e_abs = epoch + self.args.stream_offset
            start = sum(self.quota(e)[i] for e in range(1, e_abs))
            out += self.split["stream"][src][start:start + self.quota(e_abs)[i]]
        return out

    # ---- prompts
    def solver_messages(self, q: dict, kb: KB | None) -> list[dict]:
        sys_parts = [m["content"] for m in q["messages"] if m["role"] == "system"]
        if not sys_parts:
            sys_parts = ["You are solving an ArchitectureIQ benchmark question. Answer in the required format."]
        if q.get("task") == "ranking":
            sys_parts.append(RANKING_HINT)
        if self.args.solve_lab:
            sys_parts.append(SOLVE_LAB_NOTE)
        if kb is not None and self.args.render == "brief":
            body = "\n".join(f"[{c['id']}] {c['text']}" for c in sorted(kb.claims.values(), key=lambda c: c["id"]))
            reports = "\n".join(f"[{i}] {r['title']} (claims: {','.join(r.get('claim_ids', []))})"
                                for i, r in kb.reports.items())
            sys_parts.append("# Knowledge base (read-only, frozen for this solve)\n" + body +
                             "\n\n# Science reports\n" + (reports or "No saved reports yet.") +
                             "\nUse kb_open(targets=[claim IDs, report IDs or 'science']) for the full evidence and report. "
                             "Reports may contain unresolved hypotheses. Apply only scoped evidence; report status does not gate access. "
                             "After answering, cite the claim IDs you used in <cited>...</cited>.")
        elif kb is not None and getattr(self.args, "render", "cred") == "tool":
            sys_parts.append((KB_TOOL_NOTE_SCIENCE if self.args.science else KB_TOOL_NOTE).format(n=len(kb.claims)))
        elif kb is not None:
            if getattr(self.args, "render", "cred") == "bm25":
                body = kb.render_for(" ".join(m["content"] for m in q["messages"] if m["role"] != "system"),
                                     boost=[q.get("family", ""), q.get("task", ""), q.get("type", ""), q.get("source", "")])
            else:
                body = kb.render()
            block = KB_BLOCK_VARIANTS.get(getattr(self.args, "kb_block", "default"), KB_BLOCK)
            sys_parts.append(block.format(claims=body))
        return [{"role": "system", "content": "\n\n".join(sys_parts)}] + [m for m in q["messages"] if m["role"] != "system"]

    # ---- solve (+ discover)
    def solve(self, qid: str, rendered_kb: KB | None, hist: History | None, vis_dir: Path | None,
              epoch: int, out_dir: Path, discover: bool) -> dict:
        path = out_dir / f"{qid}.json"
        if path.exists():
            return json.loads(path.read_text())
        q = self.pool[qid]
        messages = self.solver_messages(q, rendered_kb)
        if self.args.solver_backend == "codex":
            return self.solve_codex(q, messages, rendered_kb, hist, vis_dir, epoch, out_dir, discover)
        res = self.client.chat(messages, self.args.solve_effort, self.args.solve_max_tokens)
        text = res["message"].get("content") or ""
        pred, score, inv = grade(q, text)
        cited = sorted(set(CITE_RE.findall(tag(text, "cited") or ""))) if rendered_kb else []
        rec = {"question_id": qid, "epoch": epoch, "source": q["source"], "task": q["task"], "type": q["type"],
               "family": q["family"], "group": q["group"], "answer": q["answer"], "prediction": pred, "score": score,
               "inversions": inv, "correct": score == 1.0, "cited": cited,
               "explanation": tag(text, "explanation") or text[-3000:], "solution": text,
               "solver_reasoning": res["message"].get("reasoning_content"), "solve_usage": res["usage"],
               "solve_secs": res["secs"], "finish": res["finish"]}
        if rendered_kb is not None and discover:
            rec.update(solver_snapshot=getattr(self, "solve_snapshot_label", out_dir.name),
                       solver_selection=getattr(self, "solver_selection", None))
        if discover:
            rec.update(self.discover(q, messages, text, rec, hist, vis_dir, epoch))
        write_json(path, rec)
        return rec

    def discover_prompt(self, q: dict, rec: dict) -> str:
        right = rec["correct"]
        note = f" (ranking: {rec['inversions']} inversions vs gold)" if rec["inversions"] is not None else ""
        return DISCOVER_PROMPT.format(
            gold=q["answer"], pred=rec["prediction"], score=rec["score"], rank_note=note,
            guidance=(GUIDE_RIGHT if right else GUIDE_WRONG) + (CORPUS_NOTE if self.args.corpus else "")
            + (LAB_NOTE if self.args.lab else "")
            + (LAB_EXP_NOTE.format(n=self.args.discover_exps) if self.args.lab and self.args.discover_exps and not right else ""),
            skip_note="You may reply just <skip/>." if right else "")

    def solve_codex(self, q, messages, rendered_kb, hist, vis_dir, epoch, out_dir, discover) -> dict:
        """Solve then discover in one continuing Codex session (same model, cached prefix, kept reasoning)."""
        qid = q["question_id"]
        work = self.run / "sandbox" / (f"e{epoch:04d}" if discover else out_dir.name) / qid
        session_dir = self.run / "codex" / out_dir.parent.name / out_dir.name / qid
        solve_record = session_dir / "solve_record.json"
        saved = json.loads(solve_record.read_text()) if solve_record.exists() else None
        kb_tool = rendered_kb is not None and self.args.render in ("tool", "brief")
        searched: list[str] = list(saved["record"]["kb_retrieved"]) if saved else []
        explained: list[str] = list(saved["record"]["kb_explained"]) if saved else []
        disc_exps: list[dict] = []
        disc_exp_log: list[dict] = []

        def handler(name, a):
            if name == "kb_open" and rendered_kb:
                return self.open_kb(rendered_kb, a.get("targets", []))
            if name == "kb_search":
                if not kb_tool:
                    return "no knowledge base is available for this question"
                k = max(1, min(8, int(a.get("k") or 5)))
                hits = kb_search(rendered_kb, a.get("query", ""), k, self.args.science)
                searched.extend(c["id"] for c in hits)
                sci = self.args.science
                return ("\n".join(f"[{c['id']}] (credibility {c['credibility']:.2f}; s={c['support_count']:.1f}, "
                                   f"f={c['failure_count']:.1f}) {c['text']}"
                                   + (f"\n  regime: {regime_line(c.get('regime'))}" if sci and c.get("regime") else "")
                                   + (f"\n  flips: {c['phase'][:self.args.phase_preview]}" if sci and self.args.phase_preview and c.get("phase") else "")
                                   + ("  [kb_explain for mechanism/phase]" if sci and (c.get("mechanism") or c.get("phase")) else "")
                                   for c in hits) or "no matching claims")
            if name == "kb_explain":
                c = rendered_kb.claims.get(str(a.get("claim_id", "")).strip().strip("[]")) if kb_tool else None
                if not c:
                    return "unknown claim id"
                explained.append(c["id"])
                return json.dumps({k: c.get(k) for k in ("id", "text", "mechanism", "regime", "phase", "refines",
                                                          "support_count", "failure_count", "credibility") if c.get(k) is not None},
                                  ensure_ascii=False, indent=1)
            if not discover and name == "python" and self.args.solve_lab:
                vis = self.run / "visible" / "_solve_lab"
                if not (vis / "kb.json").exists():
                    write_jsonl(vis / "history.jsonl", [])
                    write_json(vis / "kb.json", {"claims": []})
                return self.py(a["code"], work, vis)
            if not discover:
                return "unavailable"
            if name == "run_experiment":
                if not self.args.lab or len(disc_exps) >= self.args.discover_exps:
                    return f"rejected: at most {self.args.discover_exps} experiment(s) per discovery"
                disc_exps.append(a)
                a = {**a, "variants": (a.get("variants") or [])[:self.args.research_max_cands]}
                out = self.experiment(a, disc_exp_log)
                return out
            return self.disc_tool(name, a, hist, work, vis_dir)

        sess = CodexSession(session_dir, self.args.model,
                            messages[0]["content"], tools_spec(False, bool(self.args.corpus), kb_tool, self.args.science, bool(self.args.lab),
                                       bool(self.args.lab) and self.args.discover_exps > 0), handler,
                            solve_tools=(({"kb_search", "kb_explain", "kb_open"} if self.args.science else {"kb_search", "kb_open"})
                                         if kb_tool else set()) | ({"python"} if self.args.solve_lab else set()),
                            solve_max_calls=(5 if self.args.science else 4) + (6 if self.args.solve_lab else 0),
                            resume=bool(saved))
        try:
            if saved:
                rec = saved["record"]
                if not rec.get("codex_thread"):
                    raise RuntimeError("saved solve has no exact session ID for discovery recovery")
                sess.thread_id = rec["codex_thread"]
            else:
                # A previous uncheckpointed thread may have seen gold; it must never be reused to solve.
                r1 = sess.turn(messages[1]["content"], self.args.solve_effort, "solve")
                text = r1["final"]
                pred, score, inv = grade(q, text)
                cited = sorted(set(CITE_RE.findall(tag(text, "cited") or ""))) if rendered_kb else []
                kb_queries = [t["arguments"].get("query") for t in r1["tool_log"] if t["name"] == "kb_search"]
                frozen = rendered_kb.to_doc() if rendered_kb else None
                if frozen is not None:
                    frozen.pop("updated_at", None)
                rec = {"kb_queries": kb_queries, "kb_retrieved": sorted(set(searched)), "kb_explained": sorted(set(explained)),
                       "solver_snapshot": getattr(self, "solve_snapshot_label", out_dir.name) if discover else out_dir.name,
                       "solver_selection": getattr(self, "solver_selection", None) if discover else None,
                       "kb_snapshot": hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() if frozen else None,
                       "report_versions": {i: r["repo_commit"] for i, r in rendered_kb.reports.items()} if rendered_kb else {},
                       "question_id": qid, "epoch": epoch, "source": q["source"], "task": q["task"], "type": q["type"],
                       "family": q["family"], "group": q["group"], "answer": q["answer"], "prediction": pred,
                       "score": score, "inversions": inv, "correct": score == 1.0, "cited": cited,
                       "explanation": tag(text, "explanation") or text[-3000:], "solution": text,
                       "solver_reasoning": r1["reasoning"], "solve_usage": r1["usage"], "solve_secs": r1["secs"],
                       "finish": "stop", "solver_backend": "codex", "codex_thread": sess.thread_id,
                       "solve_refused_tool_calls": len(r1["tool_log"])}
                write_json(solve_record, {"record": rec, "tool_log": r1["tool_log"]})
            if discover:
                effort = self.args.discover_effort_right if rec["correct"] else self.args.discover_effort_wrong
                r2 = sess.turn(self.discover_prompt(q, rec), effort, "discover", max_calls=self.args.discover_turns * 2,
                               warn_text="stop calling tools and reply with <comment> and optional <hypothesis>.")
                final = r2["final"]
                hyp = tag(final, "hypothesis")
                trace = [{"turn": i, "content": None, "reasoning": None,
                          "tool_calls": [{"name": t["name"], "arguments": json.dumps(t["arguments"], ensure_ascii=False)}],
                          "results": [t["result"]]} for i, t in enumerate(r2["tool_log"])]
                trace.append({"turn": len(trace), "content": final, "reasoning": r2["reasoning"], "tool_calls": [],
                              "usage": r2["usage"], "secs": r2["secs"]})
                rec.update({"discover_effort": effort, "discover_model": self.args.model,
                            "skipped": "<skip/>" in final and not hyp, "comment": tag(final, "comment"),
                            "hypothesis": hyp[:HYPOTHESIS_MAX_CHARS] if hyp else None, "discover_final": final,
                            "discover_trace": trace, "discover_usage": r2["usage"], "discover_secs": r2["secs"],
                            "discover_reasoning": r2["reasoning"], "experiments": disc_exp_log})
        finally:
            sess.close()
        write_json(out_dir / f"{qid}.json", rec)
        return rec

    def disc_tool(self, name: str, a: dict, hist: History, work: Path, vis_dir: Path) -> str:
        if name == "search_history":
            return hist.search(a)
        if name == "read_record":
            return hist.read(a)
        if name == "python":
            return self.py(a["code"], work, vis_dir)
        return f"unknown tool {name}"

    def discover(self, q, messages, text, rec, hist: History, vis_dir: Path, epoch: int) -> dict:
        right = rec["correct"]
        effort = self.args.discover_effort_right if right else self.args.discover_effort_wrong
        note = f" (ranking: {rec['inversions']} inversions vs gold)" if rec["inversions"] is not None else ""
        msgs = messages + [{"role": "assistant", "content": text}, {"role": "user", "content": DISCOVER_PROMPT.format(
            gold=q["answer"], pred=rec["prediction"], score=rec["score"], rank_note=note,
            guidance=(GUIDE_RIGHT if right else GUIDE_WRONG) + (CORPUS_NOTE if self.args.corpus else "")
            + (LAB_NOTE if self.args.lab else "")
            + (LAB_EXP_NOTE.format(n=self.args.discover_exps) if self.args.lab and self.args.discover_exps and not right else ""),
            skip_note="You may reply just <skip/>." if right else "")}]
        work = self.run / "sandbox" / f"e{epoch:04d}" / q["question_id"]

        def handler(name, a):
            if name == "search_history":
                return hist.search(a)
            if name == "read_record":
                return hist.read(a)
            if name == "python":
                return self.py(a["code"], work, vis_dir)
            return f"unknown tool {name}"

        if not right and self.args.discover_wrong_backend == "claude_code":
            sys_txt = msgs[0]["content"]
            convo = "\n\n".join(f"## {m['role'].upper()}\n{m['content']}" for m in msgs[1:-1])
            user = ("Below is a question, followed by the answer the solver model gave. You take over as the "
                    f"discoverer.\n\n{convo}\n\n## GRADING\n{msgs[-1]['content']}")
            final, trace, info = run_claude_agent(
                sys_txt, user, tools_spec(False, bool(self.args.corpus)), handler, model=self.args.discover_wrong_model, effort=effort,
                workdir=work / "_cc", max_calls=self.args.discover_turns * 2, warn_calls=2,
                warn_text="stop calling tools and reply with <comment> and optional <hypothesis>.",
                budget_usd=3.0, timeout=1800)
            with self.lock:
                self.cc_cost += info["cost_usd"] or 0.0
            model = self.args.discover_wrong_model
        else:
            client = self.client if right else self.client_wrong
            final, trace = tool_loop(client, msgs, tools_spec(False, bool(self.args.corpus)), handler, effort, self.args.discover_max_tokens,
                                     self.args.discover_turns, final_nudge="Tool budget used up. Reply now with the "
                                     "final <comment> and optional <hypothesis>; no more tool calls.")
            info, model = None, client.model
        hyp = tag(final, "hypothesis")
        return {"discover_effort": effort, "discover_model": model, "discover_cc": info,
                "skipped": "<skip/>" in final and not hyp,
                "comment": tag(final, "comment"), "hypothesis": hyp[:HYPOTHESIS_MAX_CHARS] if hyp else None,
                "discover_final": final, "discover_trace": trace}

    def py(self, code: str, work: Path, vis_dir: Path) -> str:
        prelude = (f"HISTORY_PATH = {str(vis_dir / 'history.jsonl')!r}\nKB_PATH = {str(vis_dir / 'kb.json')!r}\n"
                   "def load_history():\n    return [json.loads(l) for l in open(HISTORY_PATH) if l.strip()]\n"
                   "def load_kb():\n    return json.load(open(KB_PATH))['claims']\n")
        ro = [vis_dir]
        if self.args.corpus:
            cp = Path(self.args.corpus).resolve()
            prelude += (f"CORPUS_PATH = {str(cp)!r}\n"
                        "def load_corpus():\n    return [json.loads(l) for l in open(CORPUS_PATH) if l.strip()]\n")
            ro.append(cp.parent)
        if self.args.lab:
            lp = Path(self.args.lab).resolve()
            xp = self.run / "lab_experiments.jsonl"
            prelude += (f"LAB_PATH = {str(lp)!r}\nEXP_PATH = {str(xp)!r}\n_LAB = []\n"
                        "def load_lab(experiments=True):\n    if not _LAB:\n"
                        "        _LAB.extend(json.loads(l) for l in open(LAB_PATH) if l.strip())\n"
                        "        if os.path.exists(EXP_PATH):\n"
                        "            _LAB.extend(json.loads(l) for l in open(EXP_PATH) if l.strip())\n"
                        "    return _LAB if experiments else [r for r in _LAB if r.get('source') != 'experiment']\n")
            ro += [lp.parent, xp]
        return run_python(prelude + code, work, ro)

    # ---- visible snapshots
    def history_question(self, record: dict) -> tuple[dict, dict]:
        qid = record["question_id"]
        q = self.pool.get(qid)
        provenance = {"kind": "current_pool", "path": str(self.args.pool)}
        if q is None:
            origin = record.get("imported_from")
            cached = None
            if origin:
                with self.lock:
                    if origin not in self.historical_pools:
                        config_path = OUT_ROOT / origin / "config.json"
                        config = json.loads(config_path.read_text()) if config_path.exists() else {}
                        path = Path(config["pool"]) if config.get("pool") else None
                        if path is not None and not path.is_absolute():
                            path = CONFIG.bench_root.parent / path
                        if path is not None and path.exists():
                            raw = path.read_bytes()
                            digest = hashlib.sha256(raw).hexdigest()
                            if config.get("pool_sha256") and config["pool_sha256"] != digest:
                                raise ValueError(f"historical pool hash mismatch: {origin}")
                            questions = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
                            self.historical_pools[origin] = (
                                {row["question_id"]: row for row in questions},
                                {"kind": "historical_pool", "run": origin, "path": str(path),
                                 "sha256": digest, "hash_verified": bool(config.get("pool_sha256"))})
                        else:
                            self.historical_pools[origin] = None
                    cached = self.historical_pools[origin]
            if cached is not None:
                q, provenance = cached[0].get(qid), cached[1]
            if q is None:
                messages = record.get("messages")
                if isinstance(messages, list) and messages and all(
                        isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages):
                    q = {**record, "messages": copy.deepcopy(messages)}
                elif isinstance(record.get("question"), str) and record["question"]:
                    q = {**record, "messages": [{"role": "user", "content": record["question"]}]}
                else:
                    raise ValueError(f"historical question unavailable: {qid} from {origin or 'unspecified run'}")
                provenance = {"kind": "saved_record_question", "run": origin,
                              "sha256": hashlib.sha256(json.dumps(q["messages"], ensure_ascii=False,
                                                                   sort_keys=True).encode()).hexdigest()}
        try:
            from .sample_kb_science_release import identity_tags
        except ImportError:
            from sample_kb_science_release import identity_tags
        if not hasattr(self, "history_holdout_tags"):
            ids = self.split["eval"] + self.split.get("test", [])
            self.history_holdout_tags = set().union(*(identity_tags(self.pool[i]) for i in ids))
            release_split = Path(self.args.pool).parent / "split.json"
            if release_split.exists():
                for group in json.loads(release_split.read_text()).get("groups", {}).values():
                    if group.get("split") in ("val", "test"):
                        self.history_holdout_tags.update(group.get("identifiers", []))
        if (identity_tags(q) | identity_tags(record)) & self.history_holdout_tags:
            raise ValueError(f"historical training question overlaps current holdout: {qid}")
        return q, provenance

    def visible(self, name: str, records: list[dict], kb: KB) -> Path:
        d = self.run / "visible" / name
        keep = ["question_id", "epoch", "source", "task", "type", "family", "question", "answer", "prediction",
                "score", "inversions", "cited", "explanation", "comment", "hypothesis", "imported_from",
                "solver_snapshot", "solver_selection"]
        rows = []
        for r in records:
            q, provenance = self.history_question(r)
            rows.append({**{k: r.get(k) for k in keep}, "question": q["messages"][-1]["content"],
                         "question_provenance": provenance})
        write_jsonl(d / "history.jsonl", rows)
        write_json(d / "kb.json", kb.to_doc())
        return d

    def history_records(self, upto_epoch: int) -> list[dict]:
        rows = []
        for run in self.args.import_history:  # earlier runs' training records (training questions only; ids disjoint)
            for f in sorted((OUT_ROOT / run / "epochs").glob("e*/records.jsonl")):
                rows += [{**r, "epoch": r["epoch"] - 1000, "imported_from": run} for r in read_jsonl(f)]
        for e in range(1, upto_epoch + 1):
            rows += read_jsonl(self.run / "epochs" / f"e{e:04d}" / "records.jsonl")
        return rows

    # ---- summarizer
    def summarize(self, epoch: int, kb: KB, records: list[dict], vis_dir: Path) -> tuple[str, list]:
        hist = History(self.visible_rows(vis_dir))
        lines = []
        for r in records:
            ctl = f" noKB_score={r['nokb_score']:.2f}" if r.get("nokb_score") is not None else ""
            lines.append(f"### {r['question_id']} ({r['source']}, {r['task']}, {r['family']}) score={r['score']:.2f}{ctl} "
                         f"gold={r['answer']} pred={r['prediction']} cited={','.join(r['cited']) or 'none'} "
                         f"retrieved={','.join(r.get('kb_retrieved') or []) or 'none'}\n"
                         f"solver_snapshot={r.get('solver_snapshot', 'unknown')} selection={r.get('solver_selection')}\n"
                         f"explanation: {r['explanation'][:1500]}\ncomment: {r.get('comment') or '-'}\n"
                         f"hypothesis: {r.get('hypothesis') or '-'}")
        flagged = [c for c in kb.claims.values() if c["credibility"] < 0.45 and c["support_count"] + c["failure_count"] >= 3]
        moved = collections.Counter(cid for r in records for cid in r["cited"])
        user = (f"Epoch {epoch}: {len(records)} records, mean score "
                f"{sum(r['score'] for r in records) / max(1, len(records)):.3f}.\n\n" + "\n\n".join(lines) +
                f"\n\n## Claims cited this epoch (count)\n{dict(moved.most_common())}\n\n## Low-credibility candidates "
                f"(cred<0.45, evidence>=3)\n{', '.join(c['id'] for c in flagged) or 'none'}\n\n## KB index "
                f"({len(kb.claims)} claims; use kb_view for full text; [MRPR] = has mechanism/regime/phase/refines)\n"
                f"{kb.index()}")
        user += ("\n\nSolver feedback belongs to the frozen snapshot listed with each record. "
                 "The current research KB may contain newer text under the same claim ID; do not attribute old-version "
                 "performance to a corrected or merged claim.")
        if self.worker_job:
            user += "\n\nPersisted KB changes are already present. Continue the same session without repeating them."
            conflicts = []
            for path in (self.run / "jobs").glob("J*/conflict-*.json"):
                conflicts.append(json.loads(path.read_text()))
            if conflicts:
                user += "\n\nLate proposals conflicting with newer claims (review against current KB):\n" + json.dumps(conflicts, ensure_ascii=False)
        if self.args.train_control:
            user += "\n\n" + self.claim_utility(epoch, records, kb)
        if self.args.macro_every:
            user += "\n\n" + self.bias_table(epoch, records)
            user += (f"\n\n## Phase\nEpoch {epoch}: {'CONSOLIDATE' if self.consolidation(epoch) else 'COLLECT'} "
                     f"(consolidation every {self.args.macro_every} epochs).")
        if self.args.kb_char_budget:
            used = sum(len(c["text"]) for c in kb.claims.values())
            user += (f"\n\n## KB text budget\n{used} / {self.args.kb_char_budget} chars used by claim texts "
                     "(add_claim is rejected beyond the budget). Compression is part of the job: a law that explains "
                     "several exceptions with one mechanism and a phase table is worth more than the list.")
        prev_soa = self.run / "epochs" / f"e{epoch - 1:04d}" / "soa.json"
        if self.args.science and prev_soa.exists():
            user += "\n\n## SoA readout of the current KB (diagnostic, not a target)\n" + soa_brief(json.loads(prev_soa.read_text()))
        ep_dir = self.run / "epochs" / f"e{epoch:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "summarizer_input.md").write_text(user, encoding="utf-8")
        work = self.run / "sandbox" / f"e{epoch:04d}" / "_summarizer"
        n_research = [0]
        shown = {c["id"] for c in kb.rendered()} if self.args.render not in ("tool", "brief") else set()
        hist_ids = set(hist.rows)
        adds = [sum(op["op"] == "add" for op in kb.pending)] if self.worker_job else [0]
        science = [""]
        structural = [0]
        nudged = [False]

        def curate(name, a):
            if name == "search_history":
                return hist.search(a)
            if name == "read_record":
                return hist.read(a)
            if name == "python":
                return self.py(a["code"], work, vis_dir)
            if name == "kb_view":
                cs = list(kb.claims.values())
                if a.get("ids"):
                    cs = [kb.claims[i] for i in a["ids"] if i in kb.claims]
                if a.get("contains"):
                    cs = [c for c in cs if a["contains"].lower() in c["text"].lower()]
                key = {"evidence": lambda c: -(c["support_count"] + c["failure_count"]), "id": lambda c: c["id"]}.get(
                    a.get("sort"), lambda c: -c["credibility"])
                cs = sorted(cs, key=key)[: int(a.get("limit") or 40)]
                return "\n".join(f"[{c['id']}] cred={c['credibility']:.3f} s={c['support_count']:.2f} "
                                 f"f={c['failure_count']:.2f} ep={c['created_epoch']} src={c.get('sources', [])[:5]}\n"
                                 f"{c['text']}" for c in cs) or "no claims"
            if name == "kb_open":
                return self.open_kb(kb, a.get("targets", []), latest=bool(self.worker_job))
            fields = None
            if self.args.science and name in ("add_claim", "merge_claims", "annotate_claim"):
                fields, err = science_fields(a)
                if err:
                    return err
            if name == "annotate_claim":
                if a["claim_id"] not in kb.claims:
                    return f"conflict: {a['claim_id']} does not exist"
                if not fields:
                    return "rejected: give at least one of mechanism / regime / phase / refines"
                kb.annotate(a["claim_id"], fields, a["reason"], self.author_summ)
                return f"annotated {a['claim_id']} ({', '.join(fields)})"
            if name == "research":
                if self.jobs:
                    topic = a.get("topic") or a.get("question", "")
                    for job in self.jobs.list():
                        if job["kind"] == "research" and job["payload"].get("topic") == topic and job["status"] not in ("failed", "completed"):
                            return json.dumps({k: job.get(k) for k in ("id", "status", "origin_epoch", "label", "report_id")})
                    job = self.submit_job("research", epoch, kb, [], vis_dir, topic=topic,
                                          question=a.get("question", ""), focus=a.get("focus", ""))
                    return f"research {job['id']} launched asynchronously; topic={topic}; origin=e{epoch:04d}"
                cap = self.args.consolidate_research if self.consolidation(epoch) else self.args.research_calls
                if n_research[0] >= cap:
                    return f"rejected: at most {cap} research calls this epoch"
                n_research[0] += 1
                return self.research(epoch, n_research[0], a.get("question", ""), a.get("focus", ""), vis_dir)
            if name == "add_claim" and self.args.kb_char_budget:
                used = sum(len(c["text"]) for c in kb.claims.values())
                if used + len(a["text"].strip()) > self.args.kb_char_budget:
                    return (f"rejected: KB text budget {self.args.kb_char_budget} chars (now {used}); compress first - "
                            "merge special cases into a more general law, or delete a claim that does not help solvers")
            if name == "add_claim":
                if adds[0] >= self.args.max_add:
                    return f"rejected: at most {self.args.max_add} additions per epoch"
                text = a["text"].strip()
                adds[0] += 1
                structural[0] += 1
                return "added " + kb.add(text, a.get("source_question_ids", []), a["reason"], self.author_summ, fields)
            if name == "delete_claim":
                if a["claim_id"] not in kb.claims:
                    return f"conflict: {a['claim_id']} does not exist (already merged/deleted?)"
                c = kb.claims[a["claim_id"]]
                if a["claim_id"] in shown:
                    return (f"rejected: {a['claim_id']} is shown to solvers; merge it into a sharper claim or rewrite it "
                            "in place instead (credit will demote it if it keeps failing)")
                ev = c["support_count"] + c["failure_count"]
                cx = [q for q in a.get("counterexample_question_ids") or [] if q in hist_ids]
                if not (ev >= 5 and c["credibility"] < 0.4) and len(cx) < 2:
                    return (f"rejected: {a['claim_id']} has evidence {ev:.1f}, cred {c['credibility']:.2f}; deletion needs "
                            "s+f>=5 and cred<0.4, or >=2 valid counterexample_question_ids from history")
                kb.delete(a["claim_id"], a["reason"] + (f" | counterexamples: {cx}" if cx else ""), self.author_summ)
                structural[0] += 1
                return "deleted " + a["claim_id"]
            if name == "merge_claims":
                missing = [i for i in a["claim_ids"] if i not in kb.claims]
                if missing or not a["claim_ids"]:
                    return f"conflict: missing claims {missing}"
                new = kb.merge(a["claim_ids"], a["text"].strip(), a["reason"], self.author_summ,
                               a.get("support"), a.get("failure"), fields)
                c = kb.claims[new]
                structural[0] += 1
                return f"merged into {new} (s={c['support_count']:.2f}, f={c['failure_count']:.2f})"
            if name == "adjust_counts":
                if a["claim_id"] not in kb.claims:
                    return f"conflict: {a['claim_id']} does not exist"
                kb.adjust(a["claim_id"], float(a["support"]), float(a["failure"]), a["reason"], self.author_summ)
                structural[0] += 1
                return "adjusted " + a["claim_id"]
            if name == "finish":
                science[0] = a["science"]
                if self.worker_job:
                    (self.worker_path.parent / "science.md").write_text(science[0], encoding="utf-8")
                    update_job(self.worker_path, curation_finished=True)
                return "ok"
            return f"unknown tool {name}"

        def handler(name, a):
            with self.lock:
                return curate(name, a)

        cc = self.args.summ_backend == "claude_code"
        system = SUMMARIZER_SYSTEM.format(top=RENDER_TOP, probation=RENDER_PROBATION, max_add=self.args.max_add,
                                          max_chars=CLAIM_MAX_CHARS, turns=self.args.summ_turns)
        if self.args.corpus:
            system += CORPUS_RULE
        if self.args.science:
            system += SCIENCE_RULE.format(max_chars=CLAIM_MAX_CHARS, mech=MECH_MAX_CHARS,
                                          dims=", ".join(REGIME_DIMS), phase=PHASE_MAX_CHARS,
                                          research=self.args.research_calls)
        if self.args.train_control:
            system += CONTROL_RULE
        if self.args.macro_every:
            system += CONSOLIDATE_RULE if self.consolidation(epoch) else COLLECT_RULE
        if self.worker_job:
            system = ("You curate a continuing ArchitectureIQ knowledge base for a strong solver. The solver sees all "
                      "short claims by default and may open saved science reports. Reports are usable while their theory "
                      "is unfinished. Read this batch's failures, comments and hypotheses; prioritize observations that "
                      "could correct a systematic decision bias. Record comparison, conditions, effect size, sample size "
                      "and provenance. A scoped empirical fact does not need a mechanism yet. Avoid pooled conclusions "
                      "when width, budget, data or initialization could confound them. Merge related facts only when "
                      "the resulting conditional rule preserves their evidence and can predict a new setting. "
                      "Use research(question, topic) to launch a persistent investigation; it returns immediately. "
                      "Delegate long experiments, competing explanations, theory and falsifiable extrapolation to these "
                      "reports, which can cover several claims. Curate current facts and review conflicting late proposals "
                      "against newer evidence. All mutations checkpoint immediately; surviving earlier work must not be "
                      "repeated after recovery. No tool-call limit applies. The 40-minute soft window changes epoch "
                      "membership and keeps your process alive. Use only visible training evidence, never evaluation "
                      "answers. Claim feedback counts track solver usage, not a calibrated probability of truth. "
                      f"Aim for short claim text of about {CLAIM_MAX_CHARS} characters, preserving necessary conditions; at most {self.args.max_add} "
                      "new claims per curation. When the current batch is curated, call finish with a brief digest of "
                      "actual findings and open investigations.")
            if self.args.train_control:
                system += CONTROL_RULE
            if self.args.macro_every:
                system += CONSOLIDATE_RULE if self.consolidation(epoch) else COLLECT_RULE
            user += "\n\nSaved scientific reports (open with kb_open):\n" + json.dumps(kb.reports, ensure_ascii=False)
        if self.args.render == "tool":
            system = re.sub(r"Future solvers see only a question plus the KB rendered read-only \(.*?claims\)\.",
                            "Future solvers (a strong model) see the question and may query the KB on demand with a "
                            "keyword search tool (BM25 over claim text, top 5); claims are advisory to them, so write "
                            "each claim so that its scope keywords make it retrievable and it adds information a "
                            "strong model would not already know.", system, flags=re.S)
            system = system.replace("Claims currently shown to solvers cannot be deleted (merge or sharpen\n  them instead). ", "")
        if self.args.summ_backend == "codex":
            system = system.replace("Turn 1-3", "Calls 1-6").replace("Turn 4-8", "Calls 7-20").replace(
                "Turn 9+", "Then").replace("Last 3 turns", "Last 3 calls").replace(
                f"about {self.args.summ_turns} turns total", f"about {self.args.summ_turns} tool calls total")
            sess = CodexSession(work / "_codex", self.summ_model, system,
                                tools_spec(True, bool(self.args.corpus), science=self.args.science,
                                           lab=bool(self.args.lab), research=self.args.science), handler)
            try:
                r = sess.turn(user, self.args.summ_effort, "discover", max_calls=None if self.worker_job else self.args.summ_turns,
                              warn_text="issue any backtested operations now, then call finish(science=...).",
                              timeout=None if self.worker_job else 3600)
                if not science[0]:
                    r = sess.turn("Call finish(science=...) now with the digest; apply nothing else.",
                                  self.args.summ_effort, "discover",
                                  max_calls=None if self.worker_job else 2,
                                  timeout=None if self.worker_job else 900)
                if self.worker_job and not science[0]:
                    raise RuntimeError("curator stopped without calling finish; resume the same session")
            finally:
                sess.close()
            trace = [{"turn": i, "content": None, "reasoning": None, "tool_calls": [
                {"name": t["name"], "arguments": json.dumps(t["arguments"], ensure_ascii=False)}],
                "results": [t["result"]], "usage": None} for i, t in enumerate(sess.log)]
            trace.append({"turn": len(trace), "content": r["final"], "reasoning": r["reasoning"], "tool_calls": [],
                          "usage": r["usage"]})
            write_json(self.run / "epochs" / f"e{epoch:04d}" / "summarizer_info.json",
                       {"backend": "codex", "model": self.summ_model, "calls": len(sess.log), "usage": r["usage"],
                        "finished": bool(science[0])})
            print(f"[summarizer] codex calls={len(sess.log)} finished={bool(science[0])}", flush=True)
            return science[0] or r["final"], trace
        if cc:
            system = system.replace("Turn 1-3", "Calls 1-6").replace("Turn 4-8", "Calls 7-20").replace(
                "Turn 9+", "Then").replace("Last 3 turns", "Last 3 calls").replace(
                f"about {self.args.summ_turns} turns total", f"about {self.args.summ_turns} tool calls total")
            trace_all, calls_used, cost = [], 0, 0.0
            for attempt in range(4):
                # an upstream error mid-session (cctq 403 quota blips) ends the session early: start a continuation
                # session on the CURRENT KB (edits so far are kept), with the remaining tool budget
                prompt = user if attempt == 0 else (
                    user + f"\n\n## NOTE: continuation\nA previous curation session for this epoch was cut off by an "
                    f"API error after {calls_used} tool calls. Its edits are already applied. Current KB index:\n"
                    f"{kb.index()}\nContinue the curation (do not redo applied edits), then call finish(science=...).")
                final, trace, info = run_claude_agent(
                    system, prompt, tools_spec(True, bool(self.args.corpus), science=self.args.science, lab=bool(self.args.lab),
                    research=self.args.science and self.args.research_calls > 0), handler, model=self.summ_model, effort=self.args.summ_effort,
                    workdir=work, max_calls=None if self.worker_job else max(8, self.args.summ_turns - calls_used), stop_tool="finish", warn_calls=5,
                    warn_text="if you have backtested operations, issue them now; otherwise finish (no edit is a valid "
                              "outcome).", budget_usd=self.args.summ_budget_usd,
                    trace_path=(self.worker_path.parent / "summarizer_trace.json" if self.worker_job else
                                self.run / "epochs" / f"e{epoch:04d}" / f"summarizer_trace{'' if attempt == 0 else f'_{attempt}'}.json"),
                    durable=bool(self.worker_job), timeout=None if self.worker_job else 2400)
                trace_all += trace
                calls_used += info["calls"]
                cost += info["cost_usd"] or 0.0
                if info["stopped"] or science[0] or not info.get("is_error"):
                    break
                print(f"[summarizer] session cut off ({str(final)[:120]!r}) after {info['calls']} calls; "
                      f"continuation {attempt + 1} in 120s", flush=True)
                time.sleep(120)
            trace = trace_all
            info = {**info, "calls": calls_used, "cost_usd": cost, "attempts": attempt + 1}
            self.cc_cost += info["cost_usd"] or 0.0
            if self.worker_job and (info.get("is_error") or info.get("returncode") or not science[0]):
                raise RuntimeError(f"summarizer interrupted: {str(final)[:300]}")
            write_json(self.run / "epochs" / f"e{epoch:04d}" / "summarizer_info.json", info)
            print(f"[summarizer] {json.dumps({k: info[k] for k in ('cost_usd', 'calls', 'stopped', 'secs')})}", flush=True)
            return science[0] or final, trace
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        final, trace = tool_loop(self.client, msgs, tools_spec(True), handler, self.args.summ_effort,
                                 self.args.summ_max_tokens, self.args.summ_turns, stop_tool="finish",
                                 trace_path=self.run / "epochs" / f"e{epoch:04d}" / "summarizer_trace.json",
                                 final_nudge="Turn budget used up. Apply any remaining checked operations and call "
                                             "finish(science=...) in this turn.", warn_turns=4,
                                 warn_text="if you have backtested operations, issue them now; otherwise finish "
                                           "(no edit is a valid outcome).")
        return science[0] or final, trace

    def consolidation(self, epoch: int) -> bool:
        return bool(self.args.macro_every) and epoch % self.args.macro_every == 0

    def bias_table(self, epoch: int, records: list[dict]) -> str:
        """Direction of the solver's errors: on missed questions, is its top pick higher or lower than gold's on each
        config attribute? Built from training records only (KB-run and no-KB control solves)."""
        if __package__:
            from .lab_predictor import delta as dlt, parse_question
        else:
            from lab_predictor import delta as dlt, parse_question
        rows = self.history_records(epoch - 1) + records
        cnt = collections.defaultdict(lambda: [0, 0])

        def top(a):
            m = re.findall(r"[A-E]", a or "")
            return m[0] if m else None

        attrs = {"log_delta": lambda c: math.log10(dlt(c["opt"], c["lr"], c["steps"], c.get("momentum") or 0)),
                 "adaptive_opt": lambda c: int(c["opt"] in ("Adam", "AdamW", "RMSprop")),
                 "n_layernorm": lambda c: sum(map(bool, c["layer_norm"] or [])),
                 "depth": lambda c: len(c["layer_norm"] or []) or (c.get("num_layers") or 0),
                 "residual": lambda c: int(bool(c.get("residual"))), "width": lambda c: c.get("width") or c.get("d_model") or 0,
                 "smooth_act": lambda c: int(c.get("act") in ("gelu", "silu"))}
        for r in rows:
            q, _ = self.history_question(r)
            ch = parse_question(q)
            if not ch:
                continue
            g = top(q["answer"])
            for pred in (r.get("prediction"), r.get("nokb_prediction")):
                p = top(pred)
                if not p or p == g or p not in ch or g not in ch:
                    continue
                for k, fn in attrs.items():
                    try:
                        a, b = fn(ch[p]), fn(ch[g])
                    except (TypeError, ValueError):
                        continue
                    if a != b:
                        cnt[(k, q["family"])][0 if a > b else 1] += 1
        lines = ["## Solver bias table (training misses: is the solver's top pick HIGHER or LOWER than gold's top pick?)",
                 "bias = (higher-lower)/(higher+lower); |bias|>=0.5 with n>=15 is a systematic error to correct."]
        for (k, fam), (hi, lo) in sorted(cnt.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1])):
            if hi + lo >= 10:
                lines.append(f"{fam} {k}: higher {hi} / lower {lo} -> bias {(hi - lo) / (hi + lo):+.2f}")
        return "\n".join(lines[:40])

    def macro_review(self, epoch: int, kb: KB) -> dict:
        """Measure growing research KB; sustained regression changes solver input, never research contents."""
        a = self.args
        label = f"kb_{epoch:04d}"
        best_path = self.run / "best.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else {"label": "kb_0000", "epoch": 0, "strikes": 0}
        review_path = self.run / "epochs" / f"e{epoch:04d}" / "macro_review.json"
        if review_path.exists():
            return json.loads(review_path.read_text())
        prior = best.get("review", {})
        if prior.get("label") == label and prior.get("research_preserved"):
            write_json(review_path, prior)
            return prior
        checkpoint = self.run / "kb" / f"{label}.json"
        research = KB(json.loads(checkpoint.read_text()), self.run) if checkpoint.exists() else KB(kb.to_doc(), self.run)
        cand = self.sample(label, research, a.macro_repeats)
        nokb = self.sample("nokb", None, a.macro_repeats)
        dn, sen, n = self.paired(cand, nokb)
        rev = {"epoch": epoch, "label": label, "val_mean": round(statistics.mean(statistics.mean(v) for v in cand.values()), 4),
               "val_vs_nokb": round(dn, 4), "val_se": round(sen, 4), "n_q": n}
        if getattr(a, "macro_test_readout", False) and self.split.get("test"):
            tc = self.sample(f"test_{label}", research, a.macro_repeats, self.split["test"])
            tn = self.sample("test_nokb", None, a.macro_repeats, self.split["test"])
            td, tse, _ = self.paired(tc, tn)
            rev.update(test_mean=round(statistics.mean(statistics.mean(v) for v in tc.values()), 4),
                       test_vs_nokb=round(td, 4), test_se=round(tse, 4))
        mode = best.get("solver_mode", "live")
        inc_kb = KB(json.loads((self.run / "kb" / f"{best['label']}.json").read_text()), self.run)
        inc = self.sample(best["label"], inc_kb, a.macro_repeats)
        d, se, _ = self.paired(cand, inc)
        rev.update(vs_best=best["label"], delta_best=round(d, 4), se_best=round(se, 4))
        if d >= 0:
            rev["action"] = "new best"
            mode = "live"
            best = {"label": label, "epoch": epoch, "accepted_at": now(), "strikes": 0}
        elif d < -a.gate_z * se:
            strikes = best.get("strikes", 0) + 1
            if mode == "stable" or strikes >= 2:
                mode = "stable"
                rev["action"] = f"solver fallback to {best['label']}"
            else:
                rev["action"] = "strike 1 (keep growing)"
            best = {**best, "strikes": strikes}
        else:
            rev["action"] = "keep (not significantly worse)"
            best = {**best, "strikes": 0}
        rev.update(solver_mode=mode, solver_label=best["label"] if mode == "stable" else None,
                   research_label=label, research_preserved=True)
        write_json(best_path, {**best, "solver_mode": mode, "solver_label": rev["solver_label"], "review": rev})
        write_json(review_path, rev)
        print("[review]", json.dumps(rev), flush=True)
        return rev

    def claim_utility(self, epoch: int, records: list[dict], kb: KB) -> str:
        """Cumulative training-side utility: for each claim, KB-solve score minus the same question's no-KB score over
        all training questions (this and earlier epochs) on which the solver retrieved it. Val is never used."""
        rows = [r for r in self.history_records(epoch - 1) + records if r.get("nokb_score") is not None]
        if not rows:
            return ""
        tab = collections.defaultdict(list)
        snapshots = {}
        for r in rows:
            label = r.get("solver_snapshot")
            origin = OUT_ROOT / r["imported_from"] if r.get("imported_from") else self.run
            key = (origin, label)
            if key not in snapshots:
                path = origin / "solver_snapshots" / f"{label}.json"
                snapshots[key] = {c["id"]: c for c in json.loads(path.read_text())["claims"]} if path.exists() else {}
            for cid in r.get("kb_retrieved") or []:
                used, current = snapshots[key].get(cid), kb.claims.get(cid)
                if used is not None and current is not None and all(used.get(k) == current.get(k)
                                                                  for k in ("text", *SCIENCE_FIELDS, "report_ids")):
                    tab[cid].append(r["score"] - r["nokb_score"])
        d_all = [r["score"] - r["nokb_score"] for r in rows]
        d_q = [r["score"] - r["nokb_score"] for r in rows if r.get("kb_queries")]
        lines = [f"## Claim utility on TRAINING questions (cumulative, {len(rows)} questions with a no-KB control solve)",
                 f"KB-run minus no-KB score: all {statistics.mean(d_all):+.3f}; questions where the solver queried the KB "
                 f"{statistics.mean(d_q) if d_q else 0:+.3f} (n={len(d_q)}). Per claim retrieved (delta, n; single samples, "
                 "noisy: trust only large n; matching frozen claim versions only; missing or older versions are excluded):"]
        for cid, v in sorted(tab.items(), key=lambda kv: statistics.mean(kv[1])):
            if cid in kb.claims:
                lines.append(f"{cid}: {statistics.mean(v):+.2f} (n={len(v)})")
        return "\n".join(lines)

    # ---- research assistant (science mode; summarizer-only tool)
    def lab_dataset_params(self, ds: str) -> dict:
        spec = CONFIG.bench_root / "data/datasets" / ds / "dataset_spec.json"
        d = json.loads(spec.read_text())["params"] if spec.exists() else {}
        return {k: v for k, v in d.items() if k not in ("calibration", "point_sampling")}

    def lab_row(self, ref: str) -> dict | None:
        if not hasattr(self, "_lab_index"):
            self._lab_index = {}
            for line in Path(self.args.lab).read_text().splitlines():
                r = json.loads(line)
                self._lab_index[f"{r['set_id']}/{r['candidate_id']}"] = r
        ref = ref.strip()
        if ref not in self._lab_index and (self.run / "lab_experiments.jsonl").exists():
            for r in read_jsonl(self.run / "lab_experiments.jsonl"):
                self._lab_index.setdefault(f"{r['set_id']}/{r['candidate_id']}", r)
        return self._lab_index.get(ref)

    def experiment(self, a: dict, log: list) -> str:
        base = self.lab_row(a.get("base", ""))
        if not base:
            return "unknown base: give set_id/candidate_id exactly as in load_lab() (e.g. 'sym_00bebe/set_4096_.../c_ab12cd')"
        variants = a.get("variants") or [{}]
        if len(variants) > self.args.research_max_cands:
            return f"rejected: at most {self.args.research_max_cands} variants per experiment"
        cands = []
        for v in variants:
            c = {"model": copy.deepcopy(base["model"]), "optimizer": copy.deepcopy(base["optimizer"]),
                 "loss": copy.deepcopy(base["loss"]), "budget": {"training_steps": base["budget"]["training_steps"],
                                                                 "batch_size": base["budget"]["batch_size"]}}
            for key, val in (v or {}).items():
                head, _, rest = key.partition(".")
                if head not in c or not rest:
                    return f"rejected: override key {key!r} must start with model./optimizer./loss./budget."
                c[head][rest] = val
            if c["model"].get("type") == "mlp" and "layer_norm" in c["model"]:
                ln, d = c["model"]["layer_norm"], int(c["model"]["depth"])
                if len(ln) != d:
                    return f"rejected: model.layer_norm must have depth={d} entries (got {len(ln)})"
            cands.append(c)
        t0 = time.time()
        ds = f"{base['family']}/{base['dataset_id']}"
        if a.get("dataset"):
            want = str(a["dataset"]).strip()
            want = want if "/" in want else f"{base['family']}/{want}"
            if want.split("/")[0] != base["family"]:
                return f"rejected: dataset must be of the base's family {base['family']}"
            allowed = set((Path(self.args.lab).resolve().parent / "allowed_datasets.txt").read_text().split())
            if want not in allowed:
                return f"rejected: {want} is not a lab dataset (held out or unknown)"
            ds = want
        job = {"dataset": ds, "candidates": cands}
        if "target_transform" in a:
            if not self.worker_job:
                return "rejected: target transforms are available to durable research workers only"
            job["target_transform"] = a["target_transform"]
        if "diagnostic_fail_threshold" in a:
            if not self.worker_job:
                return "rejected: diagnostic threshold is available to durable research workers only"
            job["diagnostic_fail_threshold"] = a["diagnostic_fail_threshold"]
        if "process" in a:
            if not self.worker_job:
                return "rejected: process recording is available to durable research workers only"
            job["process"] = a["process"]
        if self.worker_job:
            job["capture_executable_sources"] = True
        jp = self.run / "lab_jobs" / f"{hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()[:12]}.json"
        write_json(jp, job)
        with self.lab_sem, (self.run.parent / "lab.lock").open("a") as lab_lock:
            fcntl.flock(lab_lock, fcntl.LOCK_EX)
            p = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "kb_lab.py"), "run", str(jp),
                                "--workers", str(self.args.lab_workers),
                                "--lab-dir", str(Path(self.args.lab).resolve().parent),
                                "--pool", str(Path(self.args.pool).resolve()),
                                "--require-allowlist"], capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            return f"[experiment failed] {p.stderr[-800:]}"
        res = json.loads(p.stdout)
        if self.worker_job:
            measurement_dir = self.worker_path.parent / "repo" / "measurements"
            for result in res["results"]:
                for filename, source in (result.get("measurement_files") or {}).items():
                    if filename not in ("candidate_spec.json", "results/summary.json", "results/curves.npz",
                                        "results/execution_manifest.json", "executed/model.py",
                                        "executed/optimizer.py", "executed/loss.py", "executed/train.py",
                                        "executed/target_transform_runner.py") and not re.fullmatch(
                                            r"results/process/seed_[0-9]+\.(json|npz)", filename):
                        raise ValueError(f"unexpected measurement file: {filename}")
                    source_path = Path(source["path"]).resolve()
                    source_path.relative_to(Path(self.args.lab).resolve().parent / "experiments")
                    content = source_path.read_bytes()
                    if len(content) != source["bytes"] or hashlib.sha256(content).hexdigest() != source["sha256"]:
                        raise ValueError(f"measurement changed before evidence capture: {filename}")
                    target = measurement_dir / (source["sha256"] + source_path.suffix)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                    temporary.write_bytes(content)
                    temporary.replace(target)
                    source["repo_path"] = str(target.relative_to(self.worker_path.parent / "repo"))
            evidence = self.worker_path.parent / "repo" / "experiments" / f"{jp.stem}.json"
            write_json(evidence, {"origin_epoch": self.worker_job["origin_epoch"],
                                  "job_id": self.worker_job["id"], "why": a.get("why"),
                                  "input": job, "dataset_params": self.lab_dataset_params(ds),
                                  "profile": "v1.5", "result": res})
        evidence_fields = ("failed_seeds", "excluded", "n_seeds", "base_seed", "cached", "seed_results",
                           "measurement_files", "source_provenance", "process_provenance", "target_transform")
        rows = [{"variant": v, "mean": r.get("mean"), "std": r.get("std"), "error": r.get("error"),
                 **{key: r.get(key) for key in evidence_fields}}
                for v, r in zip(variants, res["results"])]
        # grow the lab: every experiment becomes new comparable rows (one set per job) visible to load_lab()
        set_id = f"exp:{jp.stem}"
        new_rows = []
        for c, r in zip(cands, res["results"]):
            if r.get("mean") is None:
                continue
            b = dict(c["budget"], total_samples_seen=int(c["budget"]["training_steps"]) * int(c["budget"]["batch_size"]))
            new_rows.append({"family": base["family"], "dataset_id": ds.split("/", 1)[1], "set_id": set_id,
                             "candidate_id": hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:8],
                             "dataset": self.lab_dataset_params(ds) if ds != f"{base['family']}/{base['dataset_id']}" else base["dataset"], "budget": b, "params": None, **model_row(c["model"]),
                             "model": c["model"], "optimizer": c["optimizer"], "loss": c["loss"], "metric": base["metric"],
                             "mean": r["mean"], "std": r.get("std"), "source": "experiment", "epoch": self.cur_epoch,
                             "origin_epoch": self.worker_job["origin_epoch"] if self.worker_job else self.cur_epoch,
                             "why": a.get("why"), **{key: r.get(key) for key in evidence_fields}})
        if len(new_rows) >= 2:
            with self.lock, (self.run / "lab_experiments.jsonl").open("a", encoding="utf-8") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                for nr in new_rows:
                    fh.write(json.dumps(nr) + "\n")
        log.append({"base": a.get("base"), "why": a.get("why"), "rows": rows, "secs": round(time.time() - t0, 1)})
        return json.dumps({"dataset": res["dataset"], "dataset_params": self.lab_dataset_params(ds), "metric": base["metric"], "base_budget": base["budget"],
                           "results": rows}, ensure_ascii=False)

    def observation_files(self, path: Path, expected: dict | None = None) -> tuple[Path, dict]:
        root = (self.run / "visible").resolve()
        try:
            directory = Path(path).resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"saved observation directory missing: {path}") from exc
        if directory == root or not directory.is_relative_to(root) or not directory.is_dir():
            raise ValueError(f"saved observation path is outside visible inputs: {path}")
        names = ("history.jsonl", "kb.json")
        if expected is not None and set(expected) != set(names):
            raise ValueError("saved observation pin must contain history.jsonl and kb.json")
        files = {}
        for name in names:
            source = directory / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"saved observation file missing or unsafe: {source}")
            raw = source.read_bytes()
            files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            if expected is not None and files[name] != expected[name]:
                raise ValueError(f"saved observation bytes changed: {source}")
        return directory, files

    def worker_observations(self, job: dict) -> Path:
        if "observations_path" not in job:
            return Path(job["payload"]["vis_dir"])
        pin = job.get("observations_pin")
        if pin is not None and (pin.get("path") != job["observations_path"]
                or pin.get("epoch") != job.get("observations_epoch") or "files" not in pin):
            raise ValueError("saved observation pointer does not match its pin")
        directory, files = self.observation_files(Path(job["observations_path"]),
                                                 pin["files"] if pin is not None else None)
        if pin is None:
            pin = {"sequence": job.get("observations_sequence", 0) + 1, "epoch": job.get("observations_epoch"),
                   "origin_epoch": job["origin_epoch"], "path": str(directory), "files": files,
                   "pinned_at": now(), "migration": "legacy saved bytes pinned on recovery; no historical hash"}
            job = update_job(self.worker_path, observations_path=str(directory), observations_pin=pin,
                             observations_sequence=pin["sequence"],
                             observations_history=[*job.get("observations_history", []), pin])
            self.worker_job = job
        return directory

    def research(self, epoch: int, k: int, question: str, focus: str, vis_dir: Path, kb: KB | None = None) -> str:
        out_dir = self.worker_path.parent if self.worker_job else self.run / "epochs" / f"e{epoch:04d}" / "research"
        path = out_dir / f"r{k:02d}.json"
        if path.exists():
            prev = json.loads(path.read_text())
            saved_report = tag(prev.get("report"), "report")
            saved_report = (prev.get("report") or "").strip() if saved_report is None else saved_report
            if prev.get("question") == question and (not self.worker_job
                    or saved_report not in ("", "(no report)")):
                return prev["report"]
        work = self.worker_path.parent / "repo" if self.worker_job else self.run / "sandbox" / f"e{epoch:04d}" / f"_research{k:02d}"
        exps: list[dict] = []
        n_exp = [0]

        def investigate(name, a):
            nonlocal vis_dir
            if name == "run_experiment":
                if not self.worker_job and n_exp[0] >= self.args.research_max_exps:
                    return f"rejected: at most {self.args.research_max_exps} experiments per research task"
                n_exp[0] += 1
                return self.experiment(a, exps)
            if name == "search_history":
                return History(self.visible_rows(vis_dir)).search(a)
            if name == "read_record":
                return History(self.visible_rows(vis_dir)).read(a)
            if name == "kb_open":
                return self.open_kb(kb, a.get("targets", []), latest=bool(self.worker_job))
            if name == "python":
                return self.py(a["code"], work, vis_dir)
            if name == "publish_report" and self.worker_job:
                return self.publish_report(a["text"], a.get("claim_ids"), a.get("title") or question,
                                           self.worker_job["payload"]["topic"], kb)
            if name == "propose_changes" and self.worker_job:
                return self.propose_changes(kb, a.get("operations", []), a.get("report"))
            if name == "refresh_observations" and self.worker_job:
                job = read_job(self.worker_path)
                if "observations_path" in job:
                    self.worker_observations(job)
                    job = read_job(self.worker_path)
                live = self.run / "kb_live.json"
                current = KB(json.loads(live.read_text()), self.run) if live.exists() else kb
                saved_epochs = [int(path.parent.name[1:]) for path in
                                (self.run / "epochs").glob("e[0-9][0-9][0-9][0-9]/records.jsonl")]
                latest_epoch = max([epoch, *saved_epochs,
                                    *[m["epoch"] for m in read_jsonl(self.run / "metrics.jsonl")]])
                sequence = job.get("observations_sequence", 0) + 1
                version = f"{job['id']}_research_e{latest_epoch:04d}_v{sequence:06d}_{uuid.uuid4().hex[:12]}"
                updated = self.visible("." + version,
                                       self.history_records(latest_epoch), current)
                updated, files = self.observation_files(updated)
                destination = updated.with_name(version)
                updated.rename(destination)
                pin = {"sequence": sequence, "epoch": latest_epoch, "origin_epoch": job["origin_epoch"],
                       "path": str(destination), "files": files, "pinned_at": now()}
                self.worker_job = update_job(self.worker_path, observations_epoch=latest_epoch,
                                            observations_path=str(destination), observations_pin=pin,
                                            observations_sequence=sequence,
                                            observations_history=[*job.get("observations_history", []), pin])
                vis_dir = destination
                return (f"Saved training observations frozen at e{latest_epoch:04d}; curation may still be running. "
                        f"Future Python/history reads use {destination.name}. "
                        "load_kb() now reads current saved claims; use these to review stale proposals before publishing.")
            return f"unknown tool {name}"

        def handler(name, a):
            with self.lock:
                return investigate(name, a)

        system = RESEARCH_SYSTEM.format(max_cands=self.args.research_max_cands, max_exps=self.args.research_max_exps,
                                        max_calls=self.args.research_turns)
        tools = tools_spec(False, bool(self.args.corpus), lab=True, research=True)
        if self.worker_job:
            system = ("You maintain a continuing scientific report and reproducible research repo for ArchitectureIQ. "
                      "A report can explain multiple KB claims. State the measured phenomena, definitions and assumptions, "
                      "methods, results and uncertainty, competing explanations, distinguishable predictions and their tests, "
                      "extrapolation boundaries and unresolved questions. Complete means independently reviewable, not finished theory. "
                      "Use history, Python and controlled experiments. No tool-call limit applies. Save code and analysis files with "
                      "Python in your work directory; publish_report saves the full report and commits your repo. Publish early, "
                      "Use propose_changes to add, revise, merge or delete scoped KB claims from new evidence. "
                      "Use refresh_observations at a research checkpoint to read later completed epochs; each refresh pins a data version. "
                      "then update when new evidence changes the argument. Unfinished reports are usable next epoch. A 40-minute "
                      "window changes your epoch membership and does not terminate your work. Read load_history() and load_kb() "
                      "for the frozen research inputs, preserving measurement sources. Make predictions for unmeasured conditions "
                      "before experiments; report counterexamples instead of decorating facts with a mechanism. "
                      "Only use allowed lab datasets. Return <report>your latest full report</report> when this investigation ends.")
            tools.append({"type": "function", "function": {"name": "kb_open",
                          "description": "Read latest saved research reports by topic, job ID, claim ID or science.",
                          "parameters": {"type": "object", "properties": {
                              "targets": {"type": "array", "items": {"type": "string"}}},
                              "required": ["targets"]}}})
            tools.append({"type": "function", "function": {"name": "publish_report",
                          "description": "Save a version of the complete ongoing report plus repository commit; no readiness gate.",
                          "parameters": {"type": "object", "properties": {"text": {"type": "string"},
                          "title": {"type": "string"}, "claim_ids": {"type": "array", "items": {"type": "string"}}},
                          "required": ["text"]}}})
            tools.append({"type": "function", "function": {"name": "propose_changes",
                          "description": "Submit claim operations atomically, optionally with a full report version in the same publication.",
                          "parameters": {"type": "object", "properties": {"operations": {"type": "array",
                          "items": {"type": "object", "properties": {"op": {"type": "string",
                          "enum": ["add", "revise", "merge", "delete", "annotate"]},
                          "claim_ids": {"type": "array", "items": {"type": "string"}},
                          "text": {"type": "string"}, "reason": {"type": "string"},
                          "source_question_ids": {"type": "array", "items": {"type": "string"}},
                          "fields": {"type": "object"}}, "required": ["op", "reason"]}},
                          "report": {"type": "object", "properties": {"text": {"type": "string"},
                          "title": {"type": "string"}, "topic": {"type": "string"},
                          "claim_ids": {"type": "array", "items": {"type": "string"}}},
                          "required": ["text"]}},
                          "required": ["operations"]}}})
            tools.append({"type": "function", "function": {"name": "refresh_observations",
                          "description": "Pin the latest completed training observations for this continuing research project.",
                          "parameters": {"type": "object", "properties": {}}}})
        t0 = time.time()
        session_dir = self.worker_path.parent / "session" if self.worker_job else work / "_codex"
        sess = CodexSession(session_dir, self.args.model, system, tools, handler)
        try:
            prompt = f"Research question: {question}" + (f"\nFocus / context: {focus}" if focus else "")
            r = sess.turn(prompt, self.args.research_effort, "discover",
                          max_calls=None if self.worker_job else self.args.research_turns,
                          warn_text="stop calling tools and write the <report>.", timeout=None if self.worker_job else 2400)
            final = r.get("final") or ""
            report = tag(final, "report")
            report = final.strip() if report is None else report
            if self.worker_job and report in ("", "(no report)"):
                raise RuntimeError("research interrupted without a nonempty report; resume the same session")
            report = report or "(no report)"
            if self.worker_job:
                self.publish_report(report, None, None, self.worker_job["payload"]["topic"], kb)
        except Exception as exc:  # a failed research task should not kill the summarizer session
            if self.worker_job:
                raise
            report, r = f"[research failed] {type(exc).__name__}: {exc}"[:600], {"usage": None}
        finally:
            sess.close()
        write_json(path, {"epoch": epoch, "k": k, "question": question, "focus": focus, "report": report,
                          "experiments": exps, "tool_log": sess.log, "usage": r.get("usage"),
                          "secs": round(time.time() - t0), "model": self.args.model})
        print(f"[research] e{epoch} r{k}: {len(sess.log)} calls, {len(exps)} experiments, {round(time.time() - t0)}s",
              flush=True)
        return report

    def visible_rows(self, vis_dir: Path) -> list[dict]:
        return read_jsonl(vis_dir / "history.jsonl")

    def open_kb(self, kb: KB, targets: list[str], latest: bool = False) -> str:
        claims = [c for c in kb.claims.values() if c["id"] in targets
                  or any(t in c.get("tags", []) for t in targets)]
        reports = copy.deepcopy(kb.reports)
        if latest:
            for path in sorted((self.run / "jobs").glob("J*/publications/p*.json")):
                publication = json.loads(path.read_text())
                for identifier, report in publication["after"].get("reports", {}).items():
                    prior = reports.get(identifier, {})
                    order = (report.get("published_at", ""), publication["id"])
                    if order > (prior.get("published_at", ""), prior.get("publication_id", "")):
                        reports[identifier] = {**report, "publication_id": publication["id"]}
        wanted = {r for c in claims for r in c.get("report_ids", [])}
        wanted.update(i for i, r in reports.items() if i in targets or r.get("topic") in targets
                      or r.get("job_id") in targets
                      or any(cid in targets for cid in r.get("claim_ids", [])))
        if "science" in targets:
            wanted.update(reports)
        out = [json.dumps(claims, ensure_ascii=False, indent=1)] if claims else []
        for i in sorted(wanted):
            report = reports[i]
            out.append(f"# {i}: {report['title']} ({report['repo_commit']})\n" + Path(report["path"]).read_text())
        return "\n\n".join(out) or "no matching KB entries or reports"

    def submit_job(self, kind: str, epoch: int, kb: KB, records: list[dict], vis_dir: Path,
                   **context) -> dict:
        payload = {"args": vars(self.args), "kb": kb.to_doc(), "records": records,
                   "vis_dir": str(vis_dir.resolve()), **context}
        key = f"summary:{epoch}" if kind == "summarizer" else f"research:{epoch}:{context['topic']}"
        job = self.jobs.submit(kind, epoch, payload, key=key)
        self.jobs.poll(epoch)
        self.jobs.start_supervisor()
        return job

    def journal_commit(self, rec: dict, doc: dict) -> None:
        with self.lock:
            self.job_sequence += 1
            directory = self.worker_path.parent / "publications"
            directory.mkdir(exist_ok=True)
            publication = {"id": f"{self.worker_job['id']}:{self.job_sequence:06d}",
                           "origin_epoch": self.worker_job["origin_epoch"],
                           "job_id": self.worker_job["id"], "base": copy.deepcopy(self.job_base),
                           "after": copy.deepcopy(doc), "operation": copy.deepcopy(rec), "published_at": now()}
            write_json(directory / f"p{self.job_sequence:06d}.json", publication)
            self.job_base = copy.deepcopy(doc)
            write_json(self.worker_path.parent / "kb_checkpoint.json",
                       {"sequence": self.job_sequence, "kb": doc})

    def propose_changes(self, kb: KB, operations: list[dict], report: dict | None = None) -> str:
        author = {"role": "researcher", "model": self.args.model, "job_id": self.worker_job["id"]}
        results = []
        with self.lock:
            if not isinstance(operations, list) or any(not isinstance(op, dict) for op in operations):
                return "rejected: operations must be a list of objects"
            if report is not None:
                if not isinstance(report, dict) or set(report) - {"text", "title", "topic", "claim_ids"}:
                    return "rejected: report may contain text, title, topic and claim_ids only"
                if not isinstance(report.get("text"), str) or not report["text"].strip():
                    return "rejected: report text must be nonempty"
                if any(key in report and not isinstance(report[key], str) for key in ("title", "topic")):
                    return "rejected: report title and topic must be strings"
                if "claim_ids" in report and (not isinstance(report["claim_ids"], list)
                        or any(not isinstance(cid, str) for cid in report["claim_ids"])):
                    return "rejected: report claim_ids must be a list of strings"
            staged = KB(kb.to_doc(), self.worker_path.parent)
            start = len(staged.pending)
            for op in operations:
                if set(op) - {"op", "claim_ids", "text", "reason", "source_question_ids", "fields"}:
                    return "rejected: operation contains reserved or unknown fields"
                kind, ids, reason = op.get("op"), op.get("claim_ids", []), op.get("reason", "")
                if not isinstance(ids, list) or any(not isinstance(cid, str) for cid in ids):
                    return "rejected: operation claim_ids must be a list of strings"
                if not isinstance(reason, str):
                    return "rejected: operation reason must be a string"
                if kind not in ("add", "revise", "merge", "delete", "annotate"):
                    return "rejected: unknown operation"
                if kind != "add" and (not ids or len(set(ids)) != len(ids) or any(i not in staged.claims for i in ids)):
                    return "rejected: all target claims must exist at that step"
                if kind == "revise" and len(ids) != 1:
                    return "rejected: revise takes one claim"
                if kind in ("add", "revise", "merge") and (not isinstance(op.get("text"), str)
                        or not op["text"].strip()):
                    return "rejected: provide a nonempty scoped claim text"
                sources = op.get("source_question_ids", [])
                if not isinstance(sources, list) or any(not isinstance(qid, str) for qid in sources):
                    return "rejected: source_question_ids must be a list of strings"
                fields = op.get("fields", {})
                if not isinstance(fields, dict) or set(fields) - set(SCIENCE_FIELDS) - {"tags"}:
                    return "rejected: fields may contain scientific annotations and tags only"
                if kind == "add":
                    results.append(staged.add(op["text"], op.get("source_question_ids", []), reason, author, fields))
                elif kind in ("revise", "merge"):
                    results.append(staged.merge(ids, op["text"], reason, author, None, None, fields))
                elif kind == "delete":
                    for cid in ids:
                        staged.delete(cid, reason, author)
                else:
                    for cid in ids:
                        staged.annotate(cid, fields, reason, author)
            if not operations and report is None:
                return "rejected: provide operations or a report"
            if report is not None:
                topic = report.get("topic") or self.worker_job.get("payload", {}).get("topic")
                if not topic:
                    return "rejected: provide a report topic"
                outcome = self._publish_report(report["text"], report.get("claim_ids"),
                                               report.get("title"), topic, staged)
                if outcome.startswith("rejected"):
                    return outcome
            kb.claims, kb.retired = staged.claims, staged.retired
            kb.reports = staged.reports
            kb.pending, kb.counter = staged.pending, staged.counter
            kb.commit("proposal", author, payload={"operations": copy.deepcopy(staged.pending[start:])},
                      reason="atomic research proposal" + (" with report" if report is not None else ""))
            kb.checkpoint()
        return json.dumps({"submitted": len(operations), "claim_ids": results,
                           "report_id": "R" + hashlib.sha256(topic.encode()).hexdigest()[:10]
                           if report is not None else None})

    def publish_report(self, text: str, claim_ids: list[str] | None, title: str | None, topic: str, kb: KB) -> str:
        with self.lock:
            return self._publish_report(text, claim_ids, title, topic, kb)

    def _publish_report(self, text: str, claim_ids: list[str] | None, title: str | None, topic: str, kb: KB) -> str:
        if not text.strip():
            return "rejected: report is empty"
        report_id = "R" + hashlib.sha256(topic.encode()).hexdigest()[:10]
        previous = kb.reports.get(report_id, {})
        if claim_ids is None:
            claim_ids = previous.get("claim_ids", [])
        title = title or previous.get("title") or topic
        if any(i not in kb.claims for i in claim_ids):
            return "rejected: report claim IDs must exist"
        repo = self.worker_path.parent / "repo"
        repo.mkdir(exist_ok=True)
        (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n_analysis.py\n_codex/\ncodex_home/\n", encoding="utf-8")
        if not (repo / ".git").exists():
            subprocess.run(["git", "init", "-b", "research"], cwd=repo, check=True, capture_output=True)
        (repo / "report.md").write_text(text, encoding="utf-8")
        subprocess.run(["git", "add", "--", "."], cwd=repo, check=True, capture_output=True)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode
        if staged:
            subprocess.run(["git", "-c", "user.name=KB Research", "-c", "user.email=kb@local",
                            "commit", "-m", f"e{self.worker_job['origin_epoch']:04d}: {title}"],
                           cwd=repo, check=True, capture_output=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        version = self.run / "reports" / report_id / f"{commit}.md"
        version.parent.mkdir(parents=True, exist_ok=True)
        if not version.exists():
            version.write_text(text, encoding="utf-8")
        kb.reports[report_id] = {"id": report_id, "title": title, "topic": topic,
                                 "claim_ids": claim_ids, "repo": str(repo), "repo_commit": commit,
                                 "path": str(version), "origin_epoch": self.worker_job["origin_epoch"],
                                 "job_id": self.worker_job["id"], "published_at": now()}
        for cid, claim in kb.claims.items():
            refs = [r for r in claim.get("report_ids", []) if r != report_id]
            if cid in claim_ids:
                refs.append(report_id)
            if refs or "report_ids" in claim:
                claim["report_ids"] = refs
        kb.commit("report", {"role": "researcher", "model": self.args.model},
                  report_id=report_id, repo_commit=commit, reason=title)
        kb.checkpoint()
        update_job(self.worker_path, checkpoint_commit=commit, report_id=report_id)
        return f"published {report_id}@{commit}; unfinished science is visible next batch"

    def apply_publications(self, kb: KB, epoch: int) -> list[str]:
        applied = []
        publications = [(path, json.loads(path.read_text()))
                        for path in (self.run / "jobs").glob("J*/publications/p*.json")]
        publications.sort(key=lambda row: (row[1].get("published_at", ""),
                                           int(row[0].stem[1:]), row[1]["job_id"]))
        for path, pub in publications:
            if pub["id"] in kb.applied_publications:
                continue
            prior_map = kb.publication_id_maps.get(pub["job_id"], {})
            mappings = dict(prior_map)
            remap = {}
            old_local = {c["id"]: c for c in pub["base"]["claims"]}
            new_local = {c["id"]: c for c in pub["after"]["claims"]}
            old_retired = {c["id"]: c for c in pub["base"].get("retired", [])}
            new_retired = {c["id"]: c for c in pub["after"].get("retired", [])}
            reserved = set(kb.claims) | {r["id"] for r in kb.retired}
            reserved.update(i for mapping in kb.publication_id_maps.values() for i in mapping.values())
            created = (new_local.keys() | new_retired.keys()) - (old_local.keys() | old_retired.keys())
            for cid in sorted(created):
                if cid in mappings:
                    continue
                target = cid
                if target in reserved:
                    nums = [int(i[1:]) for i in reserved | set(new_local) if i[1:].isdigit()]
                    target = f"K{max(nums, default=0) + 1:04d}"
                mappings[cid] = target
                remap[cid] = target
                reserved.add(target)

            def map_claim(c):
                result = copy.deepcopy(c)
                result["id"] = mappings.get(c["id"], c["id"])
                for key in ("refines", "merged_from", "claim_ids"):
                    if isinstance(result.get(key), list):
                        result[key] = [mappings.get(i, i) for i in result[key]]
                    elif key == "refines" and isinstance(result.get(key), str):
                        result[key] = mappings.get(result[key], result[key])
                if result.get("merged_into"):
                    result["merged_into"] = mappings.get(result["merged_into"], result["merged_into"])
                if result.get("origin") and re.fullmatch(r"c[0-9]+", result["origin"]):
                    result["origin"] = f"{pub['job_id']}:{result['origin']}"
                return result

            old = {mappings.get(i, i): map_claim(c) for i, c in old_local.items()}
            new = {mappings.get(i, i): map_claim(c) for i, c in new_local.items()}
            changed = {i for i in old.keys() | new.keys() if old.get(i) != new.get(i)}
            structural_conflicts = []
            additive = {"support_count", "failure_count", "credibility", "last_evaluated_epoch",
                        "created_epoch", "report_ids", "origin_epoch", "origin"}
            for cid in changed & old.keys():
                current = kb.claims.get(cid)
                substantive = any(old[cid].get(k) != new.get(cid, {}).get(k)
                                  for k in old[cid].keys() | new.get(cid, {}).keys()
                                  if k != "report_ids")
                if current is None and substantive:
                    structural_conflicts.append(f"{cid}: unavailable claim dependency")
                if cid not in new and current is not None:
                    for key in current.keys() | old[cid].keys():
                        if key not in additive and current.get(key) != old[cid].get(key):
                            structural_conflicts.append(f"{cid}.{key}")
                keys = old[cid].keys() | new.get(cid, {}).keys()
                for key in keys:
                    if key in additive:
                        continue
                    before, after = old[cid].get(key), new.get(cid, {}).get(key)
                    if before != after and (current is None or current.get(key) not in (before, after)):
                        structural_conflicts.append(f"{cid}.{key}")
            if structural_conflicts:
                write_json(path.parent.parent / f"conflict-{path.stem}.json",
                           {"publication_id": pub["id"], "origin_epoch": pub["origin_epoch"],
                            "status": "deferred", "conflicts": structural_conflicts,
                            "operation": pub["operation"], "proposed_claims": pub["after"]["claims"],
                            "id_map": mappings})
                changed = set()
            transfers = collections.defaultdict(lambda: [0.0, 0.0])
            operations = (pub["operation"].get("payload", {}).get("operations", [])
                          if pub["operation"]["op"] == "proposal" else [pub["operation"]])
            merge_targets = {}
            for operation in operations:
                if operation["op"] != "merge" or not operation.get("payload", {}).get("default_counts"):
                    continue
                target = mappings.get(operation["new_id"], operation["new_id"])
                for local_id in operation["claim_ids"]:
                    cid = mappings.get(local_id, local_id)
                    merge_targets[cid] = target
            for cid in changed & old.keys() & merge_targets.keys():
                target = merge_targets[cid]
                seen = {cid}
                route = [target]
                while target in merge_targets and target not in seen:
                    seen.add(target)
                    target = merge_targets[target]
                    route.append(target)
                if cid in kb.claims:
                    for destination in route:
                        transfers[destination][0] += kb.claims[cid]["support_count"] - old[cid]["support_count"]
                        transfers[destination][1] += kb.claims[cid]["failure_count"] - old[cid]["failure_count"]
            conflicts = []
            for cid in sorted(changed):
                if cid not in old:
                    kb.claims[cid] = {**copy.deepcopy(new[cid]), "created_epoch": epoch}
                    origin = kb.claims[cid].get("origin")
                    if origin and re.fullmatch(r"c[0-9]+", origin):
                        kb.claims[cid]["origin"] = f"{pub['job_id']}:{origin}"
                    kb.claims[cid]["origin_epoch"] = pub["origin_epoch"]
                    c = kb.claims[cid]
                    kb._set(c, c["support_count"] + transfers[cid][0], c["failure_count"] + transfers[cid][1])
                elif cid not in new:
                    lineage = next((map_claim(r) for r in pub["after"].get("retired", [])
                                    if mappings.get(r["id"], r["id"]) == cid), {})
                    current = kb.claims.pop(cid)
                    kb.retired.append({**lineage, **current,
                                       **{k: lineage[k] for k in ("retired_by", "reason", "merged_into") if k in lineage},
                                       "retired_epoch": epoch, "origin_epoch": pub["origin_epoch"]})
                    if lineage.get("retired_by"):
                        kb.retired[-1]["retired_by"] = f"{pub['job_id']}:{lineage['retired_by']}"
                elif cid in kb.claims:
                    current = kb.claims[cid]
                    for key in old[cid].keys() | new[cid].keys():
                        before, after = old[cid].get(key), new[cid].get(key)
                        if before == after or key in ("credibility", "created_epoch"):
                            continue
                        if key in ("support_count", "failure_count"):
                            current[key] += after - before
                        elif key == "report_ids":
                            continue  # reconciled from the accepted report associations below
                        elif current.get(key) == before:
                            if key in new[cid]:
                                current[key] = copy.deepcopy(after)
                            else:
                                current.pop(key, None)
                        elif current.get(key) != after:
                            conflicts.append(f"{cid}.{key}")
                    kb._set(current, current["support_count"], current["failure_count"])
                else:
                    conflicts.append(cid)
            if not structural_conflicts:
                for local_id in new_retired.keys() - old_retired.keys() - old_local.keys():
                    lineage = map_claim(new_retired[local_id])
                    if lineage["id"] not in {r["id"] for r in kb.retired}:
                        lineage.update(retired_epoch=epoch, origin_epoch=pub["origin_epoch"])
                        if lineage.get("retired_by"):
                            lineage["retired_by"] = f"{pub['job_id']}:{lineage['retired_by']}"
                        delta = transfers[lineage["id"]]
                        kb._set(lineage, lineage["support_count"] + delta[0], lineage["failure_count"] + delta[1])
                        kb.retired.append(lineage)
            for report_id, report in pub["after"].get("reports", {}).items():
                if structural_conflicts and pub["operation"]["op"] == "proposal":
                    continue
                if report != pub["base"].get("reports", {}).get(report_id):
                    order = (report.get("published_at", pub.get("published_at", "")), pub["id"])
                    prior = kb.reports.get(report_id, {})
                    if order < (prior.get("published_at", ""), prior.get("publication_id", "")):
                        continue
                    wanted = [mappings.get(i, i) for i in report["claim_ids"]]
                    available = [i for i in wanted if i in kb.claims]
                    kb.reports[report_id] = {**report, "claim_ids": available,
                                             "deferred_claim_ids": [i for i in wanted if i not in kb.claims],
                                             "publication_id": pub["id"],
                                             "claim_proposal_status": "deferred" if structural_conflicts or wanted != available else "applied"}
            for claim in kb.claims.values():
                associated = sorted(i for i, report in kb.reports.items() if claim["id"] in report["claim_ids"])
                if associated or "report_ids" in claim:
                    claim["report_ids"] = associated
            kb.publication_id_maps[pub["job_id"]] = mappings
            kb.applied_publications.append(pub["id"])
            kb.commit("async_apply", pub["operation"]["author"], origin_epoch=pub["origin_epoch"],
                      applied_epoch=epoch, publication_id=pub["id"], job_id=pub["job_id"],
                      payload={"operation": pub["operation"], "id_remap": remap,
                               "claim_status": "deferred" if structural_conflicts else "applied",
                               "conflicts": structural_conflicts or conflicts})
            applied.append(pub["id"])
        if applied:
            write_json(self.run / "kb_live.json", {**kb.to_doc(), "pending_commits": kb.pending})
            kb.flush()
            write_json(self.run / "kb_live.json", kb.to_doc())
        return applied

    @classmethod
    def drain_publications(cls, run: Path) -> list[str] | None:
        run = Path(run).resolve()
        with (run / "jobs/manager.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            config = read_job(run / "jobs/supervisor.json")
            if not config.get("owner_finished") and process_alive(config.get("owner")) is not False:
                return None
            live = run / "kb_live.json"
            if not live.exists():
                return []
            obj = cls.__new__(cls)
            obj.run = run
            kb = KB(json.loads(live.read_text()), run)
            pending = bool(kb.pending)
            applied = obj.apply_publications(kb, kb.epoch + 1)
            if pending and not applied:
                kb.flush()
                write_json(live, kb.to_doc())
            return applied

    def worker(self, path: Path) -> None:
        with worker_lock(path) as job:
            self.worker_job, self.worker_path = job, path
            payload, epoch = job["payload"], job["origin_epoch"]
            checkpoint = path.parent / "kb_checkpoint.json"
            saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            publications = sorted((path.parent / "publications").glob("p*.json"))
            if publications:
                latest = json.loads(publications[-1].read_text())
                sequence = int(publications[-1].stem[1:])
                if sequence > saved.get("sequence", 0):
                    saved = {"sequence": sequence, "kb": latest["after"]}
            kb = KB(saved.get("kb", payload["kb"]), path.parent)
            kb.epoch, self.cur_epoch = epoch, epoch
            self.job_base = copy.deepcopy(kb.to_doc())
            self.job_sequence = saved.get("sequence", 0)
            kb.publish_commit = self.journal_commit
            vis = self.worker_observations(job)
            if job["kind"] == "summarizer":
                if job.get("curation_finished"):
                    finish_job(path, {"science": (path.parent / "science.md").read_text(),
                                      "publications": self.job_sequence})
                    return
                science, trace = self.summarize(epoch, kb, payload["records"], vis)
                write_json(path.parent / "summarizer_trace.json", trace)
                (path.parent / "science.md").write_text(science, encoding="utf-8")
            else:
                science = self.research(epoch, 1, payload["question"], payload.get("focus", ""), vis, kb)
            finish_job(path, {"science": science, "publications": self.job_sequence})

    # ---- eval
    def evaluate(self, label: str, kb: KB | None, ids: list[str] | None = None) -> dict:
        out_dir = self.run / "eval" / label
        ids = ids or self.split["eval"]
        with cf.ThreadPoolExecutor(self.args.concurrency) as ex:
            recs = self.retry_map(ex, lambda i: self.safe(self.solve, i, kb, None, None, 0, out_dir, False), ids)
        by = collections.defaultdict(list)
        for r in recs:
            by[r["source"]].append(r["score"])
        summary = {"label": label, "n": len(recs), "mean": round(sum(r["score"] for r in recs) / max(1, len(recs)), 4),
                   "by_source": {s: round(sum(v) / len(v), 4) for s, v in sorted(by.items())},
                   "errors": len(ids) - len(recs)}
        write_json(out_dir / "_summary.json", summary)
        return summary

    @staticmethod
    def retry_map(ex, fn, ids: list[str], passes: int = 3) -> list[dict]:
        done: dict[str, dict] = {}
        for k in range(passes):
            todo = [i for i in ids if i not in done]
            if not todo:
                break
            if k:
                print(f"[retry] pass {k + 1}: {len(todo)} failed, sleeping 90s", flush=True)
                time.sleep(90)
            for i, r in zip(todo, ex.map(fn, todo)):
                if r:
                    done[i] = r
        return [done[i] for i in ids if i in done]

    def safe(self, fn, *a):
        try:
            return fn(*a)
        except Exception as exc:
            print(f"[warn] {a[0]}: {exc}", flush=True)
            return None

    def sample(self, base: str, kb: KB | None, n: int, ids: list[str] | None = None) -> dict[str, list[float]]:
        """Ensure >= n eval samples under labels base, base_r2, ...; return per-question score lists."""
        labels = [base] + [f"{base}_r{k}" for k in range(2, n + 1)]
        for lab in labels:
            if not self.eval_done(lab):
                print(f"[eval] {lab}", self.evaluate(lab, kb, ids), flush=True)
        per_q = collections.defaultdict(list)
        k = 1
        while (d := self.run / "eval" / (base if k == 1 else f"{base}_r{k}")).exists():
            for f in d.glob("q_*.json"):
                r = json.loads(f.read_text())
                per_q[r["question_id"]].append(r["score"])
            k += 1
        return per_q

    @staticmethod
    def paired(a: dict, b: dict) -> tuple[float, float, int]:
        qs = [q for q in a if q in b]
        d = [statistics.mean(a[q]) - statistics.mean(b[q]) for q in qs]
        se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
        return statistics.mean(d), se, len(qs)

    def gate(self, epoch: int, kb: KB) -> dict:
        """Accept the epoch's KB only if it is not worse (paired, val set) than the incumbent; else roll back."""
        a = self.args
        best_path = self.run / "best.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else {"label": "kb_0000", "epoch": 0}
        inc_doc = json.loads((self.run / "kb" / f"{best['label']}.json").read_text())
        inc_kb = KB(inc_doc, self.run)
        cand = self.sample(f"kb_{epoch:04d}", kb, a.gate_repeats)
        have = len(next(iter(self.sample(best["label"], inc_kb, 1).values()), []))
        inc = self.sample(best["label"], inc_kb, have + 1)  # one fresh incumbent sample per gate (winner's curse)
        nokb = self.sample("nokb", None, 1)
        d, se, n = self.paired(cand, inc)
        dn, sen, _ = self.paired(cand, nokb)
        accept = d >= -a.gate_tol + a.gate_z * se
        g = {"epoch": epoch, "candidate": f"kb_{epoch:04d}", "incumbent": best["label"], "delta": round(d, 4),
             "se": round(se, 4), "n_q": n, "accept": accept, "cand_mean": round(statistics.mean(
             statistics.mean(v) for v in cand.values()), 4), "delta_vs_nokb": round(dn, 4), "se_vs_nokb": round(sen, 4)}
        if accept and a.test_on_accept and self.split.get("test"):
            # readout only (never gates): the accepted KB on the untouched test set vs no-KB
            tc = self.sample(f"test_kb_{epoch:04d}", kb, a.test_on_accept, self.split["test"])
            tn = self.sample("test_nokb", None, a.test_on_accept, self.split["test"])
            td, tse, _ = self.paired(tc, tn)
            g.update(test_mean=round(statistics.mean(statistics.mean(v) for v in tc.values()), 4),
                     test_delta_vs_nokb=round(td, 4), test_se=round(tse, 4))
        if accept:
            write_json(best_path, {"label": f"kb_{epoch:04d}", "epoch": epoch, "accepted_at": now(), "gate": g})
        else:
            (self.run / "kb_rejected").mkdir(exist_ok=True)
            write_json(self.run / "kb_rejected" / f"kb_{epoch:04d}.json", kb.to_doc())
            kb.rollback(inc_doc, f"gate: kb_{epoch:04d} vs {best['label']} delta {d:+.3f} ± {se:.3f} on val",
                        {"from": f"kb_{epoch:04d}", "to": best["label"], "delta": d, "se": se})
            kb.flush()
            write_json(self.run / "kb" / f"kb_{epoch:04d}.json", kb.to_doc())
        print("[gate]", json.dumps(g), flush=True)
        return g

    def final_test(self) -> dict:
        a = self.args
        ids = self.split.get("test") or []
        if not ids:
            return {}
        best_path = self.run / "best.json"
        if best_path.exists():
            best = json.loads(best_path.read_text())
            selection = "validation_best"
        else:
            latest = max((self.run / "kb").glob("kb_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
            best = {"label": latest.stem}
            selection = "latest_checkpoint"
        arms = {"nokb": None, "kb_0000": KB(json.loads((self.run / "kb" / "kb_0000.json").read_text()), self.run),
                best["label"]: KB(json.loads((self.run / "kb" / f"{best['label']}.json").read_text()), self.run)}
        for extra in a.test_extra_kb:
            name, path = extra.split("=", 1)
            arms[name] = KB(json.loads(Path(path).read_text()), self.run)
        per = {k: self.sample(f"test_{k}", kb, a.test_repeats, ids) for k, kb in arms.items()}
        out = {"best": best["label"], "selection": selection, "n_q": len(ids), "arms": {}}
        for k, v in per.items():
            out["arms"][k] = {"mean": round(statistics.mean(statistics.mean(x) for x in v.values()), 4)}
            if k != "nokb":
                d, se, _ = self.paired(v, per["nokb"])
                out["arms"][k].update(delta_vs_nokb=round(d, 4), se=round(se, 4))
            if k not in ("nokb", "kb_0000"):
                d, se, _ = self.paired(v, per["kb_0000"])
                out["arms"][k].update(delta_vs_seed=round(d, 4), se_seed=round(se, 4))
        write_json(self.run / "test_report.json", out)
        print("[test]", json.dumps(out), flush=True)
        return out

    # ---- main
    def eval_done(self, label: str) -> bool:
        p = self.run / "eval" / label / "_summary.json"
        return p.exists() and json.loads(p.read_text()).get("errors", 1) == 0

    def load_kb(self) -> KB:
        live = self.run / "kb_live.json"
        if self.args.async_agents and live.exists():
            return KB(json.loads(live.read_text()), self.run)
        snaps = sorted((self.run / "kb").glob("kb_*.json"))
        if not snaps:
            path = Path(self.args.seed_kb) if self.args.seed_kb else SEED_KB
            kb = KB.seed(self.run, path)
            write_json(self.run / "kb" / "kb_0000.json", kb.to_doc())
            kb.commit("seed", {"role": "seed", "model": "external-seed"},
                      payload={"claims": len(kb.claims), "source": str(path)})
            kb.flush()
            return kb
        return KB(json.loads(snaps[-1].read_text()), self.run)

    def solver_snapshot(self, kb: KB, epoch: int) -> KB:
        self.solve_snapshot_label = f"solve_e{epoch:04d}"
        path = self.run / "solver_snapshots" / f"{self.solve_snapshot_label}.json"
        if path.exists():
            doc = json.loads(path.read_text())
            self.solver_selection = doc.get("solver_selection", {"mode": "saved", "label": self.solve_snapshot_label})
            return KB(doc, self.run)
        best_path = self.run / "best.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else {}
        stable = bool(self.args.macro_every) and best.get("solver_mode") == "stable"
        if stable:
            label = best["solver_label"]
            doc = json.loads((self.run / "kb" / f"{label}.json").read_text())
        else:
            label, doc = "research_live", kb.to_doc()
        self.solver_selection = {"mode": "stable" if stable else "live", "label": label,
                                 "research_epoch": epoch, "review_epoch": best.get("review", {}).get("epoch")}
        write_json(path, {**doc, "solver_selection": self.solver_selection})
        return KB(doc, self.run)

    def main(self) -> None:
        a = self.args
        config_path = self.run / "config.json"
        config = {**vars(a), "harness": SCIENCE_HARNESS if a.science else HARNESS_VERSION,
                  "pool": str(Path(a.pool).resolve()),
                  "pool_sha256": hashlib.sha256(Path(a.pool).read_bytes()).hexdigest(),
                  "seed_kb": str(a.seed_kb or SEED_KB),
                  "growth_policy": "research_persistent_solver_fallback_v1" if a.macro_every else "legacy"}
        if config_path.exists():
            previous = json.loads(config_path.read_text())
            previous.pop("updated_at", None)
        else:
            previous = None
        if config != previous:
            write_json(config_path, {**config, "updated_at": now()})
        if self.jobs:
            self.jobs.start_supervisor(claim_owner=True)
        kb = self.load_kb()
        metrics_path = self.run / "metrics.jsonl"
        if self.jobs:
            kb.flush()
            completed = {m["epoch"] for m in read_jsonl(metrics_path)}
            kb.epoch = max(completed or {0})
            self.jobs.poll(kb.epoch + 1)
        if getattr(a, "publish_blog", None):
            if __package__:
                from .kb_publish import ensure_publisher
            else:
                from kb_publish import ensure_publisher
            publisher = ensure_publisher(self.run, Path(a.pool), Path(a.publish_blog), a.publish_interval)
            print(f"[publisher] {publisher['status']}", flush=True)
        if not a.no_eval and not self.eval_done("nokb"):
            print("[eval] nokb", self.evaluate("nokb", None), flush=True)
        metrics = read_jsonl(metrics_path)
        done_epochs = {m["epoch"] for m in metrics}
        if a.macro_every and not a.no_eval:
            checkpoints = sorted((self.run / "kb").glob("kb_*.json"))
            for path in checkpoints:
                epoch = int(path.stem[3:])
                due = epoch > 0 and (epoch % a.macro_every == 0 or epoch == a.epochs)
                if due and not any(m["epoch"] == epoch and m.get("review") for m in metrics):
                    review = self.macro_review(epoch, KB(json.loads(path.read_text()), self.run))
                    existing = next((m for m in reversed(metrics) if m["epoch"] == epoch), {})
                    records = read_jsonl(self.run / "epochs" / f"e{epoch:04d}" / "records.jsonl")
                    recovered = {**existing, "epoch": epoch, "resumed_review": True, "review": review}
                    if records and "n" not in recovered:
                        recovered.update(n=len(records), mean_score=round(statistics.mean(r["score"] for r in records), 4))
                    with metrics_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(recovered, ensure_ascii=False) + "\n")
                    metrics.append(recovered)
            if self.jobs:
                kb.epoch = max((m["epoch"] for m in metrics), default=kb.epoch)
        if a.gate and not a.macro_every and not a.no_eval and kb.epoch > 0 and kb.epoch not in done_epochs:
            # crashed between the epoch's snapshot and its gate: gate it now (may roll back) before continuing
            self.cur_epoch = kb.epoch
            m = {"epoch": kb.epoch, "resumed_gate": True, "gate": self.gate(kb.epoch, kb)}
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        elif not a.no_eval and not a.macro_every and not self.eval_done(f"kb_{kb.epoch:04d}"):
            print(f"[eval] kb_{kb.epoch:04d}", self.evaluate(f"kb_{kb.epoch:04d}", kb), flush=True)
        if a.science and not (self.run / "epochs/e0000/soa.json").exists():
            write_json(self.run / "epochs/e0000/soa.json", soa_compute(self.run, kb.to_doc(), f"kb_{kb.epoch:04d}"))
        while kb.epoch < a.epochs:
            epoch = kb.epoch + 1
            kb.epoch = epoch
            self.cur_epoch = epoch
            if self.jobs:
                self.apply_publications(kb, epoch)
                jobs = self.jobs.poll(epoch)
                write_json(self.run / "epochs" / f"e{epoch:04d}" / "members.json",
                           [{k: j.get(k) for k in ("id", "kind", "origin_epoch", "current_epoch",
                                                  "member_epoch", "label", "status", "report_id")} for j in jobs])
            t0 = time.time()
            ids = self.batch(epoch)
            if not ids:
                print(f"[stream] epoch {epoch}: no remaining questions; preserve live jobs", flush=True)
                kb.epoch -= 1
                break
            past = self.history_records(epoch - 1)
            snapshot = self.solver_snapshot(kb, epoch)
            vis = self.visible(f"e{epoch:04d}", past, snapshot)
            hist = History(self.visible_rows(vis))
            rec_dir = self.run / "epochs" / f"e{epoch:04d}" / "solves"
            with cf.ThreadPoolExecutor(a.concurrency) as ex:
                recs = self.retry_map(ex, lambda i: self.safe(self.solve, i, snapshot, hist, vis, epoch, rec_dir, True), ids)
            if len(recs) < len(ids):
                raise SystemExit(f"epoch {epoch}: {len(ids) - len(recs)} solves failed; rerun to resume")
            if a.train_control:
                # same training questions solved once WITHOUT the KB (solve only): per-question KB counterfactual
                ctl_dir = self.run / "epochs" / f"e{epoch:04d}" / "nokb_solves"
                with cf.ThreadPoolExecutor(a.concurrency) as ex:
                    ctl = self.retry_map(ex, lambda i: self.safe(self.solve, i, None, None, None, epoch, ctl_dir, False), ids)
                cmap = {c["question_id"]: c for c in ctl}
                for r in recs:
                    c = cmap.get(r["question_id"])
                    r["nokb_score"] = c["score"] if c else None
                    r["nokb_prediction"] = c["prediction"] if c else None
            for r in recs:
                unchanged = {cid for cid in r["cited"] if cid in snapshot.claims and cid in kb.claims and
                             all(snapshot.claims[cid].get(k) == kb.claims[cid].get(k)
                                 for k in ("text", *SCIENCE_FIELDS, "report_ids"))}
                kb.credit({cid: (r["score"], 1.0 - r["score"]) for cid in unchanged},
                          self.author_solver, r["question_id"])
            ep_dir = self.run / "epochs" / f"e{epoch:04d}"
            write_jsonl(ep_dir / "records.jsonl", [{k: v for k, v in r.items() if k not in ("discover_trace",
                        "solver_reasoning")} for r in recs])
            vis_s = self.visible(f"e{epoch:04d}_summarizer", past + recs, kb)
            before = len(kb.claims)
            if self.jobs:
                write_json(self.run / "kb_live.json", {**kb.to_doc(), "pending_commits": kb.pending})
                job = self.submit_job("summarizer", epoch, kb, recs, vis_s)
                until = time.monotonic() + a.agent_timeout
                while time.monotonic() < until:
                    self.apply_publications(kb, epoch + 1)
                    jobs = self.jobs.poll(epoch)
                    current = next(j for j in jobs if j["id"] == job["id"])
                    if current["status"] in ("completed", "failed"):
                        break
                    time.sleep(min(5, max(0.01, until - time.monotonic())))
                self.jobs.poll(epoch + 1)
                print(f"[summarizer] {job['id']} origin=e{epoch:04d}; next epoch continues live members", flush=True)
                self.apply_publications(kb, epoch + 1)
            else:
                science, trace = self.summarize(epoch, kb, recs, vis_s)
                write_json(ep_dir / "summarizer_trace.json", trace)
                (ep_dir / "science.md").write_text(f"# Science digest, epoch {epoch}\n\n{science}\n", encoding="utf-8")
            ops = collections.Counter(c["op"] for c in kb.pending)
            kb.flush()
            write_json(self.run / "kb" / f"kb_{epoch:04d}.json", kb.to_doc())
            if self.jobs:
                write_json(self.run / "kb_live.json", kb.to_doc())
            by = collections.defaultdict(list)
            for r in recs:
                by[r["source"]].append(r["score"])
            m = {"epoch": epoch, "n": len(recs), "mean_score": round(sum(r["score"] for r in recs) / len(recs), 4),
                 "by_source": {s: round(sum(v) / len(v), 4) for s, v in sorted(by.items())},
                 "hypotheses": sum(1 for r in recs if r.get("hypothesis")), "skipped": sum(1 for r in recs if r.get("skipped")),
                 "citations": sum(len(r["cited"]) for r in recs), "kb_before": before, "kb_after": len(kb.claims),
                 "ops": dict(ops), "secs": round(time.time() - t0), "usage": dict(self.client.usage),
                 "usage_wrong": dict(self.client_wrong.usage) if self.client_wrong is not self.client else None,
                 "cc_cost_usd": round(self.cc_cost, 4), "solver_selection": self.solver_selection}
            if a.macro_every and not a.no_eval:
                if epoch % a.macro_every == 0 or epoch == a.epochs:
                    m["review"] = self.macro_review(epoch, kb)
            elif not a.no_eval and (epoch % a.eval_every == 0 or epoch == a.epochs):
                if a.gate:
                    m["gate"] = self.gate(epoch, kb)
                else:
                    m["eval"] = self.evaluate(f"kb_{epoch:04d}", kb)
            if a.science:
                try:  # SoA index: a readout of the epoch's (post-gate) KB, never a gate criterion
                    soa = soa_compute(self.run, kb.to_doc(), f"kb_{epoch:04d}")
                    write_json(ep_dir / "soa.json", soa)
                    m["soa"] = {k: v for k, v in soa.items() if k not in ("claim_table", "uncovered_cells")}
                except Exception as exc:
                    print(f"[warn] soa index failed: {exc}", flush=True)
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            print(json.dumps(m, ensure_ascii=False), flush=True)
        if not a.no_eval and a.test_repeats:
            self.final_test()
        if self.jobs:
            self.jobs.finish_owner()


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="opus55_b15")
    p.add_argument("--pool", default=str(POOL), help="immutable sampled release questions.jsonl")
    p.add_argument("--async-agents", action="store_true", help="durable researcher and summarizer processes")
    p.add_argument("--agent-timeout", type=float, default=2400, help="soft epoch window; live jobs continue afterwards")
    p.add_argument("--job-workers", type=int, default=4)
    p.add_argument("--publish-blog", default=str(CONFIG.blog_root) if CONFIG.blog_root else None, help="existing blog clone; requires verified live provenance")
    p.add_argument("--publish-interval", type=float, default=1800)
    p.add_argument("--worker-job", default=None, help=argparse.SUPPRESS)
    p.add_argument("--provider", default="vapi")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--sources", nargs="+", default=["arch170", "ranking_v2", "ranking_v3", "dataflip500"])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--eval-per-source", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=3)
    p.add_argument("--test-per-source", type=int, default=0, help="extra held-out groups per source, final test only")
    p.add_argument("--test-repeats", type=int, default=0, help="opt in to final test evaluation")
    p.add_argument("--test-extra-kb", nargs="*", default=[], help="name=path KBs to add to the final test")
    p.add_argument("--gate", action="store_true", help="legacy opt-in: per-eval whole-KB acceptance and rollback; disables default macro review")
    p.add_argument("--gate-repeats", type=int, default=2)
    p.add_argument("--gate-tol", type=float, default=0.0)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--render", choices=["cred", "bm25", "tool", "brief"], default="cred",
                   help="KB rendering: cred = global credibility top-40 (default); bm25 = per-question BM25 retrieval + credibility backbone")
    p.add_argument("--solve-effort", default="low")
    p.add_argument("--discover-effort-right", default="low")
    p.add_argument("--discover-effort-wrong", default="medium")
    p.add_argument("--summ-effort", default="xhigh")
    p.add_argument("--solve-max-tokens", type=int, default=16000)
    p.add_argument("--discover-max-tokens", type=int, default=12000)
    p.add_argument("--summ-max-tokens", type=int, default=32000)
    p.add_argument("--discover-turns", type=int, default=8)
    p.add_argument("--summ-turns", type=int, default=30)
    p.add_argument("--max-add", type=int, default=8)
    p.add_argument("--seed-kb", default=None, help=f"seed KB json (default: {SEED_KB})")
    p.add_argument("--kb-block", choices=["default", "nocite", "cite_after"], default="default")
    p.add_argument("--corpus", default=None, help="questions.jsonl backtest corpus exposed as load_corpus()")
    p.add_argument("--discover-wrong-provider", default=None)
    p.add_argument("--discover-wrong-model", default=None, help="model for discovery on wrong answers")
    p.add_argument("--discover-wrong-backend", choices=["api", "claude_code"], default="api")
    p.add_argument("--solver-backend", choices=["api", "codex"], default="api",
                   help="codex: solve + discover in one continuing Codex session with --model")
    p.add_argument("--summ-backend", choices=["api", "claude_code", "codex"], default="api")
    p.add_argument("--summ-model", default=None)
    p.add_argument("--summ-budget-usd", type=float, default=15.0)
    p.add_argument("--science", action="store_true", help="mechanism/regime/phase claims, kb_explain, research tool")
    p.add_argument("--lab", default=None, help="lab_db.jsonl (kb_lab.py build) exposed as load_lab() + experiments")
    p.add_argument("--lab-workers", type=int, default=12)
    p.add_argument("--research-calls", type=int, default=3, help="research tasks per epoch (summarizer)")
    p.add_argument("--research-turns", type=int, default=14)
    p.add_argument("--research-max-exps", type=int, default=3)
    p.add_argument("--research-max-cands", type=int, default=12)
    p.add_argument("--research-effort", default="medium")
    p.add_argument("--discover-exps", type=int, default=0, help="controlled experiments per wrong-answer discovery")
    p.add_argument("--solve-lab", action="store_true", help="diagnostic: solver gets python + load_lab() while solving")
    p.add_argument("--macro-every", type=int, default=None, help="checkpoint review + CONSOLIDATE phase every N epochs "
                   "(default 5); regression selects stable solver input, preserving research KB")
    p.add_argument("--macro-repeats", type=int, default=3)
    p.add_argument("--macro-test-readout", action="store_true",
                   help="opt in to test readouts during growth; default reserves test for final evaluation")
    p.add_argument("--consolidate-research", type=int, default=6)
    p.add_argument("--import-history", nargs="*", default=[], help="earlier run names whose training records join history")
    p.add_argument("--stream-offset", type=int, default=0, help="epoch t reads the training stream as epoch t+offset")
    p.add_argument("--gate-z", type=float, default=None, help="macro regression threshold in SE (default 2); "
                   "legacy --gate acceptance threshold (default 0)")
    p.add_argument("--test-on-accept", type=int, default=0, help="repeats of the untouched test readout per accepted KB")
    p.add_argument("--kb-char-budget", type=int, default=0, help="max total claim text chars; add_claim rejected beyond")
    p.add_argument("--train-control", action="store_true", help="also solve each training question without KB once; "
                   "summarizer sees per-question and per-claim KB counterfactuals")
    p.add_argument("--phase-preview", type=int, default=0, help="chars of a claim's phase shown in kb_search results")
    args = p.parse_args(argv)
    if args.gate and args.macro_every not in (None, 0):
        p.error("--gate is a legacy alternative to --macro-every")
    if args.macro_every is None:
        args.macro_every = 0 if args.gate else 5
    if args.gate_z is None:
        args.gate_z = 0.0 if args.gate else 2.0
    if args.macro_every < 0 or args.gate_z < 0:
        p.error("--macro-every and --gate-z must be nonnegative")
    return args


def cli(argv=None) -> None:
    args = parse(argv)
    os.environ.update(AIQ_KB_DATA_ROOT=str(CONFIG.data_root),
                      AIQ_BENCH_ROOT=str(CONFIG.bench_root),
                      AIQ_KB_KEYS=str(CONFIG.keys_file), AIQ_KB_POOL=str(CONFIG.pool_file))
    if args.worker_job:
        path = Path(args.worker_job).resolve()
        job = read_job(path)
        Loop(argparse.Namespace(**job["payload"]["args"])).worker(path)
    else:
        Loop(args).main()


if __name__ == "__main__":
    cli()
