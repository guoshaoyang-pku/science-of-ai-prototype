"""No-training Science-of-AI prototype: KB evolution loop.

Per epoch:
  1. Render read-only KB text from top-credibility claims (+ probation claims).
  2. Solve a stratified batch of ArchitectureIQ questions twice per question:
     arm A with KB injected, arm B without KB (measures KB lift). Effort=low.
  3. RLVR reward: exact match against gold answer.
  4. Attribute: claims cited [Kxxxx] in correct answers get support+1,
     cited in wrong answers get failure+1. Credibility = (s+1)/(s+f+2).
  5. Discovery (effort=medium): reflect on failures, propose new quantitative
     claims; they enter the KB on probation next epoch.
  6. Reject stale low-credibility claims; write architectureiq_kb_v4 snapshot.

Resume-safe: per-question records and epoch snapshots are atomic files.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .config import KBConfig
else:
    from config import KBConfig

CONFIG = KBConfig.from_env()
ROOT = CONFIG.bench_root.parent
BENCH_ROOT = CONFIG.bench_root
EVAL_DIR = BENCH_ROOT / "data" / "evals"
RUNNER_PATH = BENCH_ROOT / "tools" / "eval"
QUESTIONS_PATH = BENCH_ROOT / "data" / "v1_review" / "v15_questions.json"
DEFAULT_KB_IN = Path(os.environ.get("AIQ_KB_SEED_KB", str(CONFIG.data_root / "seed_kb.json")))
DEFAULT_OUT = CONFIG.data_root / "kb_evolve_pilot"
QUESTION_TYPES = ("architecture_only", "mixed", "optimizer_only")

ANSWER_RE = re.compile(r"<answer>\s*([A-E])\s*</answer>", re.IGNORECASE)
CITE_RE = re.compile(r"\[(K\d{4})\]")
CREDIBILITY_REJECT = 0.45
MIN_EVALUATIONS_REJECT = 3
# Rendering mirrors the original kb_pipeline: PUCT score = credibility + exploration
# bonus, top-N injection. PUCT_EXPLORATION matches kb_pipeline.py.
PUCT_EXPLORATION = 0.3
RENDER_LIMIT = 20


def puct_score(claim: dict[str, Any], history_count: int) -> float:
    evidence = float(claim.get("support_count", 0.0)) + float(claim.get("failure_count", 0.0))
    exploration = PUCT_EXPLORATION * min(
        1.0,
        math.sqrt(math.log1p(history_count) / (1.0 + evidence)),
    )
    return float(claim.get("credibility", 0.5)) + exploration


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_spec(provider: str, model: str, effort: str, concurrency: int) -> Any:
    sys.path.insert(0, str(RUNNER_PATH))
    from benchmark_eval_runner import ModelSpec

    config = json.loads(CONFIG.keys_file.read_text(encoding="utf-8"))
    block = config[provider]
    if not block.get("api_key"):
        raise RuntimeError(f"No API key configured for provider {provider}")
    return ModelSpec(
        provider=provider,
        name=model,
        base_url=block["base_url"],
        api_key=block["api_key"],
        concurrency=concurrency,
        extra_headers=block.get("extra_headers", {}),
        params={"reasoning_effort": effort},
    )


# ---------------------------------------------------------------- KB state

def credibility(support: float, failure: float) -> float:
    return (support + 1.0) / (support + failure + 2.0)


class KB:
    def __init__(self, claims: list[dict[str, Any]], epoch: int):
        self.claims = {c["id"]: dict(c) for c in claims}
        self.epoch = epoch
        self.history_count = 0
        self.delta = {"added": [], "reinforced": [], "penalized": [], "rejected": []}

    @classmethod
    def load(cls, path: Path) -> "KB":
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("schema_version") != "architectureiq_kb_v4":
            raise ValueError(f"Unexpected KB schema {doc.get('schema_version')!r}")
        return cls(doc["claims"], int(doc.get("epoch", 0)) + 1)

    def next_id(self) -> str:
        numeric = [int(cid[1:]) for cid in self.claims if cid.startswith("K") and cid[1:].isdigit()]
        return f"K{(max(numeric) + 1) if numeric else 1:04d}"

    def render(self, limit: int = RENDER_LIMIT) -> str:
        # PUCT ranking like the original kb_pipeline: credibility plus an
        # exploration bonus that fades with evidence, so fresh claims surface
        # automatically without a separate probation slot.
        selected = sorted(
            self.claims.values(),
            key=lambda c: (-puct_score(c, self.history_count), str(c["id"])),
        )[:limit]
        selected = sorted(selected, key=lambda c: str(c["id"]))
        if not selected:
            return "(empty knowledge base)"
        lines = [
            f"[{c['id']}] (credibility {c['credibility']:.2f}) {c['text']}" for c in selected
        ]
        return "\n".join(lines)

    def attribute(self, cited_ids: list[str], correct: bool) -> None:
        for cid in set(cited_ids):
            claim = self.claims.get(cid)
            if not claim:
                continue
            if correct:
                claim["support_count"] += 1.0
                claim["last_supported_epoch"] = self.epoch
                self.delta["reinforced"].append(cid)
            else:
                claim["failure_count"] += 1.0
                self.delta["penalized"].append(cid)
            claim["last_evaluated_epoch"] = self.epoch
            claim["credit"] = claim["support_count"] - claim["failure_count"]
            claim["credibility"] = credibility(claim["support_count"], claim["failure_count"])

    def add_claims(self, texts: list[str], max_new: int) -> list[str]:
        added = []
        for text in texts[:max_new]:
            text = str(text).strip()
            if len(text) < 40:
                continue
            cid = self.next_id()
            self.claims[cid] = {
                "id": cid,
                "text": text,
                "support_count": 0.0,
                "failure_count": 0.0,
                "credit": 0.0,
                "credibility": credibility(0.0, 0.0),
                "created_epoch": self.epoch,
                "last_supported_epoch": None,
                "last_evaluated_epoch": None,
            }
            self.delta["added"].append(cid)
            added.append(cid)
        return added

    def reject_stale(self) -> None:
        for cid, claim in list(self.claims.items()):
            evaluated = claim["support_count"] + claim["failure_count"]
            if evaluated >= MIN_EVALUATIONS_REJECT and claim["credibility"] < CREDIBILITY_REJECT:
                del self.claims[cid]
                self.delta["rejected"].append(cid)

    def snapshot(self, solver_model: str) -> dict[str, Any]:
        claims = sorted(self.claims.values(), key=lambda c: c["id"])
        return {
            "schema_version": "architectureiq_kb_v4",
            "epoch": self.epoch,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "solver_model": solver_model,
            "claims": claims,
            "delta": self.delta,
        }


# ---------------------------------------------------------------- prompting

def solve_prompt(question_prompt: str, kb_text: str | None) -> str:
    kb_block = (
        "Below is a read-only knowledge base of quantitative claims from prior experiments. "
        "Use it as evidence; when a claim materially informs your reasoning, cite it inline as [K0001].\n"
        "<knowledge_base>\n" + (kb_text or "") + "\n</knowledge_base>\n\n"
        if kb_text is not None
        else "You are given no external knowledge base; rely on your own knowledge.\n\n"
    )
    return (
        "Solve the benchmark question below. Reasoning effort is low: be concise and quantitative. "
        "Give a brief evidence-based explanation of the key mechanism, then follow the question's "
        "required final answer format exactly.\n\n"
        + kb_block
        + "<benchmark_question>\n"
        + question_prompt
        + "\n</benchmark_question>"
    )


def discovery_prompt(failures: list[dict[str, Any]], weak_claims: list[dict[str, Any]], max_new: int) -> str:
    failure_payload = [
        {
            "question_id": f["question_id"],
            "family": f["family"],
            "gold_answer": f["gold_answer"],
            "predicted_answer": f["predicted_answer"],
            "explanation": (f["explanation"] or "")[:1200],
        }
        for f in failures
    ]
    weak_payload = [{"id": c["id"], "credibility": round(c["credibility"], 2), "text": c["text"][:300]} for c in weak_claims]
    return (
        "You are the discovery module of a science-of-AI system. A solver just failed the benchmark questions "
        "listed below (predicted vs gold). Your job: propose precise, quantitative, falsifiable claims about "
        "training/optimization/architecture mechanisms that would have prevented these failures and that will "
        "generalize to unseen questions. Force yourself to give exact quantitative relationships (thresholds, "
        "ratios, scaling laws, ordering of methods) rather than vague advice. Do not restate existing weak "
        "claims; replace or sharpen them. "
        f"Return only JSON: {{\"claims\": [\"...\", ...]}} with at most {max_new} claims, each a single "
        "self-contained sentence of 40-600 characters.\n\n"
        "<failures>\n" + json.dumps(failure_payload, ensure_ascii=False, indent=1) + "\n</failures>\n\n"
        "<weak_existing_claims>\n" + json.dumps(weak_payload, ensure_ascii=False, indent=1) + "\n</weak_existing_claims>"
    )


def parse_claims_json(text: str) -> list[str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return []
    claims = payload.get("claims")
    return [c for c in claims if isinstance(c, str)] if isinstance(claims, list) else []


# ---------------------------------------------------------------- API calls

def api_text(raw: dict[str, Any]) -> tuple[str, str | None, dict[str, Any] | None]:
    choices = raw.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content), choice.get("finish_reason"), raw.get("usage")


async def call_model(spec: Any, semaphore: asyncio.Semaphore, prompt: str, *, temperature: float,
                     max_tokens: int, timeout_s: float) -> dict[str, Any]:
    sys.path.insert(0, str(RUNNER_PATH))
    from benchmark_eval_runner import call_backend

    started = time.monotonic()
    try:
        async with semaphore:
            raw = await asyncio.to_thread(
                call_backend, spec, prompt,
                temperature=temperature, max_tokens=max_tokens,
                timeout_s=timeout_s, max_retries=2,
            )
        text, finish_reason, usage = api_text(raw)
        return {"text": text, "finish_reason": finish_reason, "usage": usage,
                "seconds": round(time.monotonic() - started, 2), "error": None}
    except Exception as exc:
        return {"text": "", "finish_reason": None, "usage": None,
                "seconds": round(time.monotonic() - started, 2),
                "error": f"{type(exc).__name__}: {exc}"}


async def solve_one(row: dict[str, Any], arm: str, spec: Any, semaphore: asyncio.Semaphore,
                    out_dir: Path, epoch: int, *, kb_text: str | None, temperature: float,
                    max_tokens: int, timeout_s: float) -> dict[str, Any]:
    record_path = out_dir / "solves" / f"epoch{epoch:03d}" / f"{row['question_id']}_{arm}.json"
    if record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))
    item = await call_model(spec, semaphore, solve_prompt(row["prompt"], kb_text),
                            temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
    answers = ANSWER_RE.findall(item["text"])
    predicted = answers[-1].upper() if answers else None
    record = {
        "question_id": row["question_id"],
        "question_type": row["type"],
        "family": row["family"],
        "arm": arm,
        "gold_answer": row["correct_letter"].upper(),
        "predicted_answer": predicted,
        "exact": predicted == row["correct_letter"].upper(),
        "cited_claims": sorted(set(CITE_RE.findall(item["text"]))),
        "explanation": item["text"][-2000:],
        "finish_reason": item["finish_reason"],
        "usage": item["usage"],
        "seconds": item["seconds"],
        "error": item["error"],
    }
    if not item["error"]:
        write_json_atomic(record_path, record)
    return record


# ---------------------------------------------------------------- epoch loop

def select_questions(source: list[dict[str, Any]], types: tuple[str, ...], per_type: int, seed: int, epoch: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + epoch * 7919)
    selected = []
    for qtype in types:
        candidates = [q for q in source if q["type"] == qtype]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_type])
    rng.shuffle(selected)
    return selected


async def run_epoch(epoch: int, kb: KB, rows: list[dict[str, Any]], args: argparse.Namespace,
                    solve_spec: Any, discover_spec: Any) -> dict[str, Any]:
    sem = asyncio.Semaphore(args.concurrency)
    kb_text = kb.render(limit=args.render_limit)
    tasks = []
    for row in rows:
        tasks.append(solve_one(row, "kb", solve_spec, sem, args.output, epoch, kb_text=kb_text,
                               temperature=args.temperature, max_tokens=args.max_tokens, timeout_s=args.timeout_s))
        if not args.no_baseline:
            tasks.append(solve_one(row, "nokb", solve_spec, sem, args.output, epoch, kb_text=None,
                                   temperature=args.temperature, max_tokens=args.max_tokens, timeout_s=args.timeout_s))
    records = await asyncio.gather(*tasks)
    kb.history_count += len(rows)

    kb_arm = [r for r in records if r["arm"] == "kb"]
    nokb_arm = [r for r in records if r["arm"] == "nokb"]

    for record in kb_arm:
        if record["cited_claims"]:
            kb.attribute(record["cited_claims"], record["exact"])

    failures = [r for r in kb_arm if not r["exact"] and not r["error"]]
    rng = random.Random(args.seed + epoch * 31)
    rng.shuffle(failures)
    weak_claims = sorted(kb.claims.values(), key=lambda c: c["credibility"])[:8]
    new_claim_ids: list[str] = []
    discovery_record: dict[str, Any] = {"skipped": True}
    if failures and not args.no_discovery:
        item = await call_model(
            discover_spec, sem,
            discovery_prompt(failures[: args.discovery_failures], weak_claims, args.max_new_claims),
            temperature=0.4, max_tokens=args.discovery_max_tokens, timeout_s=args.timeout_s * 2,
        )
        new_texts = parse_claims_json(item["text"])
        new_claim_ids = kb.add_claims(new_texts, args.max_new_claims)
        discovery_record = {
            "skipped": False,
            "failures_shown": min(len(failures), args.discovery_failures),
            "proposed": len(new_texts),
            "accepted": new_claim_ids,
            "seconds": item["seconds"],
            "usage": item["usage"],
            "error": item["error"],
            "raw": item["text"][:4000],
        }
    kb.reject_stale()
    snapshot = kb.snapshot(solver_model=args.model)
    write_json_atomic(args.output / "snapshots" / f"kb_{epoch:04d}.json", snapshot)

    def acc(arm_records: list[dict[str, Any]]) -> float | None:
        valid = [r for r in arm_records if not r["error"]]
        return (sum(r["exact"] for r in valid) / len(valid)) if valid else None

    summary = {
        "epoch": epoch,
        "questions": len(rows),
        "kb_accuracy": acc(kb_arm),
        "nokb_accuracy": acc(nokb_arm),
        "kb_lift": (acc(kb_arm) - acc(nokb_arm)) if (acc(kb_arm) is not None and acc(nokb_arm) is not None) else None,
        "accuracy_by_type": {
            qtype: {
                "kb": acc([r for r in kb_arm if r["question_type"] == qtype]),
                "nokb": acc([r for r in nokb_arm if r["question_type"] == qtype]),
            }
            for qtype in QUESTION_TYPES
        },
        "citations": sum(len(r["cited_claims"]) for r in kb_arm),
        "api_errors": sum(bool(r["error"]) for r in records),
        "delta_sizes": {k: len(v) for k, v in kb.delta.items()},
        "kb_size": len(kb.claims),
        "discovery": discovery_record,
        "token_usage": {
            "prompt": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in records),
            "completion": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in records),
        },
    }
    write_json_atomic(args.output / "epochs" / f"epoch_{epoch:04d}_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-in", type=Path, default=DEFAULT_KB_IN)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="qwen3.8-max-0902")
    parser.add_argument("--provider", default="aliyun")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--questions-per-type", type=int, default=10)
    parser.add_argument("--types", nargs="+", default=list(QUESTION_TYPES),
                        choices=list(QUESTION_TYPES))
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--discovery-failures", type=int, default=8)
    parser.add_argument("--discovery-max-tokens", type=int, default=4096)
    parser.add_argument("--max-new-claims", type=int, default=5)
    parser.add_argument("--render-limit", type=int, default=RENDER_LIMIT)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-discovery", action="store_true")
    args = parser.parse_args()

    source_doc = json.loads(args.questions.read_text(encoding="utf-8"))
    source_rows = source_doc["questions"]
    args.output.mkdir(parents=True, exist_ok=True)

    kb = KB.load(args.kb_in)
    # Resume: if snapshots already exist in output dir, load the latest one instead.
    existing = sorted((args.output / "snapshots").glob("kb_*.json"))
    if existing:
        kb = KB.load(existing[-1])
    start_epoch = kb.epoch

    solve_spec = load_spec(args.provider, args.model, "low", args.concurrency)
    discover_spec = load_spec(args.provider, args.model, "medium", args.concurrency)

    manifest = {
        "prototype": "no-training KB evolution (solve=low, discovery=medium)",
        "kb_seed": str(args.kb_in.resolve()),
        "questions": str(args.questions.resolve()),
        "model": args.model,
        "start_epoch": start_epoch,
        "epochs_planned": args.epochs,
        "questions_per_type": args.questions_per_type,
        "baseline_arm": not args.no_baseline,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(args.output / "run_manifest.json", manifest)

    for offset in range(args.epochs):
        epoch = start_epoch + offset
        kb.epoch = epoch
        kb.delta = {"added": [], "reinforced": [], "penalized": [], "rejected": []}
        rows = select_questions(source_rows, tuple(args.types), args.questions_per_type, args.seed, epoch)
        print(f"EPOCH_START {epoch} questions={len(rows)} kb_size={len(kb.claims)}", flush=True)
        summary = asyncio.run(run_epoch(epoch, kb, rows, args, solve_spec, discover_spec))
        print(
            f"EPOCH_DONE {epoch} kb_acc={summary['kb_accuracy']:.3f} nokb_acc={summary['nokb_accuracy'] if summary['nokb_accuracy'] is None else round(summary['nokb_accuracy'],3)} "
            f"lift={summary['kb_lift'] if summary['kb_lift'] is None else round(summary['kb_lift'],3)} "
            f"citations={summary['citations']} delta={summary['delta_sizes']} kb_size={summary['kb_size']}",
            flush=True,
        )

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "completed"
    write_json_atomic(args.output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
