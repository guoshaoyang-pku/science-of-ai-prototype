#!/usr/bin/env python3
"""Paired control at one checkpoint: eval with thinking enabled vs disabled.

Usage (on cluster):
  AIQ_BASE_MODEL=... CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_ALLREDUCE_USE_FLASHINFER=0 python ctrl_think_vs_direct.py \
      --ckpt runs/<trial>/checkpoint-180 --out tmp/ctrl.json
"""
import argparse
import json
import os
import re

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ALLREDUCE_USE_FLASHINFER", "0")

from vllm import LLM, SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from eval_mcq import build_shim  # noqa: E402


def parse(t):
    ms = re.findall(r"<answer>\s*([A-D])", t)
    if ms:
        return ms[-1]
    bs = re.findall(r"\*\*([A-D])\*\*", t)
    return bs[-1] if bs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/dataREMOTE_HOME/aiq_rl/data/eval100.jsonl")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gpu-frac", type=float, default=0.45)
    a = ap.parse_args()

    base = os.environ["AIQ_BASE_MODEL"]
    model = build_shim(a.ckpt)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    tok = AutoTokenizer.from_pretrained(base)
    by_id = {r["question_id"]: {"qid": r["question_id"], "gold": r["answer"]} for r in rows}
    llm = LLM(model=model, gpu_memory_utilization=a.gpu_frac,
              max_model_len=24576, disable_log_stats=True)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=0)
    for arm, think in [("think", True), ("direct", False)]:
        prompts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=think)
                   for r in rows]
        outs = llm.generate(prompts, sp)
        for r, o in zip(rows, outs):
            d = by_id[r["question_id"]]
            d["pred_" + arm] = parse(o.outputs[0].text)
            d["tok_" + arm] = len(o.outputs[0].token_ids)
        print("ARM", arm, "done", flush=True)
    recs = list(by_id.values())
    n = len(recs)
    at = sum(x.get("pred_think") == x["gold"] for x in recs)
    ad = sum(x.get("pred_direct") == x["gold"] for x in recs)
    w = sum(1 for x in recs if x.get("pred_think") == x["gold"] and x.get("pred_direct") != x["gold"])
    l = sum(1 for x in recs if x.get("pred_think") != x["gold"] and x.get("pred_direct") == x["gold"])
    out = {"records": recs, "summary": {
        "n": n, "acc_think": at / n, "acc_direct": ad / n,
        "think_wins": w, "think_losses": l,
        "mean_tok_think": sum(x["tok_think"] for x in recs) / n,
        "mean_tok_direct": sum(x["tok_direct"] for x in recs) / n}}
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print("SUMMARY", json.dumps(out["summary"]), flush=True)


if __name__ == "__main__":
    main()
