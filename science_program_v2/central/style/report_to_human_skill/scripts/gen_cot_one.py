#!/usr/bin/env python3
"""Generate full CoT completions for one model on a fixed held-out question slice.

Runs ON THE CLUSTER (360-2). Each invocation loads one model (base or
checkpoint shim), samples temperature matching training rollouts, and
appends full-text records to a JSONL.

Usage:
  export AIQ_BASE_MODEL=/path/to/base_model          # tokenizer + shim source
  export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
  python gen_cot_one.py --model <path-or-"shim:CKPT_DIR"> --name ckpt180 \
      --out cot_samples.jsonl --data /dataREMOTE_HOME/aiq_rl/data/eval100.jsonl

Checkpoint paths starting with "shim:" are resolved via eval_mcq.build_shim
(TRL checkpoints lack preprocessor_config.json; vLLM refuses them directly).
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ALLREDUCE_USE_FLASHINFER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# Fixed deterministic slice of the eval set (indices), diverse families.
IDX = [0, 3, 7, 11, 19, 27, 35, 43, 55, 67, 79, 91]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="model dir, or 'shim:CKPT_DIR' to resolve via build_shim")
    ap.add_argument("--name", required=True, help="tag written to records (e.g. base_step0)")
    ap.add_argument("--out", required=True, help="JSONL output (append mode)")
    ap.add_argument("--data", default="/dataREMOTE_HOME/aiq_rl/data/eval100.jsonl")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.45)
    a = ap.parse_args()

    model = a.model
    if model.startswith("shim:"):
        sys_path = os.path.dirname(os.path.abspath(__file__))
        import sys
        sys.path.insert(0, "/dataREMOTE_HOME/aiq_rl")  # for eval_mcq
        from eval_mcq import build_shim
        model = build_shim(model[len("shim:"):])

    base = os.environ["AIQ_BASE_MODEL"]
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    sel = [rows[i] for i in IDX if i < len(rows)]

    tok = AutoTokenizer.from_pretrained(base)
    prompts = [
        tok.apply_chat_template(
            r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for r in sel
    ]
    llm = LLM(model=model, gpu_memory_utilization=a.gpu_frac,
              max_model_len=24576, disable_log_stats=True)
    outs = llm.generate(
        prompts,
        SamplingParams(temperature=a.temperature, top_p=a.top_p,
                        max_tokens=a.max_tokens, seed=a.seed),
    )
    with open(a.out, "a") as fh:
        for r, o in zip(sel, outs):
            fh.write(json.dumps({
                "model": a.name, "model_path": model,
                "qid": r["question_id"], "family": r["family"],
                "type": r.get("type"), "source": r.get("source"),
                "gold": r["answer"], "tokens": len(o.outputs[0].token_ids),
                "finish_reason": o.outputs[0].finish_reason,
                "text": o.outputs[0].text,
            }, ensure_ascii=False) + "\n")
    print("DONE", a.name, flush=True)


if __name__ == "__main__":
    main()
