#!/usr/bin/env python3
"""Build a single-file HTML viewer for a kb_science_loop run: solver / discoverer / summarizer traces + KB per step.

Usage: aiq-kb-viewer RUN_NAME   ->  <AIQ_KB_DATA_ROOT>/runs/RUN_NAME/viewer.html
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if __package__:
    from . import kb_science_loop as L
else:
    import kb_science_loop as L

def build(run_name: str, out: Path | None = None, home: str = "") -> Path:
  run = L.OUT_ROOT / run_name
  pool = {json.loads(l)["question_id"]: json.loads(l) for l in open(L.POOL)}
  texts, tix = [], {}

  def tid(t: str) -> int:
    if t not in tix:
        tix[t] = len(texts)
        texts.append(t)
    return tix[t]


  snaps = []
  for f in sorted((run / "kb").glob("kb_*.json")):
      doc = json.loads(f.read_text())
      kb = L.KB(doc, run)
      shown = {c["id"] for c in kb.rendered()}
      snaps.append({"label": f.stem, "epoch": doc.get("epoch", int(f.stem[3:])),
                    "claims": [[c["id"], round(c["support_count"], 2), round(c["failure_count"], 2),
                                round(c["credibility"], 3), tid(c["text"]), c.get("created_epoch", 0),
                                int(c["id"] in shown), c.get("origin", "")[:40]] for c in doc["claims"]],
                    "render": L.KB_BLOCK.format(claims=kb.render())})

  commits = [json.loads(l) for l in open(run / "commits.jsonl")]
  metrics = {json.loads(l)["epoch"]: json.loads(l) for l in open(run / "metrics.jsonl")} if (run / "metrics.jsonl").exists() else {}


  def trace(tr):
      return [{"turn": t["turn"], "content": t.get("content") or "", "reasoning": t.get("reasoning") or "",
               "calls": t.get("tool_calls") or [], "results": t.get("results") or [], "secs": t.get("secs"),
               "usage": t.get("usage") or {}} for t in tr]


  epochs = []
  for d in sorted((run / "epochs").glob("e*")):
      e = int(d.name[1:])
      recs = []
      for f in sorted((d / "solves").glob("*.json")):
          r = json.loads(f.read_text())
          q = pool[r["question_id"]]
          recs.append({k: r.get(k) for k in ("question_id", "source", "task", "family", "answer", "prediction", "score",
                                              "inversions", "cited", "explanation", "solution", "solver_reasoning",
                                              "solve_usage", "solve_secs", "discover_effort", "skipped", "comment",
                                              "hypothesis", "discover_final")}
                      | {"question": "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in q["messages"]),
                         "discover_trace": trace(r.get("discover_trace") or [])})
      st = d / "summarizer_trace.json"
      epochs.append({"epoch": e, "records": recs, "summarizer": trace(json.loads(st.read_text())) if st.exists() else [],
                     "science": (d / "science.md").read_text() if (d / "science.md").exists() else "",
                     "metrics": metrics.get(e, {})})

  evals = {}
  for p in sorted((run / "eval").glob("*/_summary.json")):
      evals[p.parent.name] = json.loads(p.read_text())

  data = {"run": run.name, "config": json.loads((run / "config.json").read_text()), "texts": texts, "snaps": snaps,
          "commits": commits, "epochs": epochs, "evals": evals,
          "prompts": {"解题者：KB 注入块（追加在题目 system prompt 之后）": L.KB_BLOCK,
                      "解题者：排序题格式提示（仅排序题）": L.RANKING_HINT,
                      "发现者：判分消息（揭示答案和得分）": L.DISCOVER_PROMPT, "发现者：答对时的指引": L.GUIDE_RIGHT,
                      "发现者：答错时的指引": L.GUIDE_WRONG,
                      "总结者：system prompt（当前代码版本；luna_main 运行时没有其中的删除门槛，那是跑完后加的）": L.SUMMARIZER_SYSTEM}}

  data["home"] = home
  html = (Path(__file__).resolve().parent / "kb_loop_viewer.html").read_text()
  out = out or run / "viewer.html"
  out.write_text(html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
  print(out, f"{out.stat().st_size / 1e6:.1f} MB")
  return out


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("usage: aiq-kb-viewer RUN_NAME")
    build(args[0], Path(args[1]) if len(args) > 1 else None)


if __name__ == "__main__":
    main()
