#!/usr/bin/env python3
"""Build the KB science loop website: site/index.html (dashboard, Chinese UI) + site/runs/<run>.html (per-run viewers).

Usage: aiq-kb-site   ->  <AIQ_KB_DATA_ROOT>/site/index.html
"""
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if __package__:
    from . import kb_science_loop as L
    from . import kb_loop_viewer
else:
    import kb_science_loop as L
    import kb_loop_viewer

SITE = L.CONFIG.data_root / "site"

RUNS = {  # name -> (显示名, 说明, 是否生成详情页)
    "luna_main": ("gpt-6-luna · 主实验", "cctq，12 轮 × 每轮 15 题；固定 40 题评测集，每个 KB 采样 3 次。约 84 美元、10 小时。", True),
    "qwen_6ep_40eval": ("Qwen3.8-Max · 对照", "aliyun，与主实验同一评测集；跑到第 5 轮时 aliyun 欠费中断，每个 KB 只采样 1 次。", True),
    "luna_gate": ("gpt-6-luna · 门控版", "快照择优 + 回滚 + 删除门槛 + 独立测试集；第 1 轮中途停止，暂无结果（表中种子 KB 和不带 KB 的评测是从主实验复制的）。", True),
    "smoke_qwen": ("Qwen3.8-Max · 烟测", "首次端到端跑通 2 轮；评测集只有 8 题，不可用于结论。", True),
    "luna_b15": ("gpt-6-luna · 试跑", "cctq 首次试跑 1 轮；评测集 20 题（每题源 5 题）。", True),
    "smoke": ("Claude Opus 5.5 · 烟测", "vapi 余额耗尽前只完成了 8 题评测（不带 KB 与种子 KB），循环没有跑起来。", False),
    "opus_mini": ("Claude Opus 5.5 · 迷你", "vapi 余额耗尽，8 次评测调用全部失败，没有可用结果。", False),
}
EARLY = [  # 早期循环（kb_evolve_pilot.py 时代，Qwen3.8-Max，规则式 KB 更新）
    ("混合题（150 题）", 0.660, 0.720), ("仅架构题（240 题）", 0.454, 0.596),
]
RENDER_ABL = [("可信度 top-40 + 新条目 10", 15.8), ("PUCT top-40", 3.3), ("PUCT top-20（旧默认）", -3.3)]


def per_q(run: Path, base: str) -> tuple[dict, int]:
    d, errors = collections.defaultdict(list), 0
    for ev in sorted((run / "eval").glob("*")):
        if not (ev.name == base or re.fullmatch(re.escape(base) + r"_r\d+", ev.name)):
            continue
        sm = ev / "_summary.json"
        if sm.exists():
            errors += json.loads(sm.read_text()).get("errors", 0)
        for f in ev.glob("q_*.json"):
            r = json.loads(f.read_text())
            if r.get("error"):
                continue
            d[r["question_id"]].append((r["score"], r["source"]))
    return d, errors


def stats(run: Path) -> dict:
    bases = sorted({re.sub(r"_r\d+$", "", p.parent.name) for p in (run / "eval").glob("*/_summary.json")})
    bases = [b for b in bases if b == "nokb" or b.startswith("kb_")]
    pq = {b: per_q(run, b) for b in bases}
    nokb = {q: statistics.mean(s for s, _ in v) for q, v in pq.get("nokb", ({}, 0))[0].items()}
    out = []
    for b in bases:
        d, err = pq[b]
        if not d:
            continue
        m = {q: statistics.mean(s for s, _ in v) for q, v in d.items()}
        vals = list(m.values())
        se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0
        src = collections.defaultdict(list)
        for q, v in d.items():
            src[v[0][1]].append(m[q])
        row = {"label": b, "epoch": -1 if b == "nokb" else int(b[3:]), "n_q": len(vals),
               "samples": sum(len(v) for v in d.values()), "errors": err, "mean": statistics.mean(vals), "se": se,
               "by_source": {k: statistics.mean(v) for k, v in sorted(src.items())}}
        if b != "nokb" and nokb:
            qs = [q for q in m if q in nokb]
            dd = [m[q] - nokb[q] for q in qs]
            row["delta"] = statistics.mean(dd)
            row["delta_se"] = statistics.stdev(dd) / math.sqrt(len(dd)) if len(dd) > 1 else 0
        out.append(row)
    return out


def epoch_rows(run: Path) -> list[dict]:
    rows = []
    for d in sorted((run / "epochs").glob("e*")):
        recs = [json.loads(f.read_text()) for f in (d / "solves").glob("*.json")]
        if not recs or not (d / "summarizer_trace.json").exists():
            continue
        e = int(d.name[1:])
        ops = collections.Counter(c["op"] for c in map(json.loads, open(run / "commits.jsonl"))
                                  if c["epoch"] == e and c["op"] not in ("credit", "seed"))
        kb = run / "kb" / f"kb_{e:04d}.json"
        rows.append({"epoch": e, "mean": statistics.mean(r["score"] for r in recs), "n": len(recs),
                     "hyp": sum(bool(r.get("hypothesis")) for r in recs), "ops": dict(ops),
                     "kb_size": len(json.loads(kb.read_text())["claims"]) if kb.exists() else None})
    return rows


def main():
    (SITE / "runs").mkdir(parents=True, exist_ok=True)
    runs = []
    for name, (title, note, viewer) in RUNS.items():
        run = L.OUT_ROOT / name
        if not (run / "config.json").exists():
            continue
        cfg = json.loads((run / "config.json").read_text())
        same_val = (run / "split.json").exists() and json.loads((run / "split.json").read_text())["eval"] == \
            json.loads((L.OUT_ROOT / "luna_main" / "split.json").read_text())["eval"]
        page = None
        if viewer and (run / "epochs").exists():
            kb_loop_viewer.build(name, SITE / "runs" / f"{name}.html", home="../index.html")
            page = f"runs/{name}.html"
        runs.append({"name": name, "title": title, "note": note, "model": cfg["model"], "provider": cfg["provider"],
                     "batch": cfg["batch_size"], "eval_n": cfg["eval_per_source"] * len(cfg["sources"]),
                     "same_val": same_val, "page": page, "evals": stats(run), "epochs": epoch_rows(run)})
    data = {"runs": runs, "early": EARLY, "render_abl": RENDER_ABL, "built": L.now()}
    html = (Path(__file__).resolve().parent / "kb_site.html").read_text()
    (SITE / "index.html").write_text(html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)))
    print(SITE / "index.html")


if __name__ == "__main__":
    main()
