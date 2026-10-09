#!/usr/bin/env python3
"""One-off content-realignment rewrite for D1-D4 (user directive 2026-10-07 21:0x).

The four reports answered "what was measured" instead of the user's core
questions. Each session rewrites directions/Dx/report.md to answer the core
question first (formula where possible), in human language. Zero new
training; post-hoc fits of SAVED curves allowed and must be labelled.
Sessions never commit and never touch central/ or other directions; this
runner commits only the direction's paths, then republishes the blog.

Runs alongside the research line (disjoint file sets); git index contention
handled by retry. Run: nohup python central/rewrite_core.py >> central/rewrite.out 2>&1 &
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parent.parent
CENTRAL = PROGRAM_DIR / "central"
LOG = CENTRAL / "rewrite.log.jsonl"
LOCK = CENTRAL / "rewrite.lock"
SESSION_TIMEOUT = 2400

TARGETS = ["D1_u_effective_time", "D2_overfitting_u_curve",
           "D3_capacity_resolution", "D4_sigmoid_training_curve"]

PROMPT = """你是报告内容对齐重写会话（不是研究轮）。目标：directions/{d}/report.md。

先读（按序）：directions/{d}/task.md 的"用户指定核心问题"节、central/style/REPORT_STYLE.md、AGENTS.md 的骨架与核心问题直答规则、directions/{d}/findings/ 全部文件、directions/{d}/studies/ 各 summary.json 的数字（只读，不重跑）。

任务：重写 report.md，使其直接回答用户核心问题：
1. "结论"节第一段直接回答核心问题：能给公式就给公式（分段/列表），公式下一行逐符号一句话定义；随后逐条展开，每条 = 一句主张 + 一句数字证据。
2. 与核心问题无关的发现全部压缩进 support 节（方法与条件/失败与反例/未决问题/证据），不得占据结论区。
3. 人话：一句一事实；行话首次出现即解释；数字代替形容词；外行只读结论节能拿走 90% 图像。
4. 若回答需要新数字（如上升支指数、停点尾部验证），允许对**已保存曲线**做零训练分析：脚本与 summary 存 directions/{d}/studies/rewrite_20261007/；所得公式必须标注"保存数据事后拟合（development）"。禁止新训练、新 cell、新 seed 实验。
5. 铁律：不改动、不删除任何既有已测数字与结论（可重组、改写表述、补 post-hoc 标注）；结论与成立程度两节不出现文件路径/commit/轮次号。
6. 禁止：写 central/、写其他方向、git commit、改 kb.json/state.json/PROGRESS.md。

自查后写 directions/{d}/findings/report_rewrite_audit.md（核心问题是什么、原报告哪里跑题、重写如何对齐、人话检验结果）。
最终回复只写一行：REWRITE_RESULT: <ok|failed> | <一句话>"""


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(kind, **kw):
    rec = {"ts": now(), "event": kind, **kw}
    line = json.dumps(rec, ensure_ascii=False)
    with LOG.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def commit_direction(d, tries=6):
    for i in range(tries):
        subprocess.run(["git", "-C", str(PROGRAM_DIR), "add",
                        f"directions/{d}"], capture_output=True)
        r = subprocess.run(["git", "-C", str(PROGRAM_DIR), "diff", "--cached",
                            "--quiet"], capture_output=True)
        if r.returncode == 0:
            return "clean"
        c = subprocess.run(["git", "-C", str(PROGRAM_DIR), "commit", "-m",
                            f"rewrite: {d} report realigned to user core question"],
                           capture_output=True, text=True)
        if c.returncode == 0:
            return "committed"
        if "index.lock" in (c.stderr + c.stdout):
            time.sleep(6)
            continue
        return f"error: {c.stderr.strip()[:120]}"
    return "error: lock retries exhausted"


def rewrite_one(cfg, d):
    rdir = CENTRAL / "rewrite_rounds" / d
    rdir.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT.format(d=d)
    (rdir / "prompt.md").write_text(prompt)
    last_msg = rdir / "last_message.md"
    cmd = [cfg.get("codex_bin", "codex"),
           "exec", "-C", str(PROGRAM_DIR),
           "--sandbox", "workspace-write",
           "-c", "approval_policy=never",
           "-m", cfg.get("model", "gpt-6.1-sol"),
           "-c", f"model_reasoning_effort={cfg.get('rewrite_effort', 'high')}",
           "-o", str(last_msg), prompt]
    t0 = time.time()
    with (rdir / "output.log").open("wb") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                cwd=str(PROGRAM_DIR), start_new_session=True)
        try:
            proc.wait(timeout=SESSION_TIMEOUT)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            import os, signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    elapsed = round(time.time() - t0, 1)
    result, summary = ("timeout", "") if timed_out else ("unknown", "")
    if last_msg.exists():
        for line in last_msg.read_text(errors="replace").splitlines():
            if line.strip().startswith("REWRITE_RESULT:"):
                parts = line.split("|", 1)
                result = parts[0].split(":", 1)[1].strip()
                summary = parts[1].strip() if len(parts) > 1 else ""
    return result, summary, elapsed


def main():
    if LOCK.exists():
        try:
            import os
            os.kill(int(LOCK.read_text().strip()), 0)
            print("rewrite already running; exit", file=sys.stderr)
            sys.exit(1)
        except (ValueError, ProcessLookupError):
            pass
    LOCK.write_text(str(__import__("os").getpid()))
    cfg = json.loads((CENTRAL / "config.json").read_text())
    try:
        log("rewrite_start", targets=TARGETS)
        results = {}
        for d in TARGETS:
            log("rewrite_begin", direction=d)
            result, summary, elapsed = rewrite_one(cfg, d)
            cstat = commit_direction(d)
            results[d] = result
            log("rewrite_end", direction=d, result=result,
                summary=summary[:200], elapsed_sec=elapsed, commit=cstat)
        log("rewrite_done", results=results)
        pub = subprocess.run(["python3",
                              str(PROGRAM_DIR / "reports/publish_to_blog.py")],
                             capture_output=True, text=True, cwd=str(PROGRAM_DIR))
        log("blog_republish", rc=pub.returncode, tail=pub.stdout.strip()[-200:])
    finally:
        try:
            if LOCK.read_text().strip() == str(__import__("os").getpid()):
                LOCK.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise RuntimeError('Historical one-off repair runner is disabled in the recipient handoff')
    main()
