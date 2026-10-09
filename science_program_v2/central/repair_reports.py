#!/usr/bin/env python3
"""One-off report repair pass (user directive 2026-10-07).

For every direction: run one codex session that self-audits report.md
against central/style/REPORT_STYLE.md and rewrites it to the standard
(结论 → Formulation → 成立程度 → support 后置, L4 prose). Sessions never
change measured numbers and never commit; this runner verifies number
preservation (soft check) and commits per direction.

Waits for the research supervisor to exit first (central/STOP must be set).
Restarts the supervisor when all repairs finish, unless central/STOP_REPAIR
is present (then the line stays stopped for the user).

Run:  nohup python central/repair_reports.py >> central/repair.out 2>&1 &
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parent.parent
CENTRAL = PROGRAM_DIR / "central"
STOP = CENTRAL / "STOP"
STOP_REPAIR = CENTRAL / "STOP_REPAIR"
LOCK = CENTRAL / "repair.lock"
LOG = CENTRAL / "repair.log.jsonl"
SESSION_TIMEOUT = 1800
WAIT_SUPERVISOR_MAX = 3600

DIRECTIONS = [
    "D1_u_effective_time", "D2_overfitting_u_curve", "D3_capacity_resolution",
    "D4_sigmoid_training_curve", "D5_ln_geometry", "D6_kernel_drift",
    "D7_train_gain_test_harm", "D8_activation_pathway", "D9_minibatch_noise",
    "D10_depth_propagation", "D11_width_scaling", "D12_cascade_sigmoid",
]

NUM_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(kind, **kw):
    rec = {"ts": now(), "event": kind, **kw}
    line = json.dumps(rec, ensure_ascii=False)
    with LOG.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def supervisor_alive():
    r = subprocess.run(["pgrep", "-f", "python central/supervisor.py"],
                       capture_output=True, text=True, cwd=str(PROGRAM_DIR))
    return bool(r.stdout.strip())


def numbers_of(path):
    try:
        return set(NUM_RE.findall(Path(path).read_text(errors="replace")))
    except OSError:
        return set()


def git(*args):
    return subprocess.run(["git", "-C", str(PROGRAM_DIR), *args],
                          capture_output=True, text=True)


def commit_with_retry(msg, tries=6):
    for i in range(tries):
        git("add", "-A")
        if not git("status", "--porcelain").stdout.strip():
            return "clean"
        r = git("commit", "-m", msg)
        if r.returncode == 0:
            return "committed"
        if "index.lock" in (r.stderr + r.stdout):
            time.sleep(5)
            continue
        return f"error: {r.stderr.strip()[:120]}"
    return "error: lock retries exhausted"


def repair_one(cfg, direction):
    rdir = CENTRAL / "repair_rounds" / direction
    rdir.mkdir(parents=True, exist_ok=True)
    prompt = f"""你是报告修复会话（不是研究轮）。禁止：跑实验、新测量、预注册、git commit、写 directions/{direction}/ 之外的文件。

任务：把 directions/{direction}/report.md 自查并重写到统一标准。

步骤：
1. 读 central/style/REPORT_STYLE.md（结构+语言规则+自查 checklist）与 AGENTS.md 的 report.md 骨架节。
2. 读 directions/{direction}/report.md、directions/{direction}/findings/ 全部轮次记录、central/kb.json 中 direction 匹配的 claim。
3. 自查：按 checklist 逐项列出不符合点。
4. 重写 report.md：结构 = 结论 → Formulation → 成立程度 → 方法与条件 → 失败与反例 → 未决问题 → 证据；语言 = L4 给人汇报风格（结论先行、短句、行话首次出现即解释、数字代替形容词、无填充）。
5. 铁律：不得改动、新增或删除任何已测数字与科学结论。原报告的每个数字都必须在新文中出现（允许格式归一，如 .9526→0.9526）。拿不准的句子原样保留并放进合适的节。
6. 把自查发现与修复动作写入 directions/{direction}/findings/report_style_audit.md。
7. 若报告已完全符合标准，只做最小编辑。

最终回复只写一行：REPAIR_RESULT: <ok|noop|failed> | <一句话>"""
    (rdir / "prompt.md").write_text(prompt)
    last_msg = rdir / "last_message.md"
    cmd = [
        cfg.get("codex_bin", "codex"),
        "exec", "-C", str(PROGRAM_DIR),
        "--sandbox", "workspace-write",
        "-c", "approval_policy=never",
        "-m", cfg.get("model", "gpt-6.1-sol"),
        "-c", f"model_reasoning_effort={cfg.get('repair_effort', 'high')}",
        "-o", str(last_msg),
        prompt,
    ]
    t0 = time.time()
    with (rdir / "output.log").open("wb") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                cwd=str(PROGRAM_DIR), start_new_session=True)
        try:
            proc.wait(timeout=SESSION_TIMEOUT)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, 15)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
                proc.wait()
    elapsed = round(time.time() - t0, 1)
    result, summary = ("timeout", "") if timed_out else ("unknown", "")
    if last_msg.exists():
        for line in last_msg.read_text(errors="replace").splitlines():
            if line.strip().startswith("REPAIR_RESULT:"):
                parts = line.split("|", 1)
                result = parts[0].split(":", 1)[1].strip()
                summary = parts[1].strip() if len(parts) > 1 else ""
    return result, summary, elapsed


def main():
    if LOCK.exists():
        try:
            os.kill(int(LOCK.read_text().strip()), 0)
            print("repair already running; exit", file=sys.stderr)
            sys.exit(1)
        except (ValueError, ProcessLookupError):
            pass
    LOCK.write_text(str(os.getpid()))
    cfg = json.loads((CENTRAL / "config.json").read_text())
    try:
        log("repair_start", pid=os.getpid(), directions=len(DIRECTIONS))
        # wait for research supervisor to exit (STOP must already be set)
        waited = 0
        while supervisor_alive() and waited < WAIT_SUPERVISOR_MAX:
            time.sleep(15)
            waited += 15
        if supervisor_alive():
            log("abort", reason="supervisor still alive after wait window")
            return
        log("supervisor_exited", waited_sec=waited)

        results = {}
        for d in DIRECTIONS:
            if STOP_REPAIR.exists():
                log("abort", reason="STOP_REPAIR set; line stays stopped")
                results[d] = "skipped"
                continue
            report = PROGRAM_DIR / "directions" / d / "report.md"
            old_nums = numbers_of(report)
            log("repair_begin", direction=d)
            result, summary, elapsed = repair_one(cfg, d)
            new_nums = numbers_of(report)
            missing = sorted(old_nums - new_nums)
            cstat = commit_with_retry(f"style: {d} report rebuilt to L4 standard")
            results[d] = result
            log("repair_end", direction=d, result=result, summary=summary[:200],
                elapsed_sec=elapsed, numbers_old=len(old_nums),
                numbers_missing=len(missing),
                missing_sample=missing[:12], commit=cstat)
            time.sleep(5)

        ok = sum(1 for v in results.values() if v in ("ok", "noop"))
        log("repair_done", ok=ok, total=len(DIRECTIONS), results=results)
        if not STOP_REPAIR.exists():
            STOP.unlink(missing_ok=True)
            time.sleep(2)
            with (CENTRAL / "supervisor.log").open("ab") as slog:
                sup = subprocess.Popen(
                    ["python3",
                     str(CENTRAL / "supervisor.py")],
                    stdout=slog, stderr=subprocess.STDOUT,
                    cwd=str(PROGRAM_DIR), start_new_session=True)
            log("supervisor_restarted", pid=sup.pid)
        else:
            log("left_stopped", reason="STOP_REPAIR present")
    finally:
        try:
            if LOCK.read_text().strip() == str(os.getpid()):
                LOCK.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise RuntimeError('Historical one-off repair runner is disabled in the recipient handoff')
    main()
