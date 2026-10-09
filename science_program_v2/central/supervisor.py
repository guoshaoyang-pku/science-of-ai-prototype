#!/usr/bin/env python3
"""Central supervisor for science_program_v2.

Serially schedules research rounds. Each round = one `codex exec` session
(Sol 6.1 Ultra) that works on one direction, following AGENTS.md.
All memory lives in files + git; this process only schedules, times,
logs, and backstop-commits.

Stop gracefully:  touch central/STOP        (finishes current round)
Stop now:         kill <supervisor_pid>     (SIGTERM; kills round child)
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parent.parent
CENTRAL = PROGRAM_DIR / "central"
ROUNDS = CENTRAL / "rounds"
LOCK = CENTRAL / "supervisor.lock"
STOP = CENTRAL / "STOP"
LOG = CENTRAL / "log.jsonl"

# rounds shorter than this are startup/infra failures (e.g. DNS outage
# 2026-10-07 07:12-08:00 burned r52-r72 in ~40s each); they must not
# consume the per-direction science budget.
INFRA_FAIL_MAX_SEC = 120.0
RELAY_HEALTH_URL = os.environ.get("SCIENCE_RELAY_HEALTH_URL", "")
UPSTREAM_HOSTS = tuple(h for h in os.environ.get("SCIENCE_UPSTREAM_HOSTS", "").split(",") if h)


def network_healthy(timeout=10):
    """Relay up AND at least one model upstream resolvable."""
    if not RELAY_HEALTH_URL:
        return True  # the configured CLI owns connectivity unless a preflight URL is supplied
    try:
        with urllib.request.urlopen(RELAY_HEALTH_URL, timeout=timeout) as r:
            if r.status != 200:
                return False
    except Exception:
        return False
    if not UPSTREAM_HOSTS:
        return True
    for h in UPSTREAM_HOSTS:
        try:
            socket.getaddrinfo(h, 443)
            return True
        except OSError:
            continue
    return False

_stop = False


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log_event(kind, **kw):
    rec = {"ts": now_iso(), "event": kind, **kw}
    line = json.dumps(rec, ensure_ascii=False)
    with LOG.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(p, obj):
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def acquire_lock():
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            os.kill(old, 0)
            print(f"supervisor already running (pid {old}); exit", file=sys.stderr)
            sys.exit(1)
        except (ValueError, ProcessLookupError):
            pass  # stale lock
        except PermissionError:
            print("another live process owns the lock; exit", file=sys.stderr)
            sys.exit(1)
    LOCK.write_text(str(os.getpid()))


def release_lock():
    try:
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()
    except OSError:
        pass


def git(*args):
    r = subprocess.run(["git", "-C", str(PROGRAM_DIR), *args],
                       capture_output=True, text=True)
    return r


def backstop_commit(round_id):
    git("add", "-A")
    dirty = git("status", "--porcelain").stdout.strip()
    if dirty:
        git("commit", "-m", f"supervisor backstop {round_id}")


def ensure_directions(cfg, state):
    """Auto-initialize state entries for directions listed in config order but
    missing from state.json. Safe: called at loop top while no round runs."""
    changed = False
    for d in cfg.get("direction_order") or []:
        if d not in state["directions"]:
            state["directions"][d] = {"rounds_done": 0, "status": "active",
                                      "last_round": None, "next_question": None}
            changed = True
    return changed


def budget_for(cfg, d):
    return int(cfg.get("round_overrides", {}).get(d,
                 cfg.get("rounds_per_direction", 3)))


def pick_direction(cfg, state):
    order = cfg.get("direction_order") or list(state["directions"])
    # round-robin: prefer the direction with fewest rounds_done among active ones
    active = [d for d in order
              if state["directions"].get(d, {}).get("status") == "active"
              and state["directions"][d]["rounds_done"] < budget_for(cfg, d)]
    if not active:
        return None
    return min(active, key=lambda d: (state["directions"][d]["rounds_done"], order.index(d)))


def build_prompt(round_id, direction, state):
    d = state["directions"][direction]
    python_bin = load_json(CENTRAL / "config.json").get("python_bin", sys.executable)
    return f"""你是中心化研究程序 science_program_v2 的第 {round_id} 轮（方向 {direction}，该方向第 {d['rounds_done']+1} 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/{direction}/task.md、该方向已有 findings/ 与 report.md、directions/{direction}/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 {now_iso()}]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：{d.get('next_question') or '见 task.md'}。
3. 预注册 JSON 先 git commit，再跑实验（{python_bin}，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/{direction}/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。"""


def run_round(cfg, round_id, direction, state):
    rdir = ROUNDS / f"r{round_id:03d}_{direction}"
    rdir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(round_id, direction, state)
    (rdir / "prompt.md").write_text(prompt)
    last_msg = rdir / "last_message.md"
    cmd = [
        cfg.get("codex_bin", "codex"),
        "exec",
        "-C", str(PROGRAM_DIR),
        "--sandbox", "workspace-write",
        "-c", "approval_policy=never",
        # workspace-write denies .git writes by default; explicitly allow this
        # repo's .git so rounds can make their own prereg/closeout commits.
        "-c", f'sandbox_workspace_write.writable_roots=["{PROGRAM_DIR / ".git"}"]',
        "-m", cfg.get("model", "gpt-6.1-sol"),
        "-c", f"model_reasoning_effort={cfg.get('reasoning_effort', 'ultra')}",
        "-o", str(last_msg),
        prompt,
    ]
    log_event("round_start", round=round_id, direction=direction,
              timeout_sec=cfg.get("round_timeout_sec", 7200))
    t0 = time.time()
    timed_out = False
    with (rdir / "output.log").open("wb") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                cwd=str(PROGRAM_DIR), start_new_session=True)
        try:
            proc.wait(timeout=float(cfg.get("round_timeout_sec", 7200)))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    elapsed = round(time.time() - t0, 1)
    result = "timeout" if timed_out else "unknown"
    summary = ""
    if last_msg.exists():
        txt = last_msg.read_text(errors="replace")
        for line in txt.splitlines():
            if line.strip().startswith("ROUND_RESULT:"):
                parts = line.split("|", 1)
                result = parts[0].split(":", 1)[1].strip()
                summary = parts[1].strip() if len(parts) > 1 else ""
    if timed_out:
        result = "timeout"
    backstop_commit(f"r{round_id:03d}")
    log_event("round_end", round=round_id, direction=direction,
              result=result, summary=summary[:300],
              elapsed_sec=elapsed, exit_ok=not timed_out)
    return {"round": round_id, "direction": direction, "result": result,
            "summary": summary[:300], "elapsed_sec": elapsed}


def main():
    global _stop
    if os.environ.get("SCIENCE_ENABLE_AGENT") != "1":
        raise RuntimeError('Parked handoff. Prepare a new external run and set SCIENCE_ENABLE_AGENT=1 only for an explicitly configured research round.')
    repo_root = Path(git("rev-parse", "--show-toplevel").stdout.strip())
    if repo_root != PROGRAM_DIR:
        raise RuntimeError('Run supervisor only from a dedicated research Git repo prepared outside the source checkout')

    def on_term(signum, frame):
        global _stop
        _stop = True
        log_event("signal", signum=signum, note="will stop after current round")

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    acquire_lock()
    try:
        state = load_json(CENTRAL / "state.json")
        state["supervisor_pid"] = os.getpid()
        if not state.get("started_at"):
            state["started_at"] = now_iso()
        state["status"] = "running"
        save_json(CENTRAL / "state.json", state)
        log_event("supervisor_start", pid=os.getpid())

        while not _stop and not STOP.exists():
            cfg = load_json(CENTRAL / "config.json")  # re-read: live budget edits
            state = load_json(CENTRAL / "state.json")
            state["supervisor_pid"] = os.getpid()
            if ensure_directions(cfg, state):
                save_json(CENTRAL / "state.json", state)
                log_event("directions_autoinit",
                          order=list(cfg.get("direction_order", [])))
            total_max = int(cfg.get("max_total_rounds", 24))
            if state["round"] >= total_max:
                state["status"] = "budget_complete"
                save_json(CENTRAL / "state.json", state)
                log_event("budget_complete", rounds=state["round"])
                break
            direction = pick_direction(cfg, state)
            if direction is None:
                # park directions that hit per-direction budget, then stop
                for d, s in state["directions"].items():
                    if s["status"] == "active" and s["rounds_done"] >= budget_for(cfg, d):
                        s["status"] = "parked"
                state["status"] = "rounds_complete"
                save_json(CENTRAL / "state.json", state)
                log_event("rounds_complete",
                          note="all directions hit per-direction budget; parked for user review")
                break
            if not network_healthy():
                log_event("network_down",
                          note="round not started; budget not consumed; retry in 300s")
                for _ in range(30):
                    if _stop or STOP.exists():
                        break
                    time.sleep(10)
                continue
            state["round"] += 1
            round_id = state["round"]
            state["status"] = "running"
            save_json(CENTRAL / "state.json", state)

            rec = run_round(cfg, round_id, direction, state)

            state = load_json(CENTRAL / "state.json")  # round may have updated it
            ds = state["directions"].setdefault(direction, {})
            hist = state.setdefault("history", [])
            hist.append(rec)
            # supervisor owns rounds_done: derive from history, never +=1.
            # infra failures (<INFRA_FAIL_MAX_SEC, session never really started)
            # stay in history for audit but do not consume the budget.
            ds["rounds_done"] = sum(
                1 for h in hist
                if h.get("direction") == direction
                and float(h.get("elapsed_sec") or 0) >= INFRA_FAIL_MAX_SEC)
            ds["last_round"] = rec
            state["updated_at"] = now_iso()
            save_json(CENTRAL / "state.json", state)

            time.sleep(float(load_json(CENTRAL / "config.json").get("inter_round_sleep_sec", 30)))

        state = load_json(CENTRAL / "state.json")
        if state.get("status") == "running":
            state["status"] = "stopped"
        state["supervisor_pid"] = None
        save_json(CENTRAL / "state.json", state)
        log_event("supervisor_exit", status=state["status"], rounds=state["round"])
    finally:
        release_lock()


if __name__ == "__main__":
    main()
