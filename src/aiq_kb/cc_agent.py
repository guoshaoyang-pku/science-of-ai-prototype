#!/usr/bin/env python3
"""Run one tool-using agent turn-loop through the real Claude Code CLI (`claude -p --bare`).

cctq serves Claude models only to the Claude Code client, so strong-model roles (summarizer, seed distiller) run as a
headless Claude Code session whose ONLY tools are ours: built-in tools are disabled (`--tools ""`), and our tools are
exposed by kb_mcp_bridge.py, which forwards each call to `handler` running in this process. The agent therefore sees
exactly what the handler returns (same visibility control as the API backend).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

if __package__:
    from .config import KBConfig
else:
    from config import KBConfig

CONFIG = KBConfig.from_env()
KEYS = CONFIG.keys_file
BRIDGE = Path(__file__).resolve().parent / "kb_mcp_bridge.py"
PREFIX = "mcp__kb__"


def run_claude_agent(*args, retries: int = 6, **kw) -> tuple[str, list[dict], dict]:
    """Retry sessions that die on an upstream API error before any tool call (cctq has transient 403/5xx)."""
    for k in range(retries + 1):
        final, trace, info = _run_claude_agent(*args, **kw)
        transient = info.get("is_error") and info["calls"] == 0 and not info.get("cost_usd")
        if not transient or k == retries:
            info["attempts"] = k + 1
            return final, trace, info
        wait = min(600, 30 * 2 ** k)
        print(f"[cc] upstream error before any tool call ({final[:120]!r}); retry {k + 1}/{retries} in {wait}s", flush=True)
        time.sleep(wait)


def _run_claude_agent(system: str, user: str, tools: list, handler, *, model: str, effort: str, workdir: Path,
                     max_calls: int | None, stop_tool: str | None = None, warn_calls: int = 4,
                     warn_text: str = "wrap up now.", trace_path: Path | None = None, provider: str = "cctq_claude",
                     budget_usd: float = 30.0, timeout: float | None = 5400, durable: bool = False,
                     on_event: Callable | None = None) -> tuple[str, list[dict], dict]:
    """Returns (final_text, trace, info) where info has cost_usd, num_turns, calls, stopped."""
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(KEYS.read_text())[provider]
    log: list[dict] = []
    state = {"calls": 0, "stopped": False, "stop_at": None}
    lock = threading.Lock()
    session_path = workdir / "_claude_session.json"
    session = json.loads(session_path.read_text()) if durable and session_path.exists() else {}
    events_path = workdir / "_cc_events.jsonl"
    if durable and not session.get("initialized") and events_path.exists():
        for line in events_path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "system" and event.get("subtype") == "init" and event.get("session_id"):
                session.update(session_id=event["session_id"], initialized=True)
    session.setdefault("session_id", str(uuid.uuid4()))

    def checkpoint(**updates):
        if durable:
            session.update(updates, updated_at=time.time())
            temp = session_path.with_suffix(".tmp")
            temp.write_text(json.dumps(session, ensure_ascii=False))
            temp.replace(session_path)

    checkpoint()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name = req["name"].removeprefix(PREFIX)
            with lock:
                state["calls"] += 1
                k = state["calls"]
            left = None if max_calls is None else max_calls - k
            if left is not None and left < 0 and name != stop_tool:
                out = f"[budget] tool budget exhausted; call {stop_tool or 'nothing more'} now."
            else:
                try:
                    out = handler(name, req["arguments"])
                except Exception as exc:  # tool errors go back to the model
                    out = f"[tool error] {type(exc).__name__}: {exc}"
                if left is not None and 0 <= left <= warn_calls and name != stop_tool:
                    out += f"\n[budget] {left} tool call(s) left: {warn_text}"
            with lock:
                entry = {"k": k, "name": name, "arguments": req["arguments"], "result": out[:4000], "t": time.time()}
                log.append(entry)
                if durable:
                    with (workdir / "_cc_tools.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if name == stop_tool and not out.startswith("rejected"):
                    state["stopped"], state["stop_at"] = True, time.time()
            if on_event:
                on_event({"type": "tool.result", **entry})
            print(f"[cc:{workdir.name}] call {k}: {name}", flush=True)
            data = out.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    spec_path, mcp_path = workdir / "_tools.json", workdir / "_mcp.json"
    spec_path.write_text(json.dumps(tools))
    mcp_path.write_text(json.dumps({"mcpServers": {"kb": {"command": sys.executable, "args": [
        str(BRIDGE), str(spec_path), f"http://127.0.0.1:{srv.server_port}/"]}}}))
    cmd = ["claude", "-p", "--bare", "--model", model, "--effort", effort, "--tools", "", "--strict-mcp-config",
           "--mcp-config", str(mcp_path), "--allowedTools", "mcp__kb", "--permission-prompts", "none",
           "--disable-slash-commands", "--max-budget-usd", str(budget_usd),
           "--output-format", "stream-json", "--verbose", "--system-prompt", system]
    if durable:
        cmd += ["--resume" if session.get("initialized") else "--session-id", session["session_id"]]
    else:
        cmd += ["--no-session-persistence"]
    env = {**os.environ, "ANTHROPIC_BASE_URL": cfg["base_url"], "ANTHROPIC_API_KEY": cfg["api_key"],
           "ANTHROPIC_AUTH_TOKEN": "", "MCP_TOOL_TIMEOUT": "1800000", "MCP_TIMEOUT": "60000",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    if durable:
        config_dir = workdir / "claude_home"
        config_dir.mkdir(exist_ok=True)
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    err_file = (workdir / "_cc_stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=workdir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=err_file, text=True)
    events: list[dict] = []
    t0 = time.time()
    checkpoint(pid=proc.pid, status="running", started_at=t0, returncode=None)
    done = threading.Event()

    def watchdog():
        while proc.poll() is None:
            if done.wait(1):
                return
            over = max_calls is not None and state["calls"] > max_calls + 8
            late = state["stop_at"] and time.time() - state["stop_at"] > 180
            timed_out = timeout is not None and time.time() - t0 > timeout
            if over or late or timed_out:
                print(f"[cc:{workdir.name}] terminating (over={over}, late_after_stop={bool(late)})", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        proc.stdin.write(user)
        proc.stdin.close()
        mode = "a" if durable else "w"
        with (workdir / "_cc_events.jsonl").open(mode, encoding="utf-8") as event_file:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(event)
                event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                event_file.flush()
                if event.get("type") == "system" and event.get("subtype") == "init":
                    checkpoint(session_id=event.get("session_id") or session["session_id"], initialized=True)
                if on_event:
                    on_event(event)
                if trace_path and event.get("type") == "assistant":
                    from_events = build_trace(events, log)
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text(json.dumps(from_events, ensure_ascii=False, indent=1))
        proc.wait()
    finally:
        done.set()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        proc.stdout.close()
        if not proc.stdin.closed:
            proc.stdin.close()
        err_file.close()
        checkpoint(status="interrupted" if proc.returncode else "completed", returncode=proc.returncode)
        srv.shutdown()
        srv.server_close()
    err = (workdir / "_cc_stderr.log").read_text()
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    trace = build_trace(events, log)
    if trace_path:
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=1))
    info = {"cost_usd": result.get("total_cost_usd", 0.0), "num_turns": result.get("num_turns"),
            "calls": state["calls"], "stopped": state["stopped"], "secs": round(time.time() - t0),
            "is_error": result.get("is_error"), "usage": result.get("usage"), "returncode": proc.returncode,
            "session_id": session["session_id"] if durable else result.get("session_id"),
            "stderr_tail": err[-1500:] if err else ""}
    if not result:
        info["is_error"] = True
    checkpoint(status="failed" if info.get("is_error") else "completed")
    return result.get("result") or "", trace, info


def build_trace(events: list[dict], log: list[dict]) -> list[dict]:
    """Group stream-json assistant blocks by message id into the loop's trace format; attach handler results."""
    by_msg: dict[str, dict] = {}
    order: list[str] = []
    for e in events:
        if e.get("type") != "assistant":
            continue
        m = e["message"]
        mid = m.get("id") or str(len(order))
        if mid not in by_msg:
            by_msg[mid] = {"turn": len(order), "content": "", "reasoning": "", "tool_calls": [], "results": [],
                           "usage": m.get("usage")}
            order.append(mid)
        ent = by_msg[mid]
        for b in m.get("content", []):
            if b.get("type") == "text":
                ent["content"] += b.get("text", "")
            elif b.get("type") == "thinking":
                ent["reasoning"] += b.get("thinking", "")
            elif b.get("type") == "tool_use":
                ent["tool_calls"].append({"name": b["name"].removeprefix(PREFIX),
                                          "arguments": json.dumps(b.get("input", {}), ensure_ascii=False)})
        ent["usage"] = m.get("usage") or ent["usage"]
    trace = [by_msg[m] for m in order]
    pending = list(log)
    for ent in trace:
        for c in ent["tool_calls"]:
            for i, r in enumerate(pending):
                if r["name"] == c["name"]:
                    ent["results"].append(r["result"])
                    pending.pop(i)
                    break
        ent["content"] = ent["content"] or None
        ent["reasoning"] = ent["reasoning"] or None
    return trace
