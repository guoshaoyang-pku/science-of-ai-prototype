#!/usr/bin/env python3
"""Solve -> discover as ONE continuing Codex session (`codex exec` then `codex exec resume`).

The same model keeps its conversation (and Responses-API reasoning items / prompt cache) across the two phases.
Codex built-in tools are disabled; our tools come from kb_mcp_bridge.py, forwarded to `handler` in this process.
The tool list is fixed for the whole session (so the prompt prefix stays cacheable); during the solve phase every
tool call is refused by the bridge, so the solver still sees only the question and the KB.
Each session gets its own CODEX_HOME (config + session files), and network goes direct (NO_PROXY=*), never via the
system proxy.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
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

CONFIG = """model_provider = "cctq"
model = "{model}"
approval_policy = "never"
sandbox_mode = "read-only"
web_search = "disabled"
model_instructions_file = "{instr}"
include_permissions_instructions = false
include_apps_instructions = false
include_collaboration_mode_instructions = false

[features]
shell_tool = false
unified_exec = false
view_image = false
apps = false
plugins = false
memories = false
multi_agent = false
multi_agent_v2 = false
browser_use = false
image_generation = false
computer_use = false
goals = false
sleep_tool = false
tool_suggest = false
skill_search = false
hooks = false

[model_providers.cctq]
name = "cctq"
base_url = "{base_url}"
wire_api = "responses"
env_key = "CCTQ_API_KEY"

[mcp_servers.kb]
command = "{python}"
args = ["{bridge}", "{spec}", "{url}"]
startup_timeout_sec = 60
tool_timeout_sec = 900
default_tools_approval_mode = "approve"
"""


class CodexSession:
    def __init__(self, workdir: Path, model: str, instructions: str, tools: list, handler,
                 provider: str = "cctq", key: str | None = None, solve_tools: set | None = None,
                 solve_max_calls: int = 4, resume: bool = True, on_event: Callable | None = None):
        self.dir = workdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.home = workdir / "codex_home"
        self.home.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(KEYS.read_text())[provider]
        self.key = key or cfg["api_key"]
        self.handler = handler
        self.solve_tools = solve_tools or set()
        self.solve_max_calls = solve_max_calls
        self.phase = "solve"
        self.calls = 0
        self.max_calls: int | None = 0
        self.warn_text = ""
        self.log: list[dict] = []
        self.thread_id: str | None = None
        self.lock = threading.Lock()
        self.on_event = on_event
        self.state_path = workdir / "_codex_session.json"
        self.state = json.loads(self.state_path.read_text()) if resume and self.state_path.exists() else {}
        self.thread_id = self.state.get("thread_id")
        events_path = workdir / "events.jsonl"
        if resume and not self.thread_id and events_path.exists():
            for line in events_path.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "thread.started":
                    self.thread_id = event["thread_id"]
        sess = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                out = sess._call(req["name"], req.get("arguments") or {})
                data = out.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        instr = workdir / "instructions.md"
        instr.write_text(instructions, encoding="utf-8")
        spec = workdir / "_tools.json"
        spec.write_text(json.dumps(tools))
        (self.home / "config.toml").write_text(CONFIG.format(
            model=model, instr=instr, base_url=cfg["base_url"].rstrip("/"), python=sys.executable, bridge=BRIDGE,
            spec=spec, url=f"http://127.0.0.1:{self.srv.server_port}/"))

    def _call(self, name: str, args: dict) -> str:
        with self.lock:
            phase = self.phase
            allowed = phase == "discover" or name in self.solve_tools
            if allowed:
                self.calls += 1
            cap = self.max_calls if phase == "discover" else self.solve_max_calls
            k, left = self.calls, None if cap is None else cap - self.calls
        if not allowed:
            out = ("[refused] only " + (", ".join(sorted(self.solve_tools)) or "no tools") +
                   " can be used while solving; answer from the question" + (" and kb_search." if self.solve_tools else "."))
        elif left is not None and left < 0:
            out = "[budget] tool budget exhausted; reply with your final answer now, no more tool calls."
        else:
            try:
                out = self.handler(name, args)
            except Exception as exc:  # tool errors go back to the model
                out = f"[tool error] {type(exc).__name__}: {exc}"
            if left is not None and left <= 1:
                out += f"\n[budget] {left} tool call(s) left: {self.warn_text}"
        with self.lock:
            entry = {"phase": phase, "k": k, "name": name, "arguments": args, "result": out[:4000]}
            self.log.append(entry)
            with (self.dir / "_codex_tools.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self.on_event:
            self.on_event({"type": "tool.result", **entry})
        return out

    def _checkpoint(self, **updates) -> None:
        self.state.update(updates, thread_id=self.thread_id, updated_at=time.time())
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, ensure_ascii=False))
        temp.replace(self.state_path)

    def turn(self, prompt: str, effort: str, phase: str, max_calls: int | None = 0, warn_text: str = "",
             timeout: float | None = 1800) -> dict:
        with self.lock:
            self.phase, self.max_calls, self.calls, self.warn_text = phase, max_calls, 0, warn_text
        n0 = len(self.log)
        common = ["--json", "--skip-git-repo-check", "-c", f'model_reasoning_effort="{effort}"']
        cmd = (["codex", "exec", "resume", self.thread_id, *common, "-"] if self.thread_id
               else ["codex", "exec", *common, "-"])
        env = {**os.environ, "CODEX_HOME": str(self.home), "CCTQ_API_KEY": self.key, "NO_PROXY": "*", "no_proxy": "*",
               "HTTPS_PROXY": "", "HTTP_PROXY": "", "ALL_PROXY": "", "https_proxy": "", "http_proxy": "",
               "all_proxy": ""}
        t0 = time.time()
        events = []
        timed_out = threading.Event()
        done = threading.Event()
        stderr_path = self.dir / "_codex_stderr.log"
        with stderr_path.open("w", encoding="utf-8") as err_file:
            p = subprocess.Popen(cmd, cwd=self.dir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=err_file, text=True)
            self._checkpoint(pid=p.pid, phase=phase, status="running", started_at=t0, returncode=None)

            def watchdog():
                if timeout is not None and not done.wait(timeout) and p.poll() is None:
                    timed_out.set()
                    p.terminate()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.kill()

            if timeout is not None:
                threading.Thread(target=watchdog, daemon=True).start()
            try:
                p.stdin.write(prompt)
                p.stdin.close()
                with (self.dir / "events.jsonl").open("a", encoding="utf-8") as event_file:
                    for line in p.stdout:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        events.append(event)
                        event_file.write(json.dumps({"phase": phase, **event}, ensure_ascii=False) + "\n")
                        event_file.flush()
                        if event.get("type") == "thread.started":
                            self.thread_id = event["thread_id"]
                            self._checkpoint()
                        if self.on_event:
                            self.on_event({"phase": phase, **event})
                p.wait()
            finally:
                done.set()
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                p.stdout.close()
                if not p.stdin.closed:
                    p.stdin.close()
                self._checkpoint(status="interrupted" if timed_out.is_set() or p.returncode else "completed",
                                 returncode=p.returncode)
        if timed_out.is_set():
            raise subprocess.TimeoutExpired(cmd, timeout)
        final, reasoning, items, usage, errors = "", [], [], {}, []
        for e in events:
            if e.get("type") == "thread.started":
                self.thread_id = self.thread_id or e["thread_id"]
            elif e.get("type") == "item.completed":
                it = e["item"]
                items.append(it)
                if it["type"] == "agent_message":
                    final = it.get("text") or ""
                elif it["type"] == "reasoning":
                    reasoning.append(it.get("text") or "")
            elif e.get("type") == "turn.completed":
                usage = e.get("usage") or {}
            elif e.get("type") in ("error", "turn.failed"):
                errors.append(str(e)[:300])
        if not usage:
            self._checkpoint(status="failed")
            raise RuntimeError(f"codex turn failed (rc={p.returncode}): {errors[-3:]} {stderr_path.read_text()[-500:]}")
        return {"final": final, "reasoning": "\n".join(r for r in reasoning if r) or None, "items": items,
                "usage": usage, "secs": round(time.time() - t0, 1), "tool_log": self.log[n0:], "errors": errors}

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()
