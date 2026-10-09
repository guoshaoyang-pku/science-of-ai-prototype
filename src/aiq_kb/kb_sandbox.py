"""Run model-issued Python for KB analysis.

On macOS, sandbox-exec confines reads to research inputs and Python runtime
files, writes to the work directory, and blocks network and process creation.
The interpreter itself is allowed to exec for macOS startup; CPython audit hooks
block model-requested exec, sockets and ctypes. Elsewhere only
audit hooks apply: best-effort checks, not an OS boundary against hostile Python.
Environment, CPU/file-size limits and timeout apply on either path. An available
OS sandbox failing to launch never falls back.
"""
from __future__ import annotations

import os
import json
import resource
import shutil
import site
import subprocess
import sys
import textwrap
from pathlib import Path

SANDBOX_EXEC = shutil.which("sandbox-exec")
OUTPUT_LIMIT = 6000

PREAMBLE = textwrap.dedent(r'''
    import sys, os, json, re, math, statistics, collections, itertools, functools, random
    try:
        import numpy as np
    except Exception:
        np = None
    _WORK = os.path.realpath(os.getcwd())
    _RO = [os.path.realpath(p) for p in __RO_DIRS__]
    def _inside(p, roots):
        return any(p == r or p.startswith(r + os.sep) for r in roots)
    def _check(path, write):
        if isinstance(path, int) or path is None:
            return
        p = os.path.realpath(os.fsdecode(path))
        if _inside(p, [_WORK]):
            if write and _inside(p, [os.path.join(_WORK, '.git')]):
                raise PermissionError("sandbox: research Git metadata is read-only")
            return
        if not write and _inside(p, _RO):
            return
        raise PermissionError("sandbox: access denied: " + p)
    _WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    _BLOCK = ("socket.", "subprocess.", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
              "os.system", "os.kill", "os.killpg", "ctypes.", "os.putenv", "os.unsetenv", "webbrowser.", "urllib.", "http.")
    _WRITE_EVENTS = ("os.remove", "os.rename", "os.rmdir", "os.mkdir", "shutil.rmtree", "shutil.copyfile", "os.chmod",
                     "os.chown", "os.symlink", "os.link", "os.truncate", "os.utime", "shutil.move")
    def _hook(event, args):
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            write = bool(mode and any(c in str(mode) for c in "wax+")) or bool(isinstance(flags, int) and flags & _WFLAGS)
            _check(path, write)
        elif event in ("os.listdir", "os.scandir"):
            if args and args[0] not in (None, ".", ""):
                _check(args[0], False)
        elif event in _WRITE_EVENTS:
            for a in args[:2]:
                if isinstance(a, (str, bytes, os.PathLike)):
                    _check(a, True)
        elif event.startswith(_BLOCK):
            raise PermissionError("sandbox: blocked " + event)
    sys.addaudithook(_hook)
''')


def run_python(code: str, work: Path, ro_dirs: list[Path], timeout_s: float = 45.0) -> str:
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    runtime = [Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, *site.getsitepackages())]
    runtime += [Path(p) for p in ("/System", "/usr/lib", "/usr/share", "/private/var/db",
                                  "/dev/null", "/dev/urandom", "/dev/random")]
    reads = sorted({str(p.resolve()) for p in [*ro_dirs, *runtime]})
    preamble = PREAMBLE.replace('__RO_DIRS__', repr(reads))
    script = work / '_analysis.py'
    script.write_text(preamble + '\n' + code, encoding='utf-8')
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'LANG', 'LC_ALL', 'SYSTEMROOT')}
    env["OPENBLAS_NUM_THREADS"] = env["OMP_NUM_THREADS"] = "1"
    command = [sys.executable, '-I', str(script)]
    if sys.platform == "darwin" and SANDBOX_EXEC:
        interpreter = Path(sys.executable).resolve()
        framework = next((p for p in interpreter.parents if p.name == "Python.framework"), None)
        profile = ['(version 1)', '(deny default)', '(allow process-info* sysctl-read mach-lookup)',
                   '(allow process-exec (literal ' + json.dumps(str(interpreter)) + '))',
                   '(allow file-read* (literal "/"))',
                   '(allow file-read* (subpath ' + json.dumps(str(work)) + '))',
                   '(allow file-write* (subpath ' + json.dumps(str(work)) + '))']
        if framework:
            profile.append('(allow process-exec (subpath ' + json.dumps(str(framework)) + '))')
        for path in reads:
            kind = "subpath" if Path(path).is_dir() else "literal"
            profile.append('(allow file-read* (' + kind + ' ' + json.dumps(path) + '))')
        profile.append('(allow file-read-metadata)')
        profile.append('(deny file-write* (subpath ' + json.dumps(str(work / ".git")) + '))')
        command = [SANDBOX_EXEC, "-p", "\n".join(profile), *command]
    try:
        proc = subprocess.run(command, cwd=work, env=env, text=True,
                              capture_output=True, timeout=timeout_s, preexec_fn=limits)
        output = proc.stdout + proc.stderr
        if proc.returncode and not output.strip():
            return f'[python sandbox failed: returncode={proc.returncode}]'
        return output[-OUTPUT_LIMIT:]
    except subprocess.TimeoutExpired:
        return f'[python timed out after {timeout_s:g} seconds]'
