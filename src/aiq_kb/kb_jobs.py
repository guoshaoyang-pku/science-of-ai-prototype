#!/usr/bin/env python3
"""Persistent research/summarizer processes; timeouts change membership, never kill."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def read_job(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write(path: Path, value: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("w") as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


@contextlib.contextmanager
def _lock(path: Path):
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def update_job(path: Path, **changes) -> dict:
    path = Path(path)
    with _lock(path.parent / "state.lock"):
        job = read_job(path)
        for field in ("id", "origin_epoch", "payload", "key"):
            if field in changes and changes[field] != job.get(field):
                raise ValueError(f"immutable job field: {field}")
        job.update(changes, updated_at=time.time())
        _write(path, job)
        return job


def process_identity(pid: int | None) -> dict | None:
    if not pid:
        return None
    try:
        os.kill(pid, 0)
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart=", "-o", "stat=", "-o", "command="],
            text=True, capture_output=True, timeout=5,
        )
        fields = result.stdout.strip().split(None, 6)
        if result.returncode or len(fields) < 7:
            os.kill(pid, 0)
            raise RuntimeError("could not inspect a process that still exists")
        if "Z" in fields[5]:
            return None
        return {"pid": pid, "started": " ".join(fields[:5]), "command": fields[6]}
    except ProcessLookupError:
        return None


def process_alive(identity: dict | None) -> bool | None:
    try:
        current = process_identity(identity["pid"]) if identity else None
    except (PermissionError, RuntimeError, subprocess.TimeoutExpired):
        return None
    return bool(current and current["started"] == identity["started"])


@contextlib.contextmanager
def worker_lock(path: Path):
    """A restarted launcher cannot duplicate a worker that still owns its lock."""
    path = Path(path).resolve()
    with (path.parent / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(0)
        job = read_job(path)
        if job["status"] == "completed":
            raise SystemExit(0)
        job = update_job(path, status="running", pid=os.getpid(),
                         identity=process_identity(os.getpid()),
                         first_started_at=job.get("first_started_at", time.time()))
        yield job


def finish_job(path: Path, result: dict | None = None) -> dict:
    return update_job(path, status="completed", completed_at=time.time(), result=result)


def _retire_children(job: dict) -> bool:
    identity = job.get("identity") or job.get("previous_identity")
    if not identity:
        return True
    try:
        leader = process_identity(identity["pid"])
        if leader:
            return leader["started"] != identity["started"]
        result = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="],
                                capture_output=True, text=True, timeout=5, check=True)
        members = [line.split() for line in result.stdout.splitlines()]
        alive = any(int(pgid) == identity["pid"] and "Z" not in status
                    for pid, pgid, status in members)
        if not alive:
            return True
        ended = job.get("children_term_at")
        os.killpg(identity["pid"], signal.SIGKILL if ended and time.time() - ended > 5
                  else signal.SIGTERM)
        if not ended:
            update_job(Path(job["path"]), children_term_at=time.time())
        return False
    except ProcessLookupError:
        return True
    except (PermissionError, RuntimeError, subprocess.SubprocessError):
        return False


class Jobs:
    def __init__(self, run: Path, program: Path, timeout: float = 2400, max_workers: int = 4,
                 retry_limit: int = 6, retry_delay: float = 30):
        self.run, self.program = Path(run).resolve(), Path(program).resolve()
        self.directory = self.run / "jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout, self.max_workers = timeout, max_workers
        self.retry_limit, self.retry_delay = retry_limit, retry_delay
        self.processes: dict[str, subprocess.Popen] = {}
        if timeout <= 0 or max_workers < 1:
            raise ValueError("timeout and max_workers must be positive")

    def list(self) -> list[dict]:
        jobs = [read_job(path) for path in self.directory.glob("J*/job.json")]
        return sorted(jobs, key=lambda job: (job["created_at"], job["id"]))

    def submit(self, kind: str, origin_epoch: int, payload: dict, key: str | None = None) -> dict:
        json.dumps(payload)
        with _lock(self.directory / "manager.lock"):
            if key is not None:
                for job in self.list():
                    if job.get("key") == key:
                        if job["kind"] != kind or job["origin_epoch"] != origin_epoch:
                            raise ValueError("idempotency key belongs to another job")
                        return job
            identifier = "J" + uuid.uuid4().hex[:12]
            path = self.directory / identifier / "job.json"
            path.parent.mkdir()
            job = {"id": identifier, "kind": kind, "origin_epoch": origin_epoch,
                   "current_epoch": origin_epoch, "member_epoch": origin_epoch,
                   "carryover": False, "label": f"e{origin_epoch:04d}", "key": key,
                   "payload": payload, "status": "queued", "attempt": 0,
                   "created_at": time.time(), "updated_at": time.time(), "path": str(path)}
            _write(path, job)
            return job

    def _config(self, epoch: int) -> dict:
        path = self.directory / "supervisor.json"
        config = read_job(path) if path.exists() else {}
        config.update(run=str(self.run), program=str(self.program), timeout=self.timeout,
                      max_workers=self.max_workers, retry_limit=self.retry_limit,
                      retry_delay=self.retry_delay,
                      current_epoch=max(epoch, config.get("current_epoch", 0)))
        _write(path, config)
        return config

    def poll(self, epoch: int) -> list[dict]:
        with _lock(self.directory / "manager.lock"):
            config = self._config(epoch)
            epoch = config["current_epoch"]
            for identifier, process in list(self.processes.items()):
                if process.poll() is not None:
                    del self.processes[identifier]
            active, active_research = 0, 0
            for job in self.list():
                path = Path(job["path"])
                alive = process_alive(job.get("identity"))
                if alive is None:
                    active += 1
                    active_research += job["kind"] == "research"
                    continue
                if alive:
                    active += 1
                    active_research += job["kind"] == "research"
                    if job["status"] not in ("completed", "failed"):
                        carried = time.time() - job["first_started_at"] >= self.timeout
                        update_job(path, current_epoch=epoch, carryover=carried,
                                   member_epoch=epoch if carried else job["origin_epoch"],
                                   label=(f"e{epoch:04d} [from e{job['origin_epoch']:04d}]"
                                          if carried else f"e{job['origin_epoch']:04d}"))
                    continue
                if job["status"] in ("completed", "failed", "queued", "retry_wait"):
                    continue
                if not _retire_children(job):
                    active += 1
                    active_research += job["kind"] == "research"
                    continue
                with _lock(path.parent / "state.lock"):
                    current = read_job(path)
                    if (current["status"] not in ("running", "launching")
                            or current.get("identity") != job.get("identity")
                            or current["attempt"] != job["attempt"]):
                        continue
                    failed = current["attempt"] > self.retry_limit
                    delay = min(600, self.retry_delay * 2 ** max(0, current["attempt"] - 1))
                    now = time.time()
                    current.update(status="failed" if failed else "retry_wait",
                                   next_retry_at=now + delay, updated_at=now,
                                   last_error="worker process stopped before durable completion",
                                   previous_identity=current.get("identity"), pid=None, identity=None)
                    _write(path, current)
            prefer_research = self.max_workers == 1 and config.get("last_started_kind") == "summarizer"
            preferred = "research" if prefer_research else "summarizer"
            candidates = sorted(self.list(), key=lambda job: job["kind"] != preferred)
            for job in candidates:
                if active >= self.max_workers:
                    break
                if (job["kind"] == "research" and self.max_workers > 1
                        and active_research >= self.max_workers - 1):
                    continue
                if job["status"] not in ("queued", "retry_wait"):
                    continue
                if time.time() < job.get("next_retry_at", 0):
                    continue
                if not _retire_children(job):
                    continue
                path = Path(job["path"])
                with _lock(path.parent / "state.lock"):
                    job = read_job(path)
                    now = time.time()
                    job.update(status="launching", attempt=job["attempt"] + 1,
                               started_at=now, first_started_at=job.get("first_started_at", now),
                               current_epoch=epoch, updated_at=now)
                    _write(path, job)
                    with (path.parent / "worker.log").open("ab") as log:
                        process = subprocess.Popen(
                            [sys.executable, str(self.program), "--worker-job", str(path)],
                            cwd=path.parent, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            start_new_session=True, close_fds=True,
                        )
                    job.update(status="running", pid=process.pid,
                               identity=process_identity(process.pid), updated_at=time.time())
                    _write(path, job)
                self.processes[job["id"]] = process
                active += 1
                active_research += job["kind"] == "research"
                config["last_started_kind"] = job["kind"]
                _write(self.directory / "supervisor.json", config)
            return self.list()

    def start_supervisor(self, *, claim_owner: bool = False) -> dict:
        with _lock(self.directory / "manager.lock"):
            config = self._config(0)
            if claim_owner or not config.get("owner"):
                config["owner"] = process_identity(os.getpid())
                config["owner_finished"] = False
            if process_alive(config.get("identity")) is not False:
                _write(self.directory / "supervisor.json", config)
                return config
            with (self.directory / "supervisor.log").open("ab") as log:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "supervise", str(self.run),
                     str(self.program), "--timeout", str(self.timeout),
                     "--max-workers", str(self.max_workers), "--retry-limit", str(self.retry_limit),
                     "--retry-delay", str(self.retry_delay)],
                    cwd=self.run, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True, close_fds=True,
                )
            config.update(pid=process.pid, identity=process_identity(process.pid), started_at=time.time())
            _write(self.directory / "supervisor.json", config)
            self.processes["supervisor"] = process
            return config

    def finish_owner(self) -> None:
        with _lock(self.directory / "manager.lock"):
            config = read_job(self.directory / "supervisor.json")
            if config.get("owner") == process_identity(os.getpid()):
                config["owner_finished"] = True
                _write(self.directory / "supervisor.json", config)


def supervise(args) -> None:
    jobs = Jobs(args.run, args.program, args.timeout, args.max_workers, args.retry_limit, args.retry_delay)
    with (jobs.directory / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        while True:
            config = read_job(jobs.directory / "supervisor.json")
            current = jobs.poll(config.get("current_epoch", 0))
            if config.get("owner_finished") or process_alive(config.get("owner")) is False:
                if __package__:
                    from .kb_science_loop import Loop
                else:
                    from kb_science_loop import Loop
                applied = Loop.drain_publications(jobs.run)
                if applied:
                    print(f"[publications] drained {len(applied)} after main loop", flush=True)
                if applied is not None and all(
                        job["status"] in ("completed", "failed") for job in current):
                    return
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["supervise"])
    parser.add_argument("run", type=Path)
    parser.add_argument("program", type=Path)
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retry-limit", type=int, default=6)
    parser.add_argument("--retry-delay", type=float, default=30)
    supervise(parser.parse_args())
