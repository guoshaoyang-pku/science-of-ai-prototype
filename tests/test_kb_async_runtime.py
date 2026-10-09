"""Actual async loop subprocesses with explicit synthetic inputs and fake CLIs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from aiq_kb import kb_science_loop as loop, cc_agent, codex_agent, soa_index
from aiq_kb.kb_jobs import Jobs, process_alive, read_job

SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPT = SRC / "aiq_kb/sample_kb_science_release.py"
if SCRIPT.is_file():
    SPEC = importlib.util.spec_from_file_location("runtime_release", SCRIPT)
    release = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(release)
else:
    release = None


FAKE = r'''#!/usr/bin/env python3
import json, os, sys, time, re, uuid, urllib.request
from pathlib import Path
args = sys.argv[1:]
prompt = sys.stdin.read()
codex = Path(sys.argv[0]).name == "codex"
def emit(event):
    print(json.dumps(event), flush=True)
if codex:
    cfg = (Path(os.environ["CODEX_HOME"]) / "config.toml").read_text()
    url = re.search(r"http://127[.]0[.]0[.]1:[0-9]+/", cfg).group(0)
    sid = args[2] if args[:2] == ["exec", "resume"] else str(uuid.uuid4())
    emit({"type": "thread.started", "thread_id": sid})
    summary = "_summarizer" in str(Path.cwd())
else:
    cfg = json.loads(Path(args[args.index("--mcp-config") + 1]).read_text())
    url = cfg["mcpServers"]["kb"]["args"][-1]
    flag = "--resume" if "--resume" in args else "--session-id"
    sid = args[args.index(flag) + 1]
    emit({"type": "system", "subtype": "init", "session_id": sid})
    summary = True
role = "summary" if summary else "research"
e = re.search(r"e(\d{4})", str(Path.cwd()))
epoch = int(e.group(1)) if e else 0
with (Path(os.environ["FAKE_RUNTIME_ROOT"]) / "cli_sessions.jsonl").open("a") as out:
    out.write(json.dumps({"role": role, "sid": sid, "epoch": epoch, "args": args, "cwd": str(Path.cwd())}) + "\n")
def call(name, values):
    if not codex:
        emit({"type": "assistant", "message": {"id": name, "content": [
            {"type": "tool_use", "id": name, "name": "mcp__kb__" + name, "input": values}]}})
    body = json.dumps({"name": name if codex else "mcp__kb__" + name, "arguments": values}).encode()
    with urllib.request.urlopen(urllib.request.Request(url, body), timeout=5) as response:
        answer = response.read().decode()
    emit({"type": "item.completed", "item": {"type": "mcp_tool_call", "result": answer}})
    return answer
if summary:
    specs = json.loads(Path("_tools.json").read_text())
    if "research" not in [tool["function"]["name"] for tool in specs]:
        raise RuntimeError("summarizer research tool is missing from its advertised tools")
    marker = Path("fake_curated")
    if not marker.exists():
        result = call("add_claim", {"text": "Synthetic scoped observation", "source_question_ids": [],
                                   "reason": "Runtime test only; no scientific claim."})
        if not result.startswith("added "):
            raise RuntimeError(result)
        marker.write_text(result)
    result = call("research", {"question": "Synthetic unresolved research", "topic": "synthetic-runtime"})
    if "unknown tool" in result:
        raise RuntimeError(result)
    time.sleep(float(os.environ.get("FAKE_SUMMARY_DELAY", "0.7")))
    call("finish", {"science": "Synthetic runtime curation; report remains unresolved."})
    final = "finished"
else:
    text = "# Synthetic ongoing report\nObserved fixture. Competing explanations remain unresolved.\nPrediction not yet tested."
    result = call("publish_report", {"text": text, "claim_ids": ["K1001"], "title": "Unfinished fixture report"})
    if not result.startswith("published "):
        raise RuntimeError(result)
    time.sleep(float(os.environ.get("FAKE_RESEARCH_DELAY", "1.5")))
    final = "<report>" + text + "</report>"
    marker = Path("fake_empty_once")
    if os.environ.get("FAKE_EMPTY_RESEARCH_ONCE") == "1" and not marker.exists():
        marker.write_text(sid)
        final = "<report>  </report>"
if codex:
    emit({"type": "item.completed", "item": {"type": "agent_message", "text": final}})
    emit({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
else:
    emit({"type": "result", "result": final, "is_error": False, "session_id": sid,
          "total_cost_usd": .01, "usage": {"input_tokens": 1}, "num_turns": 1})
'''


WRAPPER = r'''
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, SRC)
from aiq_kb import kb_science_loop as loop, cc_agent, codex_agent
from aiq_kb.kb_jobs import read_job
root = Path(TEST_ROOT)
loop.KEYS = cc_agent.KEYS = codex_agent.KEYS = root / "keys.json"
loop.OUT_ROOT = root / "runs"
path = Path(sys.argv[sys.argv.index("--worker-job") + 1])
job = read_job(path)
worker = loop.Loop(argparse.Namespace(**job["payload"]["args"]))
worker.jobs = loop.Jobs(worker.run, Path(__file__), timeout=.2, retry_delay=.03)
worker.worker(path)
'''


@unittest.skipUnless(release is not None, "configure AIQ_BENCH_ROOT for release fixtures")
class AsyncRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        binary = self.root / "bin"
        binary.mkdir()
        for name in ("claude", "codex"):
            target = binary / name
            target.write_text(FAKE.replace("#!/usr/bin/env python3", "#!" + sys.executable, 1))
            target.chmod(0o755)
        self.keys = self.root / "keys.json"
        self.keys.write_text(json.dumps({p: {"api_key": "unused-fixture", "base_url": "http://localhost.invalid"}
                                        for p in ("vapi", "cctq", "cctq_claude")}))
        questions = []
        for source in release.SOURCES:
            for i in range(12):
                questions.append({"question_id": source + "_" + str(i), "source": source,
                    "group": source + "_group_" + str(i), "family": "synthetic_test_only", "type": "mlp",
                    "task": "ranking" if source.startswith("ranking") else "select",
                    "num_choices": 5 if source.startswith("ranking") else 3,
                    "answer": "A<B<C<D<E" if source.startswith("ranking") else "A",
                    "messages": [{"role": "user", "content": "Synthetic runtime fixture only."}],
                    "meta": {"gate": "synthetic_test_only"}})
        raw = self.root / "synthetic.jsonl"
        raw.write_text("".join(json.dumps(q) + "\n" for q in questions))
        self.release = self.root / "release"
        release.build({s: raw for s in release.SOURCES}, {s: 8 for s in release.SOURCES}, [],
                      self.release, seed=12, val_fraction=0, test_fraction=0, name="synthetic_test_only")
        self.seed = self.root / "seed.json"
        self.seed.write_text(json.dumps({"claims": [{"id": "K1001", "text": "Synthetic seed observation",
                                          "support_count": 1, "failure_count": 0}]}))
        with patch.object(sys, "argv", ["loop", "--run-name", "synthetic_test_only", "--pool",
             str(self.release / "questions.jsonl"), "--seed-kb", str(self.seed), "--async-agents",
             "--agent-timeout", ".2", "--batch-size", "4", "--epochs", "2", "--concurrency", "1",
             "--no-eval", "--summ-backend", "claude_code", "--model", "fixture-model",
             "--science", "--render", "brief", "--test-repeats", "0"]):
            self.args = loop.parse()
        self.wrapper = self.root / "worker.py"
        self.wrapper.write_text("SRC = " + repr(str(SRC)) + "\nTEST_ROOT = " + repr(str(self.root)) + "\n" + WRAPPER)
        self.patches = [patch.object(loop, "KEYS", self.keys), patch.object(cc_agent, "KEYS", self.keys),
                        patch.object(codex_agent, "KEYS", self.keys), patch.object(loop, "OUT_ROOT", self.root / "runs"),
                        patch.dict(os.environ, {"PATH": str(binary) + os.pathsep + os.environ["PATH"],
                            "FAKE_RUNTIME_ROOT": str(self.root), "FAKE_SUMMARY_DELAY": ".7",
                            "FAKE_RESEARCH_DELAY": "1.5", "AIQ_KB_DATA_ROOT": str(self.root),
                            "AIQ_KB_KEYS": str(self.keys)})]
        for value in self.patches:
            value.start()
        self.obj = loop.Loop(self.args)
        self.obj.jobs = Jobs(self.obj.run, self.wrapper, timeout=.2, max_workers=4, retry_delay=.03)
        loop.write_json(self.obj.run / "config.json", vars(self.args))

    def tearDown(self):
        for path in self.obj.run.glob("jobs/J*/job.json"):
            state = read_job(path)
            if process_alive(state.get("identity")):
                try:
                    os.killpg(state["identity"]["pid"], signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        supervisor = self.obj.run / "jobs/supervisor.json"
        if supervisor.exists():
            state = read_job(supervisor)
            if process_alive(state.get("identity")):
                try:
                    os.kill(state["identity"]["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in self.obj.jobs.processes.values():
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for value in reversed(self.patches):
            value.stop()
        self.temp.cleanup()

    def wait(self, condition, timeout=8, epoch=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            jobs = self.obj.jobs.poll(epoch)
            if condition(jobs):
                return jobs
            time.sleep(.03)
        logs = [(str(p), p.read_text()) for pattern in ("jobs/J*/worker.log", "sandbox/e*/_summarizer/_cc_stderr.log",
                "jobs/J*/sessions/*/_codex_stderr.log", "jobs/J*/conflict-*.json") for p in self.obj.run.glob(pattern)]
        self.fail("runtime condition timed out: " + json.dumps(jobs) + "\n" + repr(logs))

    def stub_solve(self, qid, kb, hist, vis, epoch, directory, discover):
        q = self.obj.pool[qid]
        record = {"question_id": qid, "source": q["source"], "family": q["family"], "task": q["task"],
            "epoch": epoch, "score": 1., "answer": q["answer"], "prediction": q["answer"], "explanation": "Synthetic",
            "cited": ["K1001"], "discover_trace": [], "solver_reasoning": "",
            "solver_snapshot": self.obj.solve_snapshot_label, "comment": "Synthetic test",
            "hypothesis": None, "skipped": True}
        loop.write_json(directory / (qid + ".json"), record)
        return record

    def test_packaged_worker_entrypoint_uses_external_data_and_keys(self):
        self.obj.jobs = Jobs(self.obj.run, Path(loop.__file__), timeout=.2,
                             max_workers=4, retry_delay=.03)
        kb = self.obj.load_kb()
        kb.epoch = 1
        vis = self.obj.visible("package-entrypoint", [], kb)
        job = self.obj.submit_job("research", 1, kb, [], vis,
                                  question="Synthetic external-path fixture", topic="entrypoint")
        finished = self.wait(lambda current: next(j for j in current if j["id"] == job["id"])
                             ["status"] == "completed", epoch=1)
        state = next(j for j in finished if j["id"] == job["id"])
        self.assertEqual(state["attempt"], 1)
        self.assertTrue(Path(state["path"]).is_relative_to((self.root / "runs").resolve()))
        sessions = loop.read_jsonl(self.root / "cli_sessions.jsonl")
        self.assertTrue(sessions)
        self.assertTrue(all(Path(s["cwd"]).is_relative_to((self.root / "runs").resolve()) for s in sessions))
        self.assertEqual(self.keys.read_text(), json.dumps({p: {"api_key": "unused-fixture",
                         "base_url": "http://localhost.invalid"} for p in ("vapi", "cctq", "cctq_claude")}))

    def test_two_epochs_use_unfinished_report_and_keep_processes(self):
        self.args.summ_backend = "codex"
        os.environ["FAKE_RESEARCH_DELAY"] = "4"
        kb = self.obj.load_kb()
        kb.epoch = 1
        records = [{"question_id": next(iter(self.obj.pool)), "source": "arch170",
                    "family": "synthetic_test_only", "task": "select", "score": 1.,
                    "prediction": "A", "answer": "A", "cited": ["K1001"],
                    "explanation": "Synthetic", "comment": "Synthetic", "hypothesis": None}]
        vis = self.obj.visible("runtime_before", records, kb)
        self.obj.submit_job("summarizer", 1, kb, records, vis)
        jobs = self.wait(lambda current: any(j.get("report_id") for j in current), epoch=1)
        running_report = next(j for j in jobs if j["kind"] == "research")
        self.assertTrue(process_alive(running_report["identity"]))
        with patch.object(self.obj, "solve", self.stub_solve):
            self.obj.main()
        jobs = self.obj.jobs.list()
        summaries = [j for j in jobs if j["kind"] == "summarizer"]
        self.assertEqual(len(summaries), 2)
        self.assertTrue(any(j["carryover"] for j in jobs))
        reports = [j for j in jobs if j["kind"] == "research"]
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertTrue(process_alive(report["identity"]))
        self.assertEqual(report["attempt"], 1)
        next_epoch = loop.KB(json.loads((self.obj.run / "solver_snapshots/solve_e0002.json").read_text()), self.obj.run)
        text = self.obj.open_kb(next_epoch, ["science"])
        self.assertIn("Competing explanations remain unresolved", text)
        frozen = loop.KB(json.loads((self.obj.run / "solver_snapshots/solve_e0001.json").read_text()), self.obj.run)
        self.assertEqual(self.obj.open_kb(frozen, ["science"]), text)
        owner = read_job(self.obj.run / "jobs/supervisor.json")["owner"]
        self.assertEqual(owner["pid"], os.getpid())
        self.assertIs(read_job(self.obj.run / "jobs/supervisor.json")["owner_finished"], True)
        frozen_paths = [self.obj.run / "kb/kb_0002.json", self.obj.run / "metrics.jsonl"]
        frozen_bytes = {path: path.read_bytes() for path in frozen_paths}
        self.wait(lambda current: all(j["status"] == "completed" for j in current))
        self.assertEqual(read_job(Path(report["path"]))["pid"], report["pid"])

        def drained(current):
            live = json.loads((self.obj.run / "kb_live.json").read_text())
            publications = [json.loads(path.read_text())["id"]
                            for path in self.obj.run.glob("jobs/J*/publications/p*.json")]
            return set(publications).issubset(live["applied_publications"])

        self.wait(drained)
        for path, contents in frozen_bytes.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_brief_curator_can_retire_refuted_claim_without_changing_solver_snapshot(self):
        self.args.summ_backend = "codex"
        kb = self.obj.load_kb()
        kb.epoch = 1
        kb.claims["K1001"].update(support_count=0., failure_count=5., credibility=1/7)
        frozen = loop.KB(kb.to_doc(), self.obj.run)
        vis = self.obj.visible("delete_regression", [], kb)
        job = self.obj.jobs.submit("summarizer", 1, {})
        self.obj.worker_job, self.obj.worker_path = job, Path(job["path"])
        outcomes = []

        class Session:
            def __init__(self, work, model, system, tools, handler):
                self.handler, self.log = handler, []

            def turn(self, *args, **kwargs):
                outcomes.append(self.handler("delete_claim", {"claim_id": "K1001",
                                "reason": "Synthetic refutation"}))
                self.handler("finish", {"science": "Synthetic curation complete"})
                return {"final": "done", "reasoning": None, "usage": {}}

            def close(self):
                pass

        with patch.object(loop, "CodexSession", Session):
            self.obj.summarize(1, kb, [], vis)
        self.assertEqual(outcomes, ["deleted K1001"])
        self.assertNotIn("K1001", kb.claims)
        self.assertIn("K1001", frozen.claims)
        self.assertEqual(kb.retired[-1]["id"], "K1001")

    def test_summary_interruption_resumes_session_and_no_duplicate_credit(self):
        kb = self.obj.load_kb()
        kb.epoch = 1
        qid = next(iter(self.obj.pool))
        kb.credit({"K1001": (1., 0)}, self.obj.author_solver, qid)
        saved = {**kb.to_doc(), "pending_commits": kb.pending}
        loop.write_json(self.obj.run / "kb_live.json", saved)
        restored = self.obj.load_kb()
        restored.credit({"K1001": (1., 0)}, self.obj.author_solver, qid)
        restored.flush()
        records = [{"question_id": qid, "source": "arch170", "family": "synthetic_test_only", "task": "select",
                    "score": 1., "prediction": "A", "answer": "A", "cited": ["K1001"], "explanation": "Synthetic",
                    "comment": "Synthetic", "hypothesis": None}]
        vis = self.obj.visible("runtime_summary", records, restored)
        job = self.obj.submit_job("summarizer", 1, restored, records, vis)
        running = self.wait(lambda current: (self.obj.run / "sandbox/e0001/_summarizer/_claude_session.json").exists()
                            and list(Path(job["path"]).parent.glob("publications/p*.json")))[0]
        original = read_job(Path(job["path"]))
        session = json.loads((self.obj.run / "sandbox/e0001/_summarizer/_claude_session.json").read_text())["session_id"]
        os.kill(original["pid"], signal.SIGKILL)
        done = self.wait(lambda current: next(j for j in current if j["id"] == job["id"])["status"] == "completed")
        finished = next(j for j in done if j["id"] == job["id"])
        self.assertEqual(finished["attempt"], 2)
        sessions = loop.read_jsonl(self.root / "cli_sessions.jsonl")
        summaries = [s for s in sessions if s["role"] == "summary"]
        self.assertEqual({s["sid"] for s in summaries}, {session})
        self.assertTrue(any("--resume" in s["args"] for s in summaries))
        self.obj.apply_publications(restored, 3)
        commits = loop.read_jsonl(self.obj.run / "commits.jsonl")
        self.assertEqual(len([c for c in commits if c["op"] == "credit"]), 1)
        self.assertEqual(len([c for c in commits if c["op"] == "async_apply" and c["payload"]["operation"]["op"] == "add"]), 1)

    def test_empty_research_retries_same_session_and_keeps_saved_report(self):
        os.environ.update(FAKE_EMPTY_RESEARCH_ONCE="1", FAKE_RESEARCH_DELAY="0.05")
        kb = self.obj.load_kb()
        kb.epoch = 1
        vis = self.obj.visible("empty-final-fixture", [], kb)
        job = self.obj.submit_job("research", 1, kb, [], vis, question="Synthetic empty-final recovery",
                                  topic="synthetic-empty-final")
        states = self.wait(lambda current: next(j for j in current if j["id"] == job["id"])["status"] == "retry_wait")
        interrupted = next(j for j in states if j["id"] == job["id"])
        self.assertEqual(interrupted["attempt"], 1)
        work = Path(job["path"]).parent
        saved_session = json.loads((work / "session/_codex_session.json").read_text())["thread_id"]
        checkpoint = json.loads((work / "kb_checkpoint.json").read_text())
        saved = next(iter(checkpoint["kb"]["reports"].values()))
        saved_text = Path(saved["path"]).read_text()
        self.assertIn("Competing explanations remain unresolved", saved_text)
        self.assertEqual((work / "repo/report.md").read_text(), saved_text)
        self.assertFalse((work / "r01.json").exists())
        done = self.wait(lambda current: next(j for j in current if j["id"] == job["id"])["status"] == "completed")
        finished = next(j for j in done if j["id"] == job["id"])
        self.assertEqual(finished["attempt"], 2)
        self.assertEqual(json.loads((work / "session/_codex_session.json").read_text())["thread_id"], saved_session)
        sessions = [s for s in loop.read_jsonl(self.root / "cli_sessions.jsonl") if s["role"] == "research"]
        self.assertEqual({s["sid"] for s in sessions}, {saved_session})
        self.assertEqual(sessions[-1]["args"][:3], ["exec", "resume", saved_session])
        final = json.loads((work / "r01.json").read_text())
        self.assertEqual(final["report"], saved_text)
        report = next(iter(json.loads((work / "kb_checkpoint.json").read_text())["kb"]["reports"].values()))
        self.assertEqual(report["repo_commit"], saved["repo_commit"])
        self.assertEqual(Path(saved["path"]).read_text(), saved_text)

    def test_batch_extension_preserves_assignments_and_excludes_imported_history(self):
        self.args.epochs = 1
        first = self.obj.batch(1)
        existing = json.loads((self.obj.run / "batches.json").read_text())
        available = [i for ids in self.obj.split["stream"].values() for i in ids if i not in first]
        prior = self.root / "runs/prior/epochs/e0001/records.jsonl"
        loop.write_jsonl(prior, [{"question_id": available[0]}])
        self.args.import_history = ["prior"]
        self.args.epochs = 3
        second = self.obj.batch(2)
        third = self.obj.batch(3)
        schedule = json.loads((self.obj.run / "batches.json").read_text())
        self.assertEqual(existing["1"], schedule["1"])
        self.assertEqual(len(second), self.args.batch_size)
        self.assertEqual(len(third), self.args.batch_size)
        self.assertEqual(len(set(first + second + third)), len(first + second + third))
        self.assertNotIn(available[0], second + third)

    def test_soa_mass_uses_configured_pool_and_explicit_override(self):
        kb = self.obj.load_kb().to_doc()
        kb["claims"][0]["regime"] = {"family": ["synthetic_test_only"], "type": ["mlp"]}
        qid = next(iter(self.obj.pool))
        fixture = self.root / "soa_pool.jsonl"
        fixture.write_text(json.dumps({"question_id": qid, "family": "different_fixture", "type": "gru"}) + "\n")
        original = soa_index.compute(self.obj.run, kb)
        self.assertEqual(original["regime_completeness_cells"], 1)
        changed = soa_index.compute(self.obj.run, kb, pool=fixture)
        self.assertEqual(changed["regime_completeness_cells"], 0)
        self.assertEqual(changed["uncovered_cells"], ["different_fixture/gru"])
        config = self.obj.run / "config.json"
        loop.write_json(config, {"pool": str(fixture)})
        self.assertEqual(soa_index.compute(self.obj.run, kb)["uncovered_cells"], ["different_fixture/gru"])
        fixture.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "configured question pool"):
            soa_index.compute(self.obj.run, kb)

    def test_sampled_pool_imports_exact_historical_question_without_expanding_solver_pool(self):
        old = next(json.loads(line) for line in (self.root / "synthetic.jsonl").read_text().splitlines()
                   if json.loads(line)["question_id"] not in self.obj.pool)
        old["messages"] = [{"role": "user", "content": "Original historical prompt, preserved exactly."}]
        old_pool = self.root / "historical.jsonl"
        old_pool.write_text(json.dumps(old) + "\n")
        prior = loop.OUT_ROOT / "prior"
        loop.write_json(prior / "config.json", {"pool": str(old_pool),
                        "pool_sha256": hashlib.sha256(old_pool.read_bytes()).hexdigest()})
        record = {k: old[k] for k in ("question_id", "group", "source", "task", "type", "family", "answer")}
        record.update(epoch=1, prediction="B", score=0, cited=[], solver_snapshot="solve_e0001",
                      solver_selection={"mode": "live", "label": "research_live"})
        loop.write_jsonl(prior / "epochs/e0001/records.jsonl", [record])
        self.args.import_history = ["prior"]
        records = self.obj.history_records(0)
        pool_ids = set(self.obj.pool)
        kb = self.obj.load_kb()
        visible = self.obj.visible("imported", records, kb)
        row = loop.read_jsonl(visible / "history.jsonl")[0]
        self.assertEqual(row["question"], old["messages"][-1]["content"])
        self.assertEqual(row["imported_from"], "prior")
        self.assertEqual(row["solver_snapshot"], "solve_e0001")
        self.assertEqual(row["solver_selection"], record["solver_selection"])
        self.assertEqual(row["question_provenance"]["path"], str(old_pool))
        self.assertTrue(row["question_provenance"]["hash_verified"])
        old_pool.unlink()
        self.assertEqual(self.obj.history_question(records[0])[0], old)
        self.assertEqual(set(self.obj.pool), pool_ids)
        self.assertNotIn(old["question_id"], self.obj.batch(1))
        with patch("aiq_kb.lab_predictor.parse_question", return_value=None) as parser:
            self.obj.bias_table(1, [])
        parser.assert_called_once_with(old)

    def test_historical_question_fallback_and_missing_or_changed_source_are_explicit(self):
        record = {"question_id": "old_saved", "question": "Exact saved question.", "imported_from": "saved"}
        question, provenance = self.obj.history_question(record)
        self.assertEqual(question["messages"][-1]["content"], record["question"])
        self.assertEqual(provenance["kind"], "saved_record_question")
        with self.assertRaisesRegex(ValueError, "historical question unavailable"):
            self.obj.history_question({"question_id": "missing", "imported_from": "saved"})
        changed = self.root / "changed.jsonl"
        changed.write_text(json.dumps({"question_id": "changed"}) + "\n")
        loop.write_json(loop.OUT_ROOT / "changed/config.json", {"pool": str(changed), "pool_sha256": "incorrect"})
        with self.assertRaisesRegex(ValueError, "historical pool hash mismatch"):
            self.obj.history_question({"question_id": "changed", "imported_from": "changed"})

    def test_historical_group_cannot_enter_current_holdout_even_with_different_question_id(self):
        held = next(iter(self.obj.pool.values()))
        self.obj.split["test"] = [held["question_id"]]
        historical = {"question_id": "old_same_group", "group": held["group"],
                      "question": "Saved related question", "imported_from": "missing_pool"}
        with self.assertRaisesRegex(ValueError, "overlaps current holdout"):
            self.obj.history_question(historical)


if __name__ == "__main__":
    unittest.main()
