"""Exercise carryover and recovery with real detached worker processes."""
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

from aiq_kb.kb_jobs import Jobs, finish_job, process_alive, read_job, update_job


WORKER = '''
import json, sys, time
from pathlib import Path
sys.path.insert(0, MODULE)
from aiq_kb.kb_jobs import worker_lock, finish_job, read_job, update_job
path = Path(sys.argv[2])
with worker_lock(path) as job:
    checkpoint = path.parent / "checkpoint.json"
    count = json.loads(checkpoint.read_text())["count"] if checkpoint.exists() else 0
    while count < job["payload"]["steps"]:
        count += 1
        checkpoint.write_text(json.dumps({"count": count}))
        update_job(path, checkpoint={"count": count})
        time.sleep(job["payload"].get("delay", 0.04))
    commit = path.parent / "commit.json"
    if not commit.exists():
        commit.write_text(json.dumps({"origin_epoch": job["origin_epoch"], "count": count}))
    finish_job(path, {"commit": str(commit)})
'''


class JobsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.program = self.directory / "worker.py"
        self.program.write_text("MODULE = " + repr(str(Path(__file__).resolve().parents[1] / "src")) + "\n" + WORKER)
        self.jobs = Jobs(self.directory / "run", self.program, timeout=.15,
                         max_workers=2, retry_delay=.03)

    def tearDown(self):
        paths = list(self.jobs.directory.glob("J*/job.json"))
        supervisor = self.jobs.directory / "supervisor.json"
        if supervisor.exists():
            paths.append(supervisor)
        for path in paths:
            state = read_job(path)
            if process_alive(state.get("identity")):
                os.kill(state["identity"]["pid"], signal.SIGTERM)
        for process in self.jobs.processes.values():
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self.temp.cleanup()

    def wait(self, job, condition, epoch=1, timeout=6, poll=True):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if poll:
                self.jobs.poll(epoch)
            state = read_job(Path(job["path"]))
            if condition(state):
                return state
            time.sleep(.03)
        self.fail("job condition timed out: " + json.dumps(state))

    def test_timeout_carries_same_worker_and_new_manager_attaches(self):
        job = self.jobs.submit("research", 1, {"steps": 12}, key="topic1")
        duplicate = self.jobs.submit("research", 1, {"steps": 12}, key="topic1")
        self.assertEqual(job["id"], duplicate["id"])
        running = self.wait(job, lambda value: value.get("checkpoint", {}).get("count", 0) >= 2)
        pid = running["pid"]
        replacement = Jobs(self.jobs.run, self.program, timeout=.15, retry_delay=.03)
        replacement.poll(3)
        carried = self.wait(job, lambda value: value["carryover"], epoch=3)
        self.assertEqual(carried["pid"], pid)
        self.assertEqual(carried["member_epoch"], 3)
        self.assertEqual(carried["origin_epoch"], 1)
        self.assertEqual(carried["attempt"], 1)
        finished = self.wait(job, lambda value: value["status"] == "completed", epoch=3)
        self.assertTrue(Path(finished["result"]["commit"]).exists())

    def test_killed_process_resumes_checkpoint_without_duplicate_commit(self):
        job = self.jobs.submit("summarizer", 2, {"steps": 16})
        running = self.wait(job, lambda value: value.get("checkpoint", {}).get("count", 0) >= 3)
        before = running["checkpoint"]["count"]
        os.kill(running["pid"], signal.SIGKILL)
        finished = self.wait(job, lambda value: value["status"] == "completed", epoch=4)
        self.assertEqual(finished["attempt"], 2)
        self.assertGreater(finished["checkpoint"]["count"], before)
        self.assertEqual(json.loads(Path(finished["result"]["commit"]).read_text())["count"], 16)
        self.assertEqual(len(list(Path(job["path"]).parent.glob("commit*"))), 1)

    def test_supervisor_survives_parent_exit_and_finishes(self):
        host_script = self.directory / "host.py"
        host_script.write_text(
            "import sys\nfrom pathlib import Path\nsys.path.insert(0, "
            + repr(str(Path(__file__).resolve().parents[1] / "src")) + ")\nfrom aiq_kb.kb_jobs import Jobs\n"
            + "jobs = Jobs(Path(" + repr(str(self.jobs.run)) + "), Path("
            + repr(str(self.program)) + "), timeout=.15, retry_delay=.03)\n"
            + "jobs.submit('research', 1, {'steps': 15}, key='orphan')\n"
            + "jobs.start_supervisor()\n")
        result = subprocess.run([sys.executable, str(host_script)], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        job = self.jobs.list()[0]
        finished = self.wait(job, lambda value: value["status"] == "completed", timeout=8, poll=False)
        self.assertEqual(finished["attempt"], 1)
        self.assertTrue(Path(finished["result"]["commit"]).exists())

    def test_failed_worker_uses_bounded_retries(self):
        self.program.write_text("raise SystemExit(2)\n")
        self.jobs.retry_limit = 1
        job = self.jobs.submit("research", 1, {})
        failed = self.wait(job, lambda value: value["status"] == "failed")
        self.assertEqual(failed["attempt"], 2)

    def test_observation_failure_does_not_restart_worker(self):
        job = self.jobs.submit("research", 1, {"steps": 8})
        running = self.wait(job, lambda value: value.get("checkpoint"))
        with patch("aiq_kb.kb_jobs.process_identity", side_effect=subprocess.TimeoutExpired("ps", 5)):
            self.jobs.poll(2)
        attached = read_job(Path(job["path"]))
        self.assertEqual(attached["pid"], running["pid"])
        self.assertEqual(attached["attempt"], 1)
        self.wait(job, lambda value: value["status"] == "completed", epoch=2)

    def test_completion_during_liveness_check_keeps_saved_result(self):
        job = self.jobs.submit("summarizer", 12, {"session": "saved-session"})
        path = Path(job["path"])
        identity = {"pid": 99999999, "started": "fixture", "command": "fixture"}
        update_job(path, status="running", attempt=1, identity=identity, pid=identity["pid"],
                   first_started_at=time.time())
        result = {"science": "saved result", "publications": 11}

        def complete_during_check(observed):
            self.assertEqual(observed, identity)
            finish_job(path, result)
            return False

        with patch("aiq_kb.kb_jobs.process_alive", side_effect=complete_during_check), \
                patch("aiq_kb.kb_jobs._retire_children", return_value=True), \
                patch("aiq_kb.kb_jobs.subprocess.Popen") as launch:
            self.jobs.poll(13)
            launch.assert_not_called()
        state = read_job(path)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["result"], result)
        self.assertEqual(state["attempt"], 1)
        self.assertEqual(state["origin_epoch"], 12)
        self.assertEqual(state["payload"], job["payload"])
        self.assertNotIn("last_error", state)

    def test_replacement_during_liveness_check_is_not_overwritten(self):
        identity = {"pid": 99999999, "started": "old", "command": "fixture"}
        for field, value in [("identity", {**identity, "started": "replacement"}), ("attempt", 2)]:
            with self.subTest(field=field):
                jobs = Jobs(self.directory / field, self.program, retry_delay=30)
                job = jobs.submit("research", 1, {"session": "continuing-session"})
                path = Path(job["path"])
                before = update_job(path, status="running", attempt=1, identity=identity,
                                    pid=identity["pid"], first_started_at=time.time())

                def replace_during_check(observed):
                    self.assertEqual(observed, identity)
                    update_job(path, **{field: value})
                    return False

                with patch("aiq_kb.kb_jobs.process_alive", side_effect=replace_during_check), \
                        patch("aiq_kb.kb_jobs._retire_children", return_value=True), \
                        patch("aiq_kb.kb_jobs.subprocess.Popen") as launch:
                    jobs.poll(2)
                    launch.assert_not_called()
                state = read_job(path)
                self.assertEqual(state["status"], "running")
                self.assertEqual(state[field], value)
                self.assertEqual(state["identity"], value if field == "identity" else identity)
                self.assertEqual(state["attempt"], value if field == "attempt" else 1)
                self.assertEqual(state["origin_epoch"], before["origin_epoch"])
                self.assertEqual(state["payload"], before["payload"])
                self.assertNotIn("last_error", state)

    def test_stopped_worker_still_enters_retry_wait(self):
        self.jobs.retry_delay = 30
        job = self.jobs.submit("research", 3, {"session": "resume-this"})
        path = Path(job["path"])
        identity = {"pid": 99999999, "started": "fixture", "command": "fixture"}
        before = update_job(path, status="running", attempt=1, identity=identity,
                            pid=identity["pid"], first_started_at=time.time())
        with patch("aiq_kb.kb_jobs.process_alive", return_value=False), \
                patch("aiq_kb.kb_jobs._retire_children", return_value=True), \
                patch("aiq_kb.kb_jobs.subprocess.Popen") as launch:
            self.jobs.poll(4)
            launch.assert_not_called()
        state = read_job(path)
        self.assertEqual(state["status"], "retry_wait")
        self.assertEqual(state["previous_identity"], identity)
        self.assertIsNone(state["identity"])
        self.assertIsNone(state["pid"])
        self.assertEqual(state["attempt"], 1)
        self.assertEqual(state["origin_epoch"], 3)
        self.assertEqual(state["payload"], before["payload"])
        self.assertGreater(state["next_retry_at"], time.time())

    def test_dead_worker_children_retired_before_resume(self):
        original = self.program.read_text()
        self.program.write_text(original.replace(
            "    checkpoint = path.parent",
            "    import os, subprocess\n"
            "    if job[\"attempt\"] == 1:\n"
            "        child = subprocess.Popen([sys.executable, \"-c\", \"import time; time.sleep(30)\"])\n"
            "        update_job(path, child_identity=__import__(\"aiq_kb.kb_jobs\", fromlist=[\"process_identity\"]).process_identity(child.pid))\n"
            "        os._exit(2)\n"
            "    checkpoint = path.parent"))
        job = self.jobs.submit("research", 1, {"steps": 4})
        finished = self.wait(job, lambda value: value["status"] == "completed")
        self.assertEqual(finished["attempt"], 2)
        self.assertFalse(process_alive(finished["child_identity"]))

    def test_worker_attach_preserves_live_main_owner(self):
        original = self.jobs.start_supervisor()
        code = ("import sys;from pathlib import Path;sys.path.insert(0,"
                + repr(str(Path(__file__).resolve().parents[1] / "src")) + ");from aiq_kb.kb_jobs import Jobs;"
                + "Jobs(Path(" + repr(str(self.jobs.run)) + "),Path("
                + repr(str(self.program)) + ")).start_supervisor()")
        subprocess.run([sys.executable, "-c", code], check=True, timeout=5)
        current = read_job(self.jobs.directory / "supervisor.json")
        self.assertEqual(current["owner"], original["owner"])
        self.assertEqual(current["identity"], original["identity"])

    def test_curator_slot_available_with_three_long_researchers(self):
        self.jobs.max_workers = 4
        researchers = [self.jobs.submit("research", 1, {"steps": 40}) for _ in range(4)]
        jobs = self.jobs.poll(1)
        running = [job for job in jobs if process_alive(job.get("identity"))]
        self.assertEqual(len(running), 3)
        original_pids = {job["id"]: job["pid"] for job in running}
        self.assertEqual(read_job(Path(researchers[3]["path"]))["status"], "queued")
        curator = self.jobs.submit("summarizer", 2, {"steps": 12})
        self.jobs.start_supervisor()
        replacement = Jobs(self.jobs.run, self.program, timeout=.15, max_workers=4, retry_delay=.03)
        jobs = replacement.poll(2)
        current = next(job for job in jobs if job["id"] == curator["id"])
        self.assertTrue(process_alive(current["identity"]))
        self.assertEqual(len([job for job in jobs if process_alive(job.get("identity"))]), 4)
        for job in jobs:
            if job["id"] in original_pids:
                self.assertEqual(job["pid"], original_pids[job["id"]])
                self.assertEqual(job["attempt"], 1)
        self.wait(curator, lambda value: value["status"] == "completed", epoch=2)
        self.assertEqual(read_job(Path(researchers[3]["path"]))["status"], "queued")
        for process in replacement.processes.values():
            process.wait(timeout=2)

    def test_single_worker_alternates_curator_and_research_without_starvation(self):
        self.jobs.max_workers = 1
        research = self.jobs.submit("research", 1, {"steps": 4})
        first = self.jobs.submit("summarizer", 1, {"steps": 3})
        later = self.jobs.submit("summarizer", 2, {"steps": 12})
        self.jobs.poll(2)
        self.assertTrue(process_alive(read_job(Path(first["path"]))["identity"]))
        self.assertEqual(read_job(Path(research["path"]))["status"], "queued")
        finished_research = self.wait(research, lambda value: value["status"] == "completed", epoch=2)
        finished_later = self.wait(later, lambda value: value["status"] == "completed", epoch=2)
        self.assertGreaterEqual(finished_research["started_at"], read_job(Path(first["path"]))["completed_at"])
        self.assertGreaterEqual(finished_later["started_at"], finished_research["completed_at"])


if __name__ == "__main__":
    unittest.main()
