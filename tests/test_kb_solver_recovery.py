import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiq_kb import kb_science_loop as loop


class FakeSession:
    calls = []
    crash_discovery = False
    crash_solve = False

    def __init__(self, directory, model, instructions, tools, handler, *, resume=True, **kwargs):
        self.dir = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "fake_session.json"
        self.state = json.loads(self.path.read_text()) if resume and self.path.exists() else {}
        self.thread_id = self.state.get("thread_id")
        self.handler = handler
        self.log = []

    def turn(self, prompt, effort, phase, **kwargs):
        thread = self.thread_id
        self.calls.append({"phase": phase, "resumed_id": thread, "had_gold": self.state.get("gold", False)})
        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())
        if phase == "solve":
            if self.state.get("gold"):
                raise AssertionError("attempted to solve from a gold-bearing thread")
            self.state = {"thread_id": self.thread_id, "gold": False}
            self.path.write_text(json.dumps(self.state))
            if self.crash_solve:
                type(self).crash_solve = False
                raise RuntimeError("solve transport interrupted")
            self.handler("kb_search", {"query": "measured"})
            tools = [{"name": "kb_search", "arguments": {"query": "measured"}, "result": "K1001"}]
            self.log += tools
            return {"final": "<answer>B</answer><cited>K1001</cited>", "tool_log": tools,
                    "reasoning": "initial reasoning", "usage": {"input_tokens": 1}, "secs": 1}
        self.state.update(thread_id=self.thread_id, gold=True)
        self.path.write_text(json.dumps(self.state))
        self.assert_gold_prompt(prompt)
        if self.crash_discovery:
            type(self).crash_discovery = False
            raise RuntimeError("crash after gold was revealed")
        return {"final": "<comment>checked</comment>", "tool_log": [], "reasoning": "discovery",
                "usage": {"input_tokens": 1}, "secs": 1}

    def assert_gold_prompt(self, prompt):
        if "Gold answer: A" not in prompt:
            raise AssertionError("discovery did not receive gold")

    def close(self):
        pass


class SolverRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        self.obj = loop.Loop.__new__(loop.Loop)
        self.obj.run = self.run
        self.obj.args = SimpleNamespace(render="brief", science=True, phase_preview=100, lab=None,
                                        discover_exps=0, solve_lab=False, corpus=None, model="test-model",
                                        solve_effort="low", discover_effort_right="low",
                                        discover_effort_wrong="medium", discover_turns=4)
        self.obj.solve_snapshot_label = "solve_e0001"
        self.q = {"question_id": "q_test", "task": "select", "answer": "A", "num_choices": 2,
                  "source": "arch170", "type": "select", "family": "regression", "group": "g1"}
        claim = {"id": "K1001", "text": "measured", "support_count": 0, "failure_count": 0,
                 "credibility": .5, "created_epoch": 0, "sources": [], "merged_from": []}
        self.kb = loop.KB({"epoch": 1, "claims": [claim]}, self.run)
        self.out_dir = self.run / "epochs/e0001/solves"
        self.messages = [{"role": "system", "content": "solve the question"},
                         {"role": "user", "content": "choose A or B"}]
        FakeSession.calls = []
        FakeSession.crash_discovery = False
        FakeSession.crash_solve = False
        self.mock = patch.object(loop, "CodexSession", FakeSession)
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.temp.cleanup()

    @property
    def session_dir(self):
        return self.run / "codex/e0001/solves/q_test"

    def solve(self, discover=True, q=None):
        return self.obj.solve_codex(q or self.q, self.messages, self.kb, None, None, 1, self.out_dir, discover)

    def test_crash_after_gold_skips_solve_and_resumes_exact_discovery(self):
        FakeSession.crash_discovery = True
        with self.assertRaisesRegex(RuntimeError, "after gold"):
            self.solve()
        self.assertFalse((self.out_dir / "q_test.json").exists())
        saved = json.loads((self.session_dir / "solve_record.json").read_text())
        thread_id = saved["record"]["codex_thread"]
        self.assertEqual(saved["record"]["prediction"], "B")
        self.assertEqual(saved["record"]["score"], 0)
        self.assertEqual(saved["tool_log"][0]["name"], "kb_search")
        # The saved solve ID is authoritative even if a session pointer is accidentally replaced.
        (self.session_dir / "fake_session.json").write_text(json.dumps({"thread_id": "wrong-pointer", "gold": True}))
        rec = self.solve()
        self.assertEqual([c["phase"] for c in FakeSession.calls], ["solve", "discover", "discover"])
        self.assertEqual(FakeSession.calls[-1]["resumed_id"], thread_id)
        self.assertEqual(rec["prediction"], "B")
        self.assertEqual(rec["score"], 0)
        self.assertEqual(rec["kb_queries"], ["measured"])
        self.assertEqual(rec["kb_retrieved"], ["K1001"])
        self.assertEqual(rec["comment"], "checked")

    def test_missing_solve_checkpoint_uses_fresh_thread_despite_stale_gold(self):
        self.session_dir.mkdir(parents=True)
        (self.session_dir / "fake_session.json").write_text(json.dumps({"thread_id": "old-thread", "gold": True}))
        rec = self.solve()
        self.assertIsNone(FakeSession.calls[0]["resumed_id"])
        self.assertFalse(FakeSession.calls[0]["had_gold"])
        self.assertNotEqual(rec["codex_thread"], "old-thread")
        self.assertEqual(rec["prediction"], "B")

    def test_failed_uncheckpointed_solve_restarts_fresh(self):
        FakeSession.crash_solve = True
        with self.assertRaisesRegex(RuntimeError, "transport"):
            self.solve()
        self.assertFalse((self.session_dir / "solve_record.json").exists())
        self.solve()
        self.assertIsNone(FakeSession.calls[0]["resumed_id"])
        self.assertIsNone(FakeSession.calls[1]["resumed_id"])

    def test_hash_is_identical_across_time_and_questions(self):
        with patch.object(loop, "now", return_value="2026-10-04T12:00:00Z"):
            first = self.solve(False)
        q2 = {**self.q, "question_id": "q_test2"}
        with patch.object(loop, "now", return_value="2026-10-04T12:02:00Z"):
            second = self.solve(False, q=q2)
        self.assertEqual(first["kb_snapshot"], second["kb_snapshot"])
        self.kb.claims["K1001"]["text"] = "new measured evidence"
        third = self.solve(False, q={**self.q, "question_id": "q_test3"})
        self.assertNotEqual(first["kb_snapshot"], third["kb_snapshot"])


if __name__ == "__main__":
    unittest.main()
