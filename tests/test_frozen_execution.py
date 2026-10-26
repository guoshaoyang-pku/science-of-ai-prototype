"""Offline tests for an explicitly preregistered scheduling-only amendment."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from aiq_kb import frozen_eval as ev
from aiq_kb import frozen_execution as execution


fixture_spec = importlib.util.spec_from_file_location("frozen_eval_fixture", Path(__file__).with_name("test_frozen_eval.py"))
fixture_module = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture_module)


class FrozenExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.FrozenEvaluationTests("test_successful_raw_is_saved_and_resume_never_resends")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.run
        self.fixture.execute(max_jobs=1)
        self.amendment = {"schema_version": 1,
            "parent_registration_sha256": ev.file_hash(self.run / "registration.json"),
            "frozen_eval_source_sha256": ev.file_hash(Path(ev.__file__)),
            "controller_source_sha256": ev.file_hash(Path(execution.__file__)),
            "workers": 4, "reason": "Observed request latency; no labels or accuracy were inspected.",
            "basis": "latency_only", "labels_unopened": True}
        self.write_amendment()

    def write_amendment(self):
        self.fixture.write_json("execution-amendment.json", self.amendment)

    def execute(self, **kwargs):
        return execution.run_execution(self.run, transport=kwargs.pop("transport", lambda *a:
            (200, fixture_module.response())), **kwargs)

    def test_partial_cache_takeover_never_resends_and_preserves_original_identity(self):
        original_registration = (self.run / "registration.json").read_bytes()
        original_freeze = (self.run / "freeze.json").read_bytes()
        original_prediction = next((self.run / "predictions").glob("*.json")).read_bytes()
        original_raw = next((self.run / "requests").glob("*/call_*/response.raw"))
        calls = []
        old_executor = ev.ThreadPoolExecutor
        with patch.object(execution, "ThreadPoolExecutor", wraps=old_executor) as executor:
            status = self.execute(transport=lambda *a: (calls.append(a) or (200, fixture_module.response())))
        executor.assert_called_once_with(max_workers=4)
        self.assertIs(ev.ThreadPoolExecutor, old_executor)
        self.assertEqual(len(calls), 17)
        self.assertEqual(status["completed_before"], 1)
        self.assertEqual(status["completed_after"], 18)
        self.assertEqual(status["requests_before"], 1)
        self.assertEqual(status["requests_after"], 18)
        self.assertEqual(status["actual_worker_limit"], 4)
        self.assertEqual(status["amendment_sha256"], ev.file_hash(self.run / "execution-amendment.json"))
        self.assertEqual((self.run / "registration.json").read_bytes(), original_registration)
        self.assertEqual((self.run / "freeze.json").read_bytes(), original_freeze)
        self.assertIn(original_prediction, [p.read_bytes() for p in (self.run / "predictions").glob("*.json")])
        self.assertEqual(original_raw.read_bytes(), fixture_module.response())
        self.execute(transport=lambda *a: self.fail("completed cache resent"))

    def test_mid_tool_round_successful_raw_is_reused(self):
        job = self.fixture.jobs[3]  # F arm, first repetition
        registration, jobs, lookups, freeze_hash = ev._contract(self.run)
        calls = 0
        def stop_after_tool(*args):
            nonlocal calls
            calls += 1
            (self.run / "STOP").write_text("stop safely between requests")
            return 200, fixture_module.response(tokens=40, finish="tool_calls", tools=fixture_module.tool())
        runner = ev._Runner(self.run, registration, jobs, lookups, freeze_hash, stop_after_tool)
        with self.assertRaises(ev.EvaluationStopped):
            runner.solve(job)
        self.assertEqual(calls, 1)
        tool_raw = self.run / "requests" / job["key"] / "call_01/response.raw"
        preserved = tool_raw.read_bytes()
        (self.run / "STOP").unlink()
        observed = []
        self.execute(transport=lambda endpoint, payload, *args:
            (observed.append(payload) or (200, fixture_module.response())))
        self.assertEqual(len(observed), 17)
        self.assertEqual(tool_raw.read_bytes(), preserved)
        self.assertEqual(sum(p["max_tokens"] == 2960 for p in observed), 1)

    def test_concurrency_amendment_cannot_change_other_configuration(self):
        self.amendment["model"] = "unregistered-model"
        self.write_amendment()
        with self.assertRaisesRegex(ValueError, "fields"):
            self.execute(transport=lambda *a: self.fail("extra configuration sent"))

    def test_changed_parent_registration_refuses_dispatch(self):
        self.fixture.registration["model"]["name"] = "changed-model"
        self.fixture.register()
        self.amendment["parent_registration_sha256"] = ev.file_hash(self.run / "registration.json")
        self.write_amendment()
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.execute(transport=lambda *a: self.fail("changed model sent"))

    def test_stop_file_refuses_without_dispatch(self):
        (self.run / "STOP").write_text("do not remove automatically")
        with self.assertRaisesRegex(ev.EvaluationStopped, "STOP"):
            self.execute(transport=lambda *a: self.fail("STOP ignored"))
        self.assertTrue((self.run / "STOP").exists())
        self.assertEqual(len(ev.load_predictions(self.run)), 1)

    def test_all_three_preregistered_hashes_are_checked(self):
        for field in ("parent_registration_sha256", "frozen_eval_source_sha256", "controller_source_sha256"):
            with self.subTest(field=field):
                saved = self.amendment[field]
                self.amendment[field] = "0" * 64
                self.write_amendment()
                with self.assertRaisesRegex(ValueError, "hash"):
                    self.execute(transport=lambda *a: self.fail("wrong source sent"))
                self.amendment[field] = saved

    def test_frozen_amendment_cannot_be_changed_on_resume(self):
        self.execute()
        self.amendment["workers"] = 3
        self.write_amendment()
        with self.assertRaisesRegex(ValueError, "amendment changed"):
            self.execute(transport=lambda *a: self.fail("amendment silently replaced"))

    def test_labels_attestation_and_existing_scores_block_amendment(self):
        self.amendment["labels_unopened"] = False
        self.write_amendment()
        with self.assertRaisesRegex(ValueError, "latency"):
            self.execute()
        self.amendment["labels_unopened"] = True
        self.write_amendment()
        (self.run / "summary.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "scor"):
            self.execute()

    def test_existing_request_budget_is_not_reset(self):
        original_runner = execution.ev._Runner
        observed = []
        def runner(*args, **kwargs):
            instance = original_runner(*args, **kwargs)
            observed.append(instance.requests)
            return instance
        with patch.object(execution.ev, "_Runner", side_effect=runner):
            self.execute()
        self.assertEqual(observed, [1])

    def test_explicit_v2_uses_its_own_archive_and_preserves_v1(self):
        archive = self.run / "source/frozen_execution.py"
        archive.write_bytes(b"original preregistered controller")
        old_freeze = self.run / "execution-amendment.freeze.json"
        old_status = self.run / "execution-amendment.status.json"
        old_freeze.write_bytes(b"original freeze")
        old_status.write_bytes(b"original status")
        self.fixture.write_json("execution-amendment-v2.json", self.amendment)
        execution.run_execution(self.run, amendment_filename="execution-amendment-v2.json",
                                transport=lambda *a: (200, fixture_module.response()))
        self.assertEqual(archive.read_bytes(), b"original preregistered controller")
        self.assertEqual(old_freeze.read_bytes(), b"original freeze")
        self.assertEqual(old_status.read_bytes(), b"original status")
        self.assertTrue((self.run / "execution-amendment-v2.freeze.json").exists())
        self.assertTrue((self.run / "execution-amendment-v2.status.json").exists())
        self.assertTrue((self.run / "source/frozen_execution-execution-amendment-v2.py").exists())


if __name__ == "__main__":
    unittest.main()
