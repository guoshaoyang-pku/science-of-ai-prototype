"""Explicit transport repair tests; all physical requests use a fake transport."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from aiq_kb import frozen_eval as ev
from aiq_kb import frozen_recovery as repair


spec = importlib.util.spec_from_file_location("recovery_fixture", Path(__file__).with_name("test_frozen_eval.py"))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


class FrozenRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.FrozenEvaluationTests("test_successful_raw_is_saved_and_resume_never_resends")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.run
        self.fixture.execute(max_jobs=2)
        registration, jobs, lookups, freeze = ev._contract(self.run)
        self.jobs = [self.fixture.jobs[6], self.fixture.jobs[3]]
        self.targets = []
        for job, token_counts in zip(self.jobs, ([80, 81], [62])):
            remaining = list(token_counts)
            def interrupted(*args):
                if remaining:
                    return 200, fixture_module.response(tokens=remaining.pop(0), finish="tool_calls",
                                                         tools=fixture_module.tool())
                raise ev.TransportFailure("transport_error")
            runner = ev._Runner(self.run, registration, jobs, lookups, freeze, interrupted)
            with self.assertRaises(ev.EvaluationStopped):
                runner.solve(job)
            path = self.run / "requests" / job["key"] / f"call_{len(token_counts) + 1:02d}"
            request, status = ev._json(path / "request.json"), ev._json(path / "status.json")
            self.targets.append({"path": path.relative_to(self.run).as_posix(),
                "request_sha256": ev.file_hash(path / "request.json"),
                "status_sha256": ev.file_hash(path / "status.json"),
                "payload_sha256": request["sha256"],
                "reserved_completion_tokens": status["reserved_completion_tokens"]})
        (self.run / "STOP").write_text("operator stopped the previous controller")
        self.plan = {"schema_version": 1,
            "parent_registration_sha256": ev.file_hash(self.run / "registration.json"),
            "frozen_eval_source_sha256": ev.file_hash(Path(ev.__file__)),
            "repair_source_sha256": ev.file_hash(Path(repair.__file__)),
            "basis": "transport_only", "labels_unopened": True,
            "reason": "Transport failed; no labels or scores inspected.", "external_requests": 1,
            "external_reserved_completion_tokens": 3000, "max_additional_requests": 2,
            "max_additional_reserved_completion_tokens": 5777, "repairs": self.targets}
        self.write_plan()

    def write_plan(self, name="recovery-plan.json"):
        self.fixture.write_json(name, self.plan)

    def execute(self, **kwargs):
        return repair.run_recovery(self.run, transport=kwargs.pop("transport", lambda *a:
            (200, fixture_module.response())), **kwargs)

    def test_exact_payload_retry_preserves_before_and_every_successful_artifact(self):
        before = {p: p.read_bytes() for p in self.run.glob("requests/*/call_*/response.raw")}
        before.update({p: p.read_bytes() for p in self.run.glob("predictions/*.json")})
        target_before = [(self.run / t["path"] / "request.json").read_bytes() for t in self.targets]
        status_before = [(self.run / t["path"] / "status.json").read_bytes() for t in self.targets]
        payloads = []
        result = self.execute(transport=lambda endpoint, payload, *a:
            (payloads.append(payload) or (200, fixture_module.response())))
        self.assertEqual(len(payloads), 2)
        self.assertEqual([ev.digest(ev.encoded(p)) for p in payloads], [t["payload_sha256"] for t in self.targets])
        self.assertTrue(all(p.read_bytes() == raw for p, raw in before.items()))
        attempts = sorted((self.run / "recovery/recovery-plan/attempts").glob("*"))
        self.assertEqual(len(attempts), 2)
        for index, directory in enumerate(attempts):
            self.assertEqual((directory / "before/request.json").read_bytes(), target_before[index])
            self.assertEqual((directory / "before/status.json").read_bytes(), status_before[index])
            self.assertEqual((directory / "response.raw").read_bytes(), fixture_module.response())
        self.assertEqual(result["repair_attempts"], 2)
        self.assertEqual(result["original_unknown_reserved_completion_tokens"], 5777)
        self.assertEqual(result["external_reserved_completion_tokens"], 3000)
        self.assertLessEqual(result["worstcase_total_requests"], 54)
        self.assertEqual(result["strict_budget_excluded_qids"], ["q_a"])
        self.execute(transport=lambda *a: self.fail("successful repair was resent"))
        # Original runtime can reconstruct both complete solves solely from saved raw.
        (self.run / "STOP").unlink()
        registration, jobs, lookups, freeze = ev._contract(self.run)
        runner = ev._Runner(self.run, registration, jobs, lookups, freeze,
                            lambda *a: self.fail("original successful calls resent"))
        for job in self.jobs:
            self.assertTrue(runner.solve(job)["valid"])

    def test_failure_cost_is_kept_and_same_plan_never_automatically_retries(self):
        def failed(*args):
            raise ev.TransportFailure("transport_error")
        with self.assertRaises(ev.EvaluationStopped):
            self.execute(transport=failed)
        attempts = list((self.run / "recovery/recovery-plan/attempts").glob("*/request.json"))
        self.assertEqual(len(attempts), 1)
        with self.assertRaisesRegex(ev.EvaluationStopped, "already attempted"):
            self.execute(transport=lambda *a: self.fail("failed repair auto-retried"))
        budget = ev._json(self.run / "recovery-budget.json")
        self.assertEqual(budget["repair_unknown_reserved_completion_tokens"], 2839)
        self.assertEqual(budget["repair_attempts"], 1)
        self.assertEqual(budget["original_unknown_reserved_completion_tokens"], 5777)
        self.plan.update(repairs=self.targets[:1], max_additional_requests=1,
                         max_additional_reserved_completion_tokens=2839)
        self.write_plan("recovery-plan-v2.json")
        self.execute(plan_filename="recovery-plan-v2.json")
        budget = ev._json(self.run / "recovery-budget.json")
        self.assertEqual(budget["repair_attempts"], 2)
        self.assertEqual(budget["repair_unknown_reserved_completion_tokens"], 2839)

    def test_saved_repair_raw_survives_crash_before_publish_without_recall(self):
        original_write = ev._write
        def interrupted(path, value):
            if path.name == "status.json" and value.get("state") == "response_saved":
                raise KeyboardInterrupt()
            return original_write(path, value)
        with patch.object(ev, "_write", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.execute()
        attempt = self.run / "recovery/recovery-plan/attempts/0001"
        self.assertTrue((attempt / "response.raw").exists())
        self.assertFalse((self.run / self.targets[0]["path"] / "response.raw").exists())
        calls = []
        self.execute(transport=lambda *a: (calls.append(a) or (200, fixture_module.response())))
        self.assertEqual(len(calls), 1)
        self.assertEqual(ev._json(attempt / "status.json")["state"], "response_saved")

    def test_only_exact_unknown_transport_errors_without_raw_are_eligible(self):
        target = self.run / self.targets[1]["path"]
        status = ev._json(target / "status.json")
        status["reason"] = "http_error"
        ev._write(target / "status.json", status)
        self.targets[1]["status_sha256"] = ev.file_hash(target / "status.json")
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "transport_error"):
            self.execute(transport=lambda *a: self.fail("ineligible plan partly executed"))
        self.assertFalse((self.run / self.targets[0]["path"] / "response.raw").exists())

    def test_hash_or_budget_mismatch_refuses_before_network(self):
        self.plan["repair_source_sha256"] = "0" * 64
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "hash"):
            self.execute()
        self.plan["repair_source_sha256"] = ev.file_hash(Path(repair.__file__))
        self.plan["external_requests"] = 3
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "ceiling"):
            self.execute(transport=lambda *a: self.fail("over-budget request sent"))

    def test_stop_and_no_grader_access(self):
        (self.run / "STOP").unlink()
        with self.assertRaisesRegex(ev.EvaluationStopped, "STOP"):
            self.execute()
        (self.run / "STOP").write_text("stop")
        (self.run / "grader.json").unlink()
        self.execute()

    def test_changed_primary_payload_or_existing_raw_is_never_overwritten(self):
        target = self.run / self.targets[0]["path"]
        (target / "response.raw").write_bytes(b"existing durable raw")
        with self.assertRaisesRegex(ValueError, "raw"):
            self.execute(transport=lambda *a: self.fail("existing raw overwritten"))
        self.assertEqual((target / "response.raw").read_bytes(), b"existing durable raw")

    def test_original_payload_hash_is_checked_before_any_repair(self):
        target = self.run / self.targets[0]["path"] / "request.json"
        record = ev._json(target)
        record["payload"]["temperature"] = .5
        ev._write(target, record)
        with self.assertRaisesRegex(ValueError, "payload hash"):
            self.execute(transport=lambda *a: self.fail("changed payload sent"))

    def test_failed_http_attempt_has_no_raw_body_and_is_still_reserved(self):
        with self.assertRaises(ev.EvaluationStopped):
            self.execute(transport=lambda *a: (503, b"private provider failure body"))
        attempt = self.run / "recovery/recovery-plan/attempts/0001"
        self.assertFalse((attempt / "response.raw").exists())
        status = ev._json(attempt / "status.json")
        self.assertEqual(status["http_status"], 503)
        self.assertEqual(status["state"], "unknown")
        self.assertEqual(ev._json(self.run / "recovery-budget.json")["repair_unknown_reserved_completion_tokens"], 2839)
        self.assertNotIn("private provider", (attempt / "status.json").read_text())


if __name__ == "__main__":
    unittest.main()
