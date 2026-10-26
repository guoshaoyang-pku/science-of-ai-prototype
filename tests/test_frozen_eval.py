"""Offline contract tests; no real endpoint or credentials are used."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiq_kb import frozen_eval as ev


def response(text="<answer>A</answer>", tokens=20, finish="stop", tools=None):
    message = {"role": "assistant", "content": text}
    if tools is not None:
        message.update(content=None, tool_calls=tools)
    return json.dumps({"model": "offline-model", "choices": [{"message": message,
        "finish_reason": finish}], "usage": {"prompt_tokens": 10,
        "completion_tokens": tokens, "total_tokens": tokens + 10}}).encode()


def tool(target="K1"):
    return [{"id": "call_1", "type": "function", "function": {"name": "kb_open",
        "arguments": json.dumps({"targets": [target]})}}]


class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.questions = [{"qid": "q_a", "group": "g_a", "family": "f_a"},
                          {"qid": "q_b", "group": "g_b", "family": "f_b"}]
        self.jobs = [{**q, "key": f"{q['qid']}__{arm}__r{rep}", "arm": arm, "rep": rep,
                      "messages": [{"role": "user", "content": "Choose A or B."}]}
                     for q in self.questions for arm in ("N", "F", "L") for rep in (1, 2, 3)]
        self.write_jsonl("questions.jsonl", self.questions)
        self.write_jsonl("jobs.jsonl", self.jobs)
        for arm in ("N", "F", "L"):
            self.write_json(f"lookup_{arm}.json", {} if arm == "N" else {"K1": "Measured rule."})
        self.write_json("grader.json", {q["qid"]: {"task": "select", "answer": "A", "num_choices": 2}
                                         for q in self.questions})
        self.registration = {"schema_version": 1, "source_sha256": ev.file_hash(Path(ev.__file__)),
            "model": {"name": "offline-model", "endpoint": "https://example.invalid/v1/chat/completions",
                      "api_key_env": "FROZEN_EVAL_TEST_KEY"},
            "limits": {"max_tokens": 3000, "max_calls_per_solve": 3, "max_requests": 54,
                       "workers": 1},
            "expected": {"question_count": 2, "arms": ["N", "F", "L"], "repeats": 3},
            "inputs": {"questions": self.pin("questions.jsonl"), "jobs": self.pin("jobs.jsonl"),
                       "kb_lookup": {arm: self.pin(f"lookup_{arm}.json") for arm in ("N", "F", "L")},
                       "grader": self.pin("grader.json")},
            "grading": {"bootstrap_seed": 42, "bootstrap_replicates": 100}}
        self.write_json("registration.json", self.registration)
        self.env = patch.dict(os.environ, {"FROZEN_EVAL_TEST_KEY": "synthetic-not-a-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def write_json(self, name, value):
        (self.run / name).write_text(json.dumps(value), encoding="utf-8")

    def write_jsonl(self, name, rows):
        (self.run / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def pin(self, name):
        return {"path": name, "sha256": hashlib.sha256((self.run / name).read_bytes()).hexdigest()}

    def register(self):
        self.registration["inputs"]["jobs"] = self.pin("jobs.jsonl")
        self.write_json("registration.json", self.registration)

    def execute(self, **kwargs):
        return ev.run_evaluation(self.run, transport=kwargs.pop("transport", lambda *a: (200, response())),
                                 **kwargs)

    def test_successful_raw_is_saved_and_resume_never_resends(self):
        calls = []
        self.execute(max_jobs=1, transport=lambda *a: (calls.append(a) or (200, response())))
        raw = list((self.run / "requests").glob("*/call_*/response.raw"))
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0].read_bytes(), response())
        prediction = next((self.run / "predictions").glob("*.json"))
        prediction.unlink()  # crash after response save, before prediction save
        self.execute(max_jobs=1, transport=lambda *a: self.fail("cached call was resent"))
        self.assertEqual(len(calls), 1)
        record = json.loads(prediction.read_text())
        self.assertEqual(record["answer"], "A")
        self.assertTrue(record["valid"])
        self.assertNotIn("synthetic-not-a-key", "".join(p.read_text() for p in self.run.rglob("*.json")))

    def test_raw_saved_before_success_status_is_recovered_without_credentials(self):
        self.execute(max_jobs=1)
        directory = next((self.run / "requests").glob("*/call_*"))
        (directory / "status.json").write_text(json.dumps({"state": "sent"}))
        next((self.run / "predictions").glob("*.json")).unlink()
        with patch.dict(os.environ, {"FROZEN_EVAL_TEST_KEY": ""}):
            self.execute(max_jobs=1, transport=lambda *a: self.fail("durable raw was resent"))
        self.assertEqual(json.loads((directory / "status.json").read_text())["state"], "success")

    def test_two_tool_rounds_share_one_completion_budget_and_force_final(self):
        payloads = []
        outputs = [response(tokens=1100, finish="tool_calls", tools=tool()),
                   response(tokens=1200, finish="tool_calls", tools=tool()), response(tokens=700)]
        def send(endpoint, payload, key, timeout):
            payloads.append(payload)
            return 200, outputs.pop(0)
        # Put one KB job first, retaining the complete registered Cartesian product.
        self.jobs.insert(0, self.jobs.pop(3))
        self.write_jsonl("jobs.jsonl", self.jobs)
        self.register()
        self.execute(max_jobs=1, transport=send)
        self.assertEqual([p["max_tokens"] for p in payloads], [3000, 1900, 700])
        self.assertEqual(payloads[-1]["tool_choice"], "none")
        self.assertEqual(payloads[0]["thinking"], {"type": "disabled"})
        self.assertEqual(payloads[0]["temperature"], 0)
        self.assertIn("Measured rule.", payloads[1]["messages"][-1]["content"])
        self.assertEqual(next(iter(ev.load_predictions(self.run).values()))["completion_tokens"], 3000)

    def test_repetitions_have_separate_cache_identity(self):
        payloads = []
        self.execute(max_jobs=3, transport=lambda endpoint, payload, *a:
                     (payloads.append(payload) or (200, response())))
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[1], payloads[2])
        predictions = ev.load_predictions(self.run)
        self.assertEqual(len({r["cache_key"] for r in predictions.values()}), 3)
        self.assertTrue(all("tools" not in p for p in payloads))

    def test_stop_file_prevents_dispatch_and_next_tool_round(self):
        self.jobs.insert(0, self.jobs.pop(3))
        self.write_jsonl("jobs.jsonl", self.jobs)
        self.register()
        (self.run / "STOP").write_text("stop")
        with self.assertRaisesRegex(ev.EvaluationStopped, "STOP"):
            self.execute(transport=lambda *a: self.fail("STOP was ignored"))
        (self.run / "STOP").unlink()
        def send(*args):
            (self.run / "STOP").write_text("stop between tool rounds")
            return 200, response(tokens=10, finish="tool_calls", tools=tool())
        with self.assertRaisesRegex(ev.EvaluationStopped, "STOP"):
            self.execute(max_jobs=1, transport=send)
        self.assertEqual(len(list((self.run / "requests").glob("*/call_*/request.json"))), 1)

    def test_last_valid_xml_tag_is_case_insensitive(self):
        self.execute(max_jobs=1, transport=lambda *a: (200, response(
            "<answer>A</answer><AnSwEr>b</aNsWeR><answer>A<B</answer><answer>not a letter</answer>")))
        self.assertEqual(next(iter(ev.load_predictions(self.run).values()))["answer"], "B")

    def test_public_ranking_task_parses_the_last_valid_permutation(self):
        self.questions[0]["task"] = "ranking"
        self.write_jsonl("questions.jsonl", self.questions)
        self.registration["inputs"]["questions"] = self.pin("questions.jsonl")
        self.register()
        self.execute(max_jobs=1, transport=lambda *a: (200, response(
            "<answer>A<B</answer><answer>b<a</answer>")))
        self.assertEqual(next(iter(ev.load_predictions(self.run).values()))["answer"], "B<A")

    def test_tool_result_is_bounded_with_explicit_truncation(self):
        self.write_json("lookup_F.json", {"K1": "x" * 40000})
        self.registration["inputs"]["kb_lookup"]["F"] = self.pin("lookup_F.json")
        self.jobs.insert(0, self.jobs.pop(3))
        self.write_jsonl("jobs.jsonl", self.jobs)
        self.register()
        calls = []
        def send(endpoint, payload, *args):
            calls.append(payload)
            return 200, response(finish="tool_calls", tools=tool()) if len(calls) == 1 else response()
        self.execute(max_jobs=1, transport=send)
        output = calls[1]["messages"][-1]["content"]
        self.assertEqual(len(output), 30000)
        self.assertIn("TRUNCATED", output)

    def test_http_failure_stops_without_body_or_automatic_retry_or_scoring(self):
        calls = []
        def fail(*args):
            calls.append(args)
            raise ev.TransportFailure("http_error", 401)
        with self.assertRaises(ev.EvaluationStopped):
            self.execute(transport=fail)
        with self.assertRaises(ev.EvaluationStopped):
            self.execute(transport=lambda *a: self.fail("failure retried"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(ev.load_predictions(self.run), {})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ev.grade_evaluation(self.run)

    def test_sent_without_raw_is_unknown_and_never_resent(self):
        def crash(*args):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.execute(max_jobs=1, transport=crash)
        with self.assertRaisesRegex(ev.EvaluationStopped, "indeterminate"):
            self.execute(max_jobs=1, transport=lambda *a: self.fail("unknown request resent"))

    def test_actual_truncation_or_bad_xml_counts_zero_in_full_denominator(self):
        sent = 0
        def send(*args):
            nonlocal sent
            sent += 1
            return 200, response("<answer>A</answer>", finish="length") if sent == 1 else response("A")
        self.execute(transport=send)
        predictions = ev.load_predictions(self.run)
        self.assertEqual(len(predictions), 18)
        self.assertTrue(all(not r["valid"] and r["invalid_reason"] for r in predictions.values()))
        summary = ev.grade_evaluation(self.run)
        self.assertEqual(summary["n_jobs"], 18)
        self.assertEqual(summary["arms"]["N"]["mean"], 0)
        self.assertEqual(summary["invalid_jobs"], 18)

    def test_unknown_usage_stops_and_preserves_raw(self):
        raw = json.loads(response())
        del raw["usage"]
        with self.assertRaisesRegex(ev.EvaluationStopped, "usage"):
            self.execute(transport=lambda *a: (200, json.dumps(raw).encode()))
        self.assertEqual(len(list((self.run / "requests").glob("*/call_*/response.raw"))), 1)
        self.assertEqual(ev.load_predictions(self.run), {})

    def test_unknown_model_is_reported_and_request_metadata_is_retained(self):
        raw = json.loads(response())
        del raw["model"]
        self.execute(transport=lambda *a: (200, json.dumps(raw).encode()))
        summary = ev.grade_evaluation(self.run)
        self.assertEqual(summary["unknown_model_responses"], 18)
        self.assertEqual(summary["response_models"], ["unknown"])
        status = json.loads(next((self.run / "requests").glob("*/call_*/status.json")).read_text())
        self.assertEqual(status["usage"]["completion_tokens"], 20)
        self.assertEqual(status["finish_reason"], "stop")
        self.assertEqual(len(status["request_sha256"]), 64)

    def test_extra_request_ledger_entry_refuses_scoring_before_grader(self):
        self.execute()
        directory = self.run / "requests/unregistered/call_01"
        directory.mkdir(parents=True)
        (directory / "request.json").write_text("{}")
        (self.run / "grader.json").unlink()
        with self.assertRaisesRegex(ValueError, "request ledger"):
            ev.grade_evaluation(self.run)

    def test_request_cap_is_persistent_hard_stop(self):
        self.registration["limits"]["max_requests"] = 1
        self.register()
        with self.assertRaisesRegex(ev.EvaluationStopped, "request budget"):
            self.execute()
        self.assertEqual(len(list((self.run / "requests").glob("*/call_*/request.json"))), 1)
        with self.assertRaises(ev.EvaluationStopped):
            self.execute(transport=lambda *a: self.fail("cap was reset"))

    def test_resume_rejects_registration_source_or_input_changes(self):
        self.execute(max_jobs=1)
        self.registration["model"]["name"] = "changed-model"
        self.register()
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.execute()

    def test_source_hash_must_match_before_first_request(self):
        self.registration["source_sha256"] = "0" * 64
        self.register()
        with self.assertRaisesRegex(ValueError, "source"):
            self.execute(transport=lambda *a: self.fail("unregistered source sent"))

    def test_corrupt_saved_raw_is_rejected_without_recall(self):
        self.execute(max_jobs=1)
        raw = next((self.run / "requests").glob("*/call_*/response.raw"))
        raw.write_bytes(b"changed")
        next((self.run / "predictions").glob("*.json")).unlink()
        with self.assertRaisesRegex(ValueError, "response hash"):
            self.execute(max_jobs=1, transport=lambda *a: self.fail("corruption resent"))

    def test_changed_prediction_is_rejected_before_grader_is_opened(self):
        self.execute()
        path = next((self.run / "predictions").glob("*.json"))
        record = json.loads(path.read_text())
        record["answer"] = "B"
        path.write_text(json.dumps(record))
        (self.run / "grader.json").unlink()
        with self.assertRaisesRegex(ValueError, "prediction"):
            ev.grade_evaluation(self.run)

    def test_concurrent_workers_obey_persistent_global_request_cap(self):
        self.registration["limits"].update(workers=6, max_requests=4)
        self.register()
        with self.assertRaises(ev.EvaluationStopped):
            self.execute()
        self.assertEqual(len(list((self.run / "requests").glob("*/call_*/request.json"))), 4)

    def test_public_input_change_is_rejected_even_with_unchanged_registration(self):
        self.execute(max_jobs=1)
        self.write_jsonl("questions.jsonl", self.questions[::-1])
        with self.assertRaisesRegex(ValueError, "input hash"):
            self.execute(transport=lambda *a: self.fail("changed public input sent"))

    def test_missing_extra_duplicate_or_unknown_job_is_rejected(self):
        for label, rows in (("missing", self.jobs[:-1]), ("duplicate", self.jobs + [self.jobs[0]]),
                            ("unknown", [{**self.jobs[0], "qid": "other"}] + self.jobs[1:])):
            with self.subTest(label=label):
                self.write_jsonl("jobs.jsonl", rows)
                self.register()
                with self.assertRaises(ValueError):
                    self.execute(transport=lambda *a: self.fail("bad job sent"))

    def test_grader_is_never_opened_before_every_prediction_exists(self):
        (self.run / "grader.json").unlink()
        self.execute(max_jobs=1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ev.grade_evaluation(self.run)

    def test_complete_grading_is_paired_reproducible_and_checks_extra_predictions(self):
        self.execute()
        summary = ev.grade_evaluation(self.run)
        self.assertEqual(summary, ev.grade_evaluation(self.run))
        self.assertEqual(summary["n_questions"], 2)
        self.assertEqual(summary["comparisons"]["N_to_F"]["delta"], 0)
        self.assertEqual(summary["comparisons"]["N_to_F"]["n_groups"], 2)
        self.assertEqual(set(summary["by_family"]), {"f_a", "f_b"})
        extra = next((self.run / "predictions").glob("*.json")).read_text()
        (self.run / "predictions/extra.json").write_text(extra)
        with self.assertRaisesRegex(ValueError, "prediction"):
            ev.grade_evaluation(self.run)

    def test_cluster_uncertainty_uses_group_units(self):
        # Duplicating perfectly correlated variants must not invent independent groups.
        measured = ev.paired_summary([("g1", 1.), ("g1", 1.), ("g2", -1.), ("g2", -1.)], 7, 300)
        self.assertEqual(measured["delta"], 0)
        self.assertEqual(measured["cluster_se"], 1)
        self.assertEqual(measured["bootstrap_95"], [-1., 1.])
        self.assertEqual(measured, ev.paired_summary([("g1", 1.), ("g1", 1.), ("g2", -1.), ("g2", -1.)], 7, 300))
        self.assertIsNone(ev.paired_summary([("same", 1.), ("same", 0.)], 7, 300)["cluster_se"])

    def test_ranking_partial_credit_and_invalid_permutations(self):
        rubric = {"task": "ranking", "answer": "A<B<C<D<E", "num_choices": 5}
        self.assertEqual(ev.score_answer("B<A<C<D<E", rubric), .75)
        self.assertEqual(ev.score_answer("A<A<C<D<E", rubric), 0)
        self.assertEqual(ev.score_answer("E<D<C<B<A", rubric), 0)

    def test_https_only_and_redirect_rejection(self):
        self.registration["model"]["endpoint"] = "http://example.invalid/v1/chat/completions"
        self.register()
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            self.execute()
        self.assertIsNone(ev.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://else.invalid"))


if __name__ == "__main__":
    unittest.main()
