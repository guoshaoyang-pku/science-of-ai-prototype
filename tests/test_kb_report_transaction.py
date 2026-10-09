"""A full report and related short claims publish through one proposal."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

from aiq_kb import kb_science_loop as loop


def claim(cid="K1001", text="Measured local observation"):
    return {"id": cid, "text": text, "support_count": 3., "failure_count": 1.,
            "credibility": 4 / 6, "created_epoch": 0, "sources": [], "merged_from": []}


class ReportTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        self.obj = loop.Loop.__new__(loop.Loop)
        self.obj.run = self.run
        self.obj.lock = threading.RLock()
        self.obj.args = SimpleNamespace(model="synthetic-test-model")
        self.obj.worker_path = self.run / "jobs/Jfixture/job.json"
        self.obj.worker_path.parent.mkdir(parents=True)
        self.obj.worker_job = {"id": "Jfixture", "origin_epoch": 3, "payload": {"topic": "fixture"}}
        self.obj.worker_path.write_text(json.dumps({**self.obj.worker_job, "key": None, "status": "running"}))
        self.obj.job_sequence = 0
        self.kb = loop.KB({"epoch": 3, "claims": [claim()]}, self.obj.worker_path.parent)
        self.obj.job_base = copy.deepcopy(self.kb.to_doc())
        self.kb.publish_commit = self.obj.journal_commit
        self.report = {"text": "# Unfinished report\nMeasured fixture. Competing explanations unresolved.",
                       "title": "Fixture theory", "claim_ids": ["K1001"]}

    def tearDown(self):
        self.temp.cleanup()

    def publications(self):
        return list(self.obj.worker_path.parent.glob("publications/p*.json"))

    def test_revision_and_report_publish_once_and_preserve_new_credits(self):
        result = self.obj.propose_changes(self.kb, [{"op": "revise", "claim_ids": ["K1001"],
            "text": "Scoped extrapolatable prediction", "reason": "New controlled evidence"}], self.report)
        outcome = json.loads(result)
        self.assertEqual(outcome["submitted"], 1)
        self.assertEqual(len(self.publications()), 1)
        publication = json.loads(self.publications()[0].read_text())
        self.assertEqual(publication["operation"]["op"], "proposal")
        self.assertEqual([c["op"] for c in publication["operation"]["payload"]["operations"]], ["revise", "report"])
        current = loop.KB({"epoch": 7, "claims": [claim()]}, self.run)
        current.credit({"K1001": (2., 0)}, {"role": "solver"}, "q")
        self.obj.apply_publications(current, 7)
        self.assertEqual(current.claims["K1001"]["text"], "Scoped extrapolatable prediction")
        self.assertEqual(current.claims["K1001"]["support_count"], 5)
        self.assertEqual(current.claims["K1001"]["report_ids"], [outcome["report_id"]])
        report = current.reports[outcome["report_id"]]
        self.assertEqual(report["origin_epoch"], 3)
        self.assertIn("Competing explanations unresolved", self.obj.open_kb(current, ["K1001"]))
        self.assertEqual(self.obj.apply_publications(current, 7), [])

    def test_invalid_later_operation_or_report_leaves_claims_and_repo_untouched(self):
        original = copy.deepcopy(self.kb.to_doc())
        result = self.obj.propose_changes(self.kb, [
            {"op": "revise", "claim_ids": ["K1001"], "text": "Tentative", "reason": "fixture"},
            {"op": "delete", "claim_ids": ["K9999"], "reason": "missing"}], self.report)
        self.assertTrue(result.startswith("rejected"))
        self.assertEqual(self.kb.claims, {c["id"]: c for c in original["claims"]})
        self.assertEqual(self.publications(), [])
        operation = {"op": "revise", "claim_ids": ["K1001"], "text": "fixture",
                     "reason": "fixture", "support_count": 900}
        self.assertTrue(self.obj.propose_changes(self.kb, [operation], self.report).startswith("rejected"))
        self.assertEqual(self.kb.claims["K1001"]["support_count"], 3)
        self.assertFalse((self.obj.worker_path.parent / "repo").exists())
        bad = {**self.report, "claim_ids": ["K9999"]}
        result = self.obj.propose_changes(self.kb, [{"op": "revise", "claim_ids": ["K1001"],
            "text": "Tentative", "reason": "fixture"}], bad)
        self.assertTrue(result.startswith("rejected"))
        self.assertEqual(self.publications(), [])
        self.assertFalse((self.obj.worker_path.parent / "repo").exists())

    def test_reserved_report_and_claim_fields_are_rejected(self):
        for report in ({**self.report, "path": "/tmp/unsafe"}, {**self.report, "repo_commit": "invented"}):
            self.assertTrue(self.obj.propose_changes(self.kb, [], report).startswith("rejected"))
        result = self.obj.propose_changes(self.kb, [{"op": "annotate", "claim_ids": ["K1001"],
            "reason": "fixture", "fields": {"support_count": 900}}], self.report)
        self.assertTrue(result.startswith("rejected"))
        self.assertEqual(self.publications(), [])

    def test_long_scientific_conditions_survive_revision_without_truncation(self):
        text = ("Scoped evidence and necessary conditions. " * 40).strip()
        fields = {"mechanism": ("Competing explanations remain unresolved. " * 30).strip(),
                  "phase": ("Measured boundary and counterexamples. " * 30).strip(),
                  "regime": {"batch_size": 64, "weight_decay": .001,
                             "input_dim": 8, "note": "Necessary condition. " * 40}}
        preserved, error = loop.science_fields(fields)
        self.assertIsNone(error)
        self.assertEqual(preserved, fields)
        outcome = self.obj.propose_changes(self.kb, [{"op": "revise", "claim_ids": ["K1001"],
            "text": text, "reason": "Preserve scope", "fields": preserved}], self.report)
        self.assertEqual(json.loads(outcome)["submitted"], 1)
        self.assertEqual(self.kb.claims["K1001"]["text"], text.strip())
        for key, value in fields.items():
            self.assertEqual(self.kb.claims["K1001"][key], value)

    def test_report_only_transaction_is_available_while_unfinished(self):
        outcome = json.loads(self.obj.propose_changes(self.kb, [], self.report))
        self.assertEqual(outcome["submitted"], 0)
        self.assertEqual(len(self.publications()), 1)
        current = loop.KB({"epoch": 6, "claims": [claim()]}, self.run)
        self.obj.apply_publications(current, 6)
        self.assertIn("Competing explanations unresolved", self.obj.open_kb(current, ["science"]))

    def test_existing_report_reads_claim_semantics_atomically_on_conflict(self):
        self.obj.propose_changes(self.kb, [], self.report)
        current = loop.KB({"epoch": 6, "claims": [claim()]}, self.run)
        self.obj.apply_publications(current, 6)
        report_id = next(iter(current.reports))
        version = current.reports[report_id]["repo_commit"]
        current.claims["K1001"]["text"] = "Newer independently measured condition"
        update = {**self.report, "text": self.report["text"] + "\nNew tentative explanation."}
        self.obj.propose_changes(self.kb, [{"op": "revise", "claim_ids": ["K1001"],
            "text": "Stale claim revision", "reason": "fixture"}], update)
        self.obj.apply_publications(current, 7)
        self.assertEqual(current.claims["K1001"]["text"], "Newer independently measured condition")
        self.assertEqual(current.reports[report_id]["repo_commit"], version)


if __name__ == "__main__":
    unittest.main()
