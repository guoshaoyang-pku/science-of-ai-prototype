import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("prepare_frozen_v15", Path(__file__).parents[1] / "tools/prepare_frozen_v15.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PublicPoolTests(unittest.TestCase):
    def pack(self):
        return {"questions": [{"id": "q_one"}], "byId": {"q_one": {
            "id": "q_one", "datasetId": "dataset", "profile": "v1.5", "family": "regression",
            "numChoices": 2, "type": "architecture", "evaluation": {"n_seeds": 10},
            "detail": {"prompt": "Public question: <answer>A</answer>", "llmCot": "PRIVATE"},
            "reveal": {"correctLetter": "B", "private_measurement": 1234}, "llmConsensusAcc": 0.9}}}

    def test_answers_and_historical_reasoning_never_enter_public_projection(self):
        public, grader = module.public_pool(self.pack())
        self.assertEqual(set(public[0]), {"qid", "group", "family", "prompt", "num_choices", "type"})
        self.assertEqual(grader["q_one"], dict(task="select", answer="B", num_choices=2))
        self.assertNotIn("PRIVATE", str(public))
        self.assertNotIn("1234", str(public))

    def test_duplicate_questions_are_rejected(self):
        pack = self.pack()
        pack["questions"] *= 2
        with self.assertRaises(ValueError):
            module.public_pool(pack)

    def test_wrong_seed_profile_is_rejected(self):
        pack = self.pack()
        pack["byId"]["q_one"]["evaluation"]["n_seeds"] = 1
        with self.assertRaises(ValueError):
            module.public_pool(pack)
