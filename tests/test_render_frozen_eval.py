import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("render_frozen_eval", Path(__file__).parents[1] / "tools/render_frozen_eval.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_budget_counts_recovered_usage_once_and_preserves_original_unknowns(tmp_path):
    registration = {"limits": {"max_requests": 13499, "max_tokens": 3000},
                    "expected": {"question_count": 500, "arms": ["N", "F", "L"], "repeats": 3},
                    "provenance": {"continuation": {"total_experiment_request_ceiling_including_parent": 13500}}}
    for index, (state, tokens) in enumerate((("success", 665), ("recovered_raw", 57), ("sent", None)), 1):
        root = tmp_path / "requests" / f"q{index}__L__r1" / "call_01"
        write(root / "request.json", {})
        write(root / "status.json", {"state": state, "reserved_completion_tokens": 100,
                                      "usage": {"completion_tokens": tokens} if tokens is not None else {}})
    for index, reserved in enumerate((2839, 2938), 1):
        root = tmp_path / "recovery" / "recovery-plan" / "attempts" / f"{index:04}"
        write(root / "request.json", {"target": f"requests/q{index}__L__r1/call_01"})
        write(root / "before/status.json", {"state": "unknown", "reserved_completion_tokens": reserved})
        write(root / "status.json", {"state": "response_saved", "usage": {"completion_tokens": (665, 57)[index-1]}})
    write(tmp_path / "recovery/external-budget.json", {"requests": 1, "reserved_completion_tokens": 3000})
    write(tmp_path / "recovery-budget.json", {"original_requests": 675})
    result = module.runtime_budget(tmp_path, registration)
    assert result["physical_requests_including_parent"] == 6
    assert result["known_completion_tokens"] == 722
    assert result["original_unknown_reserved_completion_tokens"] == 5777
    assert result["parent_unknown_reserved_completion_tokens"] == 3000
    assert result["currently_unknown_primary_requests"] == 0
    assert result["historical_unknown_primary_requests"] == 2
    assert result["recovery_checkpoint_failure_rate"] == pytest.approx(2 / 675)
    assert result["recovered_primary_requests"] == 2
    assert result["in_flight_reserved_completion_tokens"] == 100


def test_history_468_and_strict_budget_498_are_independent_question_filters():
    rows = [{"qid": f"q{i}", "group": f"g{i}", "means": {"N": 0, "F": .5, "L": 1}} for i in range(500)]
    options = {"bootstrap_seed": 20261026, "bootstrap_replicates": 20}
    history = module.sensitivity_summary(rows, {f"q{i}" for i in range(32)}, options)
    strict = module.sensitivity_summary(rows, {"q498", "q499"}, options)
    assert history["n_questions"] == 468
    assert strict["n_questions"] == 498
    assert strict["comparisons"]["N_to_F"]["delta"] == .5
    assert strict["comparisons"]["F_to_L"]["n_questions"] == 498
    assert strict["excluded_qids"] == ["q498", "q499"]


def test_incomplete_build_never_opens_grader_or_summary(tmp_path, monkeypatch):
    run, legacy = tmp_path / "run", tmp_path / "legacy"
    run.mkdir()
    legacy.mkdir()
    write(run / "registration.json", {"limits": {}, "expected": {}})
    write(run / "kb_F.json", {"claims": [], "reports": {}, "epoch": 0})
    write(run / "kb_L.json", {"claims": [], "reports": {}, "epoch": 15})
    write(run / "recovery-budget.json", {"strict_budget_excluded_qids": ["q1"]})
    (run / "jobs.jsonl").write_text(json.dumps({"key": "q1__N__r1", "qid": "q1", "arm": "N", "rep": 1,
                                                "family": "fixture", "group": "g1"}) + "\n", encoding="utf-8")
    (run / "questions.jsonl").write_text(json.dumps({"qid": "q1", "prompt": "Public question only"}) + "\n", encoding="utf-8")
    original = Path.read_text

    def protected(path, *args, **kwargs):
        if path.name in ("grader.json", "summary.json"):
            raise AssertionError("opened labels or unfinished summary")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", protected)
    monkeypatch.setattr(module, "make_data", lambda *a, **k: {"snaps": [], "reports": {}})
    monkeypatch.setattr(module, "render", lambda *a, **k: None)
    module.build(run, legacy)
    data = json.loads((run / "viewer-data.json").read_text(encoding="utf-8"))
    assert all("score" not in row for epoch in data["epochs"] for row in epoch["records"])
    assert "runtime_budget" in data
    assert not (run / "strict-budget-sensitivity.json").exists()


def test_complete_build_emits_separate_sensitivities_without_opening_grader(tmp_path, monkeypatch):
    run, legacy = tmp_path / "run", tmp_path / "legacy"
    run.mkdir()
    legacy.mkdir()
    options = {"bootstrap_seed": 20261026, "bootstrap_replicates": 20}
    write(run / "registration.json", {"grading": options})
    for arm in ("F", "L"):
        write(run / f"kb_{arm}.json", {"claims": [], "reports": {}, "epoch": 0})
    write(run / "exposure.json", {"historical_qid_overlap": ["q1"]})
    write(run / "recovery-budget.json", {"strict_budget_excluded_qids": ["q2"]})
    questions = [{"qid": f"q{i}", "prompt": "Public question only"} for i in (1, 2, 3)]
    jobs = [{"key": f"{q['qid']}__{arm}__r{rep}", "qid": q["qid"], "arm": arm, "rep": rep,
             "family": "fixture", "group": q["qid"]} for q in questions for arm in module.ARMS for rep in (1, 2, 3)]
    per_question = [{"qid": q["qid"], "group": q["qid"], "means": {"N": 0, "F": .5, "L": 1}} for q in questions]
    for job in jobs:
        write(run / "predictions" / (job["key"] + ".json"), {**job, "answer": "A"})
    for filename, values in (("questions", questions), ("jobs", jobs), ("per_question", per_question),
                             ("scores", [{**j, "score": {"N": 0, "F": .5, "L": 1}[j["arm"]]} for j in jobs])):
        (run / (filename + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in values), encoding="utf-8")
    write(run / "summary.json", {"n_questions": 3, "n_jobs": 27,
                                 "arms": {a: {"mean": v} for a, v in (("N", 0), ("F", .5), ("L", 1))},
                                 "comparisons": module.sensitivity_summary(per_question, set(), options)["comparisons"]})
    original = Path.read_text

    def protected(path, *args, **kwargs):
        if path.name == "grader.json":
            raise AssertionError("renderer opened grader")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", protected)
    monkeypatch.setattr(module, "make_data", lambda *a, **k: {"snaps": [], "reports": {}})
    monkeypatch.setattr(module, "render", lambda *a, **k: None)
    module.build(run, legacy)
    history = json.loads((run / "sensitivity.json").read_text(encoding="utf-8"))
    strict = json.loads((run / "strict-budget-sensitivity.json").read_text(encoding="utf-8"))
    data = json.loads((run / "viewer-data.json").read_text(encoding="utf-8"))
    assert history["n_questions"] == strict["n_questions"] == 2
    assert history["excluded_qids"] == ["q1"]
    assert strict["excluded_qids"] == ["q2"]
    assert len(data["comparisons"]) == 6
    assert data["strict_budget_sensitivity"]["comparisons"]["N_to_L"]["delta"] == 1
