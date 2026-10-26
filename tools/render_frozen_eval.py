"""Render a saved N/F/L public comparison without calling models or opening its grader.

Usage: python -X utf8 tools/render_frozen_eval.py --run DIR --legacy-run DIR
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import statistics

from aiq_kb.build_run_pages import make_data, public_value, render, trace
from aiq_kb.frozen_eval import ARMS, paired_summary


def load(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def runtime_budget(run, registration):
    """Operational counts only; repaired primary usage is already included once."""
    requests = sorted((run / "requests").glob("*/call_*/request.json"))
    primary = {p.parent.relative_to(run).as_posix(): load(p.parent / "status.json", {}) for p in requests}
    attempts = sorted((run / "recovery").glob("*/attempts/*/request.json"))
    original_unknown = {}
    for path in attempts:
        before = load(path.parent / "before/status.json", {})
        if before.get("state") == "unknown":
            original_unknown[load(path)["target"]] = before
    current_unknown = {key: status for key, status in primary.items() if status.get("state") == "unknown"}
    history = {**original_unknown, **current_unknown}
    resolved = {"success", "recovered_raw", "response_saved"}
    limits, expected = registration.get("limits", {}), registration.get("expected", {})
    continuation = registration.get("provenance", {}).get("continuation", {})
    external = load(run / "recovery/external-budget.json", {})
    parent_requests = external.get("requests", continuation.get("parent_requests", 0))
    parent_reserved = external.get("reserved_completion_tokens", parent_requests * limits.get("max_tokens", 0))
    checkpoint = load(run / "recovery-budget.json", {}).get("original_requests")
    ceiling = continuation.get("total_experiment_request_ceiling_including_parent", limits.get("max_requests"))
    known = [s.get("usage", {}).get("completion_tokens") for s in primary.values() if s.get("state") in resolved]
    physical = len(requests) + len(attempts) + parent_requests
    return {"as_of_utc": datetime.now(timezone.utc).isoformat(), "primary_request_slots": len(requests),
            "extra_repair_attempts": len(attempts), "parent_requests": parent_requests,
            "physical_requests_including_parent": physical, "physical_request_ceiling": ceiling,
            "physical_requests_remaining": ceiling - physical if ceiling is not None else None,
            "registered_primary_request_ceiling": limits.get("max_requests"),
            "max_completion_tokens_per_solve": limits.get("max_tokens"),
            "nominal_solve_completion_ceiling": expected.get("question_count", 0) * len(expected.get("arms", [])) * expected.get("repeats", 0) * limits.get("max_tokens", 0),
            "known_completion_tokens": sum(value for value in known if type(value) is int and value >= 0),
            "original_unknown_reserved_completion_tokens": sum(s.get("reserved_completion_tokens", 0) for s in original_unknown.values()),
            "currently_unknown_reserved_completion_tokens": sum(s.get("reserved_completion_tokens", 0) for s in current_unknown.values()),
            "parent_unknown_reserved_completion_tokens": parent_reserved,
            "in_flight_reserved_completion_tokens": sum(s.get("reserved_completion_tokens", 0) for s in primary.values() if s.get("state") == "sent"),
            "currently_unknown_primary_requests": len(current_unknown),
            "historical_unknown_primary_requests": len(history),
            "historical_primary_unknown_rate": len(history) / len(requests) if requests else None,
            "recovery_checkpoint_primary_slots": checkpoint,
            "recovery_checkpoint_unknown_requests": len(original_unknown),
            "recovered_primary_requests": sum(primary.get(key, {}).get("state") in resolved for key in original_unknown),
            "recovery_checkpoint_failure_rate": len(original_unknown) / checkpoint if checkpoint else None,
            "note": "Known completion sums resolved primary statuses once, including repaired responses; original unknown and parent reservations are separate bounds, not measured usage. Counts are a live filesystem snapshot."}


def sensitivity_summary(per_question, excluded, options):
    """Remove entire questions from the complete cohort, independently per filter."""
    if not set(excluded) <= {r["qid"] for r in per_question}:
        raise ValueError("sensitivity exclusion is outside the complete question set")
    remaining = [r for r in per_question if r["qid"] not in excluded]
    return {"excluded_qids": sorted(excluded), "n_questions": len(remaining),
            "scope": "entire_question_all_arms_all_repeats_excluded", "comparisons": {
                f"{a}_to_{b}": paired_summary([(r["group"], r["means"][b]-r["means"][a]) for r in remaining],
                    options.get("bootstrap_seed", 20261026), options.get("bootstrap_replicates", 10000))
                for a, b in (("N", "F"), ("N", "L"), ("F", "L"))} if remaining else {}}


def saved_calls(run, key):
    turns, final, usage, opened = [], "", {}, []
    paths = sorted((run / "requests" / key).glob("call_*"))
    outputs = {m["tool_call_id"]: m.get("content", "") for path in paths
               for m in load(path / "request.json", {}).get("payload", {}).get("messages", []) if m.get("role") == "tool"}
    for path in paths:
        raw = (path / "response.raw").read_text(encoding="utf-8") if (path / "response.raw").exists() else ""
        try:
            response = json.loads(raw)
            message = response.get("choices", [{}])[0].get("message", {})
        except (ValueError, IndexError, AttributeError):
            response, message = {}, {}
        calls = message.get("tool_calls") or []
        for call in calls:
            if call.get("function", {}).get("name") == "kb_open":
                try:
                    opened.extend(json.loads(call["function"]["arguments"]).get("targets", []))
                except (ValueError, KeyError, TypeError):
                    pass
        turns.append({"turn": len(turns), "content": raw or json.dumps(load(path / "status.json", {})),
                      "calls": calls, "results": [outputs.get(c.get("id"), "未取得工具返回。") for c in calls]})
        final, usage = message.get("content") or final, response.get("usage") or usage
    return trace(turns), final, usage, opened


def build(run, legacy):
    if run == legacy:
        raise ValueError("output run must differ from the preserved historical run")
    registration, jobs = load(run / "registration.json"), rows(run / "jobs.jsonl")
    questions = {q["qid"]: q for q in rows(run / "questions.jsonl")}
    predictions = {p.stem: load(p) for p in (run / "predictions").glob("*.json")}
    scores = {r["key"]: r for r in rows(run / "scores.jsonl")}
    for arm in ("F", "L"):
        snapshot = load(run / f"kb_{arm}.json")
        (run / "kb").mkdir(exist_ok=True)
        save(run / "kb" / f"{arm}.json", snapshot)
        for identifier, report in snapshot.get("reports", {}).items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or not re.fullmatch(r"[0-9a-f]{40,64}", report["repo_commit"]):
                raise ValueError("invalid frozen report identity")
            relative = Path("reports") / identifier / (report["repo_commit"] + ".md")
            (run / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(legacy / relative, run / relative)
    data = make_data(run, include_test=False)
    data.update(commits=[], epochs=[], defaultSnap="L", defaultTab="overview", home="history.html",
                flow=["公开 v1.5 冻结比较：N=无 KB；F=历史种子 KB；L=历史最新 KB。",
                      "epoch 0/1/2 仅映射 N/F/L；每题三次重复，不代表学习迭代。",
                      "答案在完整评分后才可见；未评分或缺失保持未知。",
                      "本 run 没有发现、研究或 KB 修改；历史演化另见 history.html。"])
    data["snaps"].insert(0, dict(label="N", epoch=0, claims=[], render="", complete=True, note="无 KB 对照。"))
    failures = []
    for path in (run / "requests").glob("*/call_*/status.json"):
        status = load(path, {})
        if status.get("state") not in ("success", "recovered_raw", "response_saved"):
            failures.append(f"{status.get('state', '未知')} HTTP {status.get('http_status', '未知')}")
    for epoch, arm in enumerate(ARMS):
        records = []
        for job in (j for j in jobs if j["arm"] == arm):
            prediction, q = predictions.get(job["key"]), questions[job["qid"]]
            turns, solution, usage, opened = saved_calls(run, job["key"])
            record = dict(question_id=f"{job['qid']}__r{job['rep']}", original_qid=job["qid"],
                          source=f"公开 v1.5 / {arm} / rep{job['rep']}", family=job["family"], group=job["group"],
                          task=q.get("task", "select"), question=q["prompt"], solver_snapshot=arm,
                          prediction=prediction.get("answer") if prediction else "缺失", answer="未读取 grader",
                          solution=solution or "缺失：尚无保存的模型输出。", solve_usage=usage, solve_trace=turns,
                          cited=re.findall(r"K\d+", " ".join(re.findall(r"<cited>(.*?)</cited>", solution, re.S))),
                          kb_retrieved=[x for x in opened if isinstance(x, str) and x.startswith("K")],
                          kb_opened=opened, discover_trace=[], discover_effort="无发现阶段", comment="公开冻结比较；没有 KB 修改。")
            if prediction and job["key"] in scores:
                record["score"] = scores[job["key"]]["score"]
            records.append(record)
        metrics = dict(n=len(records), ops={})
        if records and all("score" in r for r in records):
            metrics["mean_score"] = statistics.mean(r["score"] for r in records)
        data["epochs"].append(dict(epoch=epoch, records=records, metrics=metrics, jobs=[], summarizer=[],
                                   science="公开冻结对照；无新研究、发现或 KB 修改。", scienceError="", research=[]))
    complete = set(predictions) == set(scores) == {j["key"] for j in jobs}
    summary = load(run / "summary.json") if complete else None
    execution = load(run / "execution-amendment-v2.status.json", load(run / "execution-amendment.status.json", load(run / "status.json", {})))
    data["availability"] = (f"公开冻结比较；预测 {len(predictions)}/{len(jobs)}，评分 {len(scores)}/{len(jobs)}；"
                            f"缺失 {len(jobs)-len(predictions)}，未评分不是 0 分。状态：{', '.join(sorted(set(failures))) or execution.get('state', '未知')}。")
    data["summary"] = json.dumps({k: summary[k] for k in ("n_questions", "n_jobs", "arms", "comparisons") if k in summary}, ensure_ascii=False) if summary else "尚无完整评分与配对结论。"
    if summary:
        data["uncertaintyNote"] = "本次为500道公开选择题；每题先平均3次，±为按dataset组聚类的1 SE；备注列给出组bootstrap 95%区间。"
        data["evals"] = {arm: dict(n=len(questions), repeats=3, mean=summary["arms"][arm]["mean"],
                                  note="冻结公开基准；完整分母") for arm in ARMS}
        data["comparisons"] = []
        for left, right in (("N", "F"), ("N", "L"), ("F", "L")):
            comparison = summary["comparisons"][f"{left}_to_{right}"]
            data["comparisons"].append(dict(label=f"{right} − {left}", n=len(questions), repeats=3,
                mean=summary["arms"][right]["mean"], delta=comparison["delta"], se=comparison["cluster_se"],
                note=f"group bootstrap 95% {comparison['bootstrap_95']}；公开题、历史曝光不完整"))
        overlap = set(load(run / "exposure.json", {}).get("historical_qid_overlap", []))
        per_question = rows(run / "per_question.jsonl")
        if len(per_question) == len(questions) and {r["qid"] for r in per_question} == set(questions):
            options = registration.get("grading", {})
            sensitivity = sensitivity_summary(per_question, overlap, options)
            sensitivity["excluded_known_qids"] = sorted(overlap)
            save(run / "sensitivity.json", sensitivity)
            data["summary"] += f"；排除已知 {len(overlap)} 道重叠题后剩 {sensitivity['n_questions']} 题，配对敏感性见 sensitivity.json；不代表曝光已完整核实。"
            data["summary"] += " " + "; ".join(f"{name}: delta={result['delta']:+.5f}, group SE={result['cluster_se']}, 95%={result['bootstrap_95']}"
                                               for name, result in sensitivity["comparisons"].items())
            excluded = set(load(run / "recovery-budget.json", {}).get("strict_budget_excluded_qids", []))
            if excluded:
                strict = sensitivity_summary(per_question, excluded, options)
                strict["exclusion_basis"] = "transport_only_recovery_registered_before_grading"
                save(run / "strict-budget-sensitivity.json", strict)
                data["strict_budget_sensitivity"] = strict
                data["summary"] += f"；严格预算单独排除 {len(excluded)} 道重试题，保留 {strict['n_questions']} 题全部 arm/reps；该过滤不叠加历史重叠过滤。"
                for name, result in strict["comparisons"].items():
                    data["comparisons"].append(dict(label=f"严格预算 {name}", n=strict["n_questions"], repeats=3,
                        delta=result["delta"], se=result["cluster_se"], note=f"整题排除重试；group bootstrap 95% {result['bootstrap_95']}"))
    budget = data["runtime_budget"] = runtime_budget(run, registration)
    data["summary"] += (f"；请求预算：主槽 {budget['primary_request_slots']} + 额外修复 {budget['extra_repair_attempts']} + 父请求 {budget['parent_requests']}"
                        f" = {budget['physical_requests_including_parent']}/{budget['physical_request_ceiling']} 次物理请求；已知 completion {budget['known_completion_tokens']}，"
                        f"原未知预留 {budget['original_unknown_reserved_completion_tokens']}、父请求未知预留 {budget['parent_unknown_reserved_completion_tokens']} 单列；"
                        f"恢复点未知 {budget['recovery_checkpoint_unknown_requests']} / 主槽 {budget['recovery_checkpoint_primary_slots']}，已恢复 {budget['recovered_primary_requests']}，"
                        f"该点失败率 {budget['recovery_checkpoint_failure_rate']}；当前未知 {budget['currently_unknown_primary_requests']}。")
    data = public_value(data)
    save(run / "viewer-data.json", data)
    render(data, run / "viewer.html")
    history = make_data(legacy, include_test=False)
    history.update(home="viewer.html", defaultTab="overview")
    render(history, run / "history.html")
    return {"viewer": str(run / "viewer.html"), "predictions": len(predictions), "expected": len(jobs), "scored": len(scores)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run.resolve(), args.legacy_run.resolve()), ensure_ascii=False))
