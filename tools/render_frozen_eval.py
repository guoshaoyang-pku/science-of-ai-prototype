"""Render a saved N/F/L public comparison without calling models or opening its grader.

Usage: python -X utf8 tools/render_frozen_eval.py --run DIR --legacy-run DIR
"""
from __future__ import annotations

import argparse
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
        if status.get("state") != "success":
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
    complete = len(predictions) == len(jobs) and set(scores) == {j["key"] for j in jobs}
    summary = load(run / "summary.json") if complete else None
    data["availability"] = (f"公开冻结比较；预测 {len(predictions)}/{len(jobs)}，评分 {len(scores)}/{len(jobs)}；"
                            f"缺失 {len(jobs)-len(predictions)}，未评分不是 0 分。状态：{', '.join(sorted(set(failures))) or load(run / 'status.json', {}).get('state', '未知')}。")
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
        remaining = [r for r in per_question if r["qid"] not in overlap]
        if len(per_question) == len(questions) and remaining:
            options = registration.get("grading", {})
            sensitivity = {"excluded_known_qids": sorted(overlap), "n_questions": len(remaining), "comparisons": {
                f"{a}_to_{b}": paired_summary([(r["group"], r["means"][b]-r["means"][a]) for r in remaining],
                    options.get("bootstrap_seed", 20261026), options.get("bootstrap_replicates", 10000))
                for a, b in (("N", "F"), ("N", "L"), ("F", "L"))}}
            save(run / "sensitivity.json", sensitivity)
            data["summary"] += f"；排除已知 {len(overlap)} 道重叠题后剩 {len(remaining)} 题，配对敏感性见 sensitivity.json；不代表曝光已完整核实。"
            data["summary"] += " " + "; ".join(f"{name}: delta={result['delta']:+.5f}, group SE={result['cluster_se']}, 95%={result['bootstrap_95']}"
                                               for name, result in sensitivity["comparisons"].items())
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
