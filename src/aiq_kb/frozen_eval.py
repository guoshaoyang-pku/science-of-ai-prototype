"""Frozen, resumable N/F/L evaluation using only the Python standard library.

Run ``python -m aiq_kb.frozen_eval run --run-dir DIR --max-jobs 1`` for the
registered first job, then omit --max-jobs to finish. ``grade`` opens the
separate grader only after the exact registered prediction set is complete.
No command retries a sent request without a saved response. Such a request
has unknown cost/outcome and requires an explicit, separately reviewed repair.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import ssl
import statistics
import threading
import urllib.error
import urllib.parse
import urllib.request


ARMS = ("N", "F", "L")
JOB_FIELDS = {"key", "qid", "group", "family", "arm", "rep", "messages"}
TOOL = {"type": "function", "function": {"name": "kb_open",
    "description": "Open frozen knowledge-base entries or reports by target ID.",
    "parameters": {"type": "object", "properties": {"targets": {"type": "array",
        "items": {"type": "string"}}}, "required": ["targets"], "additionalProperties": False}}}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def _atomic(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write(path, value):
    _atomic(path, encoded(value))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _run_lock(run):
    """OS locks release after a crash; the harmless lock file may remain."""
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".frozen_eval.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise EvaluationStopped("another evaluator holds the run lock") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class EvaluationStopped(RuntimeError):
    """Incomplete evaluation; its unknown outcomes must never become zero scores."""


class TransportFailure(Exception):
    def __init__(self, kind, status=None):
        self.kind, self.status = kind, status
        super().__init__(kind)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_transport(endpoint, payload, key, timeout):
    """TLS verification stays enabled. Failure bodies and exception text are discarded."""
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(
        context=ssl.create_default_context()))
    request = urllib.request.Request(endpoint, encoded(payload), method="POST", headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise TransportFailure("response_too_large")
            return response.status, raw
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        raise TransportFailure("http_error", code) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise TransportFailure("transport_error") from None


def _pin_path(run, pin):
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"} or not isinstance(pin["path"], str):
        raise ValueError("inputs require path and sha256")
    if not isinstance(pin["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", pin["sha256"]):
        raise ValueError("invalid input sha256")
    return (run / pin["path"]).resolve()


def _read_pin(run, pin):
    path = _pin_path(run, pin)
    raw = path.read_bytes()
    if digest(raw) != pin["sha256"]:
        raise ValueError(f"frozen input hash mismatch: {path.name}")
    return raw


def _rows(raw):
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def _positive(value, maximum, name):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"invalid {name}: expected integer from 1 to {maximum}")


def _contract(run):
    """Verify public inputs and freeze source/configuration; never read the grader."""
    registration_raw = (run / "registration.json").read_bytes()
    registration = json.loads(registration_raw)
    if registration.get("schema_version") != 1:
        raise ValueError("unsupported registration schema")
    source = Path(__file__).read_bytes()
    inputs = registration["inputs"]
    source_hash = registration.get("source_sha256")
    if "source" in inputs:
        if _read_pin(run, inputs["source"]) != source:
            raise ValueError("registered source differs from executing source")
        source_hash = inputs["source"]["sha256"]
    if source_hash != digest(source):
        raise ValueError("registered source hash differs from executing source")
    freeze = {"registration_sha256": digest(registration_raw), "source_sha256": digest(source)}
    freeze_path = run / "freeze.json"
    if freeze_path.exists() and _json(freeze_path) != freeze:
        raise ValueError("frozen registration or source changed; refuse resume")
    model, limits, expected = (registration[k] for k in ("model", "limits", "expected"))
    url = urllib.parse.urlsplit(model["endpoint"])
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment or url.query:
        raise ValueError("endpoint must be an HTTPS URL without credentials, query or fragment")
    if not url.path.endswith("/chat/completions"):
        raise ValueError("endpoint must include the complete /chat/completions path")
    if not isinstance(model["name"], str) or not model["name"]:
        raise ValueError("model name is required")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", model["api_key_env"]):
        raise ValueError("api_key_env must be an environment variable name")
    _positive(limits["max_tokens"], 3000, "completion budget")
    _positive(limits["max_calls_per_solve"], 3, "calls per solve")
    _positive(limits["max_requests"], 13500, "request budget")
    _positive(limits["workers"], 64, "workers")
    _positive(limits.get("max_open_targets", 4), 16, "kb_open target limit")
    _positive(limits.get("max_tool_chars", 30000), 30000, "kb_open character limit")
    _positive(expected["question_count"], 500, "question count")
    if expected["arms"] != list(ARMS) or expected["repeats"] != 3:
        raise ValueError("expected arms N/F/L and three independent repetitions")
    questions = _rows(_read_pin(run, inputs["questions"]))
    by_qid = {}
    for question in questions:
        if any(not isinstance(question.get(k), str) or not question[k] for k in ("qid", "group", "family")):
            raise ValueError("questions need public qid/group/family")
        if question["qid"] in by_qid:
            raise ValueError("duplicate question")
        if question.get("task", "select") not in ("select", "ranking"):
            raise ValueError("public task must be select or ranking")
        by_qid[question["qid"]] = question
    if len(by_qid) != expected["question_count"]:
        raise ValueError("question count differs from registration")
    jobs = _rows(_read_pin(run, inputs["jobs"]))
    keys, tuples = set(), set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) != JOB_FIELDS:
            raise ValueError("job contains missing or nonpublic fields")
        if not isinstance(job["key"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", job["key"]):
            raise ValueError("invalid job key")
        if job["key"] in keys or job["qid"] not in by_qid:
            raise ValueError("duplicate key or unknown job question")
        question = by_qid[job["qid"]]
        if any(job[k] != question[k] for k in ("group", "family")):
            raise ValueError("job metadata differs from question")
        combination = (job["qid"], job["arm"], job["rep"])
        if job["arm"] not in ARMS or type(job["rep"]) is not int or job["rep"] not in (1, 2, 3):
            raise ValueError("unknown arm or repetition")
        if combination in tuples:
            raise ValueError("duplicate job condition")
        messages = job["messages"]
        if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or
            set(m) != {"role", "content"} or m["role"] not in ("system", "user") or
            not isinstance(m["content"], str) for m in messages):
            raise ValueError("job messages must contain public system/user text only")
        keys.add(job["key"])
        tuples.add(combination)
    wanted = {(q, arm, rep) for q in by_qid for arm in ARMS for rep in (1, 2, 3)}
    if tuples != wanted:
        raise ValueError("missing or extra registered jobs")
    lookups = {}
    if set(inputs["kb_lookup"]) != set(ARMS):
        raise ValueError("one frozen lookup is required for each arm")
    for arm in ARMS:
        lookup = json.loads(_read_pin(run, inputs["kb_lookup"][arm]))
        if not isinstance(lookup, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                               for k, v in lookup.items()):
            raise ValueError("lookup must map target IDs to text")
        lookups[arm] = lookup
    if lookups["N"]:
        raise ValueError("no-KB arm must have an empty lookup")
    for pin in inputs.get("extra", {}).values():
        _read_pin(run, pin)
    _pin_path(run, inputs["grader"])  # syntax only; opening labels is deferred
    grading = registration.get("grading", {})
    _positive(grading.get("bootstrap_replicates", 2000), 100000, "bootstrap replicates")
    if type(grading.get("bootstrap_seed", 20261026)) is not int:
        raise ValueError("bootstrap seed must be an integer")
    if not freeze_path.exists():
        _write(freeze_path, freeze)
        _atomic(run / "source/frozen_eval.py", source)
    return registration, jobs, lookups, digest(encoded(freeze))


def load_predictions(run):
    result = {}
    for path in sorted((Path(run) / "predictions").glob("*.json")):
        value = _json(path)
        if value.get("key") != path.stem or path.stem in result:
            raise ValueError("unexpected or duplicate prediction file")
        result[path.stem] = value
    return result


def _answer(text, task="select"):
    if not isinstance(text, str):
        return None
    for answer in reversed(re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)):
        pattern = r"[A-E](?:\s*<\s*[A-E])*" if task == "ranking" else r"[A-E]"
        if re.fullmatch(pattern, answer, re.I):
            letters = re.findall(r"[A-E]", answer.upper())
            if len(set(letters)) == len(letters):
                return "<".join(letters)
    return None


class _Runner:
    def __init__(self, run, registration, jobs, lookups, freeze_hash, transport, allow_network=True):
        self.run, self.registration, self.jobs, self.lookups = run, registration, jobs, lookups
        self.freeze_hash, self.transport = freeze_hash, transport
        self.allow_network = allow_network
        self.tasks = {q["qid"]: q.get("task", "select") for q in
                      _rows(_read_pin(run, registration["inputs"]["questions"]))}
        self.lock, self.stopped = threading.Lock(), threading.Event()
        self.requests = len(list((run / "requests").glob("*/call_*/request.json")))

    def request(self, job, call, payload, cache_key):
        if self.allow_network and (self.run / "STOP").exists():
            self.stopped.set()
            raise EvaluationStopped("STOP file present; no more requests")
        directory = self.run / "requests" / job["key"] / f"call_{call:02d}"
        request_path, raw_path, status_path = (directory / name for name in
                                              ("request.json", "response.raw", "status.json"))
        request_hash = digest(encoded(payload))
        if request_path.exists():
            saved = _json(request_path)
            if saved != {"key": job["key"], "call": call, "cache_key": cache_key,
                         "sha256": request_hash, "payload": payload}:
                raise ValueError("frozen request hash or payload changed")
            status = _json(status_path) if status_path.exists() else {}
            if not raw_path.exists():
                raise EvaluationStopped("indeterminate sent request has no saved response; no retry")
            raw = raw_path.read_bytes()
            if status.get("response_sha256") and status["response_sha256"] != digest(raw):
                raise ValueError("saved response hash mismatch")
        else:
            if not self.allow_network:
                raise ValueError("prediction has no complete saved request; cache-only verification")
            if raw_path.exists() or status_path.exists():
                raise ValueError("response/status without registered request")
            # Read only at execution time; no secret enters a request record or status.
            key = os.environ.get(self.registration["model"]["api_key_env"])
            if not key:
                raise EvaluationStopped("registered API key environment variable is unset")
            with self.lock:
                if self.stopped.is_set():
                    raise EvaluationStopped("evaluation stopped after an unknown request")
                if (self.run / "STOP").exists():
                    self.stopped.set()
                    raise EvaluationStopped("STOP file present; no more requests")
                if self.requests >= self.registration["limits"]["max_requests"]:
                    raise EvaluationStopped("hard request budget exhausted")
                self.requests += 1
                _write(request_path, {"key": job["key"], "call": call, "cache_key": cache_key,
                                      "sha256": request_hash, "payload": payload})
                _write(status_path, {"state": "sent", "request_sha256": request_hash,
                                     "reserved_completion_tokens": payload["max_tokens"]})
            try:
                status_code, raw = self.transport(self.registration["model"]["endpoint"], payload, key,
                                                  self.registration["limits"].get("timeout_seconds", 180))
                if not 200 <= status_code < 300:
                    raise TransportFailure("http_error", status_code)
            except TransportFailure as error:
                self.stopped.set()
                _write(status_path, {"state": "unknown", "reason": error.kind, "http_status": error.status,
                    "request_sha256": request_hash, "reserved_completion_tokens": payload["max_tokens"]})
                raise EvaluationStopped(f"request stopped: {error.kind}; no automatic retry") from None
            except Exception:
                self.stopped.set()
                _write(status_path, {"state": "unknown", "reason": "transport_exception",
                    "request_sha256": request_hash, "reserved_completion_tokens": payload["max_tokens"]})
                raise EvaluationStopped("request stopped: transport exception; no automatic retry") from None
            _atomic(raw_path, raw)  # durable before interpretation, next call, or prediction
            status = {"http_status": status_code}
        try:
            result = json.loads(raw)
            usage = result["usage"]
            used = usage["completion_tokens"]
            if type(used) is not int or used < 0:
                raise ValueError()
            choice = result["choices"][0]
            if not isinstance(choice["message"], dict):
                raise ValueError()
        except (ValueError, KeyError, TypeError, IndexError):
            self.stopped.set()
            _write(status_path, {**status, "state": "unknown", "reason": "response_or_usage_unknown",
                "request_sha256": request_hash, "response_sha256": digest(raw),
                "reserved_completion_tokens": payload["max_tokens"]})
            raise EvaluationStopped("response or usage unknown; saved raw retained, no retry") from None
        _write(status_path, {"state": "success", "http_status": status.get("http_status", 200),
            "request_sha256": request_hash, "response_sha256": digest(raw), "usage": usage,
            "model": result.get("model"), "finish_reason": choice.get("finish_reason")})
        return choice, used, digest(raw)

    def solve(self, job):
        if self.allow_network and (self.run / "STOP").exists():
            self.stopped.set()
            raise EvaluationStopped("STOP file present; dispatch halted")
        path = self.run / "predictions" / f"{job['key']}.json"
        job_hash = digest(encoded(job))
        saved = None
        if path.exists():
            saved = _json(path)
            if saved.get("job_sha256") != job_hash or saved.get("freeze_sha256") != self.freeze_hash:
                raise ValueError("prediction belongs to another frozen job")
        limits = self.registration["limits"]
        cache_key = digest(encoded([self.freeze_hash, job["key"], job["rep"]]))
        # Repetitions intentionally have identical model inputs. The local cache
        # identity isolates saved requests; no nonce changes the model's prompt.
        messages = [{"role": "system", "content": "Answer with an XML <answer>...</answer> field. "
                     "Use letters and < between letters for a ranking."}] + list(job["messages"])
        consumed, response_hashes, answer, reason = 0, [], None, "no_final_answer"
        for call in range(1, limits["max_calls_per_solve"] + 1):
            remaining = limits["max_tokens"] - consumed
            if remaining <= 0:
                reason = "completion_budget_exhausted"
                break
            payload = {"model": self.registration["model"]["name"], "messages": messages,
                       "temperature": 0, "thinking": {"type": "disabled"}, "max_tokens": remaining}
            if job["arm"] != "N":
                tool_schema = copy.deepcopy(TOOL)
                tool_schema["function"]["parameters"]["properties"]["targets"]["maxItems"] = limits.get("max_open_targets", 4)
                payload.update(tools=[tool_schema], tool_choice="none" if call == limits["max_calls_per_solve"] else "auto")
            if saved is not None and not (self.run / "requests" / job["key"] / f"call_{call:02d}" / "request.json").exists():
                raise ValueError("prediction has no matching saved request")
            choice, used, response_hash = self.request(job, call, payload, cache_key)
            consumed += used
            response_hashes.append(response_hash)
            message, finish = choice["message"], choice.get("finish_reason")
            if used > remaining:
                self.stopped.set()
                reason = "provider_exceeded_completion_budget"
                break
            if finish == "length":
                reason = "truncated"
                break
            tools = message.get("tool_calls")
            if not tools:
                if finish != "stop":
                    reason = "not_final"
                else:
                    answer = _answer(message.get("content") or "", self.tasks[job["qid"]])
                    reason = None if answer is not None else "invalid_xml_answer"
                break
            if job["arm"] == "N" or call == limits["max_calls_per_solve"]:
                reason = "tool_limit_or_unavailable"
                break
            try:
                if finish != "tool_calls" or not isinstance(tools, list) or len(tools) != 1:
                    raise ValueError()
                tool = tools[0]
                arguments = json.loads(tool["function"]["arguments"])
                if tool["function"]["name"] != "kb_open" or not isinstance(tool["id"], str):
                    raise ValueError()
                if not isinstance(arguments, dict) or set(arguments) != {"targets"} or not isinstance(arguments["targets"], list):
                    raise ValueError()
                if any(not isinstance(target, str) for target in arguments["targets"]):
                    raise ValueError()
                if len(arguments["targets"]) > limits.get("max_open_targets", 4):
                    output = f"Rejected: at most {limits.get('max_open_targets', 4)} frozen targets per call."
                else:
                    output = "\n\n".join(self.lookups[job["arm"]].get(target, f"Unknown frozen target: {target}")
                                         for target in arguments["targets"])
                limit = limits.get("max_tool_chars", 30000)
                marker = "\n[TRUNCATED: frozen kb_open character limit]"
                if len(output) > limit:
                    output = output[:max(0, limit - len(marker))] + marker[:limit]
            except (KeyError, TypeError, ValueError):
                reason = "invalid_tool_call"
                break
            messages = messages + [{"role": "assistant", "content": message.get("content"), "tool_calls": tools},
                                   {"role": "tool", "tool_call_id": tool["id"], "content": output}]
        result = {k: job[k] for k in ("key", "qid", "group", "family", "arm", "rep")}
        result.update(job_sha256=job_hash, freeze_sha256=self.freeze_hash, cache_key=cache_key,
            valid=reason is None, invalid_reason=reason, answer=answer,
            completion_tokens=consumed, requests=len(response_hashes), response_sha256=response_hashes)
        if saved is not None:
            if result != saved:
                raise ValueError("prediction differs from cache-only replay of saved raw responses")
            return saved
        _write(path, result)
        return result


def run_evaluation(run_dir, *, max_jobs=None, transport=None):
    run = Path(run_dir).resolve()
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1):
        raise ValueError("max_jobs must be a positive integer")
    with _run_lock(run):
        registration, jobs, lookups, freeze_hash = _contract(run)
        predictions = load_predictions(run)
        if set(predictions) - {job["key"] for job in jobs}:
            raise ValueError("extra prediction files")
        runner = _Runner(run, registration, jobs, lookups, freeze_hash, transport or http_transport)
        selected = jobs if max_jobs is None else jobs[:max_jobs]
        with ThreadPoolExecutor(max_workers=registration["limits"]["workers"]) as executor:
            list(executor.map(runner.solve, selected))
        return {"registered_jobs": len(jobs), "completed_jobs": len(load_predictions(run)),
                "requests_reserved": runner.requests, "complete": len(load_predictions(run)) == len(jobs)}


def score_answer(answer, rubric):
    if not isinstance(answer, str):
        return 0.0
    if rubric["task"] == "select":
        return float(answer == rubric["answer"])
    predicted, gold = answer.split("<"), rubric["answer"].split("<")
    if len(predicted) != rubric["num_choices"] or sorted(predicted) != sorted(gold):
        return 0.0
    ranks = {letter: i for i, letter in enumerate(gold)}
    inversions = sum(ranks[predicted[i]] > ranks[predicted[j]]
                     for i in range(len(predicted)) for j in range(i + 1, len(predicted)))
    return {0: 1., 1: .75, 2: .5, 3: .25}.get(inversions, 0.)


def paired_summary(rows, seed, replicates):
    """Question-weighted mean; CR1 cluster SE and percentile group bootstrap."""
    groups = defaultdict(list)
    for group, value in rows:
        groups[group].append(value)
    n, g = len(rows), len(groups)
    mean = statistics.mean(value for _, value in rows)
    se, interval = None, None
    if g > 1:
        residual_sums = [sum(value - mean for value in values) for values in groups.values()]
        se = math.sqrt(g / (g - 1) * sum(value * value for value in residual_sums)) / n
        clusters, rng, samples = [groups[key] for key in sorted(groups)], random.Random(seed), []
        for _ in range(replicates):
            draw = [clusters[rng.randrange(g)] for _ in range(g)]
            samples.append(sum(map(sum, draw)) / sum(map(len, draw)))
        samples.sort()
        def percentile(p):
            index = (len(samples) - 1) * p
            low, high = math.floor(index), math.ceil(index)
            return samples[low] + (samples[high] - samples[low]) * (index - low)
        interval = [percentile(.025), percentile(.975)]
    return {"delta": mean, "cluster_se": se, "bootstrap_95": interval,
            "n_questions": n, "n_groups": g, "bootstrap_replicates": replicates, "bootstrap_seed": seed}


def grade_evaluation(run_dir):
    run = Path(run_dir).resolve()
    with _run_lock(run):
        registration, jobs, lookups, freeze_hash = _contract(run)
        predictions = load_predictions(run)
        if set(predictions) != {job["key"] for job in jobs}:
            raise ValueError("incomplete or extra prediction set; grader remains unopened")
        verifier = _Runner(run, registration, jobs, lookups, freeze_hash, None, allow_network=False)
        for job in jobs:
            verifier.solve(job)  # reconstruct every tool round and final result, without network or labels
        expected_requests = {run / "requests" / job["key"] / f"call_{call:02d}" / "request.json"
                             for job in jobs for call in range(1, predictions[job["key"]]["requests"] + 1)}
        if set((run / "requests").glob("*/call_*/request.json")) != expected_requests:
            raise ValueError("request ledger differs from complete registered predictions")
        request_status = [_json(path.parent / "status.json") for path in sorted(expected_requests)]
        # All registered predictions and their successful raw responses are now accounted for.
        grader = json.loads(_read_pin(run, registration["inputs"]["grader"]))
        if not isinstance(grader, dict) or set(grader) != {job["qid"] for job in jobs}:
            raise ValueError("grader question set differs from frozen jobs")
        for rubric in grader.values():
            if rubric.get("task") not in ("select", "ranking") or type(rubric.get("num_choices")) is not int:
                raise ValueError("invalid grader rubric")
            letters = list("ABCDE"[:rubric["num_choices"]])
            answer = rubric.get("answer", "")
            if not 2 <= rubric["num_choices"] <= 5 or (rubric["task"] == "select" and answer not in letters) or (
                rubric["task"] == "ranking" and sorted(answer.split("<")) != letters):
                raise ValueError("invalid grader answer")
        scores, per_job, meta = defaultdict(lambda: defaultdict(dict)), [], {}
        for job in jobs:
            prediction = predictions[job["key"]]
            score = score_answer(prediction["answer"], grader[job["qid"]]) if prediction["valid"] else 0.
            scores[job["qid"]][job["arm"]][job["rep"]] = score
            meta[job["qid"]] = {k: job[k] for k in ("qid", "group", "family")}
            per_job.append({**prediction, "score": score})
        per_question = [{**meta[qid], "scores": {arm: [scores[qid][arm][rep] for rep in (1, 2, 3)] for arm in ARMS},
                         "means": {arm: statistics.mean(scores[qid][arm].values()) for arm in ARMS}}
                        for qid in sorted(scores)]
        options = registration.get("grading", {})
        seed, replicates = options.get("bootstrap_seed", 20261026), options.get("bootstrap_replicates", 2000)
        def aggregate(rows):
            return {"n_questions": len(rows), "arms": {arm: {"mean": statistics.mean(r["means"][arm] for r in rows)}
                    for arm in ARMS}, "comparisons": {f"{left}_to_{right}": paired_summary(
                        [(r["group"], r["means"][right] - r["means"][left]) for r in rows], seed, replicates)
                        for left, right in (("N", "F"), ("N", "L"), ("F", "L"))}}
        summary = {**aggregate(per_question), "schema_version": 1, "freeze_sha256": freeze_hash,
            "n_jobs": len(jobs), "invalid_jobs": sum(not r["valid"] for r in per_job),
            "successful_responses": len(request_status),
            "total_completion_tokens": sum(r["usage"]["completion_tokens"] for r in request_status),
            "response_models": sorted({r.get("model") or "unknown" for r in request_status}),
            "unknown_model_responses": sum(not r.get("model") for r in request_status),
            "uncertainty": "question-weighted paired means; CR1 group cluster SE; seeded percentile group bootstrap",
            "by_family": {family: aggregate([r for r in per_question if r["family"] == family])
                          for family in sorted({r["family"] for r in per_question})}}
        _write(run / "summary.json", summary)
        _atomic(run / "scores.jsonl", b"".join(encoded(r) + b"\n" for r in per_job))
        _atomic(run / "per_question.jsonl", b"".join(encoded(r) + b"\n" for r in per_question))
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "grade"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-jobs", type=int, help="run only the first N registered jobs, including cached jobs")
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(args.run_dir, max_jobs=args.max_jobs) if args.command == "run" else grade_evaluation(args.run_dir)
    except (ValueError, EvaluationStopped) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
