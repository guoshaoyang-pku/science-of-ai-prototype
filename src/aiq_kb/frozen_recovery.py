"""Preregistered, single-attempt repair of an unknown transport outcome.

Requires STOP and the original run lock. No successful raw or prediction is
replaced. Every additional physical attempt and its unknown cost remain in
recovery/, even when a later, separately registered plan succeeds.
"""
import argparse
import json
import os
from pathlib import Path
import re

from . import frozen_eval as ev


FIELDS = {"schema_version", "parent_registration_sha256", "frozen_eval_source_sha256",
          "repair_source_sha256", "basis", "labels_unopened", "reason", "external_requests",
          "external_reserved_completion_tokens", "max_additional_requests",
          "max_additional_reserved_completion_tokens", "repairs"}
TARGET_FIELDS = {"path", "request_sha256", "status_sha256", "payload_sha256", "reserved_completion_tokens"}


def _once(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("immutable recovery evidence differs: " + path.name)
    else:
        ev._atomic(path, raw)


def _target(run, value, jobs):
    if not isinstance(value, dict) or set(value) != TARGET_FIELDS:
        raise ValueError("invalid repair target fields")
    match = re.fullmatch(r"requests/([A-Za-z0-9_-]{1,200})/call_(0[1-3])", value["path"])
    if not match or match[1] not in jobs:
        raise ValueError("unknown repair request path")
    directory = run / value["path"]
    raw = (directory / "request.json").read_bytes()
    request = json.loads(raw)
    if (ev.digest(raw) != value["request_sha256"] or request["sha256"] != value["payload_sha256"] or
        ev.digest(ev.encoded(request["payload"])) != request["sha256"] or
        request["key"] != match[1] or request["call"] != int(match[2])):
        raise ValueError("original request or exact payload hash differs")
    reserved = value["reserved_completion_tokens"]
    if type(reserved) is not int or not 1 <= reserved <= 3000 or request["payload"]["max_tokens"] != reserved:
        raise ValueError("repair reservation differs from original payload")
    return directory, request, raw


def _budget(run, registration, jobs, freeze_hash, external):
    """Worst-case remaining original calls plus *all* physical repair attempts."""
    by_key = {job["key"]: job for job in jobs}
    calls, unknown, count = {}, {}, 0
    for path in (run / "requests").glob("*/call_*/request.json"):
        request = ev._json(path)
        key, call = request["key"], request["call"]
        if key not in by_key or type(call) is not int or not 1 <= call <= registration["limits"]["max_calls_per_solve"]:
            raise ValueError("unknown original request in budget ledger")
        if path != run / "requests" / key / f"call_{call:02d}" / "request.json":
            raise ValueError("request ledger path differs")
        calls.setdefault(key, set()).add(call)
        count += 1
        status = ev._json(path.parent / "status.json") if (path.parent / "status.json").exists() else {}
        if not (path.parent / "response.raw").exists() and status.get("state", "sent") in ("unknown", "sent"):
            unknown[path.parent.relative_to(run).as_posix()] = request["payload"]["max_tokens"]
    predictions = ev.load_predictions(run)
    if set(predictions) - set(by_key):
        raise ValueError("unknown prediction in budget ledger")
    future = 0
    for key, job in by_key.items():
        saved = calls.get(key, set())
        if saved != set(range(1, len(saved) + 1)):
            raise ValueError("original request ledger has missing call slots")
        if key in predictions:
            prediction = predictions[key]
            if (prediction.get("job_sha256") != ev.digest(ev.encoded(job)) or
                prediction.get("freeze_sha256") != freeze_hash or prediction["requests"] != len(saved) or
                len(prediction["response_sha256"]) != len(saved)):
                raise ValueError("completed prediction cannot justify request-budget slack")
            for call, expected in enumerate(prediction["response_sha256"], 1):
                if ev.file_hash(run / "requests" / key / f"call_{call:02d}/response.raw") != expected:
                    raise ValueError("completed raw hash cannot justify request-budget slack")
        else:
            future += registration["limits"]["max_calls_per_solve"] - len(saved)
    attempts, repair_unknown, excluded = 0, 0, set()
    for path in (run / "recovery").glob("*/attempts/*/request.json"):
        record = ev._json(path)
        if record["key"] not in by_key or record["qid"] != by_key[record["key"]]["qid"]:
            raise ValueError("unknown repair attempt in budget ledger")
        original = run / record["target"] / "request.json"
        request = ev._json(original)
        if record["payload"] != request["payload"] or record["reserved_completion_tokens"] != request["payload"]["max_tokens"]:
            raise ValueError("repair attempt differs from original request budget")
        attempts += 1
        excluded.add(record["qid"])
        before = ev._json(path.parent / "before/status.json")
        if before.get("state") == "unknown":
            unknown[record["target"]] = record["reserved_completion_tokens"]
        status = ev._json(path.parent / "status.json") if (path.parent / "status.json").exists() else {}
        if status.get("state") != "response_saved" or not status.get("usage_known"):
            repair_unknown += record["reserved_completion_tokens"]
    return {"original_requests": count, "worstcase_future_original_requests": future,
        "repair_attempts": attempts, "external_requests": external["requests"],
        "worstcase_total_requests": count + future + attempts + external["requests"],
        "request_ceiling": registration["limits"]["max_requests"],
        "original_unknown_reserved_completion_tokens": sum(unknown.values()),
        "repair_unknown_reserved_completion_tokens": repair_unknown,
        "external_reserved_completion_tokens": external["reserved_completion_tokens"],
        "strict_budget_excluded_qids": sorted(excluded)}


def _response_status(attempt, raw, payload_hash):
    previous = ev._json(attempt / "status.json") if (attempt / "status.json").exists() else {}
    response_hash = ev.digest(raw)
    if previous.get("response_sha256") and previous["response_sha256"] != response_hash:
        raise ValueError("immutable repair response hash differs")
    details = {}
    try:
        parsed = json.loads(raw)
        if type(parsed["usage"]["completion_tokens"]) is int and parsed["usage"]["completion_tokens"] >= 0:
            details = {"usage": parsed["usage"], "model": parsed.get("model"),
                       "finish_reason": (parsed.get("choices") or [{}])[0].get("finish_reason")}
    except (ValueError, KeyError, TypeError, IndexError):
        pass
    ev._write(attempt / "status.json", {"state": "response_saved", "http_status": previous.get("http_status", 200),
        "response_sha256": response_hash, "payload_sha256": payload_hash, "usage_known": bool(details), **details})
    return response_hash


def _publish_raw(primary, attempt, response_hash, plan_hash):
    raw = (attempt / "response.raw").read_bytes()
    if ev.digest(raw) != response_hash:
        raise ValueError("immutable repair response hash differs")
    destination = primary / "response.raw"
    existed = destination.exists()
    _once(destination, raw)
    if not existed:
        ev._write(primary / "status.json", {"state": "recovered_raw", "http_status": ev._json(attempt / "status.json")["http_status"],
            "response_sha256": response_hash, "request_sha256": ev._json(primary / "request.json")["sha256"],
            "recovery_plan_sha256": plan_hash})


def run_recovery(run_dir, *, plan_filename="recovery-plan.json", transport=None):
    run = Path(run_dir).resolve()
    if not re.fullmatch(r"recovery-plan(?:-[A-Za-z0-9_-]+)?\.json", plan_filename):
        raise ValueError("invalid recovery plan filename")
    with ev._run_lock(run):
        if not (run / "STOP").exists():
            raise ev.EvaluationStopped("STOP must exist before an operator transport repair")
        registration, jobs, _, freeze_hash = ev._contract(run)
        raw = (run / plan_filename).read_bytes()
        plan, plan_hash = json.loads(raw), ev.digest(raw)
        if not isinstance(plan, dict) or set(plan) != FIELDS or type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
            raise ValueError("invalid recovery plan fields or schema")
        hashes = {"parent_registration_sha256": ev.file_hash(run / "registration.json"),
                  "frozen_eval_source_sha256": ev.file_hash(Path(ev.__file__)),
                  "repair_source_sha256": ev.file_hash(Path(__file__))}
        if any(plan[name] != value for name, value in hashes.items()):
            raise ValueError("recovery source or parent registration hash mismatch")
        if (plan["basis"] != "transport_only" or plan["labels_unopened"] is not True or
            not isinstance(plan["reason"], str) or not plan["reason"].strip()):
            raise ValueError("recovery requires transport-only reasoning and unopened labels")
        if any((run / name).exists() for name in ("summary.json", "scores.jsonl", "per_question.jsonl")):
            raise ValueError("scoring artifacts exist; labels-unopened repair is unavailable")
        targets = plan["repairs"]
        if not isinstance(targets, list) or not targets or len({t["path"] for t in targets}) != len(targets):
            raise ValueError("repair targets must be a nonempty unique list")
        if (plan["max_additional_requests"] != len(targets) or
            plan["max_additional_reserved_completion_tokens"] != sum(t["reserved_completion_tokens"] for t in targets)):
            raise ValueError("additional repair budget differs from exact planned attempts")
        external = {"requests": plan["external_requests"], "reserved_completion_tokens": plan["external_reserved_completion_tokens"]}
        if any(type(v) is not int or v < 0 for v in external.values()):
            raise ValueError("invalid external request budget")
        root = run / "recovery" / Path(plan_filename).stem
        frozen = {**hashes, "plan_sha256": plan_hash}
        if (root / "freeze.json").exists() and ev._json(root / "freeze.json") != frozen:
            raise ValueError("frozen recovery plan changed")
        by_key, prepared, new_attempts = {j["key"]: j for j in jobs}, [], 0
        # Validate the entire plan before any physical request or credential read.
        for index, target in enumerate(targets, 1):
            primary, request, request_raw = _target(run, target, by_key)
            attempt = root / "attempts" / f"{index:04d}"
            if (attempt / "request.json").exists():
                if not (attempt / "response.raw").exists():
                    raise ev.EvaluationStopped("repair already attempted with unknown outcome; register a new plan")
                state = ev._json(attempt / "status.json") if (attempt / "status.json").exists() else {}
                record = ev._json(attempt / "request.json")
                if (record["plan_sha256"] != plan_hash or record["target"] != target["path"] or
                    record["payload_sha256"] != request["sha256"] or
                    (attempt / "before/request.json").read_bytes() != request_raw):
                    raise ValueError("archived repair request differs from its frozen plan")
                if ev.file_hash(attempt / "before/status.json") != target["status_sha256"]:
                    raise ValueError("archived original status hash differs")
                response_hash = ev.file_hash(attempt / "response.raw")
                if state.get("response_sha256") and state["response_sha256"] != response_hash:
                    raise ValueError("immutable repair response hash differs")
                prepared.append((target, primary, request, request_raw, attempt, response_hash))
                continue
            if (primary / "response.raw").exists():
                raise ValueError("successful or existing raw cannot be repaired or overwritten")
            status_raw = (primary / "status.json").read_bytes()
            status = json.loads(status_raw)
            if ev.digest(status_raw) != target["status_sha256"]:
                raise ValueError("original status hash differs")
            if (status.get("state") != "unknown" or status.get("reason") != "transport_error" or
                status.get("reserved_completion_tokens") != target["reserved_completion_tokens"]):
                raise ValueError("only unknown transport_error without raw is eligible")
            prepared.append((target, primary, request, request_raw, attempt, None))
            new_attempts += 1
        external_path = run / "recovery/external-budget.json"
        if external_path.exists() and ev._json(external_path) != external:
            raise ValueError("external request budget cannot be reset across recovery plans")
        budget = _budget(run, registration, jobs, freeze_hash, external)
        if budget["worstcase_total_requests"] + new_attempts > budget["request_ceiling"]:
            raise ValueError("worst-case physical requests exceed the original request ceiling")
        _once(root / "freeze.json", ev.encoded(frozen))
        _once(root / "plan.json", raw)
        _once(root / "source.py", Path(__file__).read_bytes())
        _once(external_path, ev.encoded(external))
        try:
            for target, primary, request, request_raw, attempt, saved_hash in prepared:
                if saved_hash:
                    _response_status(attempt, (attempt / "response.raw").read_bytes(), request["sha256"])
                    _publish_raw(primary, attempt, saved_hash, plan_hash)
                    continue
                key = os.environ.get(registration["model"]["api_key_env"])
                if not key:
                    raise ev.EvaluationStopped("registered API key environment variable is unset")
                _once(attempt / "before/request.json", request_raw)
                _once(attempt / "before/status.json", (primary / "status.json").read_bytes())
                record = {"key": request["key"], "qid": by_key[request["key"]]["qid"], "target": target["path"],
                    "plan_sha256": plan_hash, "payload": request["payload"], "payload_sha256": request["sha256"],
                    "reserved_completion_tokens": target["reserved_completion_tokens"]}
                _once(attempt / "request.json", ev.encoded(record))
                ev._write(attempt / "status.json", {"state": "sent", "reserved_completion_tokens": target["reserved_completion_tokens"]})
                try:
                    code, response = (transport or ev.http_transport)(registration["model"]["endpoint"], request["payload"],
                        key, registration["limits"].get("timeout_seconds", 180))
                    if not 200 <= code < 300:
                        raise ev.TransportFailure("http_error", code)
                except Exception as error:
                    ev._write(attempt / "status.json", {"state": "unknown", "reason": getattr(error, "kind", "transport_exception"),
                        "http_status": getattr(error, "status", None), "reserved_completion_tokens": target["reserved_completion_tokens"]})
                    raise ev.EvaluationStopped("operator repair transport failed; attempt and unknown cost retained") from None
                ev._write(attempt / "status.json", {"state": "received", "http_status": code,
                    "response_sha256": ev.digest(response), "payload_sha256": request["sha256"],
                    "reserved_completion_tokens": target["reserved_completion_tokens"]})
                _once(attempt / "response.raw", response)
                response_hash = _response_status(attempt, response, request["sha256"])
                _publish_raw(primary, attempt, response_hash, plan_hash)
        finally:
            budget = _budget(run, registration, jobs, freeze_hash, external)
            ev._write(run / "recovery-budget.json", budget)
        return budget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--plan-filename", default="recovery-plan.json")
    args = parser.parse_args(argv)
    try:
        result = run_recovery(args.run_dir, plan_filename=args.plan_filename)
    except (ValueError, ev.EvaluationStopped) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
