from __future__ import annotations

import inspect
import hashlib
import json
import math
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from architecture_iq.candidates.generator import write_candidate, validate_process_request
from architecture_iq.profile import Profile, validate_execution_device
from architecture_iq.registry import get_dataset_family, get_model_type
from architecture_iq.runtime.loader import load_candidate_train
from architecture_iq.significance.validator import final_metric_key, mean_metric_key
from architecture_iq.util import git_commit_hash, write_json
from architecture_iq.paths import ROOT


ProgressCallback = Callable[[dict[str, Any]], None]


def validate_target_transform(request: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or set(request) - {"center", "offset"}:
        raise ValueError("target_transform accepts only center and offset")
    if get_dataset_family(spec["family"]).train_loop_kind != "regression" or spec["loss"] != {"loss_id": "mse"}:
        raise ValueError("target_transform requires plain regression MSE")
    center, offset = request.get("center", False), request.get("offset", 0.0)
    if not isinstance(center, bool) or isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
        raise ValueError("target_transform requires boolean center and finite offset")
    return {"center": center, "offset": float(offset)}


def _validate_regression_diagnostics(diagnostics, spec, tensors, result):
    record, arrays = diagnostics["record"], diagnostics["evaluation"]
    request, optimizer = spec["process"], spec["optimizer"]
    defaults = record.get("optimizer_defaults", {})
    expected = {"lr": optimizer["lr"], "weight_decay": optimizer.get("weight_decay", 0.0)}
    if optimizer["type"] == "Adam":
        expected.update(betas=optimizer.get("betas", [.9, .999]), eps=1e-8)
    else:
        expected.update(momentum=optimizer.get("momentum", 0.0), dampening=0, nesterov=False)
    if record.get("optimizer_type") != optimizer["type"] or any(defaults.get(key) != value for key, value in expected.items()):
        raise ValueError("regression actual optimizer defaults mismatch")
    if len(json.dumps(record, allow_nan=False).encode()) > 8 * 1024 * 1024:
        raise ValueError("regression process scalar record exceeds bound")
    observed = [row.get("step") for row in record.get("steps", [])]
    if observed != sorted(set(observed)) or any(step not in request["steps"] for step in observed):
        raise ValueError("invalid regression observation steps")
    if not result["failed"] and observed != request["steps"]:
        raise ValueError("requested regression observations missing")
    for row in record.get("steps", []):
        parameters = row.get("parameters", [])
        if not 1 <= len(parameters) <= 256 or len({p.get("name") for p in parameters}) != len(parameters):
            raise ValueError("invalid regression parameter records")
        for p in parameters:
            if not isinstance(p.get("name"), str) or p.get("role") not in {"head", "hidden"} or not isinstance(p.get("active"), bool):
                raise ValueError("invalid regression parameter identity")
            keys = ("parameter_norm", "descent_norm", "displacement_norm") + (("data_gradient_norm",) if p["active"] else ())
            if any(isinstance(p.get(key), bool) or not isinstance(p.get(key), (int, float)) or not math.isfinite(p[key]) or p[key] < 0 for key in keys):
                raise ValueError("invalid regression process norm")
    for name, value in zip(("train_x", "train_y", "test_x", "test_y"), tensors):
        value = value.detach().cpu().contiguous()
        pin = {"shape": list(value.shape), "dtype": str(value.dtype),
               "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}
        if record.get("inputs", {}).get(name) != pin:
            raise ValueError("regression input hash mismatch")
    if result["failed"]:
        if record.get("status") != "failed" or arrays:
            raise ValueError("failed regression run returned evaluation")
        return
    required = {"indices", "predictions", "targets", "initial_predictions",
                "train_predictions", "train_targets", "initial_train_predictions"}
    if not isinstance(arrays, dict) or set(arrays) != required or any(not isinstance(v, torch.Tensor) for v in arrays.values()):
        raise ValueError("invalid regression evaluation tensor schema")
    train_x, train_y, test_x, test_y = tensors
    if not torch.equal(arrays["indices"], torch.arange(test_x.shape[0])):
        raise ValueError("regression evaluation order mismatch")
    for prefix, target in (("", test_y), ("train_", train_y)):
        target = target.detach().cpu()
        if not torch.equal(arrays[prefix + "targets"], target):
            raise ValueError("regression evaluation targets mismatch")
        for key in (prefix + "predictions", "initial_" + prefix + "predictions"):
            if arrays[key].shape != target.shape or not torch.isfinite(arrays[key]).all():
                raise ValueError("regression evaluation shape or finiteness mismatch")
    initial = record.get("initial_metrics", {})
    for prefix in ("train", "test"):
        stem = "train_" if prefix == "train" else ""
        prediction, target = arrays["initial_" + stem + "predictions"], arrays[stem + "targets"]
        metrics = {prefix + "_mse": float(((prediction - target) ** 2).mean()),
                   prefix + "_prediction_mean": float(prediction.double().mean()),
                   prefix + "_target_mean": float(target.double().mean())}
        if any(not math.isclose(initial.get(key, float("nan")), value, rel_tol=1e-6, abs_tol=1e-7) for key, value in metrics.items()):
            raise ValueError("regression step-zero metrics mismatch")
    mse = float(((arrays["predictions"] - arrays["targets"]) ** 2).mean())
    if record.get("status") != "completed" or not math.isclose(record.get("final_metrics", {}).get("mse", float("nan")), mse, rel_tol=1e-6, abs_tol=1e-7) or not math.isclose(result["final_test_mse"], mse, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("regression predictions disagree with final metric")

# Each generate-candidates worker process runs its own interpreter; without a
# clamp every process defaults torch to all cores and N parallel workers
# oversubscribe OpenMP into a spin deadlock (observed with 8 workers).
# Override with ARCHITECTURE_IQ_TORCH_THREADS if the host needs a different cap.
DEFAULT_TORCH_THREADS = 8


def _clamp_process_torch_threads() -> int:
    raw = os.environ.get("ARCHITECTURE_IQ_TORCH_THREADS")
    try:
        threads = int(raw) if raw is not None else DEFAULT_TORCH_THREADS
    except ValueError as exc:
        raise ValueError(
            "ARCHITECTURE_IQ_TORCH_THREADS must be a positive integer"
        ) from exc
    if threads <= 0:
        raise ValueError("ARCHITECTURE_IQ_TORCH_THREADS must be a positive integer")
    torch.set_num_threads(threads)
    return threads


def _emit_progress(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    """Report optional UI progress without affecting ground-truth execution."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        # Progress is observational; a UI refresh failure must not invalidate GT.
        return


def _sync_candidate_files(candidate_path: Path, spec: dict[str, Any]) -> None:
    """Rewrite on-disk .py files from spec so execution matches candidate_spec.json."""
    from architecture_iq.registry import ensure_registries

    ensure_registries()
    model_family = get_model_type(spec["model"]["type"])
    write_candidate(spec, candidate_path, model_family)


def _resolve_execution_device(candidate_spec: dict[str, Any], profile: Profile) -> torch.device:
    requested = validate_execution_device(
        str(candidate_spec.get("execution", {}).get("device", "cpu"))
    )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for this candidate but is unavailable "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    return torch.device(requested)
def _seed_parallelism_config(
    device: torch.device,
    n_seeds: int,
    progress_callback: ProgressCallback | None,
) -> tuple[int, int | None]:
    """Return opt-in CPU seed workers and their per-worker Torch threads."""
    if device.type != "cpu" or progress_callback is not None:
        return 1, None
    raw_workers = os.environ.get("ARCHITECTURE_IQ_SEED_WORKERS")
    if raw_workers is None:
        return 1, None
    try:
        requested_workers = int(raw_workers)
    except ValueError as exc:
        raise ValueError("ARCHITECTURE_IQ_SEED_WORKERS must be a positive integer") from exc
    if requested_workers <= 0:
        raise ValueError("ARCHITECTURE_IQ_SEED_WORKERS must be a positive integer")
    if requested_workers == 1:
        return 1, None
    raw_threads = os.environ.get("ARCHITECTURE_IQ_SEED_TORCH_THREADS", "1")
    try:
        torch_threads = int(raw_threads)
    except ValueError as exc:
        raise ValueError(
            "ARCHITECTURE_IQ_SEED_TORCH_THREADS must be a positive integer"
        ) from exc
    if torch_threads <= 0:
        raise ValueError("ARCHITECTURE_IQ_SEED_TORCH_THREADS must be a positive integer")
    return min(requested_workers, n_seeds), torch_threads


def run_single_seed(
    candidate_path: Path,
    candidate_spec: dict[str, Any],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    fail_threshold: float,
    *,
    selection_metric: str,
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    target_transform = None
    if "target_transform" in candidate_spec:
        request = validate_target_transform(candidate_spec["target_transform"], candidate_spec)
        if request != candidate_spec["target_transform"]:
            raise ValueError("target_transform must be normalized")
        if not train_y.is_floating_point() or not test_y.is_floating_point() or not train_y.numel() or not test_y.numel():
            raise ValueError("target_transform requires nonempty floating labels")
        original = {}
        for name, value in (("train_y", train_y), ("test_y", test_y)):
            value = value.detach().cpu().contiguous()
            original[name] = {"sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                              "mean": float(value.double().mean())}
        center = float(train_y.double().mean()) if request["center"] else 0.0
        shift = request["offset"] - center
        train_y, test_y = train_y + shift, test_y + shift
        if not torch.isfinite(train_y).all() or not torch.isfinite(test_y).all():
            raise ValueError("target_transform produced nonfinite labels")
        target_transform = {"request": request, "center_source": "train_mean" if request["center"] else None, "subtracted_mean": center,
                            "applied_offset": shift, "original_labels": original}
        target_transform["transformed_labels"] = {}
        for name, value in (("train_y", train_y), ("test_y", test_y)):
            value = value.detach().cpu().contiguous()
            target_transform["transformed_labels"][name] = {"sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                                                          "mean": float(value.double().mean())}
        target_transform["runner_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    process = validate_process_request(candidate_spec["process"], candidate_spec) if "process" in candidate_spec else None
    if process is not None and (process != candidate_spec["process"] or device.type != "cpu"):
        raise ValueError("process execution requires normalized CPU request")
    if process is not None and (test_x.shape[0] > process["max_eval_samples"] or not test_x.shape[0]):
        raise ValueError("process evaluation sample count exceeds bound or is empty")
    train_mod = load_candidate_train(candidate_path)
    if not hasattr(train_mod, "train_and_eval"):
        raise RuntimeError(
            f"{candidate_path}/train.py must define train_and_eval(); regenerate the candidate"
        )

    kwargs = {
        "steps": int(candidate_spec["budget"]["training_steps"]),
        "batch_size": int(candidate_spec["budget"]["batch_size"]),
        "seed": seed,
        "fail_threshold": fail_threshold,
    }
    parameters = inspect.signature(train_mod.train_and_eval).parameters
    if process is not None:
        if "process" not in parameters:
            raise ValueError("generated train.py lacks requested process capability")
        kwargs["process"] = process
    supports_device = "device" in parameters
    if supports_device:
        kwargs["device"] = str(device)
    elif device.type != "cpu":
        raise RuntimeError(
            f"{candidate_path}/train.py predates device-aware execution; "
            "regenerate this CUDA candidate from candidate_spec.json"
        )
    if progress_callback is not None and "progress_callback" in parameters:
        kwargs["progress_callback"] = progress_callback
    result = train_mod.train_and_eval(
        train_x,
        train_y,
        test_x,
        test_y,
        **kwargs,
    )
    final_key = final_metric_key(selection_metric)
    if final_key not in result:
        raise KeyError(f"train_and_eval missing {final_key!r}")
    seed_result: dict[str, Any] = {
        "seed": seed,
        "failed": bool(result["failed"]),
        final_key: float(result[final_key]),
        "eval_samples": list(result["eval_samples"]),
        "step_metrics": list(result["step_metrics"]),
    }
    if "final_test_accuracy" in result:
        seed_result["final_test_accuracy"] = float(result["final_test_accuracy"])
    if target_transform is not None:
        seed_result["target_transform"] = target_transform
    if process is not None:
        diagnostics = result.get("process_diagnostics")
        if not isinstance(diagnostics, dict) or set(diagnostics) != {"record", "evaluation"}:
            raise ValueError("training returned invalid process diagnostics")
        record, evaluation = diagnostics["record"], diagnostics["evaluation"]
        if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("recorder_version") != process["version"] or record.get("seed") != seed or record.get("request") != process:
            raise ValueError("process record identity mismatch")
        if process["version"] == "regression_mse_v1":
            _validate_regression_diagnostics(diagnostics, candidate_spec, (train_x, train_y, test_x, test_y), seed_result)
            seed_result["process_diagnostics"] = diagnostics
            return seed_result
        defaults = record.get("optimizer_defaults", {})
        optimizer = candidate_spec["optimizer"]
        if record.get("optimizer_type") != "Adam" or record.get("decay_mode") != "coupled" or any(
            defaults.get(key) != value for key, value in {"lr": optimizer["lr"],
            "weight_decay": optimizer.get("weight_decay", 0.0), "betas": optimizer.get("betas", [0.9, 0.999]), "eps": 1e-8}.items()
        ):
            raise ValueError("process actual optimizer defaults mismatch")
        if len(json.dumps(record, allow_nan=False).encode()) > 8 * 1024 * 1024:
            raise ValueError("process scalar record exceeds 8 MiB bound")
        observed = [row.get("step") for row in record.get("steps", [])]
        if observed != sorted(set(observed)) or any(step not in process["steps"] for step in observed):
            raise ValueError("invalid process observation steps")
        if not seed_result["failed"] and observed != process["steps"]:
            raise ValueError("requested process observations are missing")
        for row in record.get("steps", []):
            parameters = row.get("parameters", [])
            if not 1 <= len(parameters) <= 256 or len({p.get("name") for p in parameters}) != len(parameters):
                raise ValueError("invalid process parameter records")
            for parameter in parameters:
                if not isinstance(parameter.get("name"), str) or parameter.get("role") not in {"head", "hidden"} or not isinstance(parameter.get("active"), bool):
                    raise ValueError("invalid process parameter identity")
                scalars = ("parameter_norm", "descent_norm") + (("data_gradient_norm", "decay_gradient_norm",
                          "coupled_gradient_norm", "preconditioned_direction_norm", "predicted_descent_norm") if parameter["active"] else ())
                if any(isinstance(parameter.get(key), bool) or not isinstance(parameter.get(key), (int, float)) or parameter[key] < 0 for key in scalars):
                    raise ValueError("invalid process norm")
                for key in ("moment_before", "moment_after"):
                    if set(parameter.get(key, {})) != {"exp_avg", "exp_avg_sq"} or any(
                        not isinstance(v, (int, float)) or v < 0 for v in parameter[key].values()
                    ):
                        raise ValueError("invalid process moment norm")
                if parameter["active"] and parameter.get("adam_step_check", {}).get("passed") is not True:
                    raise ValueError("process Adam step check failed")
                for key, value in parameter.items():
                    if key.endswith(("_cosine", "_ratio")) or key == "relative_descent":
                        if not isinstance(value, dict) or set(value) != {"value", "reason"} or (
                            value["value"] is None and not isinstance(value["reason"], str)
                        ) or (value["value"] is not None and (not isinstance(value["value"], (int, float)) or value["reason"] is not None)):
                            raise ValueError("invalid process ratio/angle schema")
        for name, value in zip(("train_x", "train_y", "test_x", "test_y"), (train_x, train_y, test_x, test_y)):
            cpu = value.detach().cpu().contiguous()
            pin = {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
                   "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest()}
            if record.get("inputs", {}).get(name) != pin:
                raise ValueError("process dataset input hash mismatch")
        if not isinstance(evaluation, dict):
            raise ValueError("invalid process evaluation payload")
        if seed_result["failed"]:
            if record.get("status") != "failed" or evaluation:
                raise ValueError("failed process execution returned final evaluation")
        else:
            expected = {"indices", "logits", "targets", "predictions", "errors", "true_class_margins"}
            if set(evaluation) != expected or any(not isinstance(v, torch.Tensor) for v in evaluation.values()):
                raise ValueError("invalid process evaluation tensor schema")
            size, classes = test_x.shape[0], int(candidate_spec["model"]["output_dim"])
            logits = evaluation["logits"]
            targets = test_y.reshape(-1).cpu()
            if logits.shape != (size, classes) or not torch.isfinite(logits).all() or any(
                evaluation[key].shape != (size,) for key in expected - {"logits"}
            ):
                raise ValueError("invalid process evaluation shapes or nonfinite logits")
            if not torch.equal(evaluation["indices"], torch.arange(size)) or not torch.equal(evaluation["targets"], targets):
                raise ValueError("process evaluation order mismatch")
            predictions = logits.argmax(dim=-1)
            other = logits.clone()
            other.scatter_(1, targets[:, None], float("-inf"))
            margins = logits.gather(1, targets[:, None]).reshape(-1) - other.max(dim=1).values
            ce = float(torch.nn.functional.cross_entropy(logits, targets))
            accuracy = float((predictions == targets).float().mean())
            if not torch.equal(evaluation["predictions"], predictions) or not torch.equal(evaluation["errors"], predictions != targets) or not torch.equal(evaluation["true_class_margins"], margins):
                raise ValueError("process decision readback mismatch")
            metrics = record.get("final_metrics", {})
            if record.get("status") != "completed" or not math.isclose(metrics.get("ce", float("nan")), ce, rel_tol=1e-6, abs_tol=1e-7) or not math.isclose(metrics.get("accuracy", float("nan")), accuracy, abs_tol=1e-7):
                raise ValueError("process final metric readback mismatch")
            if not math.isclose(ce, seed_result[final_key], rel_tol=1e-6, abs_tol=1e-7) or accuracy != seed_result["final_test_accuracy"]:
                raise ValueError("process logits disagree with final training metric")
        seed_result["process_diagnostics"] = diagnostics
    return seed_result
def _run_single_seed_cpu_worker(
    payload: tuple[str, str, dict[str, Any], int, float, str, int],
) -> dict[str, Any]:
    """Run one CPU seed in an isolated process without writing artifacts."""
    candidate_text, dataset_text, spec, seed, fail_threshold, selection_metric, torch_threads = payload
    torch.set_num_threads(torch_threads)
    from architecture_iq.registry import ensure_registries

    ensure_registries()
    candidate_path = Path(candidate_text)
    dataset_path = Path(dataset_text)
    family = get_dataset_family(spec["family"])
    train_x, train_y, test_x, test_y = family.load_tensors(dataset_path)
    return run_single_seed(
        candidate_path,
        spec,
        train_x,
        train_y,
        test_x,
        test_y,
        seed,
        fail_threshold,
        selection_metric=selection_metric,
        device=torch.device("cpu"),
    )


def run_ground_truth(
    candidate_path: Path,
    profile: Profile,
    dataset_path: Path | None = None,
    *,
    sync_files: bool = True,
    fail_threshold_override: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    from architecture_iq.util import read_json

    candidate_path = candidate_path.resolve()
    spec = read_json(candidate_path / "candidate_spec.json")
    if "target_transform" in spec and validate_target_transform(spec["target_transform"], spec) != spec["target_transform"]:
        raise ValueError("target_transform must be normalized")
    if "process" in spec and validate_process_request(spec["process"], spec) != spec["process"]:
        raise ValueError("process request must be normalized")
    device = _resolve_execution_device(spec, profile)
    if sync_files:
        _sync_candidate_files(candidate_path, spec)

    family_name = spec["family"]
    family = get_dataset_family(family_name)
    if dataset_path is None:
        dataset_path = candidate_path.parents[2]
    dataset_path = dataset_path.resolve()
    train_x, train_y, test_x, test_y = family.load_tensors(dataset_path)
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    # Candidate specs are reusable across counterfactual datasets, but only
    # when the paired datasets preserve the tensor contract. Fail before
    # importing/running train.py so a malformed pair is recorded as a clean
    # compatibility rejection rather than a low-level matmul traceback.
    model_spec = spec.get("model", {})
    if model_spec.get("type") == "mlp" and train_x.ndim >= 2:
        expected_dim = int(model_spec.get("input_dim", train_x.shape[-1]))
        actual_dim = int(train_x.shape[-1])
        if expected_dim != actual_dim:
            raise ValueError(
                "candidate/dataset shape mismatch: "
                f"model.input_dim={expected_dim}, dataset.feature_dim={actual_dim}"
            )
    if model_spec.get("type") == "mlp" and train_y.ndim >= 1:
        output_dim = model_spec.get("output_dim")
        if output_dim is not None and train_y.ndim == 1:
            labels = train_y.detach().cpu()
            if labels.numel() and int(labels.max().item()) >= int(output_dim):
                raise ValueError(
                    "candidate/dataset label mismatch: "
                    f"model.output_dim={int(output_dim)}, dataset.max_label={int(labels.max().item())}"
                )
    selection_metric = dataset_spec["selection_metric"]
    final_key = final_metric_key(selection_metric)
    sig_cfg = dataset_spec.get("significance", {})
    gt_cfg = profile.ground_truth
    fail_threshold = (
        float(fail_threshold_override)
        if fail_threshold_override is not None
        else float(sig_cfg.get("fail_threshold", gt_cfg["fail_threshold"]))
    )
    batch_size = int(spec["budget"]["batch_size"])

    n_seeds = profile.n_seeds
    base_seed = profile.base_seed
    seed_results: list[dict[str, Any]] = []
    seed_workers, seed_torch_threads = _seed_parallelism_config(
        device, n_seeds, progress_callback
    )
    if seed_workers == 1:
        # Seed-parallel workers set their own per-worker thread counts; the
        # plain path clamps so N parallel candidate processes cannot each
        # grab every core (see DEFAULT_TORCH_THREADS above).
        _clamp_process_torch_threads()

    training_steps = int(spec["budget"]["training_steps"])
    total_samples_seen = int(spec["budget"]["total_samples_seen"])
    if seed_workers > 1:
        assert seed_torch_threads is not None
        payloads = [
            (
                str(candidate_path),
                str(dataset_path),
                spec,
                base_seed + i,
                fail_threshold,
                selection_metric,
                seed_torch_threads,
            )
            for i in range(n_seeds)
        ]
        with ProcessPoolExecutor(max_workers=seed_workers) as executor:
            # executor.map preserves input order for paired seed significance.
            seed_results = list(executor.map(_run_single_seed_cpu_worker, payloads))
    else:
        for i in range(n_seeds):
            seed_index = i + 1
            seed = base_seed + i
            base_event = {
                "seed_index": seed_index,
                "n_seeds": n_seeds,
                "seed": seed,
                "training_steps": training_steps,
                "total_samples_seen": total_samples_seen,
                "selection_metric": selection_metric,
            }
            _emit_progress(progress_callback, {"phase": "seed_started", **base_event})

            def report_evaluation(
                event: dict[str, Any],
                *,
                context: dict[str, Any] = base_event,
            ) -> None:
                _emit_progress(progress_callback, {"phase": "evaluation", **context, **event})

            seed_result = run_single_seed(
                candidate_path,
                spec,
                train_x,
                train_y,
                test_x,
                test_y,
                seed,
                fail_threshold,
                selection_metric=selection_metric,
                device=device,
                progress_callback=report_evaluation,
            )
            seed_results.append(seed_result)
            _emit_progress(
                progress_callback,
                {
                    "phase": "seed_finished",
                    **base_event,
                    "failed": bool(seed_result["failed"]),
                    "metric": float(seed_result[final_key]),
                },
            )
    ok = [r for r in seed_results if not r["failed"]]
    failed_count = len(seed_results) - len(ok)
    finals = [r[final_key] for r in ok] or [float("inf")]
    accuracies = [r["final_test_accuracy"] for r in ok if "final_test_accuracy" in r]

    max_len = max((len(r["step_metrics"]) for r in ok), default=0)
    curves = np.full((n_seeds, max_len), np.nan, dtype=np.float64)
    sample_axis: list[int] | None = None
    for i, r in enumerate(seed_results):
        if r["failed"]:
            continue
        curves[i, : len(r["step_metrics"])] = r["step_metrics"]
        if sample_axis is None:
            sample_axis = r["eval_samples"]

    mean_key = mean_metric_key(selection_metric)
    std_key = f"std_{selection_metric}"
    summary = {
        "schema_version": profile.schema_version,
        "candidate_id": spec["candidate_id"],
        "selection_metric": selection_metric,
        "execution": "candidate_py_files",
        **({"target_transform": spec["target_transform"]} if "target_transform" in spec else {}),
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "failed_seeds": failed_count,
        "excluded": failed_count >= int(profile.ground_truth["max_failed_seeds"]),
        mean_key: float(np.mean(finals)) if ok else float("inf"),
        std_key: float(np.std(finals)) if ok else float("inf"),
        **(
            {
                "mean_test_mse": float(np.mean(finals)) if ok else float("inf"),
                "std_test_mse": float(np.std(finals)) if ok else float("inf"),
            }
            if selection_metric == "test_mse"
            else {}
        ),
        **(
            {
                "mean_test_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
                "std_test_accuracy": float(np.std(accuracies)) if accuracies else float("nan"),
            }
            if any("final_test_accuracy" in r for r in seed_results)
            else {}
        ),
        "seed_results": [
            {
                "seed": r["seed"],
                "failed": r["failed"],
                final_key: r[final_key],
                **(
                    {"final_test_mse": r[final_key], "mean_test_mse": r[final_key]}
                    if selection_metric == "test_mse"
                    else {}
                ),
                **({"final_test_accuracy": r["final_test_accuracy"]} if "final_test_accuracy" in r else {}),
                **({"target_transform": r["target_transform"]} if "target_transform" in r else {}),
            }
            for r in seed_results
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "requested_device": spec.get("execution", {}).get("device", "cpu"),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_device_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "seed_workers": seed_workers,
            "torch_threads_per_seed": seed_torch_threads or torch.get_num_threads(),
            "git_commit": git_commit_hash(ROOT),
        },
    }
    summary = {k: v for k, v in summary.items() if v is not None}

    results_dir = candidate_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if "process" in spec:
        process_files = {}
        for row in seed_results:
            diagnostics = row["process_diagnostics"]
            record = dict(diagnostics["record"], profile=profile.name, candidate_id=spec["candidate_id"],
                          selection_metric=selection_metric, dataset_id=spec["dataset_id"])
            stem = f"seed_{row['seed']}"
            directory = results_dir / "process"
            directory.mkdir(exist_ok=True)
            json_path, tensor_path = directory / (stem + ".json"), directory / (stem + ".npz")
            json_path.write_text(json.dumps(record, indent=2, allow_nan=False))
            np.savez(tensor_path, **{key: value.detach().cpu().numpy() for key, value in diagnostics["evaluation"].items()})
            for path in (json_path, tensor_path):
                payload = path.read_bytes()
                if len(payload) > 8 * 1024 * 1024:
                    raise ValueError("process artifact exceeds 8 MiB bound")
                process_files[str(path.relative_to(candidate_path))] = {"bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest()}
        summary["process"] = {"recorder_version": spec["process"]["version"], "request": spec["process"],
                              "files": process_files}
    write_json(results_dir / "summary.json", summary)
    np.savez(
        results_dir / "curves.npz",
        curves=curves,
        samples=np.asarray(sample_axis or [], dtype=np.int64),
        batch_size=batch_size,
    )
    return summary
