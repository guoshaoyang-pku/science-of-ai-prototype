"""Training loop for this candidate — executed by the ground-truth runner."""
from __future__ import annotations

import math
import hashlib

import torch

from loss import loss_fn
from model import Model
from optimizer import build_optimizer


def _resolve_device(device: str) -> torch.device:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported execution device {device!r}; choose 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    return torch.device(device)


def _test_mse(model: torch.nn.Module, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    model.eval()
    with torch.inference_mode():
        pred = model(test_x)
        return float(torch.mean((pred - test_y) ** 2).item())


def train_and_eval(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int = 0,
    fail_threshold: float = float("inf"),
    device: str = "cpu",
    progress_callback=None,
    process=None,
) -> dict:
    torch.manual_seed(seed)
    run_device = _resolve_device(device)
    if run_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Model().to(run_device)
    optimizer = build_optimizer(model)
    train_x = train_x.to(run_device)
    train_y = train_y.to(run_device)
    test_x = test_x.to(run_device)
    test_y = test_y.to(run_device)
    recorder = _RegressionRecorder(model, optimizer, process, seed, (train_x, train_y, test_x, test_y))
    n = train_x.shape[0]
    step_metrics: list[float] = []
    eval_samples: list[int] = []
    failed = False
    progress_interval = max(1, steps // 100)

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch_size,), device=run_device)
        pred = model(train_x[idx])
        loss = loss_fn(model, pred, train_y[idx])
        if not torch.isfinite(loss):
            failed = True
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        observation = recorder.before(step, idx)
        optimizer.step()
        recorder.after(observation)

        metric = _test_mse(model, test_x, test_y)
        if not math.isfinite(metric):
            failed = True
            break
        eval_samples.append(step * batch_size)
        step_metrics.append(metric)
        if progress_callback is not None and (
            step == 1 or step % progress_interval == 0 or step == steps
        ):
            progress_callback(
                {
                    "step": step,
                    "training_steps": steps,
                    "samples_seen": step * batch_size,
                    "total_samples_seen": steps * batch_size,
                    "metric": metric,
                }
            )

    final_metric = step_metrics[-1] if step_metrics else float("inf")
    if final_metric > fail_threshold:
        failed = True

    return {
        "process_diagnostics": recorder.finish(model, test_x, test_y, failed),
        "failed": failed,
        "final_test_mse": final_metric,
        "eval_samples": eval_samples,
        "step_metrics": step_metrics,
    }


def train(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    """Minimal training entrypoint (no evaluation)."""
    train_and_eval(
        train_x,
        train_y,
        test_x=train_x,
        test_y=train_y,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        fail_threshold=float("inf"),
        device=device,
    )

def _tensor_pin(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def _norm(value):
    return float(torch.linalg.vector_norm(value.detach().double()).item())


def _ratio(numerator, denominator):
    if denominator == 0:
        return {"value": None, "reason": "zero denominator"}
    return {"value": numerator / denominator, "reason": None}


def _cosine(left, right):
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        return {"value": None, "reason": "zero vector"}
    return {"value": float(torch.sum(left.double() * right.double()).item()) / denominator, "reason": None}



class _RegressionRecorder:
    def __init__(self, model, optimizer, request, seed, tensors):
        if request.get("version") != "regression_mse_v1" or type(optimizer) not in (torch.optim.Adam, torch.optim.SGD) or len(optimizer.param_groups) != 1:
            raise ValueError("unsupported regression recorder")
        if any(t.shape[0] > request["max_eval_samples"] or not t.numel() for t in tensors):
            raise ValueError("regression observation inputs exceed bound or are empty")
        self.model, self.optimizer, self.request = model, optimizer, request
        self.tensors, self.named = tensors, list(model.named_parameters())
        self.initial = {name: p.detach().clone() for name, p in self.named}
        self.head = f"net.{len(model.net) - 1}."
        self.stream, self.steps = hashlib.sha256(), []
        self.initial_train, self.initial_test, metrics = self.evaluate()
        self.record = {"schema_version": 1, "recorder_version": request["version"], "request": request,
                       "seed": seed, "optimizer_type": type(optimizer).__name__, "initial_metrics": metrics,
                       "initial_parameters": {name: _tensor_pin(p) for name, p in self.named},
                       "optimizer_defaults": {k: list(v) if isinstance(v, tuple) else v
                                              for k, v in optimizer.param_groups[0].items() if k != "params"},
                       "inputs": {k: _tensor_pin(t) for k, t in zip(("train_x", "train_y", "test_x", "test_y"), tensors)},
                       "steps": self.steps}

    def evaluate(self):
        was_training = self.model.training
        self.model.eval()
        tx, ty, vx, vy = self.tensors
        with torch.inference_mode():
            train, test = self.model(tx).detach().clone(), self.model(vx).detach().clone()
            metrics = {"train_mse": float(((train - ty) ** 2).mean()),
                       "test_mse": float(((test - vy) ** 2).mean()),
                       "train_prediction_mean": float(train.double().mean()),
                       "test_prediction_mean": float(test.double().mean()),
                       "train_target_mean": float(ty.double().mean()),
                       "test_target_mean": float(vy.double().mean())}
        self.model.train(was_training)
        return train, test, metrics

    def before(self, step, indices):
        self.stream.update(indices.detach().cpu().contiguous().numpy().tobytes())
        if step not in self.request["steps"]:
            return None
        rows = []
        for name, p in self.named:
            weight = p.detach().clone()
            gradient = p.grad.detach() if p.grad is not None else None
            row = {"name": name, "role": "head" if name.startswith(self.head) else "hidden",
                   "active": gradient is not None, "parameter_norm": _norm(weight)}
            if gradient is not None:
                row["data_gradient_norm"] = _norm(gradient)
            rows.append((name, p, weight, row))
        return {"step": step, "minibatch": _tensor_pin(indices), "rows": rows}

    def after(self, snapshot):
        if snapshot is None:
            return
        rows = []
        for name, p, weight, row in snapshot["rows"]:
            row.update(descent_norm=_norm(weight - p.detach()),
                       displacement_norm=_norm(p.detach() - self.initial[name]))
            rows.append(row)
        _, _, metrics = self.evaluate()
        self.steps.append({"step": snapshot["step"], "minibatch": snapshot["minibatch"],
                           "parameters": rows, "metrics": metrics})

    def finish(self, model, test_x, test_y, failed):
        self.record["status"] = "failed" if failed else "completed"
        self.record["minibatch_stream_sha256"] = self.stream.hexdigest()
        if failed:
            self.record["final_metrics"] = None
            return {"record": self.record, "evaluation": {}}
        train, test, metrics = self.evaluate()
        self.record["final_metrics"] = dict(metrics, mse=metrics["test_mse"])
        _, train_y, _, test_y = self.tensors
        return {"record": self.record, "evaluation": {"indices": torch.arange(test_y.shape[0]),
                "predictions": test.cpu(), "targets": test_y.detach().cpu(),
                "initial_predictions": self.initial_test.cpu(), "train_predictions": train.cpu(),
                "train_targets": train_y.detach().cpu(), "initial_train_predictions": self.initial_train.cpu()}}
