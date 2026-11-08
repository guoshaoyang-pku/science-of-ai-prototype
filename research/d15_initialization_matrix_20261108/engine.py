"""Independent batched initialization interventions; no tasks, files, or runs.

All matrices use (batch, fan_in, fan_out). Each route minimizes its own MSE,
not its MSE divided by the route count. CPU float64, one torch CPU thread.
"""
from __future__ import annotations

import math
import numpy as np
import torch

ACTIVATIONS = ("linear", "relu", "tanh")
ARCHITECTURES = ("plain", "residual_hidden", "layernorm_hidden")
STATUS = {0: "finite", 1: "nonfinite_loss", 2: "nonfinite_gradient", 3: "nonfinite_update"}


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64, device="cpu")


def activate(h, name):
    if name == "linear":
        return h
    if name == "relu":
        return torch.relu(h)
    if name == "tanh":
        return torch.tanh(h)
    raise ValueError(f"unknown activation: {name}")


def forward_batch(weights, X, activation="linear", architecture="plain"):
    """Differentiable route-independent predictions (B,N,out).

    residual_hidden: first projection is linear; only internal square factors
    are residual branches. No extra activation follows the residual sum.
    layernorm_hidden: affine-free LayerNorm, eps=1e-5, after every nonfinal
    linear map and before its activation. There are no biases anywhere.
    """
    if activation not in ACTIVATIONS or architecture not in ARCHITECTURES:
        raise ValueError("unsupported activation or architecture")
    h = X
    depth = len(weights)
    for layer, w in enumerate(weights):
        if architecture == "residual_hidden" and 0 < layer < depth - 1:
            h = h + activate(torch.bmm(h, w), activation) / math.sqrt(max(depth - 2, 1))
        else:
            h = torch.bmm(h, w)
            if layer < depth - 1 and architecture != "residual_hidden":
                if architecture == "layernorm_hidden":
                    h = torch.nn.functional.layer_norm(h, (h.shape[-1],), eps=1e-5)
                h = activate(h, activation)
    return h


def _finite_layers(values):
    return torch.stack([torch.isfinite(w).flatten(1).all(1) for w in values]).all(0)


def _layer_norms(values):
    return torch.stack([torch.linalg.vector_norm(w.flatten(1), dim=1) for w in values], 1)


def train_batch(weights, X, y, Xv, yv, activation="linear", architecture="plain",
                optimizer="SGD", lr=0.01, steps=256, checkpoints=None):
    """Train batched independent routes, retaining all failure denominators.

    Loss is mean((prediction-target)^2) per route. SGD has no momentum; Adam
    has betas=(.9,.999), eps=1e-8, zero weight decay. Exact threshold crossings
    are checked at every integer step, independent of stored checkpoints.

    Checkpoint arrays are (B,C); layer arrays are (B,depth). first_hit fields
    equal -1 when the threshold was not reached. status_code records the first
    nonfinite event; no clipping, loss threshold, or learning-rate rescue is
    applied. Such a route is removed from subsequent training, its weights and
    moments are zeroed to protect other routes, and its subsequent losses and
    final diagnostics are NaN. Finite losses/gradients immediately preceding a
    failed update remain observations. Checkpoints always include 0 and steps.
    """
    torch.set_num_threads(1)
    if optimizer.lower() not in ("sgd", "adam"):
        raise ValueError("optimizer must be SGD or Adam")
    optimizer = optimizer.lower()
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if len(weights) < 2:
        raise ValueError("at least two weight layers required")
    X, y, Xv, yv = [tensor(v) for v in (X, y, Xv, yv)]
    if any(v.ndim != 3 for v in (X, y, Xv, yv)):
        raise ValueError("all data have shape (B,N,dim)")
    B = X.shape[0]
    params = [tensor(w).detach().clone().requires_grad_(True) for w in weights]
    d = X.shape[-1]
    for i, w in enumerate(params):
        if w.ndim != 3 or w.shape[0] != B or w.shape[1] != d:
            raise ValueError("incompatible (B,in,out) weight shape")
        if architecture == "residual_hidden" and 0 < i < len(params)-1 and w.shape[1] != w.shape[2]:
            raise ValueError("internal residual factors must be square")
        d = w.shape[2]
    if d != 1 or y.shape != (B, X.shape[1], d) or yv.shape != (B, Xv.shape[1], d):
        raise ValueError("scalar targets must match data")
    if Xv.shape[0] != B or Xv.shape[-1] != X.shape[-1]:
        raise ValueError("validation batch/input shapes must match training")
    if not all(bool(torch.isfinite(v).all()) for v in [*params, X, y, Xv, yv]):
        raise ValueError("initial parameters and data must be finite")
    eta = tensor(lr)
    if eta.ndim == 0:
        eta = eta.expand(B).clone()
    if eta.shape != (B,) or not bool(torch.isfinite(eta).all()) or bool((eta <= 0).any()):
        raise ValueError("lr must be positive finite scalar or (B,) vector")
    eta = eta[:, None, None]
    raw = range(steps+1) if checkpoints is None else list(checkpoints)
    if any(int(t) != t or not 0 <= t <= steps for t in raw):
        raise ValueError("checkpoints must be integer steps inside the run")
    checkpoint_steps = np.array(sorted(set([0, steps, *raw])), dtype=np.int64)
    index = {int(t): j for j, t in enumerate(checkpoint_steps)}
    C, depth = len(checkpoint_steps), len(params)
    train_loss = np.full((B, C), np.nan)
    val_loss = np.full((B, C), np.nan)
    status = torch.zeros(B, dtype=torch.int64)
    failed_step = torch.full((B,), -1, dtype=torch.int64)
    first_hits = {q: np.full(B, -1, dtype=np.int64) for q in (0.1, 0.01)}
    initial = [w.detach().clone() for w in params]
    initial_norms = _layer_norms(initial)
    moments = [torch.zeros_like(w) for w in params]
    squares = [torch.zeros_like(w) for w in params]
    initial_grad_norm = np.full((B, depth), np.nan)
    with torch.no_grad():
        pred0v = forward_batch(params, Xv, activation, architecture)

    def discard(bad, code, step):
        bad = bad & (status == 0)
        if not bool(bad.any()):
            return
        status[bad] = code
        failed_step[bad] = step
        with torch.no_grad():
            for p, m, v in zip(params, moments, squares):
                p[bad] = 0
                m[bad] = 0
                v[bad] = 0

    for t in range(steps+1):
        pred = forward_batch(params, X, activation, architecture)
        loss = (pred-y).square().mean(dim=(1,2))
        with torch.no_grad():
            lv = (forward_batch(params, Xv, activation, architecture)-yv).square().mean(dim=(1,2)) if t in index else None
            bad = ~torch.isfinite(loss)
            if lv is not None:
                bad |= ~torch.isfinite(lv)
        # Once a forward pass is nonfinite it cannot participate in backward;
        # regenerate the graph from sanitized parameters before differentiating.
        fresh_bad = bad & (status == 0)
        discard(fresh_bad, 1, t)
        if bool(fresh_bad.any()):
            pred = forward_batch(params, X, activation, architecture)
            loss = (pred-y).square().mean(dim=(1,2))
        alive = status == 0
        obs = loss.detach().numpy().copy()
        obs[~alive.numpy()] = np.nan
        for q, hit in first_hits.items():
            hit[(hit < 0) & (obs <= q)] = t
        if t in index:
            train_loss[:,index[t]] = obs
            vo = lv.numpy().copy()
            vo[~alive.numpy()] = np.nan
            val_loss[:,index[t]] = vo
        if not bool(alive.any()):
            break
        if t == steps and t != 0:
            break
        gradients = list(torch.autograd.grad(torch.where(alive, loss, torch.zeros_like(loss)).sum(), params))
        finite_grad = _finite_layers(gradients)
        if t == 0:
            initial_grad_norm = _layer_norms(gradients).detach().numpy().copy()
        discard(~finite_grad, 2, t)
        if t == steps:
            break
        alive = status == 0
        with torch.no_grad():
            for i, (p, grad) in enumerate(zip(params, gradients)):
                grad[~alive] = 0
                if optimizer == "sgd":
                    p.sub_(eta*grad)
                else:
                    moments[i].mul_(0.9).add_(grad, alpha=0.1)
                    squares[i].mul_(0.999).addcmul_(grad, grad, value=0.001)
                    mhat = moments[i]/(1-0.9**(t+1))
                    vhat = squares[i]/(1-0.999**(t+1))
                    p.sub_(eta*mhat/(vhat.sqrt()+1e-8))
            finite_update = _finite_layers(params) & _finite_layers(moments) & _finite_layers(squares)
        discard(~finite_update, 3, t+1)

    with torch.no_grad():
        changes = _layer_norms([p-w0 for p,w0 in zip(params,initial)])
        relative = changes/torch.where(initial_norms>0,initial_norms,torch.nan)
        finalv = forward_batch(params, Xv, activation, architecture)
        yvnorm = torch.linalg.vector_norm(yv.flatten(1),dim=1)
        prediction_change = torch.linalg.vector_norm((finalv-pred0v).flatten(1),dim=1)/torch.where(yvnorm>0,yvnorm,torch.nan)
        invalid = status != 0
        for value in (changes, relative):
            value[invalid] = torch.nan
        prediction_change[invalid] = torch.nan
        final_weights = []
        for p in params:
            arr = p.detach().numpy().copy()
            arr[invalid.numpy()] = np.nan
            final_weights.append(arr)
    return {
        "checkpoint_steps": checkpoint_steps,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "initial_loss": train_loss[:,0].copy(),
        "initial_val_loss": val_loss[:,0].copy(),
        "initial_grad_norm": initial_grad_norm,
        "initial_zero_gradient": np.all(initial_grad_norm == 0, axis=1),
        "first_hit_0_1": first_hits[0.1],
        "first_hit_0_01": first_hits[0.01],
        "status_code": status.numpy().copy(),
        "failed_step": failed_step.numpy().copy(),
        "initial_weight_norm": initial_norms.detach().numpy().copy(),
        "final_weight_delta_norm": changes.numpy().copy(),
        "final_weight_relative_change": relative.numpy().copy(),
        "final_prediction_change": prediction_change.numpy().copy(),
        "final_weights": final_weights,
    }
