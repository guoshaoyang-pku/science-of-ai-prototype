"""Vectorized deterministic quadratic optimizer dynamics, with a frozen-g0 twin.

Only in-memory simulation is implemented here. Callers own protocols and storage.
Every task has L(theta)=.5*(theta-optimum)^T H (theta-optimum)+loss_floor.
"""
from __future__ import annotations

import numpy as np


METHODS = ("SGD", "Momentum", "Adam", "AdamW", "RMSprop", "Adagrad")


def _vector(value, n: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(n, float(result), dtype=np.float64)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite scalar or shape ({n},)")
    return result


def _descent_cosine(gradient: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Positive means descent; zero-gradient or zero-update values are undefined."""
    numerator = -np.einsum("nd,nd->n", gradient, delta)
    denominator = np.linalg.norm(gradient, axis=-1) * np.linalg.norm(delta, axis=-1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def simulate(H, theta0, optimum, eta, method, params, steps, noise=None,
             loss_floor=None):
    """Simulate N independent float64 tasks using PyTorch optimizer conventions.

    Inputs: H:(N,D,D), theta0/optimum:(N,D), eta:(N,) or scalar.
    ``params`` entries can be scalars or task vectors. Supported entries:
    mu (.9), beta1 (.9), beta2 (.999), alpha (.99), eps (1e-8),
    weight_decay (.01 for AdamW, otherwise zero), adagrad_init (zero).
    SGD/Momentum use no dampening or Nesterov; RMSprop has no momentum or
    centering. Adam uses bias correction. Epsilon is outside every square root.
    Non-AdamW weight decay is coupled to the loss gradient, as in PyTorch.
    noise:(steps,N,D), if supplied, is added only to the real loss gradient.
    The frozen branch always receives the original unregularized g0; decay
    remains an optimizer operation in both branches.

    ``loss``, displacements, projection and ``descent_cosine`` have shape
    (N,steps+1). The cosine at index t compares the update ending at t to the
    unnoisy quadratic gradient at t-1; index zero is NaN. ``u_actual`` is the
    net displacement norm, not cumulative path length. Projection is NaN if
    ||g0||=0. Failed rows retain earlier valid samples and are NaN thereafter;
    their valid masks are monotone and first-failure indices use -1 for none.
    """
    H = np.asarray(H, dtype=np.float64)
    theta0 = np.asarray(theta0, dtype=np.float64)
    optimum = np.asarray(optimum, dtype=np.float64)
    if H.ndim != 3 or H.shape[1] != H.shape[2]:
        raise ValueError("H must have shape (N,D,D)")
    n, d, _ = H.shape
    if n < 1 or d < 1 or theta0.shape != (n, d) or optimum.shape != (n, d):
        raise ValueError("theta0 and optimum must match nonempty H task dimensions")
    if not all(np.isfinite(x).all() for x in (H, theta0, optimum)):
        raise ValueError("H, theta0 and optimum must be finite")
    if not np.allclose(H, H.transpose(0, 2, 1), rtol=1e-12, atol=1e-14):
        raise ValueError("H must be symmetric")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if noise is not None:
        noise = np.asarray(noise, dtype=np.float64)
        if noise.shape != (steps, n, d):
            raise ValueError("noise must have shape (steps,N,D)")
    params = {} if params is None else dict(params)
    rates = _vector(eta, n, "eta")
    if (rates < 0).any():
        raise ValueError("eta must be nonnegative")
    floor = _vector(0.0 if loss_floor is None else loss_floor, n, "loss_floor")
    defaults = {"mu": .9, "beta1": .9, "beta2": .999, "alpha": .99,
                "eps": 1e-8, "weight_decay": .01 if method == "AdamW" else 0.,
                "adagrad_init": 0.}
    unknown = set(params).difference(defaults)
    if unknown:
        raise ValueError(f"unsupported optimizer parameters: {sorted(unknown)}")
    pv = {key: _vector(params.get(key, default), n, key)
          for key, default in defaults.items()}
    for key in ("mu", "beta1", "beta2", "alpha"):
        if ((pv[key] < 0) | (pv[key] >= 1)).any():
            raise ValueError(f"{key} must lie in [0,1)")
    for key in ("eps", "weight_decay", "adagrad_init"):
        if (pv[key] < 0).any():
            raise ValueError(f"{key} must be nonnegative")
    # Branch 0 is the real dynamics; branch 1 is constant-initial-gradient.
    p = {key: value[None, :, None] for key, value in pv.items()}
    lr = rates[None, :, None]
    theta = np.stack([theta0.copy(), theta0.copy()])
    state = {}
    if method == "Momentum":
        state["velocity"] = np.zeros_like(theta)
    elif method in ("Adam", "AdamW"):
        state["m"] = np.zeros_like(theta)
        state["v"] = np.zeros_like(theta)
    elif method == "RMSprop":
        state["v"] = np.zeros_like(theta)
    elif method == "Adagrad":
        state["v"] = np.broadcast_to(p["adagrad_init"], theta.shape).copy()

    def gradient(value):
        return np.einsum("nij,nj->ni", H, value - optimum)

    def update(value, grad, t):
        # These are full optimizer updates, not gradient-only displacements.
        if method != "AdamW":
            grad = grad + p["weight_decay"] * value
        if method == "SGD":
            return value - lr * grad
        if method == "Momentum":
            state["velocity"] *= p["mu"]
            state["velocity"] += grad
            return value - lr * state["velocity"]
        if method in ("Adam", "AdamW"):
            state["m"] *= p["beta1"]
            state["m"] += (1. - p["beta1"]) * grad
            state["v"] *= p["beta2"]
            state["v"] += (1. - p["beta2"]) * grad * grad
            mhat = state["m"] / (1. - p["beta1"] ** t)
            vhat = state["v"] / (1. - p["beta2"] ** t)
            if method == "AdamW":
                value = value * (1. - lr * p["weight_decay"])
            return value - lr * mhat / (np.sqrt(vhat) + p["eps"])
        if method == "RMSprop":
            state["v"] *= p["alpha"]
            state["v"] += (1. - p["alpha"]) * grad * grad
        else:
            state["v"] += grad * grad
        return value - lr * grad / (np.sqrt(state["v"]) + p["eps"])

    shape = (n, steps + 1)
    loss = np.full(shape, np.nan)
    loss_const = np.full(shape, np.nan)
    u_const = np.full(shape, np.nan)
    projection = np.full(shape, np.nan)
    u_actual = np.full(shape, np.nan)
    cosine = np.full(shape, np.nan)
    grad_cos_initial = np.full(shape, np.nan)
    valid_actual = np.zeros(shape, dtype=bool)
    valid_const = np.zeros(shape, dtype=bool)
    first_bad_actual = np.full(n, -1, dtype=np.int64)
    first_bad_const = np.full(n, -1, dtype=np.int64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        g0 = gradient(theta0)
        gnorm2 = np.einsum("nd,nd->n", g0, g0)
        initial_excess = .5 * np.einsum("nd,nd->n", theta0 - optimum, g0)
        limit = 1e12 * np.maximum(initial_excess, 1e-30)
        loss[:, 0] = initial_excess + floor
        active = np.isfinite(loss[:, 0]) & np.isfinite(g0).all(axis=1) & np.isfinite(gnorm2)
        const_active = active.copy()
        valid_actual[:, 0] = active
        valid_const[:, 0] = const_active
        loss[~active, 0] = np.nan
        loss_const[const_active, 0] = loss[const_active, 0]
        grad_cos_initial[active & (gnorm2 > 0), 0] = 1.
        u_actual[active, 0] = 0.
        u_const[const_active, 0] = 0.
        projection[const_active & (gnorm2 > 0), 0] = 0.
        first_bad_actual[~active] = 0
        first_bad_const[~const_active] = 0
        theta[0, ~active] = np.nan
        theta[1, ~const_active] = np.nan
        current_g = g0.copy()
        initial_cosine = np.full((2, n), np.nan)
        for t in range(1, steps + 1):
            observed_g = current_g if noise is None else current_g + noise[t - 1]
            grads = np.stack([observed_g, g0])
            next_theta = update(theta, grads, t)
            delta = next_theta - theta
            step_cosine = _descent_cosine(current_g, delta[0])
            if t == 1:
                initial_cosine[0] = _descent_cosine(g0, delta[0])
                initial_cosine[1] = _descent_cosine(g0, delta[1])
            next_g = gradient(next_theta[0])
            excess = .5 * np.einsum("nd,nd->n", next_theta[0] - optimum, next_g)
            total = excess + floor
            const_g = gradient(next_theta[1])
            const_total = .5 * np.einsum("nd,nd->n", next_theta[1] - optimum, const_g) + floor
            displacement = next_theta - theta0[None, :, :]
            distances = np.linalg.norm(displacement, axis=2)
            projected = np.divide(-np.einsum("nd,nd->n", g0, displacement[1]),
                                  gnorm2, out=np.full(n, np.nan), where=gnorm2 > 0)
            finite_state = np.isfinite(next_theta).all(axis=2)
            for value in state.values():
                finite_state &= np.isfinite(value).all(axis=2)
            good = (active & finite_state[0] & np.isfinite(total)
                    & np.isfinite(distances[0]) & (excess <= limit))
            good_const = (const_active & finite_state[1] & np.isfinite(distances[1]) & np.isfinite(const_total)
                          & (np.isfinite(projected) | (gnorm2 == 0)))
            first_bad_actual[active & ~good] = t
            first_bad_const[const_active & ~good_const] = t
            loss[good, t] = total[good]
            loss_const[good_const, t] = const_total[good_const]
            u_actual[good, t] = distances[0, good]
            u_const[good_const, t] = distances[1, good_const]
            projection[good_const, t] = projected[good_const]
            cosine[good, t] = step_cosine[good]
            current_cos_initial = _descent_cosine(g0, -next_g)
            grad_cos_initial[good, t] = current_cos_initial[good]
            if t == 1:
                initial_cosine[0, ~good] = np.nan
                initial_cosine[1, ~good_const] = np.nan
            valid_actual[:, t] = good
            valid_const[:, t] = good_const
            active, const_active = good, good_const
            next_theta[0, ~active] = np.nan
            next_theta[1, ~const_active] = np.nan
            theta, current_g = next_theta, next_g
    return {
        "loss": loss, "loss_const": loss_const, "u_const": u_const, "projection_const": projection,
        "u_actual": u_actual, "descent_cosine": cosine, "grad_cos_initial": grad_cos_initial,
        "valid_actual": valid_actual, "valid_const": valid_const,
        "diverged_actual": first_bad_actual >= 0,
        "diverged_const": first_bad_const >= 0,
        "first_divergence_step_actual": first_bad_actual,
        "first_divergence_step_const": first_bad_const,
        "initial_gradient": g0, "initial_gradient_norm": np.sqrt(gnorm2),
        "initial_theta_norm": np.linalg.norm(theta0, axis=1),
        "initial_direction_cosine_actual": initial_cosine[0],
        "initial_direction_cosine_const": initial_cosine[1],
        "theta_final_actual": theta[0].copy(),
        "theta_final_const": theta[1].copy(),
    }
