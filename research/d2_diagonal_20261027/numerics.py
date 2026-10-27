"""Small numerical primitives; no data access or execution on import."""
import numpy as np
from scipy.optimize import minimize_scalar


def base_grid(max_steps=262144):
    return np.unique(np.r_[np.arange(257), np.rint(np.geomspace(257, max_steps, 4096)).astype(int)])


def diagonal_bias(f, h, linear, y2):
    return (f * f) @ np.diag(h) - 2 * (f @ linear) + y2


def select_integer(grid, risk, objective):
    j = int(np.argmin(risk))
    if j == 0 or j == len(grid) - 1:
        return int(grid[j])
    lo, hi = int(grid[j - 1]), int(grid[j + 1])
    result = minimize_scalar(lambda t: float(objective(np.array([t]))[0]),
                             bounds=(lo, hi), method='bounded', options={'xatol': .05})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Minimum refinement failed')
    candidates = np.unique(np.clip(np.r_[np.arange(int(result.x)-2, int(result.x)+4),
                                         lo, int(grid[j]), hi], grid[0], grid[-1])).astype(int)
    values = objective(candidates)
    if not np.all(np.isfinite(values)):
        raise RuntimeError('Nonfinite search candidate')
    return int(candidates[np.argmin(values)])


def time_factor(predicted, reference):
    if reference <= 0:
        return None
    if predicted <= 0:
        return None
    return float(max(predicted/reference, reference/predicted))


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(['min', 'p25', 'median', 'p75', 'p90', 'p95', 'max'],
                    map(float, np.quantile(a, [0, .25, .5, .75, .9, .95, 1])))) if len(a) else None
