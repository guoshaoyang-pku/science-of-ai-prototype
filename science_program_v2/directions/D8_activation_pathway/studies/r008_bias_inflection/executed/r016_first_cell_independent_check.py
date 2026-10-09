from pathlib import Path
import hashlib
import json
import numpy as np

study = Path(__file__).resolve().parents[1]
data_path = study / "executed/data.npz"
row_path = study / "results/s101_zero_a0.025.json"
array_path = study / "results/s101_zero_a0.025.npz"
with np.load(data_path, allow_pickle=False) as z:
    data = {name: z[name].copy() for name in z.files}
row = json.loads(row_path.read_text())
with np.load(array_path, allow_pickle=False) as z:
    saved = {name: z[name].copy() for name in z.files}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

def silu(x):
    return x / (1.0 + np.exp(-x))

u = np.einsum("nd,wd->nw", data["train_x"], saved["hidden_weight"])
bias = row["request"]["bias"]
scale = row["request"]["cell"]["scale"]
positive = silu(bias + scale * u)
negative = silu(bias - scale * u)
center = positive.mean(axis=0)
norm = np.sqrt(np.mean((positive - center) ** 2))
phi = (positive - center) / norm / saved["train_features"].shape[1] ** 0.5
kernel = np.einsum("nw,mw->nm", phi, phi) / len(phi)
residual = -data["train_y"].copy()
curve = []
for _ in range(saved["train_curve"].shape[0]):
    curve.append(np.mean(residual ** 2, axis=0))
    residual -= 2 * 0.8 * np.einsum("nm,mk->nk", kernel, residual)
curve = np.asarray(curve)
checkpoint_prediction = np.einsum("nw,wk->nk", phi, saved["final_head"])
finite = all(np.isfinite(a).all() for a in saved.values())
out = {
    "cell_id": row["cell_id"],
    "row_sha256": sha(row_path),
    "arrays_sha256": sha(array_path),
    "finite_saved_arrays": bool(finite),
    "preactivation_max_abs_error_einsum": maxerr(u, saved["train_u"]),
    "feature_max_abs_error_einsum": maxerr(phi, saved["train_features"]),
    "kernel_recurrence_max_abs_error": maxerr(curve, saved["train_curve"]),
    "checkpoint_prediction_max_abs_error_einsum": maxerr(checkpoint_prediction, saved["checkpoint_outputs"][-1]),
    "saved_spectral_curve_max_abs_error": float(np.max(np.abs(saved["train_curve"] - saved["spectral_curve"]))),
    "method": "独立NumPy einsum重算；不修改pinned run.py/analysis.py"
}
(study / "executed/r016_first_cell_independent_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
