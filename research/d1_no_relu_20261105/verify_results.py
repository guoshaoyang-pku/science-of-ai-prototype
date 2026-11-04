"""Recompute saved endpoint statistics and check pairing; never train."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    new = load_module("no_relu_protocol", SOURCE / "run.py")
    old_dir = new.DATA_ROOT / "runs/d1_nn_u_transfer_r1/directions/D1_u_effective_time/studies/r001_nn_u_transfer"
    old = load_module("relu_reference", old_dir / "run.py")
    old_summary = json.loads((old_dir / "summary.json").read_text())
    assert hashlib.sha256((old_dir / "run.py").read_bytes()).hexdigest() == old_summary["execution_source_sha256"]
    summary_path = new.STUDY / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert hashlib.sha256((SOURCE / "run.py").read_bytes()).hexdigest() == summary["execution_source_sha256"]
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    lookup = {}
    manifests = []
    for ai, arch in enumerate(new.ARCHS):
        for seed in new.SEEDS:
            current_data, old_data = new.make_data(seed), old.make_data(seed)
            assert all(torch.equal(a, b) for a, b in zip(current_data, old_data))
            torch.manual_seed(200000 + 1000 * ai + seed)
            a = new.Student(arch["width"], arch["hidden_layers"]).double()
            torch.manual_seed(200000 + 1000 * ai + seed)
            b = old.Student(arch["width"], arch["hidden_layers"]).double()
            assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
            x, y = (value.numpy() for value in current_data[:2])
            design = np.column_stack([x, np.ones(len(x))])
            coef = np.linalg.lstsq(design, y, rcond=None)[0]
            floor = float(np.mean((design @ coef - y) ** 2))
            for method in new.METHODS:
                record = next(r for r in summary["cell_records"] if (r["architecture"], r["seed"], r["optimizer"]) == (arch["name"], seed, method))
                metrics_path = new.STUDY / record["metrics_file"]
                metrics = json.loads(metrics_path.read_text())
                curve = json.loads((new.STUDY / record["curve_file"]).read_text())
                assert metrics["steps"] == list(range(2001)) and len(curve) == 401
                loss, u = np.array(metrics["train_loss_ratio"]), np.array(metrics["U_initial_gradient"])
                assert np.isfinite(loss).all() and np.isfinite(u).all() and (np.diff(u) > 0).all()
                assert np.allclose(floor, record["affine_ols_training_loss"], rtol=1e-10, atol=1e-13)
                assert (loss * record["initial_train_loss"] >= floor - 1e-10).all()
                assert record["train_endpoints"] == new.endpoints(loss)
                assert record["validation_endpoints"] == new.endpoints(metrics["validation_loss_ratio"])
                for row in curve:
                    i = row["step"]
                    assert row["train_loss_ratio"] == loss[i] and row["U_initial_gradient"] == u[i]
                lookup[(arch["name"], seed, method)] = (loss, u, record)
                manifests.append({"path": metrics_path.relative_to(new.DATA_ROOT).as_posix(), "sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest()})
    for cell in summary["endpoint_cells"]:
        arch, seed, method, q = (cell[k] for k in ["architecture", "seed", "optimizer", "q"])
        ref_loss, ref_u, _ = lookup[(arch, seed, "SGD")]
        loss, u, record = lookup[(arch, seed, method)]
        hit = np.flatnonzero(ref_loss <= q)
        tref = int(hit[0]) if hit.size else None
        observed = np.flatnonzero(loss <= q)
        observed = int(observed[0]) if observed.size else None
        predicted = np.flatnonzero(u >= ref_u[tref]) if tref is not None else np.array([])
        predicted = int(predicted[0]) if predicted.size else None
        error = abs(predicted - observed) / observed if predicted is not None and observed is not None and observed > 0 else None
        assert (tref, observed, predicted, error) == (cell["sgd_reference_step"], cell["observed_first_step"], cell["predicted_step"], cell["relative_absolute_error"])
        assert cell["class_unattainable_threshold"] == (q < record["affine_floor_loss_ratio"])
    for method, architectures in summary["primary_by_optimizer_architecture_q"].items():
        for arch, thresholds in architectures.items():
            for q, group in thresholds.items():
                cells = [c for c in summary["endpoint_cells"] if (c["optimizer"], c["architecture"], c["q"]) == (method, arch, float(q))]
                errors = [c["relative_absolute_error"] for c in cells if c["relative_absolute_error"] is not None]
                assert group["uncensored_comparisons"] == len(errors)
                assert group["within_5pct"] == sum(e <= .05 for e in errors)
                assert group["median_relative_error"] == (statistics.median(errors) if errors else None)
    report = {"status": "passed", "paired_initializations_checked": 9, "training_cells_checked": len(lookup),
              "per_step_observations_checked": 2001 * len(lookup), "endpoint_cells_checked": len(summary["endpoint_cells"]),
              "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(), "metric_files": manifests}
    (SOURCE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (SOURCE / "summary.json").write_bytes(summary_path.read_bytes())
    (SOURCE / "relu_reference_summary.json").write_bytes((old_dir / "summary.json").read_bytes())
    print(json.dumps({k: v for k, v in report.items() if k != "metric_files"}, indent=2))


if __name__ == "__main__":
    main()
