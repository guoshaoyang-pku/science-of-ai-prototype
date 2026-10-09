#!/usr/bin/env python3
"""Verify all C source/data/result contracts offline; never train or call models."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checked = []
    expected = {"C01_activation_scale": 216, "C02_parity_ood": 192, "C03_hidden_boundary": 128,
                "C04_noise_generalization": 64, "C05_risk_ood": 24}
    source_files = data_files = result_arrays = contracts = 0
    for name, count in expected.items():
        study = ROOT / "studies" / name
        manifest = json.loads((study / "source_manifest.json").read_text())
        pins = manifest["executable"]
        for filename, fingerprint in pins.items():
            pinned = study / "executed" / filename
            if sha(pinned) != fingerprint:
                raise ValueError(f"Changed execution source: {name}/{filename}")
            if filename in ["run_study.py", "preregistration.json"] and sha(study / filename) != fingerprint:
                raise ValueError(f"Live executable differs from pinned source: {name}/{filename}")
            source_files += 1
        if name == "C01_activation_scale":
            datasets = {"all": {"arrays": manifest["data"], "file_sha256": manifest["data_file_sha256"]}}
        else:
            datasets = manifest["data"]
        for key, meta in datasets.items():
            filename = "data.npz" if key == "all" else "data_" + ("seed" + key if name == "C05_risk_ood" else key) + ".npz"
            path = study / filename
            if sha(path) != meta["file_sha256"]:
                raise ValueError(f"Changed data archive: {name}/{filename}")
            with np.load(path, allow_pickle=False) as arrays:
                for tensor, pin in meta["arrays"].items():
                    value = arrays[tensor]
                    if hashlib.sha256(value.tobytes()).hexdigest() != pin["sha256"] or list(value.shape) != pin["shape"]:
                        raise ValueError(f"Changed tensor: {name}/{key}/{tensor}")
            data_files += 1
        paths = sorted((study / "results").glob("*.json"))
        if len(paths) != count:
            raise ValueError(f"{name}: expected{count}results, found{len(paths)}")
        for path in paths:
            row = json.loads(path.read_text())
            if not row["finite"] or row["status"] != "completed":
                raise ValueError(f"Invalid saved result: {path.relative_to(ROOT)}")
            if sha(path.with_suffix(".npz")) != row["arrays_sha256"]:
                raise ValueError(f"Changed result arrays: {path.relative_to(ROOT)}")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
                for key in arrays.files:
                    if not np.isfinite(arrays[key]).all():
                        raise ValueError(f"Nonfinite saved arrays: {path.relative_to(ROOT)}")
            cell = row["cell"]
            if name == "C01_activation_scale":
                datapins = manifest["data"]
            elif name in ["C02_parity_ood", "C03_hidden_boundary"]:
                datapins = manifest["data"][cell["distribution"]]["arrays"]
            elif name == "C04_noise_generalization":
                key = "n" + str(cell["n"]) + "_seed" + str(cell["seed"])
                datapins = manifest["data"][key]["arrays"]
            else:
                datapins = manifest["data"][str(cell["seed"])]["arrays"]
            contract = hashlib.sha256(json.dumps({"cell": cell, "pins": pins, "data": datapins}, sort_keys=True).encode()).hexdigest()
            if row["contract_sha256"] != contract:
                raise ValueError(f"Result contract changed: {path.relative_to(ROOT)}")
            contracts += 1
            result_arrays += 1
        summary = json.loads((study / "summary.json").read_text())
        if summary["cells"] != count or summary["analysis_sha256"] != sha(study / "analyze.py"):
            raise ValueError(f"Analysis source/count changed: {name}")
        checked.append({"study": name, "recipes": count, "finite": True, "summary_sha256": sha(study / "summary.json")})
    c01 = ROOT / "studies/C01_activation_scale"
    recovery = json.loads((c01 / "recovery_audit.json").read_text())
    first = c01 / recovery["before_resume"]["file"]
    if sha(first) != recovery["before_resume"]["sha256"] or not recovery["saved_success_unchanged"] or recovery["reused_cells"] != 1:
        raise ValueError("Saved first-success recovery evidence invalid")
    c02 = ROOT / "studies/C02_parity_ood"
    config2 = json.loads((c02 / "preregistration.json").read_text())
    if config2["development_report_sha256"] != sha(c02 / "executed/development_report.md"):
        raise ValueError("C02 development report pin mismatch")
    if config2["sealed_at"] >= min(p.stat().st_mtime for p in (c02 / "results").glob("*.json")):
        raise ValueError("C02 preregistration was not before first result")
    c05 = ROOT / "studies/C05_risk_ood"
    seal = json.loads((c05 / "forecast_seal.json").read_text())
    if seal["observed_training_results"] != 0 or len(seal["forecasts"]) != 24:
        raise ValueError("C05 forecast seal incomplete")
    for filename, fingerprint in seal["forecasts"].items():
        path = c05 / filename
        prediction = json.loads(path.read_text())
        if sha(path) != fingerprint or not prediction["saved_before_training"] or prediction["saved_at"] > seal["sealed_at"]:
            raise ValueError("C05 forecast seal changed")
        row = json.loads((c05 / "results" / path.name).read_text())
        if row["forecast_sha256"] != fingerprint or row["forecast_seal_sha256"] != sha(c05 / "forecast_seal.json"):
            raise ValueError("C05 result lost its prediction pin")
    config5 = json.loads((c05 / "preregistration.json").read_text())
    if config5["sealed_at"] > min(json.loads((c05 / p).read_text())["saved_at"] for p in seal["forecasts"]):
        raise ValueError("C05 config was not sealed before predictions")
    result = {"status": "pass", "studies": checked, "source_files_checked": source_files, "data_archives_checked": data_files,
              "result_array_archives_checked": result_arrays, "source_data_cell_contracts_checked": contracts,
              "recovery_verified": True, "OOD_seals_verified": ["C02", "C05"], "training_or_model_calls": 0}
    (Path(__file__).parent / "verification.json").write_text(json.dumps(result, indent=2) + chr(10))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
