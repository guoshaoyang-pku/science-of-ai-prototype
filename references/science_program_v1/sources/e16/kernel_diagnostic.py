#!/usr/bin/env python3
"""Measure initialization coupling of a constant target shift to centered outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call, jvp
from reproduce_cell import archived_cell

REPO = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    spec = json.loads((REPO / "preregistration.json").read_text())
    rows = []
    for dataset in spec["datasets"]:
        measurement = json.loads((REPO / "results" / (dataset.split("/")[-1] + "_c2.json")).read_text())["measurement"]["results"][0]
        files = measurement["measurement_files"]
        with archived_cell(dataset.split("/")[-1], 2, "SGD") as (module, tensors, _):
            tx, ty, vx, vy = tensors
            model_class = module.Model
        manifest = json.loads((REPO / "evidence_manifest.json").read_text())["files"]
        for seed in range(10):
            torch.manual_seed(seed)
            model = model_class().eval()
            idx = torch.randint(0, len(tx), (64,))
            pin = files[f"results/process/seed_{seed}.json"]
            process = json.loads((REPO / manifest[pin["sha256"]]["path"]).read_text())
            assert hashlib.sha256(idx.numpy().tobytes()).hexdigest() == process["steps"][0]["minibatch"]["sha256"]
            parameters = dict(model.named_parameters())
            assert all(hashlib.sha256(p.detach().numpy().tobytes()).hexdigest() == process["initial_parameters"][name]["sha256"]
                       for name, p in parameters.items())
            outputs = model(tx[idx])
            means = torch.autograd.grad(outputs.mean(), tuple(parameters.values()), retain_graph=True)
            gradient = torch.autograd.grad(((outputs - ty[idx]) ** 2).mean(), tuple(parameters.values()))
            direction = dict(zip(parameters, means))
            q = jvp(lambda p: functional_call(model, p, (vx,)), (parameters,), (direction,))[1].detach().reshape(-1).double().numpy()
            q_centered = q - q.mean()
            predicted_gradient_norms = {}
            for mu in (-3.0, 0.0, 3.0):
                components = [g - 2 * mu * mean for g, mean in zip(gradient, means)]
                predicted_gradient_norms[str(mu)] = float(np.sqrt(sum(float(g.double().square().sum()) for g in components)))
                condition = {-3.0: 1, 0.0: 2, 3.0: 3}[mu]
                saved = json.loads((REPO / "results" / (dataset.split("/")[-1] + f"_c{condition}.json")).read_text())
                pin = saved["measurement"]["results"][0]["measurement_files"][f"results/process/seed_{seed}.json"]
                measured = json.loads((REPO / manifest[pin["sha256"]]["path"]).read_text())["steps"][0]["parameters"]
                actual = np.sqrt(sum(p["data_gradient_norm"] ** 2 for p in measured))
                assert np.isclose(actual, predicted_gradient_norms[str(mu)], rtol=2e-6, atol=2e-6)
            rows.append({"dataset": dataset, "seed": seed,
                         "constant_response_mean": float(q.mean()),
                         "constant_response_centered_rms": float(np.sqrt(np.mean(q_centered ** 2))),
                         "centered_to_mean_response_ratio": float(np.sqrt(np.mean(q_centered ** 2)) / abs(q.mean())),
                         "predicted_step1_data_gradient_norm": predicted_gradient_norms})
    result = {"definition": "q_test = J_test(theta0) mean_minibatch(J_train(theta0)); first SGD response to target +mu is 2 eta mu q_test",
              "scope": "Initialization/minibatch local derivative; no full training or claim of endpoint prediction",
              "rows": rows}
    (REPO / "kernel_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    for dataset in spec["datasets"]:
        subset = [r for r in rows if r["dataset"] == dataset]
        print(dataset, {key: float(np.mean([r[key] for r in subset])) for key in
                       ("constant_response_mean", "constant_response_centered_rms", "centered_to_mean_response_ratio")})


if __name__ == "__main__":
    main()
