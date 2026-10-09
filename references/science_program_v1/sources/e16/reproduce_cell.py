#!/usr/bin/env python3
"""Replay one controlled cell using the archived executable, tensors and recipe."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import torch

REPO = Path(__file__).resolve().parent


@contextlib.contextmanager
def archived_cell(dataset, condition, optimizer):
    saved = json.loads((REPO / "results" / f"{dataset}_c{condition}.json").read_text())
    measurement = next(r for r in saved["measurement"]["results"] if r["candidate"]["optimizer"]["type"] == optimizer)
    manifest = json.loads((REPO / "evidence_manifest.json").read_text())
    with tempfile.TemporaryDirectory(prefix="kb-e16-replay-") as temp:
        directory = Path(temp)
        for name in ("model.py", "optimizer.py", "loss.py", "train.py"):
            pin = measurement["measurement_files"]["executed/" + name]
            raw = (REPO / manifest["files"][pin["sha256"]]["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
            (directory / name).write_bytes(raw)
        previous = {name: sys.modules.pop(name, None) for name in ("model", "optimizer", "loss")}
        sys.path.insert(0, str(directory))
        try:
            module_spec = importlib.util.spec_from_file_location("e16_archived_train", directory / "train.py")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            train = torch.load(REPO / "datasets" / dataset / "train.pt", weights_only=True)
            test = torch.load(REPO / "datasets" / dataset / "test.pt", weights_only=True)
            transform = saved["request"]["target_transform"]
            shift = transform["offset"] - (float(train["y"].double().mean()) if transform["center"] else 0)
            tensors = train["x"], train["y"] + shift, test["x"], test["y"] + shift
            yield module, tensors, measurement
        finally:
            sys.path.remove(str(directory))
            for name, old in previous.items():
                sys.modules.pop(name, None)
                if old is not None:
                    sys.modules[name] = old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvar_02036a", "mvar_071e42", "mvar_0852dd"], required=True)
    parser.add_argument("--condition", type=int, choices=range(4), required=True, help="0=original,1=-3,2=0,3=+3")
    parser.add_argument("--optimizer", choices=["SGD", "Adam"], required=True)
    parser.add_argument("--seed", type=int, choices=range(10), default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with archived_cell(args.dataset, args.condition, args.optimizer) as (module, tensors, measurement):
        result = module.train_and_eval(*tensors, steps=256, batch_size=64, seed=args.seed,
                                       fail_threshold=1e12, process=measurement["process_provenance"]["request"])
    expected = measurement["seed_results"][args.seed]["final_test_mse"]
    print(json.dumps({"test_mse": result["final_test_mse"], "saved_test_mse": expected,
                      "difference": result["final_test_mse"] - expected, "failed": result["failed"]}))


if __name__ == "__main__":
    main()
