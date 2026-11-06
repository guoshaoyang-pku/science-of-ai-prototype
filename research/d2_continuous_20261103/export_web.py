"""Export saved D2 evidence for the website; verify all pinned files, run no study.

Usage: python research/d2_continuous_20261103/export_web.py --data-root PATH
Large numerical arrays stay in external storage. Reports, ledgers and figures are
copied byte-for-byte, so the published evidence has the original audit hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "research/d2_continuous_20261103"
OUTPUT = ROOT / "docs/模糊方向/d2_evidence_20261106"
ROUNDS = (
    "r001_alpha_mix",
    "r002_validation_replication",
    "r003_population_risk",
    "r004_representable_target",
    "r005_nullspace_accounting",
    "r006_learning_preference",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    base = args.data_root / "runs/d2_continuous_20261103"
    checks = []
    copies = []
    # Validate every pinned artifact, including the arrays that are not exported.
    # No website files are touched until all six rounds pass this gate.
    for round_id in ROUNDS:
        pin_path = STUDY / round_id / "result_manifest.json"
        pins = json.loads(pin_path.read_text())
        if pins["status"] != "completed_and_audited":
            raise ValueError(f"Round has not completed its audit: {round_id}")
        artifacts = []
        for pin in pins["artifacts"]:
            relative = Path(pin["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Non-local artifact path: {relative}")
            source = base / round_id / relative
            digest = sha256(source)
            if digest != pin["sha256"] or source.stat().st_size != pin["bytes"]:
                raise ValueError(f"Saved artifact differs from its pin: {source}")
            publish = len(relative.parts) == 1 and relative.suffix in {
                ".json", ".md", ".png", ".svg"
            }
            record = {
                "source": f"runs/d2_continuous_20261103/{round_id}/{relative.as_posix()}",
                "sha256": digest,
                "bytes": pin["bytes"],
                "published": f"{round_id}/{relative.as_posix()}" if publish else None,
                "byte_identical": True if publish else None,
            }
            artifacts.append(record)
            if publish:
                copies.append((source, OUTPUT / round_id / relative))
        checks.append({
            "round": round_id,
            "preregistration_commit": pins["preregistration_commit"],
            "result_manifest": pin_path.relative_to(ROOT).as_posix(),
            "result_manifest_sha256": sha256(pin_path),
            "status": pins["status"],
            "verified_artifact_count": len(artifacts),
            "artifacts": artifacts,
        })
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise ValueError(f"Export copy failed verification: {destination}")
    manifest = {
        "kind": "read_only_publication_export",
        "new_experiments": False,
        "all_pinned_artifacts_verified": True,
        "exported_file_count": len(copies),
        "large_numerical_arrays_exported": False,
        "scope": "Previously observed fixed-feature development conditions; only three master seeds.",
        "rounds": checks,
    }
    (OUTPUT / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    count = sum(item["verified_artifact_count"] for item in checks)
    print(f"Verified {count} saved artifacts; exported {len(copies)} files without new experiments.")


if __name__ == "__main__":
    main()
