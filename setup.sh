#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
if [[ ! -f external/ArchitectureIQ/pyproject.toml ]]; then
  printf 'Initialize the pinned dependency first: git submodule update --init\n' >&2
  exit 1
fi
.venv/bin/python -m pip install -e external/ArchitectureIQ
.venv/bin/python -m pip install -e '.[dev,runtime,research,reports]'
printf 'Installed. Run .venv/bin/python tools/verify_handoff.py and .venv/bin/python -m pytest -q\n'
