#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,runtime,research,reports]'
printf 'Installed. Run .venv/bin/python tools/verify_handoff.py and .venv/bin/python -m pytest -q\n'
