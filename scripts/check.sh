#!/usr/bin/env bash
# Run before every pull request:  bash scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
if [ ! -d data/interim ]; then echo "== building weekly tables (needs hackathon-retail-sets/)"; "$PY" src/prepare.py; fi
echo "== metric tests";   "$PY" tests/test_validate.py
echo "== feature tests (includes the leakage test)"; "$PY" tests/test_features.py
echo "== rich-column tests"; "$PY" tests/test_rich.py
echo "all checks passed"
