#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3}" -m venv .venv-autogluon
.venv-autogluon/bin/python -m pip install -r env/requirements_autogluon.txt
.venv-autogluon/bin/python -m pip check
