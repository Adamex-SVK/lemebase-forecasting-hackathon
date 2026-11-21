#!/usr/bin/env bash
# Run on the GPU machine from the bundle root.  One virtual environment per model family, because
# their dependency pins conflict.  The GPU machine has Python 3.11.10 as `python3` (checked
# 2026-11-21), which satisfies every family including Moirai (needs 3.10 or 3.11).
#   bash env/setup_gpu.sh [chronos timesfm moirai tirex]
# Disk is tight (about 51 GB free): if `uv` is available it is used, because it shares downloaded
# packages between environments with hard links (four torch installs would otherwise take about 25 GB).
set -euo pipefail
PY="${PYTHON:-python3}"
FAMILIES="${*:-chronos timesfm moirai tirex}"
for fam in $FAMILIES; do
  echo "== $fam ($($PY --version))"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PY" ".venv-$fam"
    uv pip install --python ".venv-$fam/bin/python" -r env/requirements_base.txt -r "env/requirements_$fam.txt"
  else
    "$PY" -m venv ".venv-$fam"
    ".venv-$fam/bin/pip" install -q --upgrade pip
    ".venv-$fam/bin/pip" install -q -r env/requirements_base.txt -r "env/requirements_$fam.txt"
  fi
  ".venv-$fam/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())" || true
  df -h . | tail -1
done
echo "Next: .venv-<family>/bin/python src/foundation/smoke.py <model names>   (see src/foundation/models.py SPECS)"
