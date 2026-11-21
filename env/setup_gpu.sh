#!/usr/bin/env bash
# Run on the GPU machine from the project root.  NOT RUN YET.
# One virtual environment per model family, because their dependency pins conflict.
#   PYTHON=python3.12 PY_MOIRAI=python3.11 bash env/setup_gpu.sh [chronos timesfm moirai tirex]
set -euo pipefail
PY="${PYTHON:-python3}"
PY_MOIRAI="${PY_MOIRAI:-python3.11}"
FAMILIES="${*:-chronos timesfm moirai tirex}"
for fam in $FAMILIES; do
  py="$PY"; [ "$fam" = "moirai" ] && py="$PY_MOIRAI"
  echo "== $fam ($py)"
  "$py" -m venv ".venv-$fam"
  ".venv-$fam/bin/pip" install -q --upgrade pip
  ".venv-$fam/bin/pip" install -q -r env/requirements_base.txt -r "env/requirements_$fam.txt"
  ".venv-$fam/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
done
echo "Next: .venv-<family>/bin/python src/foundation/smoke.py <model names>   (see src/foundation/models.py SPECS)"
