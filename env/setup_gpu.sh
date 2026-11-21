#!/usr/bin/env bash
# Run on the GPU machine from the bundle root.   bash env/setup_gpu.sh [chronos timesfm tirex moirai]
#
# Facts about the GPU machine (checked 2026-11-21): conda at /opt/conda with Python 3.11.10, torch 2.5.1+cu121
# already installed and seeing the RTX 3090, numpy 2.1.2, internet to PyPI and Hugging Face works, disk is tight
# (about 51 GB free), `uv` is not installed.
#  * chronos, timesfm, tirex: venv with --system-site-packages, so they REUSE the machine's torch 2.5.1
#    (their packages accept torch >= 2.0 / >= 2.2) and no second torch is downloaded.
#  * moirai: uni2ts pins torch < 2.5 and numpy ~= 1.26, so it needs its own full environment
#    (it downloads an older torch, a few GB).
set -euo pipefail
PY="${PYTHON:-python3}"
FAMILIES="${*:-chronos timesfm tirex moirai}"
for fam in $FAMILIES; do
  echo "== $fam ($($PY --version))"
  if [ "$fam" = "moirai" ]; then
    "$PY" -m venv ".venv-$fam"
  else
    "$PY" -m venv --system-site-packages ".venv-$fam"
  fi
  ".venv-$fam/bin/pip" install -q --upgrade pip
  if [ "$fam" = "moirai" ]; then
    ".venv-$fam/bin/pip" install -r env/requirements_base.txt -r "env/requirements_$fam.txt" 2>&1 | tail -3
  else   # keep the machine's torch stack (constraints), and show pip's output so failures are visible
    ".venv-$fam/bin/pip" install -c env/constraints_gpu.txt -r env/requirements_base.txt -r "env/requirements_$fam.txt" 2>&1 | tail -3
  fi
  ".venv-$fam/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())" || true
  df -h . | tail -1
done
echo "Next: .venv-<family>/bin/python src/foundation/smoke.py <model names>   (see src/foundation/models.py SPECS)"
