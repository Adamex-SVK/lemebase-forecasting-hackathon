#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
for pair in "chronos chronos-2-ft" "moirai moirai-2.0-small" "tirex tirex" "chronos chronos-t5-small"; do
    read -r envname model <<< "$pair"
    "$HOME/hack/.venv-$envname/bin/python" -u foundation.py "$model" > "$model.log" 2>&1
done
touch complete
