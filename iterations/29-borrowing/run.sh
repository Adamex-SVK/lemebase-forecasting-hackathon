set -eu
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
for shops in A B C D ABCD; do
    "$HOME/hack/.venv-chronos/bin/python" -u fit.py "$shops" > "$shops.log" 2>&1
done
touch complete
