#!/usr/bin/env bash
# Build dist/gpu_bundle.tar.gz: everything the GPU machine needs to run the foundation models
# (code, environment files, the small weekly tables). No raw data, no credentials, no git history.
# Upload it in JupyterLab, then follow iterations/03-foundation-model/gpu-runbook.md
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d data/interim ] || .venv/bin/python src/prepare.py
mkdir -p dist
tar czf dist/gpu_bundle.tar.gz --exclude='__pycache__' --exclude='*.pyc' \
  src env requirements.txt iterations/03-foundation-model/gpu-runbook.md iterations/03-foundation-model/model-notes.md \
  data/interim/weekly_A.parquet data/interim/weekly_B.parquet data/interim/weekly_C.parquet data/interim/weekly_D.parquet data/interim/weekly_test.parquet
ls -la dist/gpu_bundle.tar.gz
tar tzf dist/gpu_bundle.tar.gz | head -40
