#!/usr/bin/env bash
set -euo pipefail
ENV_NAME=${SEGS_DEMO_ENV:-segs-demo}
if command -v mamba >/dev/null 2>&1; then CONDA=mamba; elif command -v conda >/dev/null 2>&1; then CONDA=conda; else CONDA=; fi
if [ -n "$CONDA" ] && ! $CONDA env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  $CONDA env create -n "$ENV_NAME" -f environment-demo.yml
fi
git submodule update --init --recursive
python -m pip install -q submodules/diff-gaussian-rasterization submodules/simple-knn || true
python -m tools.quickstart "$@"
