#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${SEGS_DEMO_ENV:-segs-demo}"
PYTHON_BIN="${PYTHON:-python}"

run_in_env() {
  if command -v conda >/dev/null 2>&1; then
    conda run -n "$ENV_NAME" "$@"
  else
    . .venv/bin/activate
    "$@"
  fi
}

echo "[run_demo] Initializing git submodules..."
git submodule update --init --recursive

if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "[run_demo] Reusing conda environment $ENV_NAME and applying environment-demo.yml updates..."
    conda env update -n "$ENV_NAME" -f environment-demo.yml --prune
  else
    echo "[run_demo] Creating conda environment $ENV_NAME from environment-demo.yml..."
    conda env create -n "$ENV_NAME" -f environment-demo.yml
  fi
else
  echo "[run_demo] conda not found; creating/reusing local virtual environment .venv."
  if [ ! -x .venv/bin/python ]; then
    "$PYTHON_BIN" -m venv .venv
  fi
  . .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  python -m pip install numpy pillow tqdm plyfile pyyaml lpips pytest
fi

echo "[run_demo] Installing/compiling CUDA extensions..."
run_in_env python -m pip install -e submodules/diff-gaussian-rasterization -e submodules/simple-knn -e submodules/fused-ssim

echo "[run_demo] Running quickstart..."
run_in_env python -m tools.quickstart "$@"
