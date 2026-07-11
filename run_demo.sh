#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${SEGS_DEMO_ENV:-segs-demo}"
PYTHON_BIN="${PYTHON:-python}"
SETUP_LOG_DIR="demo_output/logs/setup"
mkdir -p "$SETUP_LOG_DIR"

fail_setup() {
  local step="$1"
  local log="$2"
  echo "Demo setup failed during: $step" >&2
  echo "See log: $log" >&2
  exit 1
}

run_logged() {
  local step="$1"
  shift
  local log="$SETUP_LOG_DIR/${step}.log"
  echo "[run_demo] $step (log: $log)"
  if ! "$@" >"$log" 2>&1; then
    fail_setup "$step" "$log"
  fi
}

check_tool() {
  local tool="$1"
  local reason="$2"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Demo setup failed during: local build requirements" >&2
    echo "Missing required tool: $tool" >&2
    echo "$reason" >&2
    echo "Install it locally and re-run ./run_demo.sh." >&2
    exit 1
  fi
}

run_in_env() {
  if command -v conda >/dev/null 2>&1; then
    conda run -n "$ENV_NAME" "$@"
  else
    . .venv/bin/activate
    "$@"
  fi
}

check_tool git "git is required to initialize the repository submodules."
check_tool nvidia-smi "nvidia-smi is required to confirm that an NVIDIA driver/GPU is visible before CUDA training."
check_tool nvcc "nvcc from a CUDA toolkit is required to compile the CUDA extensions locally."
check_tool g++ "g++ is required by PyTorch/CUDA extension compilation."

run_logged git_submodules git submodule update --init --recursive

if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    run_logged conda_env_update conda env update -n "$ENV_NAME" -f environment-demo.yml --prune
  else
    run_logged conda_env_create conda env create -n "$ENV_NAME" -f environment-demo.yml
  fi
else
  echo "[run_demo] conda not found; creating/reusing local virtual environment .venv."
  if [ ! -x .venv/bin/python ]; then
    run_logged venv_create "$PYTHON_BIN" -m venv .venv
  fi
  . .venv/bin/activate
  run_logged pip_bootstrap python -m pip install --upgrade pip setuptools wheel cmake ninja
  run_logged pip_torch python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  run_logged pip_dependencies python -m pip install numpy pillow tqdm plyfile pyyaml pandas openpyxl lpips pytest
fi

run_logged cuda_extensions run_in_env python -m pip install -e submodules/diff-gaussian-rasterization -e submodules/simple-knn -e submodules/fused-ssim

echo "[run_demo] Running quickstart..."
run_in_env python -m tools.quickstart "$@"
