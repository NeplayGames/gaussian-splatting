#!/usr/bin/env python3
"""Cross-platform Conda launcher for the local SEGS demo."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

ENV_NAME_DEFAULT = "segs-demo"
REQUIRED_EXTENSIONS = ("diff_gaussian_rasterization", "simple_knn", "fused_ssim")
ROOT = Path(__file__).resolve().parent
SETUP_LOG_DIR = ROOT / "demo_output" / "logs" / "setup"


class SetupError(RuntimeError):
    def __init__(self, step: str, returncode: int, log_path: Path) -> None:
        super().__init__(f"Demo setup failed during: {step}\nExit code: {returncode}\nSee log: {log_path}")
        self.step = step
        self.returncode = returncode
        self.log_path = log_path


def is_windows() -> bool:
    return platform.system() == "Windows"


def command_display(command: Sequence[os.PathLike[str] | str]) -> str:
    return " ".join(str(part) for part in command)


def run_logged(step: str, command: Sequence[os.PathLike[str] | str], env: Mapping[str, str] | None = None) -> None:
    SETUP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SETUP_LOG_DIR / f"{step}.log"
    print(f"[run_demo] {step} (log: {log_path})")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {command_display(command)}\n\n")
        result = subprocess.run(
            [str(part) for part in command],
            cwd=ROOT,
            env=dict(env) if env is not None else None,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise SetupError(step, result.returncode, log_path)


def run_passthrough(command: Sequence[os.PathLike[str] | str], env: Mapping[str, str]) -> int:
    return subprocess.run([str(part) for part in command], cwd=ROOT, env=dict(env), check=False).returncode


def require_tool(name: str, explanation: str, env: Mapping[str, str] | None = None) -> None:
    if shutil.which(name, path=(env or os.environ).get("PATH")) is None:
        print("Demo setup failed during: local build requirements", file=sys.stderr)
        print(f"Missing required tool: {name}", file=sys.stderr)
        print(explanation, file=sys.stderr)
        raise SystemExit(1)


def capture_visual_studio_environment(base_env: Mapping[str, str]) -> dict[str, str]:
    env = dict(base_env)
    if not is_windows() or shutil.which("cl.exe", path=env.get("PATH")):
        return env

    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        print("cl.exe was not found and vswhere.exe is unavailable. Run this launcher from Developer PowerShell for Visual Studio.", file=sys.stderr)
        raise SystemExit(1)

    query = [
        vswhere,
        "-latest",
        "-products",
        "*",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-property",
        "installationPath",
    ]
    result = subprocess.run([str(part) for part in query], capture_output=True, text=True, env=env, check=False)
    install_path = result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else ""
    devcmd = Path(install_path) / "Common7" / "Tools" / "VsDevCmd.bat" if install_path else Path()
    if not devcmd.exists():
        print("Could not locate Visual Studio C++ x64/x86 tools. Run this launcher from Developer PowerShell for Visual Studio.", file=sys.stderr)
        raise SystemExit(1)

    # cmd.exe is required to execute the batch file and then print the transient environment.
    cmd = ["cmd.exe", "/s", "/c", f'"{devcmd}" -arch=x64 -host_arch=x64 >nul && set']
    captured = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if captured.returncode != 0:
        print("Failed to import the Visual Studio build environment. Run this launcher from Developer PowerShell for Visual Studio.", file=sys.stderr)
        raise SystemExit(1)
    for line in captured.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env


def conda_command(env_name: str, args: Sequence[os.PathLike[str] | str]) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", env_name, *[str(arg) for arg in args]]


def conda_env_exists(env_name: str, env: Mapping[str, str]) -> bool:
    result = subprocess.run(["conda", "env", "list", "--json"], cwd=ROOT, env=dict(env), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        SETUP_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = SETUP_LOG_DIR / "conda_env_list.log"
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        raise SetupError("conda_env_list", result.returncode, log_path)
    data = json.loads(result.stdout)
    envs = [Path(path).name for path in data.get("envs", [])]
    return env_name in envs


def ensure_conda_environment(env_name: str, env: Mapping[str, str]) -> None:
    require_tool("conda", "Conda is required; install it or add it to PATH before running the demo.", env)
    exists = conda_env_exists(env_name, env)
    if not exists:
        run_logged("conda_env_create", ["conda", "env", "create", "-n", env_name, "-f", ROOT / "environment-demo.yml"], env)
    elif os.environ.get("SEGS_UPDATE_ENV") == "1":
        run_logged("conda_env_update", ["conda", "env", "update", "-n", env_name, "-f", ROOT / "environment-demo.yml", "--prune"], env)
    else:
        print(f"[run_demo] Reusing existing Conda environment: {env_name}")


def all_extensions_importable(env_name: str, env: Mapping[str, str]) -> bool:
    code = "import " + ", ".join(REQUIRED_EXTENSIONS)
    result = subprocess.run(conda_command(env_name, ["python", "-c", code]), cwd=ROOT, env=dict(env), check=False)
    return result.returncode == 0


def build_extensions_if_needed(env_name: str, env: Mapping[str, str]) -> None:
    if os.environ.get("SEGS_FORCE_REBUILD") != "1" and all_extensions_importable(env_name, env):
        print("[run_demo] CUDA extensions already import successfully; skipping rebuild.")
        return
    run_logged("pip_bootstrap", conda_command(env_name, ["python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]), env)
    submodules = {
        "diff_gaussian_rasterization": ROOT / "submodules" / "diff-gaussian-rasterization",
        "simple_knn": ROOT / "submodules" / "simple-knn",
        "fused_ssim": ROOT / "submodules" / "fused-ssim",
    }
    for name, path in submodules.items():
        run_logged(f"cuda_extension_{name}", conda_command(env_name, ["python", "-m", "pip", "install", "--no-build-isolation", "-e", path]), env)


def prepare_environment() -> tuple[str, dict[str, str]]:
    system = platform.system()
    if system not in {"Windows", "Linux"}:
        print(f"Unsupported operating system: {system}. Supported systems are Windows and Linux.", file=sys.stderr)
        raise SystemExit(1)

    env = capture_visual_studio_environment(os.environ)
    if is_windows():
        env["DISTUTILS_USE_SDK"] = "1"
        env["MSSdk"] = "1"
        if env.get("CUDA_PATH"):
            env["CUDA_HOME"] = env["CUDA_PATH"]
        tools = ["git", "nvidia-smi", "nvcc", "cl.exe", "cmake", "ninja"]
    else:
        tools = ["git", "nvidia-smi", "nvcc", "g++", "cmake", "ninja"]
    for tool in tools:
        require_tool(tool, f"{tool} is required for the local Conda/CUDA demo setup.", env)

    env_name = os.environ.get("SEGS_DEMO_ENV", ENV_NAME_DEFAULT)
    ensure_conda_environment(env_name, env)
    run_logged("git_submodules", ["git", "submodule", "update", "--init", "--recursive"], env)
    build_extensions_if_needed(env_name, env)
    return env_name, env


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        env_name, env = prepare_environment()
        check_cmd = conda_command(env_name, ["python", "-m", "tools.quickstart", "--check-only"])
        run_logged("environment_validation", check_cmd, env)
        if "--check-only" in args:
            return 0
        return run_passthrough(conda_command(env_name, ["python", "-m", "tools.quickstart", *args]), env)
    except SetupError as error:
        print(error, file=sys.stderr)
        return error.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
