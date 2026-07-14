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
EXPECTED_CUDA_VERSION = "11.8"
PINNED_SETUPTOOLS_VERSION = "69.5.1"

REQUIRED_EXTENSIONS = (
    "diff_gaussian_rasterization",
    "simple_knn",
    "fused_ssim",
)

ROOT = Path(__file__).resolve().parent
SETUP_LOG_DIR = ROOT / "demo_output" / "logs" / "setup"


class SetupError(RuntimeError):
    def __init__(
        self,
        step: str,
        returncode: int,
        log_path: Path,
    ) -> None:
        super().__init__(
            f"Demo setup failed during: {step}\n"
            f"Exit code: {returncode}\n"
            f"See log: {log_path}"
        )
        self.step = step
        self.returncode = returncode
        self.log_path = log_path


def is_windows() -> bool:
    return platform.system() == "Windows"


def command_display(
    command: Sequence[os.PathLike[str] | str],
) -> str:
    return " ".join(str(part) for part in command)


def run_logged(
    step: str,
    command: Sequence[os.PathLike[str] | str],
    env: Mapping[str, str] | None = None,
) -> None:
    SETUP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SETUP_LOG_DIR / f"{step}.log"

    print(f"[run_demo] {step} (log: {log_path})")

    with log_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
    ) as log_file:
        log_file.write(f"$ {command_display(command)}\n\n")
        log_file.flush()

        result = subprocess.run(
            [str(part) for part in command],
            cwd=ROOT,
            env=dict(env) if env is not None else None,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        raise SetupError(
            step,
            result.returncode,
            log_path,
        )


def run_passthrough(
    command: Sequence[os.PathLike[str] | str],
    env: Mapping[str, str],
) -> int:
    result = subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        env=dict(env),
        check=False,
    )
    return result.returncode


def require_tool(
    name: str,
    explanation: str,
    env: Mapping[str, str] | None = None,
) -> None:
    active_env = env or os.environ
    tool_path = shutil.which(
        name,
        path=active_env.get("PATH"),
    )

    if tool_path is None:
        print(
            "Demo setup failed during: local build requirements",
            file=sys.stderr,
        )
        print(
            f"Missing required tool: {name}",
            file=sys.stderr,
        )
        print(
            explanation,
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"[run_demo] Found {name}: {tool_path}")


def prepend_to_path(
    env: dict[str, str],
    directory: Path,
) -> None:
    """Move a directory to the beginning of PATH, removing duplicates."""

    directory_text = str(directory)
    target = os.path.normcase(
        os.path.normpath(directory_text)
    )

    existing_parts = [
        part
        for part in env.get("PATH", "").split(os.pathsep)
        if part
    ]

    filtered_parts = [
        part
        for part in existing_parts
        if os.path.normcase(os.path.normpath(part)) != target
    ]

    env["PATH"] = os.pathsep.join(
        [directory_text, *filtered_parts]
    )


def configure_windows_cuda(
    env: dict[str, str],
) -> None:
    """Prefer CUDA 11.8 because the Conda/PyTorch environment uses cu118."""

    if not is_windows():
        return

    default_cuda_118 = Path(
        rf"C:\Program Files\NVIDIA GPU Computing Toolkit"
        rf"\CUDA\v{EXPECTED_CUDA_VERSION}"
    )

    raw_candidates = [
        env.get("CUDA_PATH_V11_8"),
        str(default_cuda_118),
        env.get("CUDA_PATH"),
        env.get("CUDA_HOME"),
    ]

    candidates: list[Path] = []
    seen: set[str] = set()

    for raw_candidate in raw_candidates:
        if not raw_candidate:
            continue

        candidate = Path(raw_candidate)
        normalized = os.path.normcase(
            os.path.normpath(str(candidate))
        )

        if normalized not in seen:
            seen.add(normalized)
            candidates.append(candidate)

    selected: Path | None = None

    for candidate in candidates:
        nvcc_path = candidate / "bin" / "nvcc.exe"

        if nvcc_path.is_file():
            selected = candidate
            break

    if selected is None:
        print(
            "[run_demo] CUDA 11.8 was not found automatically. "
            "Using the existing CUDA environment.",
        )
        return

    env["CUDA_PATH"] = str(selected)
    env["CUDA_HOME"] = str(selected)

    prepend_to_path(
        env,
        selected / "bin",
    )

    print(
        f"[run_demo] Selected CUDA toolkit: {selected}"
    )


def capture_visual_studio_environment(
    base_env: Mapping[str, str],
) -> dict[str, str]:
    env = dict(base_env)

    if not is_windows():
        return env

    if shutil.which(
        "cl.exe",
        path=env.get("PATH"),
    ):
        return env

    program_files_x86 = env.get(
        "ProgramFiles(x86)",
        r"C:\Program Files (x86)",
    )

    vswhere = (
        Path(program_files_x86)
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )

    if not vswhere.exists():
        print(
            "cl.exe was not found and vswhere.exe is unavailable. "
            "Run this launcher from Developer PowerShell for "
            "Visual Studio.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    query = [
        str(vswhere),
        "-latest",
        "-products",
        "*",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-property",
        "installationPath",
    ]

    result = subprocess.run(
        query,
        capture_output=True,
        text=True,
        encoding="mbcs",
        errors="replace",
        env=env,
        check=False,
    )

    if result.returncode != 0:
        print(
            "vswhere.exe failed while locating Visual Studio.",
            file=sys.stderr,
        )
        print(
            result.stderr.strip(),
            file=sys.stderr,
        )
        raise SystemExit(1)

    installation_lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    if not installation_lines:
        print(
            "Could not find a Visual Studio installation with "
            "the C++ x64/x86 build tools.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    installation_path = Path(
        installation_lines[0]
    )

    devcmd = (
        installation_path
        / "Common7"
        / "Tools"
        / "VsDevCmd.bat"
    )

    if not devcmd.exists():
        print(
            f"Could not locate VsDevCmd.bat at: {devcmd}",
            file=sys.stderr,
        )
        print(
            "Run this launcher from Developer PowerShell for "
            "Visual Studio.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # "call" is necessary so cmd.exe returns after VsDevCmd.bat
    # and can print the resulting transient environment.
    devcmd_text = str(devcmd)
    if any(character.isspace() for character in devcmd_text):
        devcmd_text = f'"{devcmd_text}"'

    command_text = (
        f"call {devcmd_text} "
        "-no_logo "
        "-arch=x64 "
        "-host_arch=x64 "
        ">nul && set"
    )

    command = [
        "cmd.exe",
        "/d",
        "/s",
        "/c",
        command_text,
    ]

    captured = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="mbcs",
        errors="replace",
        env=env,
        check=False,
    )

    if captured.returncode != 0:
        print(
            "Failed to import the Visual Studio build environment.",
            file=sys.stderr,
        )

        if captured.stderr.strip():
            print(
                captured.stderr.strip(),
                file=sys.stderr,
            )

        print(
            "Run this launcher from Developer PowerShell for "
            "Visual Studio.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    for line in captured.stdout.splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)

        if key:
            env[key] = value

    if not shutil.which(
        "cl.exe",
        path=env.get("PATH"),
    ):
        print(
            "Visual Studio environment was loaded, but cl.exe "
            "is still unavailable.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return env


def conda_command(
    env_name: str,
    args: Sequence[os.PathLike[str] | str],
) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        *[str(arg) for arg in args],
    ]


def conda_env_exists(
    env_name: str,
    env: Mapping[str, str],
) -> bool:
    result = subprocess.run(
        [
            "conda",
            "env",
            "list",
            "--json",
        ],
        cwd=ROOT,
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        SETUP_LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_path = (
            SETUP_LOG_DIR
            / "conda_env_list.log"
        )

        log_path.write_text(
            result.stdout + result.stderr,
            encoding="utf-8",
            errors="replace",
        )

        raise SetupError(
            "conda_env_list",
            result.returncode,
            log_path,
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        log_path = (
            SETUP_LOG_DIR
            / "conda_env_list.log"
        )

        log_path.write_text(
            result.stdout + result.stderr,
            encoding="utf-8",
            errors="replace",
        )

        raise SetupError(
            "conda_env_list",
            1,
            log_path,
        )

    environment_names = [
        Path(path).name
        for path in data.get("envs", [])
    ]

    return env_name in environment_names


def ensure_conda_environment(
    env_name: str,
    env: Mapping[str, str],
) -> None:
    require_tool(
        "conda",
        "Conda is required; install it or add it to PATH "
        "before running the demo.",
        env,
    )

    exists = conda_env_exists(
        env_name,
        env,
    )

    environment_file = (
        ROOT / "environment-demo.yml"
    )

    if not exists:
        run_logged(
            "conda_env_create",
            [
                "conda",
                "env",
                "create",
                "-n",
                env_name,
                "-f",
                environment_file,
            ],
            env,
        )
        return

    if os.environ.get("SEGS_UPDATE_ENV") == "1":
        run_logged(
            "conda_env_update",
            [
                "conda",
                "env",
                "update",
                "-n",
                env_name,
                "-f",
                environment_file,
                "--prune",
            ],
            env,
        )
        return

    print(
        f"[run_demo] Reusing existing Conda environment: "
        f"{env_name}"
    )


def all_extensions_importable(
    env_name: str,
    env: Mapping[str, str],
) -> bool:
    import_code = (
        "import "
        + ", ".join(REQUIRED_EXTENSIONS)
    )

    result = subprocess.run(
        conda_command(
            env_name,
            [
                "python",
                "-c",
                import_code,
            ],
        ),
        cwd=ROOT,
        env=dict(env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    return result.returncode == 0


def bootstrap_python_build_tools(
    env_name: str,
    env: Mapping[str, str],
) -> None:
    # PyTorch 2.1 imports pkg_resources from setuptools.
    # New setuptools releases can remove pkg_resources,
    # so use a known compatible version.
    run_logged(
        "pip_bootstrap",
        conda_command(
            env_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "wheel",
                (
                    "setuptools=="
                    f"{PINNED_SETUPTOOLS_VERSION}"
                ),
            ],
        ),
        env,
    )

    validation_code = (
        "import setuptools; "
        "import pkg_resources; "
        "import torch; "
        "from torch.utils.cpp_extension import CUDA_HOME; "
        "print('setuptools:', setuptools.__version__); "
        "print('torch:', torch.__version__); "
        "print('torch CUDA:', torch.version.cuda); "
        "print('CUDA_HOME:', CUDA_HOME)"
    )

    run_logged(
        "python_build_dependencies",
        conda_command(
            env_name,
            [
                "python",
                "-c",
                validation_code,
            ],
        ),
        env,
    )


def build_extensions_if_needed(
    env_name: str,
    env: Mapping[str, str],
) -> None:
    force_rebuild = (
        os.environ.get("SEGS_FORCE_REBUILD")
        == "1"
    )

    if (
        not force_rebuild
        and all_extensions_importable(
            env_name,
            env,
        )
    ):
        print(
            "[run_demo] CUDA extensions already import "
            "successfully; skipping rebuild."
        )
        return

    print(
        "[run_demo] CUDA extensions are missing or a rebuild "
        "was requested."
    )

    bootstrap_python_build_tools(
        env_name,
        env,
    )

    submodules = {
        "diff_gaussian_rasterization": (
            ROOT
            / "submodules"
            / "diff-gaussian-rasterization"
        ),
        "simple_knn": (
            ROOT
            / "submodules"
            / "simple-knn"
        ),
        "fused_ssim": (
            ROOT
            / "submodules"
            / "fused-ssim"
        ),
    }

    for name, path in submodules.items():
        if not path.exists():
            log_path = (
                SETUP_LOG_DIR
                / f"cuda_extension_{name}.log"
            )

            log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            log_path.write_text(
                f"Missing submodule directory: {path}\n",
                encoding="utf-8",
            )

            raise SetupError(
                f"cuda_extension_{name}",
                1,
                log_path,
            )

        run_logged(
            f"cuda_extension_{name}",
            conda_command(
                env_name,
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    "-e",
                    path,
                ],
            ),
            env,
        )


def prepare_environment() -> tuple[
    str,
    dict[str, str],
]:
    system = platform.system()

    if system not in {
        "Windows",
        "Linux",
    }:
        print(
            f"Unsupported operating system: {system}. "
            "Supported systems are Windows and Linux.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    env = capture_visual_studio_environment(
        os.environ
    )

    if is_windows():
        configure_windows_cuda(env)

        env["DISTUTILS_USE_SDK"] = "1"
        env["MSSdk"] = "1"

        tools = [
            "git",
            "nvidia-smi",
            "nvcc",
            "cl.exe",
            "cmake",
            "ninja",
        ]
    else:
        tools = [
            "git",
            "nvidia-smi",
            "nvcc",
            "g++",
            "cmake",
            "ninja",
        ]

    for tool in tools:
        require_tool(
            tool,
            f"{tool} is required for the local "
            "Conda/CUDA demo setup.",
            env,
        )

    env_name = os.environ.get(
        "SEGS_DEMO_ENV",
        ENV_NAME_DEFAULT,
    )

    ensure_conda_environment(
        env_name,
        env,
    )

    run_logged(
        "git_submodules",
        [
            "git",
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        env,
    )

    build_extensions_if_needed(
        env_name,
        env,
    )

    return env_name, env


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = list(
        sys.argv[1:]
        if argv is None
        else argv
    )

    try:
        env_name, env = prepare_environment()

        check_command = conda_command(
            env_name,
            [
                "python",
                "-m",
                "tools.quickstart",
                "--check-only",
            ],
        )

        run_logged(
            "environment_validation",
            check_command,
            env,
        )

        if "--check-only" in args:
            return 0

        demo_command = conda_command(
            env_name,
            [
                "python",
                "-m",
                "tools.quickstart",
                *args,
            ],
        )

        return run_passthrough(
            demo_command,
            env,
        )

    except SetupError as error:
        print(
            error,
            file=sys.stderr,
        )
        return error.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
