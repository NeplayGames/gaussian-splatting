import importlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_PYTHON_MODULES = [
    'torch', 'torchvision', 'lpips', 'yaml', 'PIL', 'plyfile', 'pandas', 'openpyxl',
    'diff_gaussian_rasterization', 'simple_knn', 'fused_ssim',
]


def _imp(name):
    try:
        importlib.import_module(name)
        return True, None
    except Exception as e:
        return False, str(e)


def _tool_status(tool):
    path = shutil.which(tool)
    return {"ok": path is not None, "path": path, "error": None if path else 'not found'}


def _submodule_status():
    paths = {
        'diff_gaussian_rasterization': Path('submodules/diff-gaussian-rasterization'),
        'simple_knn': Path('submodules/simple-knn'),
        'fused_ssim': Path('submodules/fused-ssim'),
    }
    status = {name: {"path": str(path), "exists": path.exists()} for name, path in paths.items()}
    try:
        result = subprocess.run(
            ['git', 'submodule', 'status', '--recursive'],
            capture_output=True,
            text=True,
            check=False,
        )
        status['git_status'] = {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        status['git_status'] = {"ok": False, "error": str(exc)}
    return status


def collect_environment(output_dir='demo_output', cache_dir='~/.cache/segs-demo', minimum_disk_gb=1, device='cuda'):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    compiler = "cl.exe" if system == "Windows" else "g++"
    checks = {
        "operating_system": system,
        "platform": platform.platform(),
        "python_version": sys.version,
        "selected_compiler": compiler,
        "time": time.time(),
        "checks": {},
        "external_tools": {},
    }

    for mod in REQUIRED_PYTHON_MODULES:
        ok, err = _imp(mod)
        checks['checks'][mod] = {"ok": ok, "error": err}

    tools = ['git', 'python', 'cmake', 'ninja', 'nvidia-smi', 'nvcc', compiler]
    for tool in tools:
        status = _tool_status(tool)
        checks['checks'][tool] = status
        checks['external_tools'][tool] = status.get('path')

    try:
        import torch
        checks['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            checks['gpu_name'] = torch.cuda.get_device_name(0)
            checks['compute_capability'] = torch.cuda.get_device_capability(0)
        else:
            checks['gpu_name'] = None
            checks['compute_capability'] = None
    except Exception as e:
        checks['cuda_available'] = False
        checks['gpu_name'] = None
        checks['compute_capability'] = None
        checks['cuda_error'] = str(e)

    usage = shutil.disk_usage(str(Path(cache_dir).expanduser().parent))
    checks['available_disk_gb'] = round(usage.free / 1024 ** 3, 2)
    checks['minimum_disk_gb'] = minimum_disk_gb
    checks['submodule_status'] = _submodule_status()
    checks['git_submodules_initialized'] = all(
        item.get('exists') for name, item in checks['submodule_status'].items() if name != 'git_status'
    )
    checks['import_status'] = {name: checks['checks'][name] for name in REQUIRED_PYTHON_MODULES}

    (out / 'environment.json').write_text(json.dumps(checks, indent=2))
    problems = []
    if device == 'cuda' and not checks.get('cuda_available'):
        problems.append('CUDA is unavailable.')
    if checks['available_disk_gb'] < minimum_disk_gb:
        problems.append('Insufficient disk space.')
    required = [*REQUIRED_PYTHON_MODULES, 'git', 'python', 'cmake', 'ninja', 'nvidia-smi', 'nvcc', compiler]
    missing = [name for name in required if not checks['checks'].get(name, {}).get('ok')]
    if missing:
        problems.append('Missing required dependencies/extensions/tools: ' + ', '.join(missing))
    if not checks.get('git_submodules_initialized'):
        problems.append('Git submodules are not initialized; clone with --recursive or run git submodule update --init --recursive.')
    return checks, problems
