import importlib, json, platform, shutil, subprocess, sys, time
from pathlib import Path

def _imp(name):
    try: importlib.import_module(name); return True, None
    except Exception as e: return False, str(e)

def collect_environment(output_dir='demo_output', cache_dir='~/.cache/segs-demo', minimum_disk_gb=1, device='cuda'):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    checks={"python_version":sys.version,"platform":platform.platform(),"time":time.time(),"checks":{}}
    for mod in ['torch','torchvision','lpips','yaml','PIL','plyfile','diff_gaussian_rasterization','simple_knn','fused_ssim']:
        ok,err=_imp(mod); checks['checks'][mod]={"ok":ok,"error":err}
    try:
        import torch
        checks['cuda_available']=bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            checks['gpu_name']=torch.cuda.get_device_name(0); checks['compute_capability']=torch.cuda.get_device_capability(0)
    except Exception as e: checks['cuda_error']=str(e)
    usage=shutil.disk_usage(str(Path(cache_dir).expanduser().parent))
    checks['available_disk_gb']=round(usage.free/1024**3,2); checks['minimum_disk_gb']=minimum_disk_gb
    checks['git_submodules_initialized']=(Path('submodules/diff-gaussian-rasterization').exists() and Path('submodules/simple-knn').exists())
    (out/'environment.json').write_text(json.dumps(checks, indent=2))
    problems=[]
    if device=='cuda' and not checks.get('cuda_available'): problems.append('CUDA is unavailable.')
    if checks['available_disk_gb'] < minimum_disk_gb: problems.append('Insufficient disk space.')
    return checks, problems
