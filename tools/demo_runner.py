import json, os, time, subprocess
from pathlib import Path
from experiments.command_builder import render_command, train_command, validate_method
from experiments.subprocess_runner import run_command, StepError


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _dir_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def _count_ply_vertices(path):
    try:
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith('element vertex'):
                    return int(line.split()[-1])
                if line.strip() == 'end_header':
                    break
    except Exception:
        return None
    return None


def _load_metrics(model_dir, split, iteration):
    metrics_path = Path(model_dir) / 'metrics.json'
    data = json.loads(metrics_path.read_text())
    vals = data[f'{split}/ours_{iteration}']
    return {'psnr': vals['PSNR'], 'ssim': vals['SSIM'], 'lpips': vals['LPIPS']}


def _fps_from_status(model_dir, render_status, split, iteration):
    render_root = Path(model_dir) / split / f'ours_{iteration}' / 'renders'
    n = len(list(render_root.glob('*.png')))
    elapsed = max(float(render_status.get('elapsed_seconds', 0.0)), 1e-9)
    return n / elapsed if n else 0.0


def _status(logs, step):
    return json.loads((Path(logs) / f'{step}_status.json').read_text())


def run_demo(config, manifest, output_dir, scene_roots=None, iterations=1000, resume=False):
    records=[]; out=Path(output_dir); scene_roots=scene_roots or {}
    split=config.get('evaluation',{}).get('split','test'); seed=config.get('seed',0)
    for scene_cfg in config['scenes']:
        scene_name = scene_cfg['scene'] if isinstance(scene_cfg, dict) else scene_cfg
        source = Path(scene_roots.get(scene_name) or scene_cfg.get('path','')).resolve()
        if not source or not source.exists():
            raise StepError(f"Scene root for {scene_name} is missing: {source}")
        dataset_label="Tanks and Temples" if scene_cfg.get('dataset')=='tanks_and_temples' or scene_name=='truck' else 'Deep Blending'
        for method in config['methods']:
            validate_method(method, ('baseline','segs_full'))
            logs=out/'logs'/scene_name/method; logs.mkdir(parents=True, exist_ok=True)
            model_dir=out/'models'/scene_name/method; model_dir.mkdir(parents=True, exist_ok=True)
            (logs/'DISCLAIMER.txt').write_text('Reduced-budget demonstration results; not final thesis results unless --full was used and validated.\n')
            train_cmd=train_command(source, model_dir, method, seed, iterations)
            render_cmd=render_command(source, model_dir, iterations, split)
            metrics_cmd=[os.sys.executable, 'metrics.py', '-m', str(model_dir)]
            (logs/'commands.json').write_text(json.dumps({'train':train_cmd,'render':render_cmd,'metrics':metrics_cmd}, indent=2))
            try:
                run_command(train_cmd, logs, 'train', resume=resume)
                run_command(render_cmd, logs, 'render', resume=resume)
                run_command(metrics_cmd, logs, 'metrics', resume=resume)
            except StepError as e:
                raise StepError(f"{e}. Log directory: {logs.resolve()}") from e
            train_status=_status(logs,'train'); render_status=_status(logs,'render')
            vals=_load_metrics(model_dir, split, iterations)
            ply=model_dir/'point_cloud'/f'iteration_{iterations}'/'point_cloud.ply'
            rec={"dataset":dataset_label,"scene":scene_name,"method":method,"seed":seed,"iteration":iterations,
                 "split":split,"psnr":round(vals['psnr'],6),"ssim":round(vals['ssim'],6),"lpips":round(vals['lpips'],6),
                 "gaussian_count":_count_ply_vertices(ply),"model_size_bytes":_dir_size(model_dir),
                 "training_time_seconds":round(train_status.get('elapsed_seconds',0),3),
                 "peak_gpu_memory_bytes":train_status.get('peak_gpu_memory_bytes','unavailable'),
                 "fps":round(_fps_from_status(model_dir, render_status, split, iterations),3),
                 "local_repository_commit":_git_commit(),"model_path":str(model_dir.resolve()),"log_path":str(logs.resolve())}
            records.append(rec)
    return records
