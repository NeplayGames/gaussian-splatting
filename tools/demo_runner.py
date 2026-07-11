import json, math, os, subprocess, time
from pathlib import Path
from experiments.command_builder import render_command, train_command, validate_method
from experiments.subprocess_runner import run_command, StepError


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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


def valid_training_output(model_dir, iteration):
    model_dir = Path(model_dir)
    ply = model_dir / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
    budget_path = model_dir / 'optimization_budget.json'
    if not ply.exists() or (ply.stat().st_size <= 0):
        return False
    count = _count_ply_vertices(ply)
    if count is None or count <= 0:
        return False
    if not budget_path.exists():
        return False
    try:
        budget = json.loads(budget_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    recorded = budget.get('final_iteration', budget.get('iteration'))
    return recorded == iteration


def _pngs(path):
    return sorted(Path(path).glob('*.png')) if Path(path).is_dir() else []


def valid_render_output(model_dir, split, iteration):
    root = Path(model_dir) / split / f'ours_{iteration}'
    renders = _pngs(root / 'renders')
    gt = _pngs(root / 'gt')
    return len(renders) > 0 and len(renders) == len(gt)


def valid_metrics_output(model_dir, split, iteration):
    metrics_path = Path(model_dir) / 'metrics.json'
    if not metrics_path.exists():
        return False
    try:
        data = json.loads(metrics_path.read_text())
        vals = data[f'{split}/ours_{iteration}']
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    for key in ('PSNR', 'SSIM', 'LPIPS'):
        value = vals.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False
    return True


def _load_metrics(model_dir, split, iteration):
    metrics_path = Path(model_dir) / 'metrics.json'
    data = json.loads(metrics_path.read_text())
    vals = data[f'{split}/ours_{iteration}']
    return {'psnr': vals['PSNR'], 'ssim': vals['SSIM'], 'lpips': vals['LPIPS']}


def _load_budget(model_dir):
    path = Path(model_dir) / 'optimization_budget.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _budget_value(budget, *names, default='unavailable'):
    for name in names:
        if name in budget and budget[name] is not None:
            return budget[name]
    return default


def _status(logs, step):
    return json.loads((Path(logs) / f'{step}_status.json').read_text())


def _run_validated(command, logs, step, resume, validator):
    if resume and validator():
        return 'skipped'
    return run_command(command, logs, step, resume=False)


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
                _run_validated(train_cmd, logs, 'train', resume, lambda: valid_training_output(model_dir, iterations))
                _run_validated(render_cmd, logs, 'render', resume, lambda: valid_render_output(model_dir, split, iterations))
                _run_validated(metrics_cmd, logs, 'metrics', resume, lambda: valid_metrics_output(model_dir, split, iterations))
            except StepError as e:
                raise StepError(f"{e}. Log directory: {logs.resolve()}") from e
            train_status=_status(logs,'train'); budget=_load_budget(model_dir)
            vals=_load_metrics(model_dir, split, iterations)
            ply=model_dir/'point_cloud'/f'iteration_{iterations}'/'point_cloud.ply'
            rec={"dataset":dataset_label,"scene":scene_name,"method":method,"seed":seed,"iteration":iterations,
                 "split":split,"psnr":round(vals['psnr'],6),"ssim":round(vals['ssim'],6),"lpips":round(vals['lpips'],6),
                 "gaussian_count":_budget_value(budget, 'gaussian_count', 'final_gaussian_count', default=_count_ply_vertices(ply)),"model_size_bytes":_budget_value(budget, 'model_file_size_bytes', 'point_cloud_file_size_bytes', default=(ply.stat().st_size if ply.exists() else 'unavailable')),
                 "training_time_seconds":round(_budget_value(budget, 'total_training_time_seconds', 'training_duration_seconds', default=train_status.get('elapsed_seconds',0)),3),
                 "peak_gpu_memory_bytes":_budget_value(budget, 'peak_gpu_memory_bytes', 'peak_gpu_memory_allocated_bytes'),
                 "fps":round(_budget_value(budget, 'render_fps', 'render_fps_recorded_during_training', default=0.0),3),
                 "local_repository_commit":_git_commit(),"model_path":str(model_dir.resolve()),"log_path":str(logs.resolve())}
            records.append(rec)
    return records
