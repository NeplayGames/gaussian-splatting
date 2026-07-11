import hashlib
import json
import os
import re
from pathlib import Path

HEX40=re.compile(r'^[0-9a-fA-F]{40}$')
HEX64=re.compile(r'^[0-9a-fA-F]{64}$')
PLACEHOLDERS=('PENDING','UNVERIFIED','REPLACE','PLACEHOLDER','unknown','metadata','manifest')

class ModelValidationError(ValueError):
    pass

def _bad_placeholder(v):
    return isinstance(v,str) and (not v.strip() or any(p.lower()==v.strip().lower() or p in v.upper() for p in PLACEHOLDERS))

def load_required_json(path):
    path=Path(path)
    try:
        value=json.loads(path.read_text())
    except Exception as error:
        raise ModelValidationError(f"Invalid JSON file: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelValidationError(f"Expected JSON object in {path}")
    return value

def _require_file(root, rel):
    p=Path(root)/rel
    if p.is_symlink(): raise ModelValidationError(f"Required file must not be a symlink: {p}")
    if not p.exists(): raise ModelValidationError(f"Missing required file: {p}")
    if not p.is_file(): raise ModelValidationError(f"Required path is not a regular file: {p}")
    if not os.access(p, os.R_OK): raise ModelValidationError(f"Required file is not readable: {p}")
    if p.stat().st_size <= 0: raise ModelValidationError(f"Required file is empty: {p}")
    return p

def _as_int(value, field):
    if isinstance(value,bool): raise ModelValidationError(f"{field} must be an integer")
    try: return int(value)
    except Exception as e: raise ModelValidationError(f"{field} must be an integer") from e

def _check_equal(label, manifest_value, package_value, scene, method):
    if str(manifest_value) != str(package_value):
        raise ModelValidationError(f"Model metadata mismatch for {scene}/{method}: manifest {label}: {manifest_value}; package {label}: {package_value}")

def _canonical_config_hash(resolved_config):
    clean=dict(resolved_config)
    clean.pop('config_hash', None)
    payload=json.dumps(clean, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def _require_fields(data, fields, path):
    for f in fields:
        if f not in data or _bad_placeholder(data[f]):
            raise ModelValidationError(f"Missing or placeholder metadata field {f} in {path}")

def _validate_training_command(data, entry):
    command=data.get('command') or data.get('argv') or data.get('args')
    if not isinstance(command, list) or not all(isinstance(x,(str,int,float)) for x in command):
        raise ModelValidationError('training_command.json must contain an argument list')
    args=[str(x) for x in command]
    joined=' '.join(args).lower()
    if '--method eggs' in joined or 'eggs' in args: raise ModelValidationError('training command must not use EGGS')
    if not any(Path(a).name == 'train.py' or a.endswith('/train.py') for a in args): raise ModelValidationError('training command must record train.py')
    if entry['scene'] not in joined: raise ModelValidationError('training command scene mismatch')
    if entry['method'] not in args and f"--method {entry['method']}" not in joined: raise ModelValidationError('training command method mismatch')
    if str(entry['seed']) not in args and f"--seed {entry['seed']}" not in joined: raise ModelValidationError('training command seed mismatch')
    if str(entry['iteration']) not in args and f"--iterations {entry['iteration']}" not in joined: raise ModelValidationError('training command iteration mismatch')
    if not any(a in args for a in ('--eval','--evaluate','--evaluation')) and 'eval' not in joined:
        raise ModelValidationError('training command must record evaluation mode')
    return args

def _validate_ply(path, expected_vertices=None):
    if path.stat().st_size <= 0: raise ModelValidationError(f"Point cloud is empty: {path}")
    with path.open('rb') as f: head=f.read(4096).decode('utf-8','replace')
    if not head.startswith('ply\n') and not head.startswith('ply\r\n'): raise ModelValidationError(f"Point cloud header does not begin with ply: {path}")
    vertex=None
    for line in head.splitlines():
        m=re.match(r'element\s+vertex\s+(\d+)', line)
        if m: vertex=int(m.group(1)); break
    if vertex is None: raise ModelValidationError(f"Point cloud header missing vertex element: {path}")
    if vertex <= 0: raise ModelValidationError(f"Point cloud vertex count must be positive: {path}")
    if expected_vertices is not None and expected_vertices > 0 and vertex != expected_vertices:
        raise ModelValidationError(f"Point cloud vertex count {vertex} does not match final_gaussian_count {expected_vertices}")
    return vertex

def directory_size(root):
    """Return the byte size of regular, non-symlink files below root.

    This intentionally mirrors the training-time model-size measurement: it sums
    file payload bytes after extraction and excludes directory entries, symlink
    targets, filesystem allocation overhead, and tar/gzip container metadata.
    """
    total=0
    for p in Path(root).rglob('*'):
        if p.is_file() and not p.is_symlink(): total += p.stat().st_size
    return total

def _validate_directory_size(root, recorded_size):
    recorded=_as_int(recorded_size,'complete_model_directory_size_bytes')
    if recorded <= 0: raise ModelValidationError('complete_model_directory_size_bytes must be positive')
    actual=directory_size(root)
    if actual <= 0: raise ModelValidationError('model directory size must be positive')
    # Packaging/extraction may add or omit very small sidecar metadata such as
    # verification markers.  Permit the larger of 4 KiB or 1% of the recorded
    # payload size, but still require the actual extracted tree to closely match
    # the optimization-budget measurement instead of merely being positive.
    tolerance=max(4096, recorded//100)
    if abs(actual-recorded) > tolerance:
        raise ModelValidationError(
            f"model directory size {actual} differs from optimization budget "
            f"complete_model_directory_size_bytes {recorded} by more than "
            f"allowed tolerance {tolerance}"
        )
    return actual

def validate_model(root, entry):
    root=Path(root); scene=entry['scene']; method=entry['method']
    minimum=['cfg_args','cfg_args.json','resolved_config.json','runtime_metadata.json','optimization_budget.json','training_command.json','MODEL_CARD.md',f"point_cloud/iteration_{entry['iteration']}/point_cloud.ply"]
    for rel in list(dict.fromkeys(minimum + list(entry.get('required_files',[])))): _require_file(root, rel)
    cfg=load_required_json(root/'cfg_args.json')
    resolved=load_required_json(root/'resolved_config.json')
    runtime=load_required_json(root/'runtime_metadata.json')
    budget=load_required_json(root/'optimization_budget.json')
    training=load_required_json(root/'training_command.json')
    _require_fields(runtime, ['dataset','scene','method','seed','iterations','training_commit','upstream_commit','config_hash','python_version','pytorch_version','cuda_runtime_version','gpu_name','operating_system','training_start_utc','training_end_utc','training_duration_seconds','dataset_archive_sha256','dataset_manifest_version','command'], root/'runtime_metadata.json')
    for k in ('scene','method','seed'): _check_equal(k, entry[k], runtime[k], scene, method)
    _check_equal('iteration', entry['iteration'], runtime['iterations'], scene, method)
    _check_equal('training_commit', entry['training_commit'], runtime['training_commit'], scene, method)
    _check_equal('upstream_commit', entry['upstream_commit'], runtime['upstream_commit'], scene, method)
    if not HEX40.fullmatch(runtime['training_commit']) or not HEX40.fullmatch(runtime['upstream_commit']): raise ModelValidationError('runtime commit SHA format is invalid')
    if not HEX64.fullmatch(runtime['config_hash']): raise ModelValidationError('runtime config_hash format is invalid')
    if _as_int(runtime['training_duration_seconds'],'training_duration_seconds') <= 0: raise ModelValidationError('training_duration_seconds must be positive')
    if not isinstance(runtime['command'], list): raise ModelValidationError('runtime_metadata command must be a list')
    actual_hash=_canonical_config_hash(resolved)
    if actual_hash != entry['config_hash'] or actual_hash != runtime['config_hash']:
        raise ModelValidationError(f"Configuration hash mismatch for {scene}/{method}: manifest {entry['config_hash']}; runtime {runtime['config_hash']}; actual {actual_hash}")
    _require_fields(budget, ['training_duration_seconds','final_gaussian_count','point_cloud_file_size_bytes','complete_model_directory_size_bytes','peak_gpu_memory_allocated_bytes','final_iteration','seed','method','scene'], root/'optimization_budget.json')
    for k in ('scene','method','seed'): _check_equal(k, entry[k], budget[k], scene, method)
    _check_equal('iteration', entry['iteration'], budget['final_iteration'], scene, method)
    if _as_int(budget['training_duration_seconds'],'budget.training_duration_seconds') <= 0: raise ModelValidationError('budget training_duration_seconds must be positive')
    gaussians=_as_int(budget['final_gaussian_count'],'final_gaussian_count')
    if gaussians <= 0: raise ModelValidationError('final_gaussian_count must be positive')
    ply=root/f"point_cloud/iteration_{entry['iteration']}/point_cloud.ply"
    if ply.stat().st_size != _as_int(budget['point_cloud_file_size_bytes'],'point_cloud_file_size_bytes'):
        raise ModelValidationError('point cloud file size does not match optimization budget')
    _validate_ply(ply, gaussians)
    actual_model_directory_size=_validate_directory_size(root, budget['complete_model_directory_size_bytes'])
    if _as_int(budget['peak_gpu_memory_allocated_bytes'],'peak_gpu_memory_allocated_bytes') < 0: raise ModelValidationError('peak GPU memory must be nonnegative')
    args=_validate_training_command(training, entry)
    return {'root': str(root), 'scene': scene, 'method': method, 'config_hash': actual_hash, 'point_cloud_vertices': gaussians, 'model_directory_size_bytes': actual_model_directory_size, 'training_command': args}

# compatibility with existing scripts/tests
def validate_model_directory(root, scene, method):
    runtime=load_required_json(Path(root)/'runtime_metadata.json')
    entry={'scene':scene,'method':method,'seed':0,'iteration':30000,'training_commit':runtime.get('training_commit',''),'upstream_commit':runtime.get('upstream_commit',''),'config_hash':runtime.get('config_hash',''),'required_files':[]}
    return validate_model(root, entry)
