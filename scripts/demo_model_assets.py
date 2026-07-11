#!/usr/bin/env python3
"""Utilities for reproducible SEGS demo model training and packaging."""
from __future__ import annotations
import argparse, gzip, hashlib, io, json, os, platform, re, subprocess, tarfile, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCENES=("truck","drjohnson"); METHODS=("baseline","segs_full"); ITERATION=30000; SEED=0
REQUIRED_FILES=["cfg_args","cfg_args.json","resolved_config.json","runtime_metadata.json","optimization_budget.json","training_command.json","MODEL_CARD.md","point_cloud/iteration_30000/point_cloud.ply"]
JSON_FILES=["cfg_args.json","resolved_config.json","runtime_metadata.json","optimization_budget.json","training_command.json"]
HEX40=re.compile(r"^[0-9a-fA-F]{40}$"); SHA256_RE=re.compile(r"^[0-9a-fA-F]{64}$")
PLACEHOLDERS=("PENDING","PLACEHOLDER","UNKNOWN","UNVERIFIED","REPLACE_ME","TODO","TBD")

def utc_now(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
def read_json(path: Path): return json.loads(path.read_text(encoding='utf-8'))
def write_json(path: Path, data: Any): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(data, indent=2, sort_keys=True)+"\n", encoding='utf-8')
def canonical_config_hash(config):
    c=dict(config); c.pop('config_hash', None)
    payload=json.dumps(c, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()
def file_sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def has_placeholder(v): return any(p in str(v).upper() for p in PLACEHOLDERS)
def assert_sha256(v, name='sha256'):
    if not SHA256_RE.fullmatch(str(v)): raise ValueError(f'{name} must be a 64-character SHA-256')
def assert_commit(v, name='commit'):
    if not HEX40.fullmatch(str(v)) or has_placeholder(v): raise ValueError(f'{name} must be a final 40-character git SHA')

def load_config(path: Path):
    # Config files are JSON documents with .yaml extension to avoid adding a YAML dependency.
    return read_json(path)

def config_path(scene, method): return Path('configs/demo_training')/f'{scene}_{method}.yaml'
def load_training_config(scene, method): return load_config(config_path(scene, method))

def validate_method_config(cfg):
    scene=cfg.get('scene'); method=cfg.get('method')
    if scene not in SCENES: raise ValueError(f'unsupported scene: {scene}')
    if method not in METHODS: raise ValueError(f'unsupported method: {method}')
    if cfg.get('seed') != SEED: raise ValueError('demo models must use seed 0')
    if cfg.get('iterations') != ITERATION: raise ValueError('demo models must use 30000 iterations')
    if cfg.get('evaluation',{}).get('enabled') is not True: raise ValueError('evaluation split must be enabled')
    segs=cfg.get('segs',{}); controls=cfg.get('controls',{})
    if method=='baseline':
        bad=[segs.get('edge_weighting'),segs.get('saliency_weighting'),segs.get('adaptive_curriculum'),segs.get('importance_aware_densification'),segs.get('segs_pruning'),controls.get('constant_scale_control'),controls.get('shuffled_map_control')]
        if any(bool(x) for x in bad): raise ValueError('baseline config must disable all SEGS and control behavior')
    else:
        required=['edge_weighting','saliency_weighting','adaptive_curriculum','importance_aware_densification','segs_pruning']
        missing=[k for k in required if not segs.get(k)]
        if missing: raise ValueError(f'segs_full config missing required SEGS features: {missing}')
        if controls.get('constant_scale_control') or controls.get('shuffled_map_control'): raise ValueError('segs_full must not use control modes')
        if segs.get('edge_only') or segs.get('saliency_only') or segs.get('densification_only'): raise ValueError('segs_full must not use ablation modes')
    return True

def training_matrix(scenes=None, methods=None):
    scenes=scenes or SCENES; methods=methods or METHODS
    return [(s,m) for s in scenes for m in methods]

def split_filter(text, allowed):
    if not text: return list(allowed)
    vals=[x.strip().lower() for x in text.split(',') if x.strip()]
    bad=[x for x in vals if x not in allowed]
    if bad: raise ValueError(f'unsupported filters: {bad}')
    return vals

def build_train_command(scene_dir: Path, model_dir: Path, cfg: dict):
    validate_method_config(cfg)
    opt=cfg['optimization']; dens=cfg['densification']; rend=cfg['rendering']; segs=cfg.get('segs',{})
    cmd=['python','train.py','-s',str(scene_dir),'-m',str(model_dir),'--images',cfg['image_directory'],'--resolution',str(cfg['image_resolution']),'--iterations',str(cfg['iterations']),'--seed',str(cfg['seed']),'--method',cfg['method'],'--eval','--disable_viewer','--lambda_dssim',str(opt['lambda_dssim']),'--position_lr_init',str(opt['position_lr_init']),'--position_lr_final',str(opt['position_lr_final']),'--position_lr_delay_mult',str(opt['position_lr_delay_mult']),'--position_lr_max_steps',str(opt['position_lr_max_steps']),'--feature_lr',str(opt['feature_lr']),'--opacity_lr',str(opt['opacity_lr']),'--scaling_lr',str(opt['scaling_lr']),'--rotation_lr',str(opt['rotation_lr']),'--percent_dense',str(opt['percent_dense']),'--densification_interval',str(dens['densification_interval']),'--opacity_reset_interval',str(dens['opacity_reset_interval']),'--densify_from_iter',str(dens['densify_from_iter']),'--densify_until_iter',str(dens['densify_until_iter']),'--densify_grad_threshold',str(dens['densify_grad_threshold']),'--test_iterations',*map(str,cfg['test_iterations']),'--save_iterations',*map(str,cfg['save_iterations'])]
    if cfg.get('checkpoint_iterations'):
        cmd += ['--checkpoint_iterations', *map(str, cfg['checkpoint_iterations'])]
    if rend.get('antialiasing'): cmd.append('--antialiasing')
    if cfg.get('background',{}).get('white_background'): cmd.append('--white_background')
    if cfg.get('exposure',{}).get('train_test_exp'): cmd.append('--train_test_exp')
    if cfg['method']=='segs_full': cmd += ['--lambda_edge',str(segs['lambda_edge']),'--lambda_saliency',str(segs['lambda_saliency']),'--segs_importance_power',str(segs['importance_power']),'--segs_error_power',str(segs['error_power']),'--segs_confidence_power',str(segs['confidence_power']),'--segs_prune_score_threshold',str(segs['prune_score_threshold'])]
    return cmd

def validate_model_directory(root: Path, scene: str, method: str, seed=SEED, iteration=ITERATION, require_load_test=True):
    root=Path(root); missing=[f for f in REQUIRED_FILES if not (root/f).exists()]
    if missing: raise ValueError(f'Missing required model files: {missing}')
    ply=root/f'point_cloud/iteration_{iteration}/point_cloud.ply'
    if ply.stat().st_size<=0: raise ValueError('Final point_cloud.ply is empty')
    parsed={name:read_json(root/name) for name in JSON_FILES}
    rc=parsed['resolved_config.json']; meta=parsed['runtime_metadata.json']; budget=parsed['optimization_budget.json']
    expected_hash=canonical_config_hash(rc)
    if rc.get('config_hash') not in (None, expected_hash): raise ValueError('resolved_config config_hash mismatch')
    for doc,nm in [(meta,'runtime_metadata'),(budget,'optimization_budget')]:
        if doc.get('scene')!=scene: raise ValueError(f'{nm} scene mismatch')
        if doc.get('method')!=method: raise ValueError(f'{nm} method mismatch')
        if int(doc.get('seed',-1))!=seed: raise ValueError(f'{nm} seed mismatch')
    if int(meta.get('iterations',-1))!=iteration or int(budget.get('final_iteration',-1))!=iteration: raise ValueError('iteration mismatch')
    if meta.get('config_hash')!=expected_hash: raise ValueError('runtime metadata config hash mismatch')
    assert_commit(meta.get('training_commit'),'training_commit'); assert_commit(meta.get('upstream_commit'),'upstream_commit')
    for key in ('python_version','pytorch_version','cuda_runtime_version','nvidia_driver_version','gpu_name','operating_system','dataset_archive_sha256','dataset_manifest_version'):
        if not meta.get(key) or has_placeholder(meta.get(key)): raise ValueError(f'invalid runtime metadata field: {key}')
    for key in ('training_duration_seconds','final_gaussian_count','point_cloud_file_size_bytes','complete_model_directory_size_bytes','peak_gpu_memory_allocated_bytes','peak_gpu_memory_reserved_bytes'):
        if key not in budget or budget[key] is None or float(budget[key]) < 0: raise ValueError(f'invalid optimization budget field: {key}')
    if require_load_test:
        status=root/'load_test_status.json'
        if not status.exists(): raise ValueError('missing load_test_status.json')
        st=read_json(status)
        if st.get('status')!='passed' or int(st.get('iteration',-1))!=iteration: raise ValueError('model load test did not pass')
    return {'config_hash':expected_hash,'point_cloud':str(ply)}

def archive_name(scene, method, seed=SEED, iteration=ITERATION): return f'{scene}_{method}_seed{seed}_iter{iteration}.tar.gz'
def model_root_name(scene, method, seed=SEED, iteration=ITERATION): return f'{scene}_{method}_seed{seed}_iter{iteration}'
def deterministic_tar_gz(model_root: Path, out: Path, top: str):
    out.parent.mkdir(parents=True, exist_ok=True)
    files=[p for p in model_root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('.tmp','.part'))]
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as tf:
        for p in sorted(files, key=lambda x: x.relative_to(model_root).as_posix()):
            rel=p.relative_to(model_root).as_posix(); info=tf.gettarinfo(str(p), arcname=f'{top}/{rel}')
            info.uid=info.gid=0; info.uname=info.gname='root'; info.mtime=0
            with p.open('rb') as f: tf.addfile(info, f)
    with out.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9) as gz: gz.write(buf.getvalue())
    return out

def inspect_archive(archive: Path, scene: str, method: str, seed=SEED, iteration=ITERATION):
    top_expected=model_root_name(scene, method, seed, iteration); roots=set()
    with tarfile.open(archive, 'r:gz') as tf:
        for m in tf.getmembers():
            n=m.name
            if n.startswith('/') or '..' in Path(n).parts: raise ValueError(f'unsafe archive member: {n}')
            roots.add(n.split('/')[0])
            if m.issym() or m.islnk(): raise ValueError(f'links are not allowed in model archive: {n}')
    if roots != {top_expected}: raise ValueError(f'archive must contain exactly one root {top_expected}, got {roots}')
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(archive,'r:gz') as tf: tf.extractall(td)
        validate_model_directory(Path(td)/top_expected, scene, method, seed, iteration, require_load_test=True)
    return True

def manifest_ready_json(archive: Path, model_root: Path, scene: str, method: str, info_url: str='', url: str='', license_url='LICENSE.md', attribution='Packaged by repository maintainers after local training.'):
    meta=read_json(model_root/'runtime_metadata.json'); rc=read_json(model_root/'resolved_config.json')
    return {'name':f'SEGS demo {scene} {method} model','version':meta.get('hosting_version','local-packaged'), 'method':method,'scene':scene,'seed':SEED,'iteration':ITERATION,'url':url,'information_url':info_url,'sha256':file_sha256(archive),'size_bytes':archive.stat().st_size,'archive_filename':archive.name,'training_commit':meta['training_commit'],'upstream_commit':meta['upstream_commit'],'config_hash':canonical_config_hash(rc),'license_url':license_url,'attribution':attribution,'required_files':REQUIRED_FILES,'expected_extracted_files':[f'{model_root_name(scene,method)}/{f}' for f in REQUIRED_FILES]}
