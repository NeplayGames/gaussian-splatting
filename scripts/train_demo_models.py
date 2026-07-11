#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo_model_assets import *
from tools.asset_manager import load_manifest, resolve_scene_root

def git_out(args): return subprocess.check_output(['git',*args], text=True).strip()
def resolve_scene(dataset_root, scene):
    manifest=load_manifest(); ds=manifest['datasets'][0]
    return resolve_scene_root(dataset_root, ds, scene), ds, manifest.get('manifest_version','')
def main():
    p=argparse.ArgumentParser(description='Train the four fixed local SEGS demo models.')
    p.add_argument('--dataset-root', required=True); p.add_argument('--output-root', required=True)
    p.add_argument('--scenes'); p.add_argument('--methods'); p.add_argument('--resume', action='store_true'); p.add_argument('--dry-run', action='store_true')
    a=p.parse_args(); outroot=Path(a.output_root); summary={'started_utc':utc_now(),'jobs':[],'training_commit':git_out(['rev-parse','HEAD']),'git_status':git_out(['status','--short']),'submodule_status':git_out(['submodule','status','--recursive'])}
    if summary['git_status'] and not a.dry_run: raise SystemExit('Working tree must be clean before final training runs')
    scenes=split_filter(a.scenes, SCENES); methods=split_filter(a.methods, METHODS)
    for scene,method in training_matrix(scenes, methods):
        job={'scene':scene,'method':method,'start_utc':utc_now()}; summary['jobs'].append(job)
        model_dir=outroot/scene/method; logdir=outroot/'logs'/scene/method; logdir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
        try:
            scene_dir, ds, manifest_version=resolve_scene(Path(a.dataset_root), scene)
            cfg=load_training_config(scene, method); validate_method_config(cfg)
            cfg['dataset_scene_directory']=str(scene_dir); cfg['config_hash']=canonical_config_hash(cfg)
            write_json(model_dir/'resolved_config.json', cfg)
            cmd=build_train_command(scene_dir, model_dir, cfg); write_json(model_dir/'training_command.json', {'command':cmd})
            job.update({'scene_dir':str(scene_dir),'model_dir':str(model_dir),'command':cmd,'config_hash':cfg['config_hash']})
            if a.dry_run:
                job['status']='dry-run'; continue
            if a.resume and (model_dir/f'point_cloud/iteration_{ITERATION}/point_cloud.ply').exists():
                validate_model_directory(model_dir, scene, method, require_load_test=False); job['status']='resumed-valid'; continue
            start=time.perf_counter()
            with (logdir/'stdout.log').open('w') as so, (logdir/'stderr.log').open('w') as se:
                subprocess.run(cmd, stdout=so, stderr=se, check=True)
            job['duration_seconds']=time.perf_counter()-start
            ply=model_dir/f'point_cloud/iteration_{ITERATION}/point_cloud.ply'
            if not ply.exists() or ply.stat().st_size<=0: raise RuntimeError(f'missing/nonempty final PLY: {ply}')
            job['status']='completed'
        except subprocess.CalledProcessError as e:
            job.update({'status':'failed','exit_code':e.returncode,'failure_utc':utc_now(),'error_summary':str(e)})
            write_json(model_dir/'training_status.json', job); write_json(outroot/'training_summary.json', summary); raise
        except Exception as e:
            job.update({'status':'failed','failure_utc':utc_now(),'error_summary':str(e),'traceback':traceback.format_exc()})
            write_json(model_dir/'training_status.json', job); write_json(outroot/'training_summary.json', summary); raise
        finally:
            write_json(outroot/'training_summary.json', summary)
    summary['finished_utc']=utc_now(); write_json(outroot/'training_summary.json', summary)
if __name__=='__main__': main()
