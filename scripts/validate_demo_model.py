#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, tempfile, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo_model_assets import write_json, utc_now

def main():
    p=argparse.ArgumentParser(description='Small render.py load test for a packaged demo model directory.')
    p.add_argument('--dataset-root', required=True); p.add_argument('--model-root', required=True); p.add_argument('--iteration', type=int, default=30000); p.add_argument('--max-views', type=int, default=2); p.add_argument('--keep-renders', action='store_true')
    a=p.parse_args(); model=Path(a.model_root); logdir=model/'validation_logs'; logdir.mkdir(parents=True, exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='demo_model_render_', dir=str(logdir)))
    cmd=['python','render.py','-s',str(Path(a.dataset_root)),'-m',str(model),'--iteration',str(a.iteration),'--skip_train']
    status={'status':'started','start_utc':utc_now(),'command':cmd,'iteration':a.iteration,'max_views':a.max_views}
    try:
        with (logdir/'render_stdout.log').open('w') as so, (logdir/'render_stderr.log').open('w') as se: subprocess.run(cmd, stdout=so, stderr=se, check=True)
        renders=list((model/'test'/(f'ours_{a.iteration}')/'renders').glob('*')) if (model/'test').exists() else []
        nonempty=[str(p) for p in renders if p.is_file() and p.stat().st_size>0]
        if not nonempty: raise RuntimeError('render.py completed but produced no nonempty render outputs')
        status.update({'status':'passed','end_utc':utc_now(),'render_count':len(nonempty),'sample_outputs':nonempty[:a.max_views]})
    except subprocess.CalledProcessError as e:
        status.update({'status':'failed','end_utc':utc_now(),'exit_code':e.returncode}); write_json(model/'load_test_status.json', status); raise SystemExit(e.returncode)
    except Exception as e:
        status.update({'status':'failed','end_utc':utc_now(),'error':str(e)}); write_json(model/'load_test_status.json', status); raise SystemExit(str(e))
    finally:
        if not a.keep_renders: shutil.rmtree(tmp, ignore_errors=True)
    write_json(model/'load_test_status.json', status)
if __name__=='__main__': main()
