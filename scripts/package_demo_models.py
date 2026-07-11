#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo_model_assets import *

def main():
    p=argparse.ArgumentParser(description='Validate and deterministically package one SEGS demo model.')
    p.add_argument('--model-root', required=True); p.add_argument('--scene', required=True, choices=SCENES); p.add_argument('--method', required=True, choices=METHODS); p.add_argument('--seed', type=int, default=SEED); p.add_argument('--iteration', type=int, default=ITERATION); p.add_argument('--output-dir', required=True); p.add_argument('--url', default=''); p.add_argument('--information-url', default='')
    a=p.parse_args(); root=Path(a.model_root)
    result=validate_model_directory(root, a.scene, a.method, a.seed, a.iteration, require_load_test=True)
    out=Path(a.output_dir)/archive_name(a.scene,a.method,a.seed,a.iteration); top=model_root_name(a.scene,a.method,a.seed,a.iteration)
    deterministic_tar_gz(root, out, top); inspect_archive(out, a.scene, a.method, a.seed, a.iteration)
    manifest=manifest_ready_json(out, root, a.scene, a.method, info_url=a.information_url, url=a.url)
    package_report={'archive_filename':out.name,'archive_sha256':manifest['sha256'],'archive_size_bytes':manifest['size_bytes'],'model_root':str(root),'scene':a.scene,'method':a.method,'seed':a.seed,'iteration':a.iteration,'config_hash':result['config_hash'],'manifest_entry':manifest}
    write_json(Path(a.output_dir)/(out.name+'.manifest.json'), package_report)
    print(json.dumps(package_report, indent=2, sort_keys=True))
if __name__=='__main__': main()
