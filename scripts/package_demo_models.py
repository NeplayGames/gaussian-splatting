#!/usr/bin/env python3
import argparse, hashlib, json, tarfile
from pathlib import Path
REQUIRED=["cfg_args","cfg_args.json","resolved_config.json","runtime_metadata.json","optimization_budget.json","point_cloud/iteration_30000/point_cloud.ply","MODEL_CARD.md"]
def sha(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('model_dir'); ap.add_argument('archive')
    a=ap.parse_args(); root=Path(a.model_dir); missing=[p for p in REQUIRED if not (root/p).exists()]
    if missing: raise SystemExit(f"Missing required model files: {missing}")
    with tarfile.open(a.archive, 'w:gz', format=tarfile.PAX_FORMAT) as tf:
        for p in sorted(root.rglob('*')):
            if p.is_file(): tf.add(p, p.relative_to(root), recursive=False)
    out=Path(a.archive)
    print(json.dumps({"archive_filename":out.name,"size_bytes":out.stat().st_size,"sha256":sha(out),"required_files":REQUIRED}, indent=2))
if __name__=='__main__': main()
