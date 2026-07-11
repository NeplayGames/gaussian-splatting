import argparse, json, shutil, sys
from pathlib import Path
try:
    import yaml
except Exception:
    yaml=None
from .asset_manager import load_manifest, ensure_cache, extract_dataset_asset, resolve_scene_root, validate_dataset_entry, validate_manifest, AssetError, download_asset
from .environment_check import collect_environment
from .demo_runner import run_demo
from .result_loader import write_results, format_table
from .report_generator import generate_report
from .dataset_validator import validate_scene_root, write_validations
from experiments.subprocess_runner import StepError

DISCLAIMER='Reduced-budget demonstration results only; not final thesis results.'

def load_config(path='configs/demo.yaml'):
    text=Path(path).read_text()
    return yaml.safe_load(text) if yaml else {'cache_dir':'~/.cache/segs-demo','output_dir':'./demo_output','iteration':30000,'seed':0,'minimum_disk_gb':80,'scenes':[{'dataset':'tanks_and_temples','scene':'truck'}],'methods':['baseline','segs_full'],'evaluation':{'split':'test'}}

def parse_csv(v): return [x.strip() for x in v.split(',') if x.strip()]

def filter_scenes(cfg, scenes):
    requested=set(parse_csv(scenes))
    cfg['scenes']=[s for s in cfg.get('scenes',[]) if (s['scene'] if isinstance(s,dict) else s) in requested]
    missing=requested-{(s['scene'] if isinstance(s,dict) else s) for s in cfg['scenes']}
    if missing: raise AssetError(f"Unsupported scene(s): {', '.join(sorted(missing))}")

def prepare_dataset_asset(config, manifest, cache, offline=False, force_download=False, resume=False):
    datasets=manifest.get('datasets', [])
    if not datasets: raise AssetError('No dataset entries are configured in the manifest')
    entry=datasets[0]; validate_dataset_entry(entry)
    archive=download_asset(entry, cache, offline=offline, force=force_download)
    dataset_root=extract_dataset_asset(archive, cache, entry, manifest.get('manifest_version',''), force=force_download)
    scene_roots={}; records=[]; scene_labels={'truck':'Truck','drjohnson':'DrJohnson'}
    selected=[s['scene'] if isinstance(s,dict) else s for s in config.get('scenes',[])] or list(entry.get('required_scenes', []))
    for scene in selected:
        scene_root=resolve_scene_root(dataset_root, entry, scene); scene_roots[scene]=scene_root
        records.append(validate_scene_root(scene_root, entry.get('name','dataset'), scene_labels.get(scene, scene), entry.get('url',''), entry.get('sha256',''), entry.get('size_bytes',0), manifest.get('manifest_version','')))
    return entry, dataset_root, scene_roots, records

def main(argv=None):
    p=argparse.ArgumentParser(description='Run the local SEGS demonstration.')
    p.add_argument('--full', action='store_true', help='30,000 iterations; optionally all configured scenes')
    p.add_argument('--resume', action='store_true', help='reuse completed local steps')
    p.add_argument('--check-only', action='store_true'); p.add_argument('--download-only', action='store_true')
    p.add_argument('--offline', action='store_true'); p.add_argument('--force-download', action='store_true')
    p.add_argument('--cache-dir'); p.add_argument('--output-dir'); p.add_argument('--device', default='cuda')
    p.add_argument('--no-open', action='store_true', default=True); p.add_argument('--clean-output', action='store_true')
    p.add_argument('--iterations', type=int); p.add_argument('--scenes'); p.add_argument('--methods')
    args=p.parse_args(argv); cfg=load_config()
    cfg['iteration']=30000 if args.full else 1000
    cfg['scenes']=[s for s in cfg['scenes'] if s['scene']=='truck'] if not args.full else cfg['scenes']
    if args.cache_dir: cfg['cache_dir']=args.cache_dir
    if args.output_dir: cfg['output_dir']=args.output_dir
    if args.iterations: cfg['iteration']=args.iterations
    if args.scenes: filter_scenes(cfg,args.scenes)
    if args.methods: cfg['methods']=parse_csv(args.methods)
    out=Path(cfg['output_dir']);
    if args.clean_output and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    env, problems=collect_environment(out, cfg['cache_dir'], cfg.get('minimum_disk_gb',1), args.device)
    if problems:
        print('\n'.join(problems), file=sys.stderr); return 2
    if args.check_only:
        print('Environment check complete.'); return 0
    manifest=load_manifest(); cache=ensure_cache(cfg['cache_dir'])
    validate_manifest(manifest, asset_scope='dataset', config=cfg)
    entry, root, scene_roots, validation_records=prepare_dataset_asset(cfg, manifest, cache, offline=args.offline, force_download=args.force_download, resume=args.resume)
    write_validations(validation_records, out/'dataset_validation.json')
    if args.download_only:
        print('Dataset download validation complete.'); return 0
    if cfg['iteration'] < 30000: print(DISCLAIMER)
    records=run_demo(cfg, manifest, out, scene_roots=scene_roots, iterations=cfg['iteration'], resume=args.resume)
    write_results(records, out)
    report=generate_report(records, env, manifest, out, open_browser=False)
    print(format_table(records)); print(); print((out/'results.csv').resolve()); print(report.resolve())
    return 0
if __name__ == '__main__':
    try: raise SystemExit(main())
    except (AssetError, StepError) as error:
        print(f"Demo failed: {error}", file=sys.stderr); raise SystemExit(2)
