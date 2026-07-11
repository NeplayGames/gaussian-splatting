import argparse, json, shutil, sys
from pathlib import Path
try:
    import yaml
except Exception:
    yaml=None
from .asset_manager import load_manifest, ensure_cache, download_asset, extract_dataset_asset, resolve_scene_root, validate_dataset_entry, validate_manifest, AssetError, prepare_model_assets, get_model_entry
from .environment_check import collect_environment
from .demo_runner import run_demo
from .result_loader import write_results, format_table
from .report_generator import generate_report
from .dataset_validator import validate_scene_root, write_validations

DISCLAIMER='These are demonstration results produced with a reduced training budget and must not be used as final paper results.'

def load_config(path='configs/demo.yaml'):
    text=Path(path).read_text()
    if yaml: return yaml.safe_load(text)
    # Minimal fallback for environments that have not installed PyYAML yet.
    return {
        'cache_dir': '~/.cache/segs-demo', 'output_dir': './demo_output',
        'iteration': 30000, 'seed': 0, 'minimum_disk_gb': 80,
        'scenes': [{'dataset':'tanks_and_temples','scene':'truck'}, {'dataset':'deep_blending','scene':'drjohnson'}],
        'methods': ['baseline','segs_full'],
        'evaluation': {'split':'test','metrics':['psnr','ssim','lpips']},
        'report': {'html': True, 'console_table': True, 'open_browser': True, 'comparison_images': 4},
    }

def parse_csv(v): return [x for x in v.split(',') if x]

def prepare_dataset_asset(config, manifest, cache, offline=False, force_download=False):
    datasets=manifest.get('datasets', [])
    if not datasets:
        raise AssetError('No dataset entries are configured in the manifest')
    entry=datasets[0]
    validate_dataset_entry(entry)
    archive=download_asset(entry, cache, offline=offline, force=force_download)
    dataset_root=extract_dataset_asset(archive, cache, entry, manifest.get('manifest_version',''), force=force_download)
    scene_roots={}
    records=[]
    scene_labels={'truck':'Truck','drjohnson':'DrJohnson'}
    for scene in entry.get('required_scenes', []):
        scene_root=resolve_scene_root(dataset_root, entry, scene)
        scene_roots[scene]=scene_root
        records.append(validate_scene_root(scene_root, entry.get('name','dataset'), scene_labels.get(scene, scene), entry.get('url',''), entry.get('sha256',''), entry.get('size_bytes',0), manifest.get('manifest_version','')))
    return entry, dataset_root, scene_roots, records

def _fmt_size(n):
    n=int(n)
    for unit in ('B','KiB','MiB','GiB'):
        if n < 1024 or unit == 'GiB': return f"{n:.1f} {unit}" if unit!='B' else f"{n} B"
        n/=1024

def print_model_summary(config, manifest, cache, model_paths):
    print('\nVerified pretrained models')
    print(f"{'Scene':<11} {'Method':<12} {'Version':<18} {'Size':<12} Cache path")
    for (scene, method), path in model_paths.items():
        entry=get_model_entry(manifest, scene, method, config.get('seed',0), config.get('iteration',30000), validate=False)
        print(f"{scene:<11} {method:<12} {entry['version']:<18} {_fmt_size(entry['size_bytes']):<12} {Path(path).resolve()}")
    print(f"\nArchives cache:\n{(Path(cache)/'downloads').resolve()}")
    print(f"\nModels cache:\n{(Path(cache)/'models').resolve()}")

def main(argv=None):
    p=argparse.ArgumentParser(description='Run the local SEGS quick-start demonstration.')
    p.add_argument('--check-only', action='store_true'); p.add_argument('--download-only', action='store_true')
    p.add_argument('--offline', action='store_true'); p.add_argument('--force-download', action='store_true')
    p.add_argument('--assets', choices=['dataset','models','all'], default='all')
    p.add_argument('--cache-dir'); p.add_argument('--output-dir'); p.add_argument('--device', default='cuda')
    p.add_argument('--no-open', action='store_true'); p.add_argument('--clean-output', action='store_true')
    p.add_argument('--train', action='store_true'); p.add_argument('--iterations', type=int, default=None)
    p.add_argument('--scenes'); p.add_argument('--methods')
    args=p.parse_args(argv)
    cfg=load_config();
    if args.cache_dir: cfg['cache_dir']=args.cache_dir
    if args.output_dir: cfg['output_dir']=args.output_dir
    if args.scenes:
        requested=parse_csv(args.scenes)
        cfg['scenes']=[{'scene': s} for s in requested]
    if args.methods: cfg['methods']=parse_csv(args.methods)
    out=Path(cfg['output_dir'])
    if args.clean_output and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    env, problems=collect_environment(out, cfg['cache_dir'], cfg.get('minimum_disk_gb',1), args.device)
    if problems and not (args.check_only or args.download_only):
        print('\n'.join(problems), file=sys.stderr); return 2
    if args.check_only:
        print('Environment check complete.'); return 0 if not problems else 2
    manifest=load_manifest(); cache=ensure_cache(cfg['cache_dir'])
    validate_manifest(manifest, asset_scope=args.assets, config=cfg)
    if args.download_only:
        if args.assets in ('dataset','all'):
            _, _, _, validation_records = prepare_dataset_asset(cfg, manifest, cache, offline=args.offline, force_download=args.force_download)
            write_validations(validation_records, out/'dataset_validation.json')
        if args.assets in ('models','all'):
            model_paths=prepare_model_assets(cfg, manifest, cache, offline=args.offline, force_download=args.force_download)
            print_model_summary(cfg, manifest, cache, model_paths)
        print('Download validation complete.'); return 0
    if args.train and (args.iterations or cfg.get('iteration')) < 30000: print(DISCLAIMER)
    records=run_demo(cfg, manifest, out, train=args.train, iterations=args.iterations or cfg.get('iteration',30000))
    write_results(records, out)
    perf={f"{r['scene']}/{r['method']}": {"mean_fps": r['fps']} for r in records}
    (out/'render_performance.json').write_text(json.dumps(perf, indent=2))
    if args.assets in ('dataset','all'):
        _, _, _, validation_records = prepare_dataset_asset(cfg, manifest, cache, offline=args.offline, force_download=args.force_download)
        write_validations(validation_records, out/'dataset_validation.json')
    report=generate_report(records, env, manifest, out, open_browser=(cfg.get('report',{}).get('open_browser',True) and not args.no_open))
    print(format_table(records)); print(); print((out/'results.csv').resolve()); print(report.resolve())
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except AssetError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2)
