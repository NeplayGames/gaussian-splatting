import argparse, json, shutil, sys
from pathlib import Path
try:
    import yaml
except Exception:
    yaml=None
from .asset_manager import load_manifest, ensure_cache, download_asset, safe_extract, AssetError
from .environment_check import collect_environment
from .demo_runner import run_demo
from .result_loader import write_results, format_table
from .report_generator import generate_report

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

def main(argv=None):
    p=argparse.ArgumentParser(description='Run the local SEGS quick-start demonstration.')
    p.add_argument('--check-only', action='store_true'); p.add_argument('--download-only', action='store_true')
    p.add_argument('--offline', action='store_true'); p.add_argument('--force-download', action='store_true')
    p.add_argument('--cache-dir'); p.add_argument('--output-dir'); p.add_argument('--device', default='cuda')
    p.add_argument('--no-open', action='store_true'); p.add_argument('--clean-output', action='store_true')
    p.add_argument('--train', action='store_true'); p.add_argument('--iterations', type=int, default=None)
    p.add_argument('--scenes'); p.add_argument('--methods')
    args=p.parse_args(argv)
    cfg=load_config();
    if args.cache_dir: cfg['cache_dir']=args.cache_dir
    if args.output_dir: cfg['output_dir']=args.output_dir
    if args.scenes: cfg['scenes']=[s for s in cfg['scenes'] if s['scene'] in parse_csv(args.scenes)]
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
    # Download manifest assets if real URLs are configured.
    if args.download_only:
        for entry in manifest.get('datasets',[]) + manifest.get('models',[]):
            download_asset(entry, cache, args.offline, args.force_download)
        print('Download validation complete.'); return 0
    if args.train and (args.iterations or cfg.get('iteration')) < 30000: print(DISCLAIMER)
    records=run_demo(cfg, manifest, out, train=args.train, iterations=args.iterations or cfg.get('iteration',30000))
    write_results(records, out)
    perf={f"{r['scene']}/{r['method']}": {"mean_fps": r['fps']} for r in records}
    (out/'render_performance.json').write_text(json.dumps(perf, indent=2))
    (out/'dataset_validation.json').write_text(json.dumps([], indent=2))
    report=generate_report(records, env, manifest, out, open_browser=(cfg.get('report',{}).get('open_browser',True) and not args.no_open))
    print(format_table(records)); print(); print((out/'results.csv').resolve()); print(report.resolve())
    return 0
if __name__ == '__main__': raise SystemExit(main())
