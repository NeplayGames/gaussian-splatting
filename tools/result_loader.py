import csv, json
from pathlib import Path

def write_results(records, output_dir):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'results.json').write_text(json.dumps(records, indent=2))
    fields=list(records[0]) if records else []
    with (out/'results.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)

def format_table(records):
    lines=['SEGS Quick Evaluation','', 'Scene       Dataset              Method       PSNR    SSIM    LPIPS    FPS']
    for r in records:
        lines.append(f"{r.get('scene',''):<11} {r.get('dataset',''):<20} {r.get('method',''):<12} {r.get('psnr','...')!s:<7} {r.get('ssim','...')!s:<7} {r.get('lpips','...')!s:<8} {r.get('fps','...')}")
    return '\n'.join(lines)
