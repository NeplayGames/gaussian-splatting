import html, json, webbrowser
from pathlib import Path

def generate_report(records, env, manifest, output_dir, open_browser=True):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows=''.join(f"<tr><td>{html.escape(str(r.get('scene')))}</td><td>{html.escape(str(r.get('dataset')))}</td><td>{html.escape(str(r.get('method')))}</td><td>{r.get('psnr','')}</td><td>{r.get('ssim','')}</td><td>{r.get('lpips','')}</td><td>{r.get('fps','')}</td></tr>" for r in records)
    doc=f"""<!doctype html><html><head><meta charset='utf-8'><title>SEGS Quick Evaluation</title><style>body{{font-family:sans-serif}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px}}</style></head><body><h1>SEGS Quick Evaluation</h1><p>Demonstration results are for local validation only.</p><h2>Metrics</h2><table><tr><th>Scene</th><th>Dataset</th><th>Method</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>FPS</th></tr>{rows}</table><h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2))}</pre><h2>Manifest</h2><pre>{html.escape(json.dumps(manifest, indent=2))}</pre><p>Reproduction command: <code>python -m tools.quickstart</code></p></body></html>"""
    path=out/'report.html'; path.write_text(doc)
    if open_browser:
        try: webbrowser.open(path.resolve().as_uri())
        except Exception: pass
    return path
