import json
from pathlib import Path

def validate_model(root, entry):
    root=Path(root); missing=[f for f in entry.get('required_files',[]) if not (root/f).exists()]
    if missing: raise FileNotFoundError(f"Model package incomplete for {entry.get('scene')}/{entry.get('method')}: {missing}")
    meta={}
    for name in ('runtime_metadata.json','resolved_config.json'):
        p=root/name
        if p.exists():
            try: meta.update(json.loads(p.read_text()))
            except Exception: pass
    for key in ('scene','method','seed','iteration'):
        if key in meta and str(meta[key]) != str(entry.get(key)): raise ValueError(f"Model metadata mismatch for {key}")
    return {"root":str(root),"metadata":meta}
