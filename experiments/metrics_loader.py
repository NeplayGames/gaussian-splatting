import json, csv
from pathlib import Path

def load_results_json(path):
    return json.loads(Path(path).read_text())

def write_results_csv(records, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields=list(records[0].keys()) if records else []
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
