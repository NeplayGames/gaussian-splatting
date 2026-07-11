import json, statistics, time
from pathlib import Path

def summarize_frame_times(times):
    times=sorted(float(t) for t in times)
    if not times: return {}
    pct=lambda p: times[min(len(times)-1, int(round((p/100)*(len(times)-1))))]
    mean=statistics.mean(times); med=statistics.median(times)
    return {"mean_fps":1/mean if mean else 0,"median_fps":1/med if med else 0,"mean_frame_time":mean,"median_frame_time":med,"p95_frame_time":pct(95),"p99_frame_time":pct(99),"frame_times":times}

def write_performance(records, path): Path(path).write_text(json.dumps(records, indent=2))
