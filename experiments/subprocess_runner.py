import json, subprocess, time
try:
    import torch
except Exception:
    torch=None
from pathlib import Path

class StepError(RuntimeError): pass

def run_command(command, log_dir, step_name, resume=False):
    log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / f"{step_name}_status.json"
    if resume and status_path.exists():
        try:
            if json.loads(status_path.read_text()).get("status") == "success": return "skipped"
        except json.JSONDecodeError: pass
    started = time.time()
    with (log_dir/f"{step_name}.stdout.log").open("w") as out, (log_dir/f"{step_name}.stderr.log").open("w") as err:
        try:
            
            if torch and torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            subprocess.run(command, check=True, stdout=out, stderr=err)
        except subprocess.CalledProcessError as exc:
            status={"status":"failed","returncode":exc.returncode,"command":list(map(str,command)),"elapsed_seconds":time.time()-started,"stderr_log":str(log_dir/f'{step_name}.stderr.log')}
            status_path.write_text(json.dumps(status, indent=2)); raise StepError(f"{step_name} failed; see {status['stderr_log']}") from exc
    status={"status":"success","returncode":0,"command":list(map(str,command)),"stdout_log":str(log_dir/f"{step_name}.stdout.log"),"stderr_log":str(log_dir/f"{step_name}.stderr.log"),"elapsed_seconds":time.time()-started}
    if torch and torch.cuda.is_available():
        status["peak_gpu_memory_bytes"]=int(torch.cuda.max_memory_allocated())
    status_path.write_text(json.dumps(status, indent=2)); return "success"
