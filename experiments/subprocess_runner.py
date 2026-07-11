import json, subprocess, time
from pathlib import Path

class StepError(RuntimeError): pass

def _artifacts_present(expected_artifacts):
    if not expected_artifacts:
        return True
    missing=[]
    for artifact in expected_artifacts:
        path=Path(artifact)
        if not path.exists():
            missing.append(str(path)); continue
        if path.is_dir() and not any(path.iterdir()):
            missing.append(str(path))
    return not missing

def run_command(command, log_dir, step_name, resume=False, expected_artifacts=None):
    log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / f"{step_name}_status.json"
    if resume and status_path.exists():
        try:
            status=json.loads(status_path.read_text())
            if status.get("status") == "success" and _artifacts_present(expected_artifacts):
                return "skipped"
        except json.JSONDecodeError:
            pass
    started = time.time()
    with (log_dir/f"{step_name}.stdout.log").open("w") as out, (log_dir/f"{step_name}.stderr.log").open("w") as err:
        try:
            subprocess.run(command, check=True, stdout=out, stderr=err)
        except subprocess.CalledProcessError as exc:
            status={"status":"failed","returncode":exc.returncode,"command":list(map(str,command)),"elapsed_seconds":time.time()-started,"stderr_log":str(log_dir/f'{step_name}.stderr.log')}
            status_path.write_text(json.dumps(status, indent=2)); raise StepError(f"{step_name} failed; see {status['stderr_log']}") from exc
    status={"status":"success","returncode":0,"command":list(map(str,command)),"stdout_log":str(log_dir/f"{step_name}.stdout.log"),"stderr_log":str(log_dir/f"{step_name}.stderr.log"),"elapsed_seconds":time.time()-started}
    status_path.write_text(json.dumps(status, indent=2)); return "success"
