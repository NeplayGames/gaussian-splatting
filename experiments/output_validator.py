from pathlib import Path

def require_files(root, files):
    missing=[f for f in files if not (Path(root)/f).exists()]
    if missing: raise FileNotFoundError(f"Missing required files under {root}: {missing}")
    return True
