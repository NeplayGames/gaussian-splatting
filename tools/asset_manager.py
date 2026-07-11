import hashlib, json, os, shutil, tarfile, zipfile, urllib.request
from pathlib import Path
from urllib.error import URLError

class AssetError(RuntimeError): pass

def sha256_file(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def load_manifest(path='configs/demo_manifest.json'):
    return json.loads(Path(path).read_text())

def ensure_cache(cache_dir):
    root=Path(cache_dir).expanduser();
    for d in ('downloads','datasets','models','manifests'): (root/d).mkdir(parents=True, exist_ok=True)
    return root

def verify_file(path, sha256=None, size_bytes=None):
    path=Path(path)
    if not path.exists(): raise AssetError(f"Missing asset: {path}")
    if size_bytes is not None and size_bytes >= 0 and path.stat().st_size != int(size_bytes): raise AssetError(f"Size mismatch for {path}")
    if sha256 and sha256_file(path).lower()!=sha256.lower(): raise AssetError(f"Checksum mismatch for {path}")
    return True

def download_asset(entry, cache_dir, offline=False, force=False):
    root=ensure_cache(cache_dir); dest=root/'downloads'/entry['archive_filename']
    if dest.exists() and not force:
        try: verify_file(dest, entry.get('sha256'), entry.get('size_bytes')); return dest
        except AssetError:
            if offline: raise
            dest.unlink()
    if offline: raise AssetError(f"Offline mode: {dest} is not cached and valid")
    url=entry['url']; part=dest.with_suffix(dest.suffix+'.part')
    req=urllib.request.Request(url, headers={})
    if part.exists(): req.add_header('Range', f'bytes={part.stat().st_size}-')
    try:
        with urllib.request.urlopen(req, timeout=30) as r, open(part, 'ab' if 'Range' in req.headers else 'wb') as f:
            shutil.copyfileobj(r, f)
    except URLError as e: raise AssetError(f"Download failed for {url}: {e}") from e
    verify_file(part, entry.get('sha256'), entry.get('size_bytes'))
    os.replace(part, dest); return dest

def _safe_members_zip(zf, dest, wanted=None):
    dest=Path(dest).resolve()
    for m in zf.infolist():
        out=(dest/m.filename).resolve()
        if not str(out).startswith(str(dest)+os.sep) and out != dest: raise AssetError(f"Unsafe ZIP member: {m.filename}")
        if wanted and not any(m.filename.startswith(w) or f'/{w}/' in m.filename for w in wanted): continue
        yield m

def safe_extract(archive, dest, expected_prefixes=None):
    archive=Path(archive); dest=Path(dest); dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf: zf.extractall(dest, members=list(_safe_members_zip(zf, dest, expected_prefixes)))
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                out=(dest/m.name).resolve()
                if not str(out).startswith(str(dest.resolve())+os.sep): raise AssetError(f"Unsafe tar member: {m.name}")
            tf.extractall(dest)
    else: raise AssetError(f"Unsupported archive: {archive}")
    return dest
