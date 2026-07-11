import hashlib, json, os, re, shutil, tarfile, time, zipfile, urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

class AssetError(RuntimeError): pass
PLACEHOLDER_WORDS=("PENDING","UNVERIFIED","REPLACE","PLACEHOLDER")

def contains_placeholder(value): return any(w in str(value).upper() for w in PLACEHOLDER_WORDS)
def validate_sha256(value): return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value)))

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def load_manifest(path='configs/demo_manifest.json'):
    return json.loads(Path(path).read_text())

def ensure_cache(cache_dir):
    root=Path(cache_dir).expanduser()
    for d in ('downloads','datasets','models','manifests'): (root/d).mkdir(parents=True, exist_ok=True)
    return root

def verify_file(path, sha256=None, size_bytes=None):
    path=Path(path)
    if not path.exists(): raise AssetError(f"Missing asset: {path}")
    if size_bytes is not None and int(size_bytes) >= 0 and path.stat().st_size != int(size_bytes):
        raise AssetError(f"Size mismatch for {path}: expected {size_bytes}, got {path.stat().st_size}")
    if sha256 and validate_sha256(sha256) and sha256_file(path).lower()!=sha256.lower():
        raise AssetError(f"Checksum mismatch for {path}")
    return True

def _valid_url(url):
    p=urlparse(str(url)); return p.scheme in ('http','https') and bool(p.netloc)

def validate_dataset_entry(entry):
    errors=[]; url=entry.get('url')
    if not url or not _valid_url(url): errors.append('dataset URL must be HTTP(S)')
    if contains_placeholder(url): errors.append('dataset URL contains a placeholder')
    sha=entry.get('sha256')
    if not validate_sha256(sha) or contains_placeholder(sha): errors.append('dataset sha256 must be 64 hex chars and final')
    try:
        if int(entry.get('size_bytes',0))<=0: errors.append('dataset size_bytes must be positive')
    except Exception: errors.append('dataset size_bytes must be positive')
    for k in ('archive_filename','scene_paths'):
        if not entry.get(k): errors.append(f'dataset {k} is required')
    if not entry.get('required_scenes'): errors.append('dataset required_scenes is required')
    sp=entry.get('scene_paths') or {}
    for s in ('truck','drjohnson'):
        if s not in sp: errors.append(f'dataset scene_paths missing {s}')
    if not entry.get('expected_extracted_files'): errors.append('dataset expected_extracted_files is required')
    if not entry.get('extract_prefixes'): errors.append('dataset extract_prefixes is required')
    if errors: raise AssetError('Invalid dataset manifest entry:\n- '+'\n- '.join(errors))
    return True

def validate_model_entry(entry):
    if contains_placeholder(entry.get('url','')) or contains_placeholder(entry.get('sha256','')) or int(entry.get('size_bytes',-1))<=0:
        raise AssetError(f"Invalid pending model manifest entry: {entry.get('name','<unnamed>')}")
    return True

def validate_manifest(manifest, asset_scope='all'):
    if asset_scope in ('dataset','all'):
        for e in manifest.get('datasets',[]): validate_dataset_entry(e)
    if asset_scope in ('models','all'):
        for e in manifest.get('models',[]): validate_model_entry(e)
    return True

def _content_range_ok(value, start):
    return bool(value and re.match(rf"bytes\s+{start}-\d+/\d+|bytes\s+{start}-\d+/\*", value))

def download_asset(entry, cache_dir, offline=False, force=False, timeout=30):
    root=ensure_cache(cache_dir); dest=root/'downloads'/entry['archive_filename']; part=dest.with_name(dest.name+'.part')
    name=entry.get('name', dest.name); url=entry.get('url')
    if force:
        dest.unlink(missing_ok=True); part.unlink(missing_ok=True)
    if dest.exists():
        try:
            verify_file(dest, entry.get('sha256'), entry.get('size_bytes')); print(f"Reusing cached asset {name}: {dest}"); return dest
        except AssetError as e:
            if offline: raise AssetError(f"Offline mode: cached asset {name} is invalid: {e}") from e
            print(f"Replacing invalid cached asset {name}: {e}"); dest.unlink(missing_ok=True)
    if offline: raise AssetError(f"Offline mode: {name} is not cached and valid at {dest}")
    headers={}; start=part.stat().st_size if part.exists() else 0
    if start: headers['Range']=f'bytes={start}-'
    req=urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status=getattr(r,'status', r.getcode()); mode='wb'; downloaded=0
            if start and status==206:
                if not _content_range_ok(r.headers.get('Content-Range'), start): raise AssetError(f"Invalid Content-Range for {name}: {r.headers.get('Content-Range')}")
                mode='ab'; downloaded=start
            elif start and status==200:
                downloaded=0
            total=int(r.headers.get('Content-Length') or 0) + (downloaded if status==206 else 0)
            with open(part, mode) as f:
                while True:
                    chunk=r.read(1024*1024)
                    if not chunk: break
                    f.write(chunk); downloaded += len(chunk)
                    pct=f" ({downloaded/total*100:.1f}%)" if total else ''
                    print(f"Downloading {name}: {downloaded} bytes" + (f" / {total}" if total else '') + pct, end='\r')
        print()
        verify_file(part, entry.get('sha256'), entry.get('size_bytes'))
        os.replace(part, dest); return dest
    except (HTTPError, URLError, TimeoutError, OSError, AssetError) as e:
        raise AssetError(f"Download failed for {name} from {url}: {e}") from e

def _safe_zip_members(zf, dest, prefixes=None):
    dest=Path(dest).resolve(); prefixes=prefixes or []
    for m in zf.infolist():
        out=(dest/m.filename).resolve()
        if not (out == dest or str(out).startswith(str(dest)+os.sep)): raise AssetError(f"Unsafe ZIP member: {m.filename}")
        if prefixes and not any(m.filename.startswith(p) for p in prefixes): continue
        yield m

def verify_expected_files(root, expected_files):
    bad=[]; root=Path(root)
    for rel in expected_files:
        p=root/rel
        if not p.exists(): bad.append(str(p)); continue
        if not os.access(p, os.R_OK): bad.append(str(p)); continue
        if p.is_file() and p.stat().st_size<=0: bad.append(str(p))
    if bad: raise AssetError('Dataset extraction is incomplete. Missing:\n- '+'\n- '.join(bad))
    return True

def _marker_ok(root, entry, manifest_version):
    marker=Path(root)/'.asset_verified.json'
    if not marker.exists(): return False
    try: data=json.loads(marker.read_text())
    except Exception: return False
    return data.get('archive_sha256')==entry.get('sha256') and data.get('archive_size')==entry.get('size_bytes') and data.get('dataset_version')==entry.get('version') and data.get('manifest_version')==manifest_version

def extract_dataset_asset(archive, cache_dir, entry, manifest_version, force=False):
    root=ensure_cache(cache_dir); final=root/'datasets'/entry['version']; tmp=final.with_name(final.name+'.extracting')
    if final.exists() and not force and _marker_ok(final, entry, manifest_version):
        verify_expected_files(final, entry['expected_extracted_files']); print(f"Reusing verified dataset extraction: {final}"); return final
    if final.exists() and not force and not _marker_ok(final, entry, manifest_version): shutil.rmtree(final)
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as zf: zf.extractall(tmp, members=list(_safe_zip_members(zf, tmp, entry.get('extract_prefixes'))))
        verify_expected_files(tmp, entry['expected_extracted_files'])
        (tmp/'.asset_verified.json').write_text(json.dumps({'archive_sha256':entry['sha256'],'archive_size':entry['size_bytes'],'archive_filename':entry['archive_filename'],'dataset_version':entry['version'],'manifest_version':manifest_version,'expected_extracted_paths':entry['expected_extracted_files'],'extraction_time':time.time()}, indent=2))
        if final.exists(): shutil.rmtree(final)
        os.replace(tmp, final); return final
    except Exception:
        if tmp.exists(): shutil.rmtree(tmp)
        raise

def resolve_scene_root(dataset_root, dataset_entry, scene_name):
    rel=(dataset_entry.get('scene_paths') or {}).get(scene_name)
    if not rel: raise AssetError(f"Scene {scene_name} is not in dataset scene_paths")
    root=Path(dataset_root).resolve(); p=(root/rel).resolve()
    if not (p==root or str(p).startswith(str(root)+os.sep)): raise AssetError(f"Scene path escapes dataset root: {rel}")
    if not p.is_dir(): raise AssetError(f"Configured scene directory does not exist: {p}")
    return p

def safe_extract(archive, dest, expected_prefixes=None):
    dest=Path(dest); dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf: zf.extractall(dest, members=list(_safe_zip_members(zf, dest, expected_prefixes)))
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf: tf.extractall(dest)
    else: raise AssetError(f"Unsupported archive: {archive}")
    return dest
