import hashlib, json, os, re, shutil, tarfile, time, zipfile, urllib.request
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from .model_validator import validate_model, ModelValidationError

class AssetError(RuntimeError): pass
PLACEHOLDER_WORDS=("PENDING","UNVERIFIED","REPLACE","PLACEHOLDER","unknown")
SUPPORTED_SCENES=('truck','drjohnson'); SUPPORTED_METHODS=('baseline','segs_full')
REQUIRED_MODEL_FIELDS=('name','version','method','scene','seed','iteration','url','information_url','sha256','size_bytes','archive_filename','training_commit','upstream_commit','config_hash','required_files','expected_extracted_files','license_url','attribution')

def contains_placeholder(value): return any(w.lower() in str(value).lower() for w in PLACEHOLDER_WORDS)
def validate_sha256(value): return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value)))
def _valid_commit(value): return bool(re.fullmatch(r"[0-9a-fA-F]{40}", str(value)))
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def load_manifest(path='configs/demo_manifest.json'): return json.loads(Path(path).read_text())
def ensure_cache(cache_dir):
    root=Path(cache_dir).expanduser(); [(root/d).mkdir(parents=True, exist_ok=True) for d in ('downloads','datasets','models','manifests')]; return root
def verify_file(path, sha256=None, size_bytes=None):
    path=Path(path)
    if not path.exists(): raise AssetError(f"Missing asset: {path}")
    if size_bytes is not None and int(size_bytes) >= 0 and path.stat().st_size != int(size_bytes): raise AssetError(f"Size mismatch for {path}: expected {size_bytes}, got {path.stat().st_size}")
    if sha256 and validate_sha256(sha256) and sha256_file(path).lower()!=sha256.lower(): raise AssetError(f"Checksum mismatch for {path}")
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
    for s in SUPPORTED_SCENES:
        if s not in sp: errors.append(f'dataset scene_paths missing {s}')
    if not entry.get('expected_extracted_files'): errors.append('dataset expected_extracted_files is required')
    if not entry.get('extract_prefixes'): errors.append('dataset extract_prefixes is required')
    if errors: raise AssetError('Invalid dataset manifest entry:\n- '+'\n- '.join(errors))
    return True

def validate_model_entry(entry):
    errors=[]; name=entry.get('name','<unnamed>')
    for k in REQUIRED_MODEL_FIELDS:
        v=entry.get(k)
        if v is None or (isinstance(v,str) and (not v.strip() or contains_placeholder(v))): errors.append(f'{k} is required and must be final')
    if not _valid_url(entry.get('url','')): errors.append('url must use HTTP or HTTPS')
    if not _valid_url(entry.get('information_url','')): errors.append('information_url must use HTTP or HTTPS')
    if entry.get('license_url') and not (_valid_url(entry.get('license_url')) or str(entry.get('license_url')).endswith(('.md','.txt'))): errors.append('license_url must be valid')
    if not validate_sha256(entry.get('sha256')): errors.append('sha256 must be exactly 64 hexadecimal characters')
    if not validate_sha256(entry.get('config_hash')): errors.append('config_hash must be exactly 64 hexadecimal characters')
    if not _valid_commit(entry.get('training_commit')): errors.append('training_commit must be exactly 40 hexadecimal characters')
    if not _valid_commit(entry.get('upstream_commit')): errors.append('upstream_commit must be exactly 40 hexadecimal characters')
    try:
        if int(entry.get('size_bytes',0))<=0: errors.append('size_bytes must be a positive integer')
    except Exception: errors.append('size_bytes must be a positive integer')
    if entry.get('scene') not in SUPPORTED_SCENES: errors.append('scene must be truck or drjohnson')
    if entry.get('method') not in SUPPORTED_METHODS: errors.append('method must be baseline or segs_full')
    if entry.get('seed') != 0: errors.append('seed must equal 0')
    if entry.get('iteration') != 30000: errors.append('iteration must equal 30000')
    if not isinstance(entry.get('required_files'), list) or not entry.get('required_files'): errors.append('required_files must not be empty')
    if not isinstance(entry.get('expected_extracted_files'), list) or not entry.get('expected_extracted_files'): errors.append('expected_extracted_files must not be empty')
    expected=f"{entry.get('scene')}_{entry.get('method')}_seed{entry.get('seed')}_iter{entry.get('iteration')}.tar.gz"
    if entry.get('archive_filename') != expected: errors.append(f'archive_filename must be {expected}')
    if errors: raise AssetError(f"Invalid model manifest entry {name}:\n- "+'\n- '.join(errors))
    return True

def _selected(config):
    scenes=[s['scene'] if isinstance(s,dict) else s for s in config.get('scenes',[])] or list(SUPPORTED_SCENES)
    methods=list(config.get('methods',[])) or list(SUPPORTED_METHODS)
    bad_s=[s for s in scenes if s not in SUPPORTED_SCENES]; bad_m=[m for m in methods if m not in SUPPORTED_METHODS]
    if bad_s: raise AssetError(f"Unsupported scene(s): {', '.join(bad_s)}")
    if bad_m: raise AssetError(f"Unsupported method(s): {', '.join(bad_m)}")
    return scenes, methods

def validate_manifest(manifest, asset_scope='all', config=None):
    if asset_scope in ('dataset','all'):
        for e in manifest.get('datasets',[]): validate_dataset_entry(e)
    if asset_scope in ('models','all'):
        entries=manifest.get('models',[])
        if config:
            scenes,methods=_selected(config); entries=[get_model_entry(manifest,s,m,config.get('seed',0),config.get('iteration',30000), validate=False) for s in scenes for m in methods]
        for e in entries: validate_model_entry(e)
    return True

def get_model_entry(manifest, scene, method, seed, iteration, validate=True):
    if scene not in SUPPORTED_SCENES: raise AssetError(f"Unsupported scene: {scene}")
    if method not in SUPPORTED_METHODS: raise AssetError(f"Unsupported method: {method}")
    matches=[e for e in manifest.get('models',[]) if e.get('scene')==scene and e.get('method')==method and e.get('seed')==seed and e.get('iteration')==iteration]
    if not matches: raise AssetError(f"No model manifest entry matches {scene}/{method} seed={seed} iteration={iteration}")
    if len(matches)>1: raise AssetError(f"Duplicate model manifest entries match {scene}/{method} seed={seed} iteration={iteration}")
    if validate: validate_model_entry(matches[0])
    return matches[0]

def _content_range_ok(value, start): return bool(value and re.match(rf"bytes\s+{start}-\d+/\d+|bytes\s+{start}-\d+/\*", value))
def download_asset(entry, cache_dir, offline=False, force=False, timeout=30):
    root=ensure_cache(cache_dir); dest=root/'downloads'/entry['archive_filename']; part=dest.with_name(dest.name+'.part'); name=entry.get('name', dest.name); url=entry.get('url')
    if force: dest.unlink(missing_ok=True); part.unlink(missing_ok=True)
    if dest.exists():
        try: verify_file(dest, entry.get('sha256'), entry.get('size_bytes')); print(f"Reusing cached asset {name}: {dest.resolve()}"); return dest
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
            elif start and status==200: downloaded=0
            total=int(r.headers.get('Content-Length') or 0)+(downloaded if status==206 else 0)
            with open(part, mode) as f:
                while True:
                    chunk=r.read(1024*1024)
                    if not chunk: break
                    f.write(chunk); downloaded+=len(chunk)
                    pct=f" ({downloaded/total*100:.1f}%)" if total else ''
                    print(f"Downloading {name}: {downloaded} bytes"+(f" / {total}" if total else '')+pct, end='\r')
        print()
        try:
            verify_file(part, entry.get('sha256'), entry.get('size_bytes'))
        except AssetError:
            part.unlink(missing_ok=True)
            raise
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
        if not p.exists() or not os.access(p, os.R_OK) or (p.is_file() and p.stat().st_size<=0): bad.append(str(p))
    if bad: raise AssetError('Extraction is incomplete. Missing/invalid:\n- '+'\n- '.join(bad))
    return True

def _read_marker(root):
    marker=Path(root)/'.asset_verified.json'
    if not marker.exists(): return None
    try: return json.loads(marker.read_text())
    except Exception: return None

def _dataset_marker_ok(root, entry, manifest_version):
    data=_read_marker(root)
    return bool(data and data.get('archive_sha256')==entry.get('sha256') and data.get('archive_size')==entry.get('size_bytes') and data.get('dataset_version')==entry.get('version') and data.get('manifest_version')==manifest_version)

def _model_marker_ok(root, entry, manifest_version):
    data=_read_marker(root)
    if not data: return False
    expected={'manifest_version':manifest_version,'model_version':entry.get('version'),'name':entry.get('name'),'scene':entry.get('scene'),'method':entry.get('method'),'seed':entry.get('seed'),'iteration':entry.get('iteration'),'archive_filename':entry.get('archive_filename'),'archive_sha256':entry.get('sha256'),'archive_size':entry.get('size_bytes'),'training_commit':entry.get('training_commit'),'upstream_commit':entry.get('upstream_commit'),'config_hash':entry.get('config_hash'),'required_files':entry.get('required_files')}
    return all(data.get(k)==v for k,v in expected.items())

def extract_dataset_asset(archive, cache_dir, entry, manifest_version, force=False):
    root=ensure_cache(cache_dir); final=root/'datasets'/entry['version']; tmp=final.with_name(final.name+'.extracting')
    if final.exists() and not force and _dataset_marker_ok(final, entry, manifest_version): verify_expected_files(final, entry['expected_extracted_files']); print(f"Reusing verified dataset extraction: {final}"); return final
    if final.exists() and not force and not _dataset_marker_ok(final, entry, manifest_version): shutil.rmtree(final)
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as zf: zf.extractall(tmp, members=list(_safe_zip_members(zf, tmp, entry.get('extract_prefixes'))))
        verify_expected_files(tmp, entry['expected_extracted_files']); (tmp/'.asset_verified.json').write_text(json.dumps({'archive_sha256':entry['sha256'],'archive_size':entry['size_bytes'],'archive_filename':entry['archive_filename'],'dataset_version':entry['version'],'manifest_version':manifest_version,'expected_extracted_paths':entry['expected_extracted_files'],'extraction_time':time.time()}, indent=2))
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

def _member_target(dest, name):
    pure=PurePosixPath(name)
    if pure.is_absolute() or '..' in pure.parts: raise AssetError(f"Unsafe TAR member path: {name}")
    out=(Path(dest)/Path(*pure.parts)).resolve(); dest=Path(dest).resolve()
    if not (out==dest or str(out).startswith(str(dest)+os.sep)): raise AssetError(f"TAR member escapes extraction root: {name}")
    return out

def _validate_tar_members(tf, dest):
    members=tf.getmembers(); roots=set()
    for m in members:
        if not (m.isfile() or m.isdir() or m.issym() or m.islnk()): raise AssetError(f"Unsupported TAR member type: {m.name}")
        _member_target(dest, m.name); parts=PurePosixPath(m.name).parts
        if parts: roots.add(parts[0])
        if m.issym() or m.islnk():
            target=PurePosixPath(m.linkname)
            base=PurePosixPath(m.name).parent
            candidate=target if target.is_absolute() else base/target
            if target.is_absolute() or '..' in candidate.parts: raise AssetError(f"Unsafe TAR link target: {m.name} -> {m.linkname}")
            _member_target(dest, str(candidate))
    roots={r for r in roots if r not in ('', '.')}
    if len(roots)!=1: raise AssetError(f"Model archive must contain exactly one top-level directory, found: {sorted(roots)}")
    return members, next(iter(roots))

def safe_extract_tar(archive, dest):
    if not tarfile.is_tarfile(archive): raise AssetError(f"Unexpected archive format (expected TAR): {archive}")
    with tarfile.open(archive) as tf:
        members,root=_validate_tar_members(tf, dest)
        for m in members: tf.extract(m, dest, set_attrs=False)
    return Path(dest)/root

def safe_extract(archive, dest, expected_prefixes=None):
    dest=Path(dest); dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf: zf.extractall(dest, members=list(_safe_zip_members(zf, dest, expected_prefixes)))
    elif tarfile.is_tarfile(archive): safe_extract_tar(archive, dest)
    else: raise AssetError(f"Unsupported archive: {archive}")
    return dest

def _write_model_marker(root, entry, manifest_version):
    data={'manifest_version':manifest_version,'model_version':entry['version'],'name':entry['name'],'scene':entry['scene'],'method':entry['method'],'seed':entry['seed'],'iteration':entry['iteration'],'archive_filename':entry['archive_filename'],'archive_sha256':entry['sha256'],'archive_size':entry['size_bytes'],'training_commit':entry['training_commit'],'upstream_commit':entry['upstream_commit'],'config_hash':entry['config_hash'],'required_files':entry['required_files'],'verification_timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (Path(root)/'.asset_verified.json').write_text(json.dumps(data, indent=2, sort_keys=True))
    return data

def _validate_extracted_model(root, entry, manifest_version):
    verify_expected_files(root, entry.get('required_files',[])); verify_expected_files(root, entry.get('expected_extracted_files',[]))
    summary=validate_model(root, entry)
    return summary

def extract_model_asset(entry, archive_path, cache_dir, manifest_version, force=False):
    cache=ensure_cache(cache_dir); final=cache/'models'/entry['scene']/entry['method']/entry['version']; tmp=final.with_name(final.name+'.extracting')
    if final.exists() and not force and _model_marker_ok(final, entry, manifest_version):
        _validate_extracted_model(final, entry, manifest_version); print(f"Reusing verified model extraction: {final.resolve()}"); return final
    if final.exists() and not force:
        try: _validate_extracted_model(final, entry, manifest_version); _write_model_marker(final, entry, manifest_version); return final
        except Exception: shutil.rmtree(final)
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        extracted_root=safe_extract_tar(archive_path, tmp)
        children=[p for p in tmp.iterdir() if p.name != '.asset_verified.json']
        if len(children)!=1 or not children[0].is_dir(): raise AssetError(f"Model archive must extract exactly one top-level directory: {archive_path}")
        _validate_extracted_model(extracted_root, entry, manifest_version)
        final.parent.mkdir(parents=True, exist_ok=True)
        stage=final.with_name(final.name+'.staged')
        if stage.exists(): shutil.rmtree(stage)
        os.replace(extracted_root, stage); shutil.rmtree(tmp)
        _write_model_marker(stage, entry, manifest_version)
        if final.exists(): shutil.rmtree(final)
        os.replace(stage, final); return final
    except Exception as e:
        if tmp.exists(): shutil.rmtree(tmp)
        raise AssetError(f"Model extraction/validation failed for {entry['scene']}/{entry['method']} at {archive_path}: {e}") from e

def prepare_model_assets(config, manifest, cache, offline=False, force_download=False):
    scenes,methods=_selected(config); result={}; manifest_version=manifest.get('manifest_version','')
    for scene in scenes:
        for method in methods:
            entry=get_model_entry(manifest, scene, method, config.get('seed',0), config.get('iteration',30000))
            try:
                archive=download_asset(entry, cache, offline=offline, force=force_download)
                verify_file(archive, entry['sha256'], entry['size_bytes'])
                root=extract_model_asset(entry, archive, cache, manifest_version, force=force_download)
                result[(scene,method)]=root.resolve()
            except Exception as e:
                cmd=f"python -m tools.quickstart --download-only --assets models --force-download --scenes {scene} --methods {method} --no-open"
                raise AssetError(f"Model download failed for {scene}/{method}:\n{entry.get('url')}\n{e}\nSuggested recovery command:\n{cmd}") from e
    return result
