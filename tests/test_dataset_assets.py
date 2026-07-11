import hashlib, io, json, os, zipfile
from pathlib import Path
from urllib.error import URLError
import pytest

from tools.asset_manager import *
from tools.dataset_validator import validate_scene_root, write_validations
from tools.quickstart import prepare_dataset_asset


def zbytes(files):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        for k,v in files.items(): z.writestr(k,v)
    return b.getvalue()

def scene_files(prefix):
    return {f'{prefix}/images/000.png':b'img', f'{prefix}/sparse/0/cameras.bin':b'c', f'{prefix}/sparse/0/images.bin':b'i', f'{prefix}/sparse/0/points3D.bin':b'p'}

def entry_for(data):
    h=hashlib.sha256(data).hexdigest()
    return {'name':'dataset','version':'v1','url':'https://example.com/tandt_db.zip','sha256':h,'size_bytes':len(data),'archive_filename':'tandt_db.zip','required_scenes':['truck','drjohnson'],'scene_paths':{'truck':'tandt/truck','drjohnson':'db/drjohnson'},'extract_prefixes':['tandt/truck/','db/drjohnson/'],'expected_extracted_files':['tandt/truck/images','tandt/truck/sparse/0/cameras.bin','tandt/truck/sparse/0/images.bin','tandt/truck/sparse/0/points3D.bin','db/drjohnson/images','db/drjohnson/sparse/0/cameras.bin','db/drjohnson/sparse/0/images.bin','db/drjohnson/sparse/0/points3D.bin']}

def valid_zip():
    f={}; f.update(scene_files('tandt/truck')); f.update(scene_files('db/drjohnson')); return zbytes(f)

def test_manifest_dataset_validation_rejects_bad_fields():
    e=entry_for(valid_zip()); validate_dataset_entry(e)
    for key,val in [('sha256','UNVERIFIED_REPLACE_AFTER_LOCAL_ASSET_AUDIT'),('url','PENDING_URL'),('size_bytes',-1),('sha256','abc')]:
        bad=dict(e); bad[key]=val
        with pytest.raises(AssetError): validate_dataset_entry(bad)
    bad=dict(e); bad.pop('scene_paths')
    with pytest.raises(AssetError): validate_dataset_entry(bad)

def test_verify_expected_files_missing(tmp_path):
    with pytest.raises(AssetError): verify_expected_files(tmp_path, ['missing'])

def test_valid_cached_archive_reused(tmp_path):
    data=valid_zip(); e=entry_for(data); root=ensure_cache(tmp_path); (root/'downloads'/e['archive_filename']).write_bytes(data)
    assert download_asset(e, tmp_path, offline=False)==root/'downloads'/e['archive_filename']

def test_corrupt_cached_archive_offline_fails_and_online_redownloads(tmp_path, monkeypatch):
    data=valid_zip(); e=entry_for(data); root=ensure_cache(tmp_path); (root/'downloads'/e['archive_filename']).write_bytes(b'bad')
    with pytest.raises(AssetError): download_asset(e, tmp_path, offline=True)
    class R:
        status=200; headers={'Content-Length':str(len(data))}
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def read(self,n=-1):
            nonlocal data; c=data[:n]; data=data[n:]; return c
        def getcode(self): return self.status
    monkeypatch.setattr('urllib.request.urlopen', lambda *a,**k: R())
    out=download_asset(e, tmp_path)
    assert out.read_bytes()!=b'bad'

def test_missing_archive_offline_fails(tmp_path):
    with pytest.raises(AssetError): download_asset(entry_for(valid_zip()), tmp_path, offline=True)

def test_http_resume_206_and_200_overwrite(tmp_path, monkeypatch):
    data=b'abcdef'; e=entry_for(data); root=ensure_cache(tmp_path); part=root/'downloads'/'tandt_db.zip.part'; part.write_bytes(b'abc')
    calls=[]
    class R:
        status=206; headers={'Content-Length':'3','Content-Range':'bytes 3-5/6'}
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def read(self,n=-1): return b'def' if not hasattr(self,'done') else b''
        def getcode(self): return self.status
    def open206(req, timeout=0): calls.append(req.headers.get('Range')); r=R(); r.done=False; old=r.read; r.read=lambda n=-1: (setattr(r,'done',True) or b'def') if not r.done else b''; return r
    monkeypatch.setattr('urllib.request.urlopen', open206); download_asset(e,tmp_path); assert calls==['bytes=3-']
    (root/'downloads'/'tandt_db.zip').unlink(); part.write_bytes(b'xxx')
    payload=bytearray(data)
    class R200(R):
        status=200; headers={'Content-Length':'6'}
        def read(self,n=-1):
            if not payload: return b''
            c=bytes(payload[:n]); del payload[:n]; return c
    monkeypatch.setattr('urllib.request.urlopen', lambda req,timeout=0: R200())
    download_asset(e,tmp_path); assert (root/'downloads'/'tandt_db.zip').read_bytes()==data

def test_zip_path_traversal_and_temporary_extraction(tmp_path):
    data=zbytes({'../evil':b'x'}); e=entry_for(data); e['sha256']=hashlib.sha256(data).hexdigest(); e['size_bytes']=len(data); a=tmp_path/'a.zip'; a.write_bytes(data)
    with pytest.raises(AssetError): extract_dataset_asset(a,tmp_path,e,'m')
    assert not (ensure_cache(tmp_path)/'datasets'/'v1.extracting').exists()

def test_partial_extraction_not_reused_and_scene_resolution(tmp_path):
    data=valid_zip(); e=entry_for(data); a=tmp_path/'a.zip'; a.write_bytes(data); root=ensure_cache(tmp_path); partial=root/'datasets'/'v1'; partial.mkdir(parents=True)
    final=extract_dataset_asset(a,tmp_path,e,'m')
    assert (final/'.asset_verified.json').exists()
    assert resolve_scene_root(final,e,'truck').name=='truck'
    bad=dict(e); bad['scene_paths']={'truck':'../x','drjohnson':'db/drjohnson'}
    with pytest.raises(AssetError): resolve_scene_root(final,bad,'truck')

def test_scene_validation_requires_colmap_metadata(tmp_path):
    scene=tmp_path/'truck'; (scene/'images').mkdir(parents=True); (scene/'images'/'a.JPG').write_bytes(b'i'); (scene/'sparse'/'0').mkdir(parents=True)
    with pytest.raises(FileNotFoundError): validate_scene_root(scene,'d','Truck')
    for n in ['cameras.bin','images.bin','points3D.bin']: (scene/'sparse'/'0'/n).write_bytes(b'x')
    rec=validate_scene_root(scene,'d','DrJohnson','u','s',1,'m'); assert rec['image_count']==1 and rec['points3D_file']

def test_prepare_writes_two_records_and_ignores_models(tmp_path):
    data=valid_zip(); e=entry_for(data); root=ensure_cache(tmp_path); (root/'downloads'/e['archive_filename']).write_bytes(data)
    manifest={'manifest_version':'m','datasets':[e],'models':[{'url':'PENDING','sha256':'PENDING','size_bytes':-1}]}
    _,_,_,records=prepare_dataset_asset({'cache_dir':str(tmp_path)}, manifest, root, offline=True)
    assert [r['scene'] for r in records]==['Truck','DrJohnson']
    out=tmp_path/'dataset_validation.json'; write_validations(records,out); assert len(json.loads(out.read_text()))==2

def test_offline_no_network(tmp_path, monkeypatch):
    data=valid_zip(); e=entry_for(data); root=ensure_cache(tmp_path); (root/'downloads'/e['archive_filename']).write_bytes(data)
    extract_dataset_asset(root/'downloads'/e['archive_filename'], tmp_path, e, 'm')
    monkeypatch.setattr('urllib.request.urlopen', lambda *a,**k: (_ for _ in ()).throw(AssertionError('network')))
    assert download_asset(e,tmp_path,offline=True).exists()
