import zipfile, pytest
from tools.asset_manager import safe_extract, AssetError

def test_zip_traversal_rejected(tmp_path):
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f: f.writestr('../evil.txt','x')
    with pytest.raises(AssetError): safe_extract(z, tmp_path/'out')

def test_zip_extracts_safe(tmp_path):
    z=tmp_path/'ok.zip'
    with zipfile.ZipFile(z,'w') as f: f.writestr('scene/images/a.png','x')
    safe_extract(z, tmp_path/'out')
    assert (tmp_path/'out/scene/images/a.png').exists()
