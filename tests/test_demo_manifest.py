from tools.asset_manager import load_manifest

def test_manifest_shape():
    m=load_manifest(); assert m['datasets']; assert len(m['models'])==4
    for e in m['datasets']+m['models']:
        for k in ['name','version','url','sha256','size_bytes','license_url','attribution','archive_filename','expected_extracted_files']:
            assert k in e
