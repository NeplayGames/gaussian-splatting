import pytest
from tools.asset_manager import validate_model_entry, AssetError

def valid_entry():
    return {'name':'m','version':'2026-07-11','method':'baseline','scene':'truck','seed':0,'iteration':30000,'url':'https://example.com/truck_baseline_seed0_iter30000.tar.gz','information_url':'https://example.com/info','sha256':'a'*64,'size_bytes':1,'archive_filename':'truck_baseline_seed0_iter30000.tar.gz','training_commit':'b'*40,'upstream_commit':'c'*40,'config_hash':'d'*64,'required_files':['cfg_args'],'expected_extracted_files':['root/cfg_args'],'license_url':'https://example.com/license','attribution':'x'}

def test_strict_model_manifest_accepts_final_entry():
    assert validate_model_entry(valid_entry())

def test_placeholder_model_manifest_rejection():
    e=valid_entry(); e['url']='PENDING_STABLE_HOSTING_URL'
    with pytest.raises(AssetError): validate_model_entry(e)

def test_manifest_rejects_bad_fields():
    for key,value in [('sha256','bad'),('size_bytes',0),('training_commit','bad'),('upstream_commit','bad'),('config_hash','bad'),('scene','lego'),('method','eggs'),('seed',1),('iteration',7000),('required_files',[]),('archive_filename','wrong.tar.gz'),('version','pending-local-training')]:
        e=valid_entry(); e[key]=value
        with pytest.raises(AssetError): validate_model_entry(e)
