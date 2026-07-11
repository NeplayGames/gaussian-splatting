import json, tarfile
from pathlib import Path
import pytest
from scripts.demo_model_assets import *

def test_training_matrix_generation_and_command_stability(tmp_path):
    cfg=load_training_config('truck','baseline'); validate_method_config(cfg)
    cmd=build_train_command(tmp_path/'dataset'/ 'tandt/truck', tmp_path/'out', cfg)
    assert training_matrix()==[('truck','baseline'),('truck','segs_full'),('drjohnson','baseline'),('drjohnson','segs_full')]
    assert cmd[:2]==['python','train.py'] and '--method' in cmd and 'baseline' in cmd
    assert all(' ' not in x or Path(x).is_absolute() for x in cmd)

def test_baseline_and_segs_full_assertions():
    validate_method_config(load_training_config('truck','baseline'))
    validate_method_config(load_training_config('truck','segs_full'))
    bad=load_training_config('truck','baseline'); bad['segs']['edge_weighting']=True
    with pytest.raises(ValueError): validate_method_config(bad)
    bad=load_training_config('truck','segs_full'); bad['controls']['constant_scale_control']=True
    with pytest.raises(ValueError): validate_method_config(bad)

def test_canonical_config_hash_is_stable():
    a={'b':2,'a':1}; b={'a':1,'b':2,'config_hash':'ignored'}
    assert canonical_config_hash(a)==canonical_config_hash(b)

def synth_model(root, scene='truck', method='baseline', seed=0, iteration=30000):
    root.mkdir(parents=True); (root/'cfg_args').write_text('Namespace()')
    rc={'scene':scene,'method':method,'seed':seed,'iterations':iteration}; h=canonical_config_hash(rc); rc['config_hash']=h
    meta={'schema_version':'demo-model-v1','dataset':'graphdeco-tandt-db-2023','scene':scene,'method':method,'seed':seed,'iterations':iteration,'training_commit':'a'*40,'upstream_commit':'b'*40,'config_hash':h,'python_version':'3','pytorch_version':'2','torchvision_version':'1','cuda_runtime_version':'12','cuda_toolkit_version':'12','nvidia_driver_version':'driver','gpu_name':'gpu','gpu_compute_capability':'8.0','operating_system':'linux','training_start_utc':'2026-07-11T00:00:00Z','training_end_utc':'2026-07-11T01:00:00Z','training_duration_seconds':1,'dataset_archive_sha256':'c'*64,'dataset_manifest_version':'v','image_count':2,'command':['python','train.py']}
    budget={'training_duration_seconds':1,'final_gaussian_count':1,'point_cloud_file_size_bytes':10,'complete_model_directory_size_bytes':100,'peak_gpu_memory_allocated_bytes':1,'peak_gpu_memory_reserved_bytes':1,'render_fps_recorded_during_training':None,'final_iteration':iteration,'seed':seed,'method':method,'scene':scene}
    for name,data in [('cfg_args.json',{}),('resolved_config.json',rc),('runtime_metadata.json',meta),('optimization_budget.json',budget),('training_command.json',{'command':['python','train.py']}),('load_test_status.json',{'status':'passed','iteration':iteration})]: write_json(root/name,data)
    (root/'MODEL_CARD.md').write_text('model card')
    ply=root/f'point_cloud/iteration_{iteration}/point_cloud.ply'; ply.parent.mkdir(parents=True); ply.write_text('ply\nformat ascii 1.0\nelement vertex 0\nend_header\n')
    return root

def test_validate_model_directory_rejects_missing_invalid_and_mismatches(tmp_path):
    root=synth_model(tmp_path/'m')
    validate_model_directory(root,'truck','baseline')
    (root/'runtime_metadata.json').write_text('{bad')
    with pytest.raises(Exception): validate_model_directory(root,'truck','baseline')
    root=synth_model(tmp_path/'m2'); meta=read_json(root/'runtime_metadata.json'); meta['training_commit']='bad'; write_json(root/'runtime_metadata.json',meta)
    with pytest.raises(ValueError): validate_model_directory(root,'truck','baseline')
    root=synth_model(tmp_path/'m3', scene='drjohnson')
    with pytest.raises(ValueError): validate_model_directory(root,'truck','baseline')
    root=synth_model(tmp_path/'m4', method='segs_full')
    with pytest.raises(ValueError): validate_model_directory(root,'truck','baseline')
    root=synth_model(tmp_path/'m5', seed=1)
    with pytest.raises(ValueError): validate_model_directory(root,'truck','baseline')
    root=synth_model(tmp_path/'m6', iteration=7)
    with pytest.raises(ValueError): validate_model_directory(root,'truck','baseline')
    root=synth_model(tmp_path/'m7'); (root/'point_cloud/iteration_30000/point_cloud.ply').write_text('')
    with pytest.raises(ValueError): validate_model_directory(root,'truck','baseline')

def test_deterministic_archive_ordering_and_manifest_ready(tmp_path):
    root=synth_model(tmp_path/'model')
    a=deterministic_tar_gz(root,tmp_path/archive_name('truck','baseline'),model_root_name('truck','baseline'))
    sha1=file_sha256(a); a2=deterministic_tar_gz(root,tmp_path/'again.tar.gz',model_root_name('truck','baseline'))
    assert sha1==file_sha256(a2)
    inspect_archive(a,'truck','baseline')
    mj=manifest_ready_json(a,root,'truck','baseline',url='https://example.com/file.tgz',info_url='https://example.com/info')
    assert mj['size_bytes']>0 and mj['sha256']==sha1

def test_archive_path_traversal_and_root_rejection(tmp_path):
    bad=tmp_path/'bad.tar.gz'
    with tarfile.open(bad,'w:gz') as tf:
        p=tmp_path/'x'; p.write_text('x'); tf.add(p, arcname='../evil')
    with pytest.raises(ValueError): inspect_archive(bad,'truck','baseline')
    bad2=tmp_path/'bad2.tar.gz'
    with tarfile.open(bad2,'w:gz') as tf:
        p=tmp_path/'x2'; p.write_text('x'); tf.add(p, arcname='root1/file'); tf.add(p, arcname='root2/file')
    with pytest.raises(ValueError): inspect_archive(bad2,'truck','baseline')
