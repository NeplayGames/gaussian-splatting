import json
from pathlib import Path
from experiments.command_builder import train_command, render_command
from tools.quickstart import load_config, filter_scenes
from tools.asset_manager import load_manifest, validate_manifest
from tools.demo_runner import _load_metrics
from experiments.subprocess_runner import run_command


def test_command_generation_iteration_and_methods(tmp_path):
    cmd=train_command('/data/truck', tmp_path/'m', 'segs_full', 0, 1000)
    assert '--iterations' in cmd and '1000' in cmd
    assert '--save_iterations' in cmd and '--test_iterations' in cmd
    assert '--method' in cmd and 'segs_full' in cmd
    r=render_command('/data/truck', tmp_path/'m', 1000, 'test')
    assert '--iteration' in r and '1000' in r and '--skip_train' in r


def test_scene_filtering_keeps_manifest_metadata():
    cfg=load_config(); filter_scenes(cfg,'truck')
    assert cfg['scenes']==[{'dataset':'tanks_and_temples','scene':'truck'}]


def test_local_training_manifest_validates_dataset_only():
    cfg=load_config(); cfg['scenes']=[s for s in cfg['scenes'] if s['scene']=='truck']; cfg['iteration']=1000
    manifest=load_manifest()
    validate_manifest(manifest, asset_scope='dataset', config=cfg)


def test_result_parsing(tmp_path):
    model=tmp_path/'m'; model.mkdir()
    (model/'metrics.json').write_text(json.dumps({'test/ours_1000': {'PSNR': 1.2, 'SSIM': 0.3, 'LPIPS': 0.4}}))
    assert _load_metrics(model,'test',1000)['psnr']==1.2


def test_resume_behavior_skips_success(tmp_path):
    log=tmp_path/'logs'; log.mkdir()
    (log/'step_status.json').write_text(json.dumps({'status':'success'}))
    assert run_command(['python','-c','raise SystemExit(9)'], log, 'step', resume=True) == 'skipped'


def test_resume_success_requires_expected_artifacts(tmp_path):
    log=tmp_path/'logs'; log.mkdir()
    artifact=tmp_path/'artifact.txt'
    (log/'step_status.json').write_text(json.dumps({'status':'success'}))
    assert run_command(['python','-c',f"from pathlib import Path; Path(r'{artifact}').write_text('ok')"], log, 'step', resume=True, expected_artifacts=[artifact]) == 'success'
    assert artifact.read_text() == 'ok'
    assert run_command(['python','-c','raise SystemExit(9)'], log, 'step', resume=True, expected_artifacts=[artifact]) == 'skipped'


def test_demo_record_uses_budget_not_render_elapsed_or_model_directory_size(tmp_path):
    from tools.demo_runner import _budget_value
    model=tmp_path/'m'; ply=model/'point_cloud'/'iteration_1000'/'point_cloud.ply'; ply.parent.mkdir(parents=True)
    ply.write_text('ply\nelement vertex 7\nend_header\n')
    render=(model/'test'/'ours_1000'/'renders'); render.mkdir(parents=True)
    (render/'00000.png').write_bytes(b'x')
    (model/'metrics.json').write_text(json.dumps({'test/ours_1000': {'PSNR': 1, 'SSIM': 2, 'LPIPS': 3}}))
    budget={'peak_gpu_memory_bytes': 123, 'model_file_size_bytes': 456, 'render_fps': 78.9}
    assert _budget_value(budget, 'peak_gpu_memory_bytes') == 123
    assert _budget_value(budget, 'model_file_size_bytes') == 456
    assert _budget_value(budget, 'render_fps') == 78.9


def test_metrics_uses_installed_lpips_package():
    source=Path('metrics.py').read_text()
    assert 'import lpips' in source
    assert 'from lpipsPyTorch import lpips' not in source
