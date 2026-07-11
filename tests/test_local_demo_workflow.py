import json, math
from pathlib import Path
from experiments.command_builder import train_command, render_command
from tools.quickstart import load_config, filter_scenes
from tools.asset_manager import load_manifest, validate_manifest
from tools.demo_runner import _load_metrics, valid_training_output, valid_render_output, valid_metrics_output


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


def test_training_resume_requires_positive_ply_and_matching_budget(tmp_path):
    model=tmp_path/'m'; ply=model/'point_cloud'/'iteration_1000'/'point_cloud.ply'; ply.parent.mkdir(parents=True)
    ply.write_text('ply\nformat ascii 1.0\nelement vertex 7\nend_header\n')
    assert not valid_training_output(model, 1000)
    (model/'optimization_budget.json').write_text(json.dumps({'final_iteration': 999}))
    assert not valid_training_output(model, 1000)
    (model/'optimization_budget.json').write_text(json.dumps({'final_iteration': 1000}))
    assert valid_training_output(model, 1000)
    ply.write_text('ply\nformat ascii 1.0\nelement vertex 0\nend_header\n')
    assert not valid_training_output(model, 1000)


def test_render_resume_requires_matching_positive_png_counts(tmp_path):
    root=tmp_path/'m'/'test'/'ours_1000'
    (root/'renders').mkdir(parents=True); (root/'gt').mkdir()
    assert not valid_render_output(tmp_path/'m', 'test', 1000)
    (root/'renders'/'000.png').write_bytes(b'x')
    assert not valid_render_output(tmp_path/'m', 'test', 1000)
    (root/'gt'/'000.png').write_bytes(b'x')
    assert valid_render_output(tmp_path/'m', 'test', 1000)
    assert not valid_render_output(tmp_path/'m', 'train', 1000)


def test_metrics_resume_requires_finite_values_for_split_and_iteration(tmp_path):
    model=tmp_path/'m'; model.mkdir()
    (model/'metrics.json').write_text(json.dumps({'test/ours_1000': {'PSNR': 1, 'SSIM': 2, 'LPIPS': 3}}))
    assert valid_metrics_output(model, 'test', 1000)
    assert not valid_metrics_output(model, 'test', 999)
    (model/'metrics.json').write_text(json.dumps({'test/ours_1000': {'PSNR': math.inf, 'SSIM': 2, 'LPIPS': 3}}))
    assert not valid_metrics_output(model, 'test', 1000)


def test_demo_record_uses_budget_not_render_elapsed_or_model_directory_size():
    from tools.demo_runner import _budget_value
    budget={'peak_gpu_memory_bytes': 123, 'model_file_size_bytes': 456, 'render_fps': 78.9}
    assert _budget_value(budget, 'peak_gpu_memory_bytes') == 123
    assert _budget_value(budget, 'model_file_size_bytes') == 456
    assert _budget_value(budget, 'render_fps') == 78.9


def test_metrics_uses_installed_lpips_package():
    source=Path('metrics.py').read_text()
    assert 'import lpips' in source
    assert 'from lpipsPyTorch import lpips' not in source


def test_demo_dependency_declarations_include_local_requirements():
    env=Path('environment-demo.yml').read_text()
    script=Path('run_demo.sh').read_text()
    launcher=Path('run_demo.py').read_text()
    for dep in ['pandas', 'openpyxl', 'cmake', 'ninja']:
        assert dep in env
    assert 'python run_demo.py "$@"' in script
    assert 'cmake' in launcher and 'ninja' in launcher
