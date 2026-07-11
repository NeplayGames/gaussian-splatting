from tools.environment_check import collect_environment

def test_environment_json(tmp_path):
    env, problems=collect_environment(tmp_path, tmp_path, 0, device='cpu')
    assert (tmp_path/'environment.json').exists(); assert 'python_version' in env
