from experiments.command_builder import render_command

def test_render_command_no_shell_string():
    cmd=render_command('/data/truck','/models/truck',30000)
    assert isinstance(cmd, list) and 'render.py' in cmd and '--skip_train' in cmd
