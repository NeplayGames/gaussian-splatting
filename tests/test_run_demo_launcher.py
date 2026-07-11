import os

import pytest

import run_demo


def test_conda_command_uses_no_capture_and_env_name():
    assert run_demo.conda_command("demo env", ["python", "-m", "tools.quickstart", "--resume"]) == [
        "conda", "run", "--no-capture-output", "-n", "demo env", "python", "-m", "tools.quickstart", "--resume"
    ]


def test_linux_tool_selection(monkeypatch):
    seen = []
    monkeypatch.setattr(run_demo.platform, "system", lambda: "Linux")
    monkeypatch.setattr(run_demo, "capture_visual_studio_environment", lambda env: dict(env))
    monkeypatch.setattr(run_demo, "ensure_conda_environment", lambda env_name, env: None)
    monkeypatch.setattr(run_demo, "run_logged", lambda step, command, env=None: None)
    monkeypatch.setattr(run_demo, "build_extensions_if_needed", lambda env_name, env: None)
    monkeypatch.setattr(run_demo, "require_tool", lambda name, explanation, env=None: seen.append(name))

    run_demo.prepare_environment()

    assert "g++" in seen
    assert "cl.exe" not in seen


def test_windows_tool_selection_and_process_env(monkeypatch):
    seen = []
    base_env = {"PATH": os.environ.get("PATH", ""), "CUDA_PATH": r"C:\CUDA"}
    monkeypatch.setattr(run_demo.platform, "system", lambda: "Windows")
    monkeypatch.setattr(run_demo.os, "environ", base_env)
    monkeypatch.setattr(run_demo, "capture_visual_studio_environment", lambda env: dict(env))
    monkeypatch.setattr(run_demo, "ensure_conda_environment", lambda env_name, env: None)
    monkeypatch.setattr(run_demo, "run_logged", lambda step, command, env=None: None)
    monkeypatch.setattr(run_demo, "build_extensions_if_needed", lambda env_name, env: None)
    monkeypatch.setattr(run_demo, "require_tool", lambda name, explanation, env=None: seen.append(name))

    _, env = run_demo.prepare_environment()

    assert "cl.exe" in seen
    assert "g++" not in seen
    assert env["DISTUTILS_USE_SDK"] == "1"
    assert env["MSSdk"] == "1"
    assert env["CUDA_HOME"] == r"C:\CUDA"


def test_unsupported_platform_exits(monkeypatch, capsys):
    monkeypatch.setattr(run_demo.platform, "system", lambda: "Darwin")
    with pytest.raises(SystemExit) as exc:
        run_demo.prepare_environment()
    assert exc.value.code == 1
    assert "Unsupported operating system" in capsys.readouterr().err


def test_main_check_only_does_not_run_demo_twice(monkeypatch):
    commands = []
    monkeypatch.setattr(run_demo, "prepare_environment", lambda: ("segs-demo", {}))
    monkeypatch.setattr(run_demo, "run_logged", lambda step, command, env=None: commands.append((step, command)))
    monkeypatch.setattr(run_demo, "run_passthrough", lambda command, env: pytest.fail("final demo should not run for --check-only"))

    assert run_demo.main(["--check-only", "--unknown-forwarded"]) == 0
    assert [step for step, _ in commands] == ["environment_validation"]


def test_main_forwards_unknown_arguments(monkeypatch):
    forwarded = []
    monkeypatch.setattr(run_demo, "prepare_environment", lambda: ("segs-demo", {}))
    monkeypatch.setattr(run_demo, "run_logged", lambda step, command, env=None: None)
    monkeypatch.setattr(run_demo, "run_passthrough", lambda command, env: forwarded.extend(command) or 7)

    assert run_demo.main(["--iterations", "1000", "--custom", "value"]) == 7
    assert forwarded[-4:] == ["--iterations", "1000", "--custom", "value"]
