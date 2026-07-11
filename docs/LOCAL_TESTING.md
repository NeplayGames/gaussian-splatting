# Local Testing

Run manually:

```bash
python -m pytest
python -m compileall .
python -m tools.quickstart --check-only
python -m tools.quickstart --download-only
python -m tools.quickstart
python scripts/run_demo_smoke_test.py
```

Record GPU smoke output in `local_test_reports/gpu_smoke_test.md`. Do not use GitHub-hosted automation.
