#!/usr/bin/env python3
import subprocess, sys
cmd=[sys.executable,'-m','tools.quickstart','--scenes','truck','--methods','baseline,segs_full','--no-open']
open('local_test_reports/gpu_smoke_test.md','w').write('# GPU Smoke Test\n\nCommand: `'+ ' '.join(cmd)+'`\n')
raise SystemExit(subprocess.run(cmd).returncode)
