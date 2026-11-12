#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec 9>job.lock
flock -n 9 || { echo 'FINAL another launcher owns this run'; exit 73; }
trap 'result=$?; printf "{\"returncode\":%s}\n" "$result" > remote-exit.json.part; mv remote-exit.json.part remote-exit.json' EXIT
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
probe_python="$HOME/deepfake_hackathon/env-py310-torch271/bin/python"
"$probe_python" -c 'import sys,torch,torchvision,numpy,PIL; assert sys.version_info[:3]==(3,10,22); assert torch.__version__=="2.7.1+cu128"; assert torchvision.__version__=="0.22.1+cu128"; assert numpy.__version__=="2.2.6"; assert PIL.__version__=="11.2.1"; print("FINAL pinned environment verified")'
"$probe_python" - <<'PY'
import hashlib,json
from pathlib import Path
for name,expected in json.loads(Path('source-hashes.json').read_text()).items():
    path=(Path.cwd()/name).resolve()
    assert path.is_relative_to(Path.cwd()),name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
print('FINAL source hashes verified')
PY
"$probe_python" -u scripts/x1_setup_weights.py --archs convnext_tiny swin_s
"$probe_python" scripts/e1_checks.py
"$probe_python" scripts/contract_checks.py
"$probe_python" scripts/final_runner_checks_v3.py
"$probe_python" scripts/final_interface_checks_v3.py
"$probe_python" -u scripts/train_final_v3.py --dry-run
timeout 14400 "$probe_python" -u scripts/train_final_v3.py --execute-frozen
echo FINAL_JOB_COMPLETE
