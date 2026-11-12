"""Measure fresh CLI processes with explicit judge-compatible invocations."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

def run_cli(root, fixture, bundle, output, device, timeout, cli_name='inference_script.py', batch_size=None, workers=None, threshold=None):
    # runpy executes the unchanged CLI, while the exit hook records PyTorch allocator peaks.
    hook = ("import atexit,runpy,sys,torch; "
            "atexit.register(lambda: print('P1_CUDA_PEAK '+str(torch.cuda.max_memory_allocated())+' '+str(torch.cuda.max_memory_reserved())) if torch.cuda.is_initialized() else None); "
            "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
    command = [sys.executable, '-c', hook, str(root / cli_name), '--test_dir', str(fixture),
               '--weights', str(bundle), '--output', str(output)]
    if device != 'cuda':
        command.extend(['--device', device])
    for flag, value in [('--batch_size', batch_size), ('--workers', workers), ('--threshold', threshold)]:
        if value is not None:
            command.extend([flag, str(value)])
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'})
    start = time.monotonic()
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stdout, tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stderr:
        proc = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        rss_peak = gpu_peak = None
        timed_out = False
        while proc.poll() is None:
            if time.monotonic() - start > timeout:
                timed_out = True
                proc.kill()
                break
            try:
                status = Path('/proc') / str(proc.pid) / 'status'
                for line in status.read_text().splitlines():
                    if line.startswith('VmHWM:'):
                        rss_peak = max(rss_peak or 0, int(line.split()[1]) * 1024)
            except (OSError, ValueError):
                pass
            if device == 'cuda':
                try:
                    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True)
                    for line in q.stdout.splitlines():
                        pid, memory = [x.strip() for x in line.split(',')]
                        if int(pid) == proc.pid:
                            gpu_peak = max(gpu_peak or 0, int(memory) * 1024 * 1024)
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            time.sleep(.2)
        proc.wait()
        elapsed = time.monotonic() - start
        stdout.seek(0); stderr.seek(0)
        out, err = stdout.read(), stderr.read()
    result = {'device': device, 'wall_seconds': elapsed, 'timeout_seconds': timeout, 'timed_out': timed_out,
              'returncode': proc.returncode, 'stdout': out, 'stderr': err,
              'cli_name': cli_name, 'requested_batch_size': batch_size, 'requested_workers': workers, 'requested_threshold': threshold, 'peak_rss_bytes_observed': rss_peak,
              'peak_gpu_bytes_nvidia_smi_observed': gpu_peak}
    return result

