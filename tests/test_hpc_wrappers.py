"""Exercise batch-job exits with fake scheduler commands and real child processes."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def batch_job(tmp_path):
    repo = tmp_path / "repo"
    run = tmp_path / "run"
    commands = tmp_path / "bin"
    (repo / ".venv/bin").mkdir(parents=True)
    (run / "checkpoints").mkdir(parents=True)
    commands.mkdir()
    shim = f"""#!{sys.executable}
import json, os, signal, sys, time
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ['EVENTS'], 'a') as log:
    log.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
if name == 'sbatch':
    Path(os.environ['RUN'], 'scheduled').touch()
    print('200')
elif name == 'squeue':
    if Path(os.environ['RUN'], 'scheduled').exists() or os.environ.get('EXISTING') == '1':
        print('200 afternotok:100(unfulfilled)')
        print('201 afternotok:100(unfulfilled)')
    print('202 afternotok:999(unfulfilled)')
elif name == 'scontrol':
    print('   StdOut=/tmp/job.log')
elif name == 'scancel':
    sys.exit(int(os.environ.get('SCANCEL_CODE', '0')))
elif name == 'nvidia-smi':
    sys.exit(int(os.environ.get('GPU_CODE', '0')))
elif name == 'python':
    if sys.argv[1] == '-':
        os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
    if os.environ.get('SIGNAL_STOP') == '1':
        def stop(signum, frame):
            Path(os.environ['RUN'], 'checkpoints', 'signal-saved').touch()
            sys.exit(0)
        signal.signal(signal.SIGUSR1, stop)
        Path(os.environ['RUN'], 'ready').touch()
        while True:
            time.sleep(.01)
    if sys.argv[1] == 'train.py':
        Path(os.environ['RUN'], 'checkpoints', 'run_summary.json').write_text(
            json.dumps({{'completed': os.environ.get('COMPLETE', '1') == '1'}}))
    sys.exit(int(os.environ.get('PYTHON_CODE', '0')))
"""
    for name in ("sbatch", "squeue", "scontrol", "scancel", "nvidia-smi"):
        path = commands / name
        path.write_text(shim)
        path.chmod(0o755)
    python = repo / ".venv/bin/python"
    python.write_text(shim)
    python.chmod(0o755)
    events = tmp_path / "events.jsonl"
    env = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
               EVENTS=str(events), RUN=str(run), CHAIN_LEFT="1",
               SLURM_JOB_ID="100", SLURM_JOB_NAME="test", SLURM_CPUS_PER_TASK="8",
               SLURM_MEM_PER_NODE="65536", AVOID_NODES="hpc3-gpu-n54-01")

    def launch(kind="train", *, completed=False, background=False, options=(), **overrides):
        if completed:
            (run / "checkpoints/run_summary.json").write_text('{"completed": true}')
        args = ["bash", str(ROOT / f"scripts/{kind}_hpc.sbatch"), str(repo)]
        args += [str(run), "train.py"] if kind == "train" else ["eval.py"]
        args += list(options)
        process = subprocess.Popen(args, env=dict(env, **overrides),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if background:
            return process, run
        stdout, stderr = process.communicate(timeout=10)
        recorded = [json.loads(line) for line in events.read_text().splitlines()]
        return process.returncode, recorded, stdout + stderr

    return launch


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_success_cancels_known_and_requeue_successors(batch_job, kind):
    code, events, output = batch_job(kind)
    assert code == 0, output
    assert ["scancel", "200", "201"] in events
    assert not any("202" in event for event in events if event[0] == "scancel")
    submission = next(event for event in events if event[0] == "sbatch")
    assert "--dependency=afternotok:100" in submission
    assert "--export=ALL,CHAIN_LEFT=0" in submission
    assert "--exclude=hpc3-gpu-n54-01" in submission


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_python_failure_retains_successor(batch_job, kind):
    code, events, _ = batch_job(kind, PYTHON_CODE="23")
    assert code == 23
    assert any(event[0] == "sbatch" for event in events)
    assert not any(event[0] == "scancel" for event in events)

def test_train_successor_keeps_the_first_jobs_gpu_type_and_time(batch_job):
    code, events, output = batch_job(SLURM_JOB_PARTITION="free-gpu", HPC_GRES="gpu:A30:1",
                                     HPC_TIME="06:00:00")
    assert code == 0, output
    submission = next(event for event in events if event[0] == "sbatch")
    assert {"--partition=free-gpu", "--gres=gpu:A30:1", "--time=06:00:00"} <= set(submission)



@pytest.mark.parametrize("kind", ["train", "eval"])
def test_gpu_failure_already_has_successor(batch_job, kind):
    code, events, _ = batch_job(kind, GPU_CODE="9")
    assert code == 9
    names = [event[0] for event in events]
    assert names.index("sbatch") < names.index("nvidia-smi")
    assert not any(event[0] == "scancel" for event in events)


@pytest.mark.parametrize("python_code", ["0", "23"])
def test_completed_training_evaluation_has_recovery_chain(batch_job, python_code):
    code, events, _ = batch_job(completed=True, PYTHON_CODE=python_code)
    assert code == int(python_code)
    names = [event[0] for event in events]
    assert names.index("sbatch") < names.index("nvidia-smi")
    assert any(event[1] == "scripts/evaluate_checkpoints.py"
               for event in events if event[0] == "python")
    assert any(event[0] == "scancel" for event in events) == (code == 0)


def test_completed_training_honors_no_dev_eval(batch_job):
    code, events, _ = batch_job(completed=True, options=("--no-dev-eval",))
    assert code == 0
    assert not any(event[1] == "scripts/evaluate_checkpoints.py"
                   for event in events if event[0] == "python")


def test_exhausted_chain_does_not_submit(batch_job):
    code, events, _ = batch_job(CHAIN_LEFT="0", EXISTING="1")
    assert code == 0
    assert not any(event[0] == "sbatch" for event in events)
    # Clean completion still reaps any dependency left by a previous requeue.
    assert ["scancel", "200", "201"] in events


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_cleanup_failure_is_not_reported_as_success(batch_job, kind):
    code, _, _ = batch_job(kind, SCANCEL_CODE="7")
    assert code == 7


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_requeue_reuses_one_successor_even_when_job_fails(batch_job, kind):
    code, events, _ = batch_job(kind, EXISTING="1", PYTHON_CODE="23")
    assert code == 23
    assert not any(event[0] == "sbatch" for event in events)
    assert ["scancel", "201"] in events
    assert not any("200" in event for event in events if event[0] == "scancel")
    assert not any("202" in event for event in events if event[0] == "scancel")


def test_signal_checkpoint_stop_exits_75_and_keeps_successor(batch_job):
    process, run = batch_job(background=True, SIGNAL_STOP="1")
    try:
        deadline = time.monotonic() + 5
        while not (run / "ready").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (run / "ready").exists()
        process.send_signal(signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 75, stdout + stderr
        assert (run / "checkpoints/signal-saved").exists()
        events = [json.loads(line) for line in (run.parent / "events.jsonl").read_text()
                  .splitlines()]
        assert any(event[0] == "sbatch" for event in events)
        assert not any(event[0] == "scancel" for event in events)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
