#!/bin/bash
# HPC3: one RTX6000 job and at most one failure successor. A completed run goes on to
# evaluate every kept adapter on the dev tier in the same job.
# Submit with --job-name and --output; create the output directory first.
# Arguments: source-directory run-directory training-entrypoint [training options].
# Successors keep the first job's partition; export HPC_GRES and HPC_TIME when the first job
# overrides the GPU type or time limit below (e.g. HPC_GRES=gpu:A30:1 on free-gpu).
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

set -euo pipefail
REPO=$1
RUN=$2
ENTRY=$3
shift 3
SELF="$REPO/scripts/train_hpc.sbatch"
CHAIN_LEFT=${CHAIN_LEFT:-1}
cd "$REPO"
export HF_HOME=/pub/guangboy/hf
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
# A port per job: two jobs on one node both took torch.distributed's default 29500 (EADDRINUSE).
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
# Compute nodes have no internet: vLLM's usage reporting thread only fails there.
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

cancel_unused_successors() {
    code=$?
    if [ "$code" -eq 0 ]; then
        # Requeue can leave more than the successor this invocation created.
        STALE=$( { printf '%s\n' "${NEXT:-}";
            squeue -h -u "$(id -un)" -t PD -o '%i %E' \
                | awk -v dep="afternotok:$SLURM_JOB_ID(" 'index($2, dep) {print $1}';
            } | sort -u | xargs) || exit $?
        if [ -n "$STALE" ]; then
            scancel $STALE || exit $?
            printf '[job] cancelled unused successor(s) %s\n' "$STALE"
        fi
    fi
}
trap cancel_unused_successors EXIT

completed() {
    .venv/bin/python - "$RUN/checkpoints/run_summary.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
sys.exit(0 if path.exists() and json.loads(path.read_text())["completed"] else 1)
PY
}
if [ "$CHAIN_LEFT" -gt 0 ]; then
    # A failed query reads as "no successor waiting", so a fresh one is submitted.
    WAITING=$(squeue -h -u "$(id -un)" -t PD -o '%i %E' \
        | awk -v dep="afternotok:$SLURM_JOB_ID(" 'index($2, dep) {print $1}') || WAITING=""
    read -r -a SUCCESSORS <<<"$(printf '%s\n' "$WAITING" | xargs)"
    if [ "${#SUCCESSORS[@]}" -gt 0 ]; then
        NEXT=${SUCCESSORS[0]}
        if [ "${#SUCCESSORS[@]}" -gt 1 ]; then
            scancel "${SUCCESSORS[@]:1}"
        fi
    else
        NEXT=$(sbatch --parsable --job-name="$SLURM_JOB_NAME" \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="${SLURM_MEM_PER_NODE}M" \
        --output="$RUN/slurm-%j.log" --export=ALL,CHAIN_LEFT=$((CHAIN_LEFT - 1)) ${AVOID_NODES:+--exclude=$AVOID_NODES} \
        ${SLURM_JOB_PARTITION:+--partition=$SLURM_JOB_PARTITION} ${HPC_GRES:+--gres=$HPC_GRES} ${HPC_TIME:+--time=$HPC_TIME} \
        --dependency=afternotok:"$SLURM_JOB_ID" \
            "$SELF" "$REPO" "$RUN" "$ENTRY" "$@")
    fi
    printf '%s\n' "$NEXT" > "$RUN/successor-job.txt"
fi
# Fail promptly on an allocated GPU that the driver cannot access; the successor is queued.
# A node that keeps failing goes into AVOID_NODES, which every successor excludes.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if completed; then
    # Evaluation can itself fail or be preempted, so it needs the same successor.
    for option in "$@"; do
        if [ "$option" = --no-dev-eval ]; then
            exit 0
        fi
    done
    COMMAND=(scripts/evaluate_checkpoints.py "$RUN/checkpoints")
else
    COMMAND=("$ENTRY" --run-dir "$RUN" --no-wandb "$@")
    # Checkpoints go to node-local disk and are copied to $RUN in the background
    # (vqar/checkpoint_mirror.py): saves straight to /pub stalled training for minutes.
    export VQAR_LOCAL_CHECKPOINTS="${TMPDIR:-/tmp/$(id -un)/$SLURM_JOB_ID}/run"
fi
# The console log goes to node-local disk and a background tail copies it into the job's log
# on /pub, so a shared-storage stall or write error blocks the copy, not the trainer.
CONSOLE="${TMPDIR:-/tmp}/console-$SLURM_JOB_ID.log"
: > "$CONSOLE"
tail -n +1 -F "$CONSOLE" 2>/dev/null &
COPIER=$!
.venv/bin/python "${COMMAND[@]}" > "$CONSOLE" 2>&1 &
PID=$!
trap 'kill -USR1 "$PID" 2>/dev/null || true' USR1 TERM
set +e
wait "$PID"
code=$?
while [ "$code" -ge 128 ] && kill -0 "$PID" 2>/dev/null; do
    wait "$PID"
    code=$?
done
if [ "$code" -eq 0 ] && ! completed; then
    # A time-limit warning can stop Trainer cleanly before the planned last step.
    code=75
fi
sleep 2  # let the copier reach the end of the console log
kill "$COPIER" 2>/dev/null || true
exit "$code"
