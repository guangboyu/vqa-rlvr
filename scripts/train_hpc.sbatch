#!/bin/bash
# HPC3: one RTX6000 job and at most one failure successor.
# Submit with --job-name and --output; create the output directory first.
# Arguments: source-directory run-directory training-entrypoint [training options].
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
# Fail promptly on an allocated GPU that the driver cannot access.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO"
export HF_HOME=/pub/guangboy/hf
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS

completed() {
    .venv/bin/python - "$RUN/checkpoints/run_summary.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
sys.exit(0 if path.exists() and json.loads(path.read_text())["completed"] else 1)
PY
}
if completed; then
    exit 0
fi

if [ "$CHAIN_LEFT" -gt 0 ]; then
    NEXT=$(sbatch --parsable --job-name="$SLURM_JOB_NAME" \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="${SLURM_MEM_PER_NODE}M" \
        --output="$RUN/slurm-%j.log" --export=ALL,CHAIN_LEFT=$((CHAIN_LEFT - 1)) \
        --dependency=afternotok:"$SLURM_JOB_ID" \
        "$SELF" "$REPO" "$RUN" "$ENTRY" "$@")
    printf '%s\n' "$NEXT" > "$RUN/successor-job.txt"
fi

.venv/bin/python "$ENTRY" --run-dir "$RUN" --no-wandb "$@" &
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
exit "$code"
