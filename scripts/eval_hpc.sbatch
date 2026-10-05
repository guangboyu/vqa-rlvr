#!/bin/bash
# HPC3: run one evaluation command on an RTX6000 with at most one failure successor.
# Evaluations cache finished batches, so a successor resumes where its job stopped.
# Submit with --job-name and --output; create the output directory first.
# Arguments: source-directory python-module-or-script [options].
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
REPO=$1
shift
SELF="$REPO/scripts/eval_hpc.sbatch"
CHAIN_LEFT=${CHAIN_LEFT:-1}
cd "$REPO"
export HF_HOME=/pub/guangboy/hf
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cancel_unused_successors() {
    code=$?
    if [ "$code" -eq 0 ]; then
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

if [ "$CHAIN_LEFT" -gt 0 ]; then
    WAITING=$(squeue -h -u "$(id -un)" -t PD -o '%i %E' \
        | awk -v dep="afternotok:$SLURM_JOB_ID(" 'index($2, dep) {print $1}')
    read -r -a SUCCESSORS <<<"$(printf '%s\n' "$WAITING" | xargs)"
    if [ "${#SUCCESSORS[@]}" -gt 0 ]; then
        NEXT=${SUCCESSORS[0]}
        if [ "${#SUCCESSORS[@]}" -gt 1 ]; then
            scancel "${SUCCESSORS[@]:1}"
        fi
    else
        NEXT=$(sbatch --parsable --job-name="$SLURM_JOB_NAME" \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="${SLURM_MEM_PER_NODE}M" \
        --output="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *StdOut=//p')" \
        --export=ALL,CHAIN_LEFT=$((CHAIN_LEFT - 1)) ${AVOID_NODES:+--exclude=$AVOID_NODES} \
            --dependency=afternotok:"$SLURM_JOB_ID" "$SELF" "$REPO" "$@")
    fi
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "$1" in
    *.py) .venv/bin/python "$@" & ;;
    *) .venv/bin/python -m "$@" & ;;
esac
PID=$!
trap 'kill -TERM "$PID" 2>/dev/null || true' TERM
set +e
wait "$PID"
code=$?
while [ "$code" -ge 128 ] && kill -0 "$PID" 2>/dev/null; do
    wait "$PID"
    code=$?
done
exit "$code"
