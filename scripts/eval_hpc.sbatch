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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO"
export HF_HOME=/pub/guangboy/hf
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [ "$CHAIN_LEFT" -gt 0 ]; then
    sbatch --parsable --job-name="$SLURM_JOB_NAME" \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="${SLURM_MEM_PER_NODE}M" \
        --output="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *StdOut=//p')" \
        --export=ALL,CHAIN_LEFT=$((CHAIN_LEFT - 1)) \
        --dependency=afternotok:"$SLURM_JOB_ID" "$SELF" "$REPO" "$@"
fi

case "$1" in
    *.py) exec .venv/bin/python "$@" ;;
    *) exec .venv/bin/python -m "$@" ;;
esac
