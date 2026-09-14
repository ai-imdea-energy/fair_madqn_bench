#!/bin/bash
#
# submit_hpc_array.sh
# =====================
# Generic SLURM array-job submission script for a CPU-only cluster
# partition.
#
# Matches the task profile: 1 processor per job (single-threaded Python +
# NumPy, no MPI/OpenMP), thousands of independent jobs, < 1 GB memory per
# job, no inter-process communication (array tasks never talk to each
# other).
#
# Usage:
#   1) Generate the task list once (locally or on the login node):
#        python sweep.py                     # writes tasklist_full.jsonl
#   2) Edit --array below to match: 0-$(( $(wc -l < tasklist_full.jsonl) - 1 ))
#   3) sbatch submit_hpc_array.sh
#
# Because task duration is highly skewed (short median task, much longer
# worst-case task), consider submitting the largest configurations as a
# SEPARATE array with a longer --time limit than the bulk of the (much
# shorter) small-configuration tasks, to avoid wasting walltime allocation
# on thousands of short jobs queued behind a handful of long ones.

#SBATCH --job-name=fair-madqn-sweep
#SBATCH --output=logs/task_%A_%a.out
#SBATCH --error=logs/task_%A_%a.err
#SBATCH --account=<YOUR_PROJECT_ID>
#SBATCH --partition=<CPU_PARTITION_NAME>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                        # 1 core/job
#SBATCH --mem=1G                                 # < 1 GB per job
#SBATCH --time=04:00:00                          # generous for the bulk of tasks;
                                                  # submit the largest configurations
                                                  # separately with a longer --time
#SBATCH --array=0-12624                          # one array element per task
                                                  # (edit to match your task list size)

set -euo pipefail
mkdir -p logs results

# --- one-time environment setup -------------------------------------------
# Pure Python + NumPy only: no compiled scientific libraries, no GPU, no
# MPI. A user-level virtualenv is sufficient and keeps this reproducible
# across nodes without relying on system-wide modules.
module purge
module load python/3.11 2>/dev/null || true   # module name may differ per cluster

if [ ! -d "$HOME/fair_madqn_venv" ]; then
    python3 -m venv "$HOME/fair_madqn_venv"
    source "$HOME/fair_madqn_venv/bin/activate"
    pip install --quiet numpy
else
    source "$HOME/fair_madqn_venv/bin/activate"
fi

# --- run exactly this array task's simulation -------------------------------
python run_hpc_task.py \
    --tasklist tasklist_full.jsonl \
    --results-dir results
    # --task-index is intentionally omitted: run_hpc_task.py reads it from
    # $SLURM_ARRAY_TASK_ID automatically.
