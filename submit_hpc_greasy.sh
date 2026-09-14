#!/bin/bash
#
# submit_hpc_greasy.sh
# =====================
# Runs the full sweep on a CPU cluster using GREASY -- a task-farming tool
# well suited to this kind of workload, since task duration is highly
# skewed (short median task, much longer worst-case task): with thousands
# of individual array elements at that granularity, a plain array job
# spends a real share of its time in per-element scheduling overhead
# instead of computing. GREASY instead runs everything inside ONE job
# allocation and handles the fine-grained scheduling of the task list
# across that allocation's cores itself.
#
# Usage:
#   python sweep.py                                            # -> tasklist_full.jsonl
#   python make_greasy_tasklist.py --tasklist tasklist_full.jsonl \
#       --out greasy_tasklist.txt                               # -> greasy_tasklist.txt
#   # edit --account and --ntasks below, then:
#   sbatch submit_hpc_greasy.sh
#
# --ntasks below is the PARALLEL WIDTH (how many of the total tasks run at
# once), not the task count itself -- GREASY works through the full
# greasy_tasklist.txt file regardless of how many workers you give it, it
# will just take longer with fewer. Since every individual task is 1 core /
# <1GB / no MPI, --ntasks can be set as high as your allocation and queue
# policy allow; more workers = shorter wall-clock for the same total
# core-hour budget.
#
# NOTE on the long tail: the largest configurations (highest N, highest k)
# contain the longest-running tasks. Either (a) give this job a --time long
# enough to cover them, or (b) split greasy_tasklist.txt in two -- one file
# for small/medium configurations (short tasks, short --time), one for the
# largest configurations (long tasks, long --time) -- and submit each as
# its own GREASY job. Splitting is usually the better use of your
# core-hour budget, since it avoids reserving a long walltime for every
# worker just to cover a handful of heavy tasks.

#SBATCH --job-name=fair-madqn-greasy
#SBATCH --output=logs/greasy_%j.out
#SBATCH --error=logs/greasy_%j.err
#SBATCH --account=<YOUR_PROJECT_ID>
#SBATCH --partition=<CPU_PARTITION_NAME>
#SBATCH --ntasks=112                             # parallel width: e.g. one full CPU
                                                  # node's worth of cores; raise this
                                                  # to use more nodes at once
#SBATCH --cpus-per-task=1                        # 1 core per individual task
#SBATCH --time=24:00:00                          # see NOTE above re: the long tail

set -euo pipefail
mkdir -p logs results

# --- one-time environment setup (pure Python + NumPy) -----------
module purge
module load greasy 2>/dev/null || echo "WARNING: 'module load greasy' failed -- adjust module name/version"
module load python/3.11 2>/dev/null || true

if [ ! -d "$HOME/fair_madqn_venv" ]; then
    python3 -m venv "$HOME/fair_madqn_venv"
    source "$HOME/fair_madqn_venv/bin/activate"
    pip install --quiet numpy
else
    source "$HOME/fair_madqn_venv/bin/activate"
fi

# --- run the whole task list under GREASY -----------------------------------
# GREASY_LOGFILE gets one line per completed task (start/end time, exit
# code, restarts) -- useful for the same "how many tasks are missing /
# failed" check aggregate_results.py does for the array-job path.
export GREASY_LOGFILE=logs/greasy_$SLURM_JOB_ID.log

greasy greasy_tasklist.txt
