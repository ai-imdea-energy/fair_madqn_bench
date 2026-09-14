"""
run_hpc_task.py
================

Entry point for ONE task of a cluster batch job: either a SLURM array
element, or one line of a GREASY task-farming list -- either way, this
script runs exactly one (seed, composition, k, criterion) combination.

Design: single-threaded, no MPI/OpenMP, no inter-process communication --
every task is fully independent, so this can be scaled out to as many
concurrent workers as the cluster/queue allows without any coordination
between them.

submit_hpc_array.sh / submit_hpc_greasy.sh set the task index (via
SLURM_ARRAY_TASK_ID, or via a --task-index argument for GREASY); this
script:
    1. reads that one task's parameters from the shared task list file,
    2. runs simulate.run_single_task() for it,
    3. writes ONE small JSON result file, named by task index, into the
       results directory.

aggregate_results.py later merges all of these per-task JSON files into a
single CSV once the whole batch has finished. Writing one file per task
(rather than many tasks appending to one shared file) avoids any need for
file locking or coordination between concurrent workers.
"""

from __future__ import annotations
import argparse
import json
import os
import sys

from simulate import run_single_task
import sweep


def main():
    parser = argparse.ArgumentParser(description="Run exactly one sweep task.")
    parser.add_argument("--tasklist", type=str, required=True,
                         help="Path to the JSON-Lines task list produced by sweep.py.")
    parser.add_argument("--task-index", type=int, default=None,
                         help="0-based index into the task list. If omitted, read from "
                              "the SLURM_ARRAY_TASK_ID environment variable.")
    parser.add_argument("--results-dir", type=str, default="results",
                         help="Directory to write this task's result JSON file into.")
    args = parser.parse_args()

    if args.task_index is not None:
        idx = args.task_index
    else:
        idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        if idx is None:
            sys.exit("No --task-index given and SLURM_ARRAY_TASK_ID is not set.")
        idx = int(idx)

    task = sweep.load_task(args.tasklist, idx)

    result = run_single_task(
        n_producers=task["n_producers"],
        n_consumers=task["n_consumers"],
        k=task["k"],
        criterion=task["criterion"],
        seed=task["seed"],
        n_episodes=task["n_episodes"],
        alpha=task.get("alpha", 2.0),
    )
    result["task_index"] = idx

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"task_{idx:06d}.json")
    with open(out_path, "w") as f:
        json.dump(result, f)

    print(f"[task {idx}] N={result['N']} k={result['k']} "
          f"criterion={result['criterion']} seed={result['seed']} "
          f"-> welfare={result['welfare']:.3f}, "
          f"jain={result['jain_fairness']:.3f}, "
          f"wall_clock={result['wall_clock_seconds']:.2f}s "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
