"""
run_local.py
============

Desktop parallel runner: the same simulate.run_single_task() logic as
run_hpc_task.py uses on a cluster, parallelized across local CPU cores with
the standard-library multiprocessing.Pool instead of a cluster job.

Usage
-----
    # Fast smoke test (seconds): validates the whole pipeline end-to-end.
    python run_local.py --quick

    # Run (a slice of) the full campaign locally, e.g. to test scaling
    # behaviour before requesting cluster time, or to run a subset that
    # fits comfortably on a workstation:
    python run_local.py --tasklist tasklist_full.jsonl --limit 500 --workers 8

Each worker process is fully independent (no inter-process communication).
Results are streamed to a CSV as they complete so a long local run can be
interrupted/resumed without losing completed work.
"""

from __future__ import annotations
import argparse
import csv
import os
import time
import multiprocessing as mp

from simulate import run_single_task
import sweep


def _run_one(task: dict) -> dict:
    """Thin wrapper so multiprocessing can pickle a plain function call."""
    return run_single_task(
        n_producers=task["n_producers"],
        n_consumers=task["n_consumers"],
        k=task["k"],
        criterion=task["criterion"],
        seed=task["seed"],
        n_episodes=task["n_episodes"],
        alpha=task.get("alpha", 2.0),
    )


def main():
    parser = argparse.ArgumentParser(description="Run Fair-MADQN tasks in parallel on this machine.")
    parser.add_argument("--quick", action="store_true",
                         help="Build and run the tiny built-in smoke-test task list.")
    parser.add_argument("--tasklist", type=str, default=None,
                         help="Path to a JSON-Lines task list (see sweep.py).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N tasks from the task list.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                         help="Number of local worker processes (default: all cores).")
    parser.add_argument("--out", type=str, default="results_local.csv",
                         help="Output CSV path.")
    args = parser.parse_args()

    if args.quick:
        tasks = sweep.generate_full_tasklist(n_episodes=20, seeds=[0, 1])
        tasks = [t for t in tasks if t["k"] in (2, 3)][:80]
        print(f"[quick mode] running {len(tasks)} tiny tasks to sanity-check the pipeline")
    elif args.tasklist:
        n_avail = sweep.count_tasks(args.tasklist)
        n_run = min(args.limit, n_avail) if args.limit else n_avail
        tasks = [sweep.load_task(args.tasklist, i) for i in range(n_run)]
        print(f"Loaded {len(tasks)} / {n_avail} tasks from {args.tasklist}")
    else:
        parser.error("Pass --quick or --tasklist <file>")

    print(f"Running with {args.workers} local worker processes...")
    t0 = time.time()

    fieldnames = ["seed", "n_producers", "n_consumers", "N", "k", "criterion",
                  "alpha", "n_episodes", "welfare", "jain_fairness",
                  "ir_fallback_rate", "opt_out_rate", "wall_clock_seconds"]

    with open(args.out, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        # imap_unordered streams results back as soon as each finishes, so
        # progress and partial output are visible immediately on long runs.
        with mp.Pool(processes=args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_one, tasks), start=1):
                writer.writerow(result)
                f_out.flush()
                if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                    elapsed = time.time() - t0
                    print(f"  {i}/{len(tasks)} tasks done ({elapsed:.1f}s elapsed)")

    print(f"Done. Wrote {len(tasks)} rows to {args.out} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
