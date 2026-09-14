"""
aggregate_results.py
=====================

Merge the many small per-task result files produced by a cluster batch job
(one JSON file per task, from run_hpc_task.py) into a single tidy CSV ready
for analysis -- mirroring what run_local.py already writes directly.

Also prints a couple of useful sanity checks:
    - how many tasks are missing (helps spot failed/timed-out tasks)
    - the mean vs. largest task wall-clock time (useful because task
      duration in this sweep is highly skewed -- a small number of large
      configurations dominate the tail, so the arithmetic mean is not a
      good estimate of a "typical" task's runtime)

Usage:
    python aggregate_results.py --results-dir results --tasklist tasklist_full.jsonl --out results_full.csv
"""

from __future__ import annotations
import argparse
import csv
import glob
import json
import os

import sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--tasklist", type=str, default="tasklist_full.jsonl",
                         help="Used only to report how many tasks are still missing.")
    parser.add_argument("--out", type=str, default="results_full.csv")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "task_*.json")))
    if not files:
        print(f"No result files found in {args.results_dir}/")
        return

    rows = []
    for path in files:
        with open(path) as f:
            rows.append(json.load(f))

    fieldnames = ["task_index", "seed", "n_producers", "n_consumers", "N", "k",
                  "criterion", "alpha", "n_episodes", "welfare", "jain_fairness",
                  "ir_fallback_rate", "opt_out_rate", "wall_clock_seconds"]

    with open(args.out, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r.get("task_index", 0)):
            writer.writerow({k_: row.get(k_) for k_ in fieldnames})

    print(f"Merged {len(rows)} task results -> {args.out}")

    if os.path.exists(args.tasklist):
        expected = sweep.count_tasks(args.tasklist)
        missing = expected - len(rows)
        print(f"Expected {expected} tasks; {missing} missing "
              f"(re-submit those task indices if > 0).")

    wall_clocks = [r["wall_clock_seconds"] for r in rows if r.get("wall_clock_seconds") is not None]
    if wall_clocks:
        mean_s = sum(wall_clocks) / len(wall_clocks)
        max_s = max(wall_clocks)
        print(f"Wall-clock time: mean={mean_s:.2f}s, largest single task={max_s:.2f}s "
              f"(the mean is NOT representative of a typical task, due to the "
              f"skewed task-size profile across N and k).")


if __name__ == "__main__":
    main()
