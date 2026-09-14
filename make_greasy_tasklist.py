"""
make_greasy_tasklist.py
========================

GREASY is a task-farming tool: you give it a plain text file where each
LINE is one shell command, and it schedules/runs as many of them as fit in
parallel across whatever cores the enclosing job was granted, moving on to
the next queued line as soon as a slot frees up.

This is a better fit for a large sweep than one cluster-array element per
task when task duration is highly skewed (a short median task against a
much longer worst-case task): running everything inside ONE job allocation
and letting GREASY handle the fine-grained scheduling avoids the
per-element scheduling overhead a plain array job would incur at that
scale.

This script turns our existing JSON-Lines task list (sweep.py) into the
one-command-per-line text file GREASY expects. Nothing about simulate.py /
run_hpc_task.py changes -- GREASY simply calls
`run_hpc_task.py --task-index i` once per line, exactly like a single
array task would.

Usage:
    python sweep.py                       # writes tasklist_full.jsonl
    python make_greasy_tasklist.py \
        --tasklist tasklist_full.jsonl \
        --out greasy_tasklist.txt
    # then see submit_hpc_greasy.sh
"""

from __future__ import annotations
import argparse
import sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasklist", type=str, default="tasklist_full.jsonl",
                         help="JSON-Lines task list produced by sweep.py.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out", type=str, default="greasy_tasklist.txt")
    args = parser.parse_args()

    n_tasks = sweep.count_tasks(args.tasklist)

    with open(args.out, "w") as f:
        for i in range(n_tasks):
            # Each line = one fully independent, single-core invocation.
            # GREASY runs this exact command line, one per available core
            # slot, until the whole file is consumed.
            f.write(
                f"python3 run_hpc_task.py --tasklist {args.tasklist} "
                f"--task-index {i} --results-dir {args.results_dir}\n"
            )

    print(f"Wrote {n_tasks} GREASY task lines -> {args.out}")
    print("Next: submit submit_hpc_greasy.sh (edit resource settings first)")


if __name__ == "__main__":
    main()
