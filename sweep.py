"""
sweep.py
========

Generates the exhaustive task list for the full compute campaign:

    - N = 2..6 agents
    - every valid producer/consumer composition per N (n_producers = 1..N-1,
      n_consumers = N-n_producers) -> 1,2,3,4,5 compositions for N=2..6,
      15 in total.
    - k (candidate-grid resolution) = 2..9 for N<=4, capped at 8 for N=5 and
      6 for N=6, to bound the largest single task's runtime.
    - 5 criteria: utilitarian, nash, maximin, alpha_fairness, oes
    - 25 seeds per configuration

Each element of the returned task list is a small, flat, JSON-serializable
dict -- exactly the arguments `simulate.run_single_task` needs. Both
run_local.py (desktop) and run_hpc_task.py (SLURM array / GREASY task)
consume this SAME list, indexed by position, so "task #17321 on the
cluster" and "task #17321 in a local multiprocessing pool" are guaranteed
to be the identical simulation.

Only dependency: the Python standard library (json for serialization).
"""

from __future__ import annotations
import json
from typing import List, Dict, Any

N_MIN, N_MAX = 2, 6
SEEDS = list(range(25))          # 25 random seeds per configuration
CRITERIA = ["utilitarian", "nash", "maximin", "alpha_fairness", "oes"]
ALPHA_DEFAULT = 2.0              # fixed alpha for alpha_fairness in this sweep;
                                  # trivially extendable to a list of alphas if
                                  # you want alpha itself swept as well.
N_EPISODES_FULL = 300            # training episodes per run


def k_range_for_N(N: int) -> List[int]:
    """Capped-k schedule: full 2..9 for N<=4, tapering to 2..8 at N=5 and
    2..6 at N=6 to bound the largest single task's runtime."""
    if N <= 4:
        return list(range(2, 10))   # 2..9 inclusive
    elif N == 5:
        return list(range(2, 9))    # 2..8 inclusive
    else:  # N == 6
        return list(range(2, 7))    # 2..6 inclusive


def compositions_for_N(N: int):
    """Every valid (n_producers, n_consumers) split with both roles present."""
    return [(p, N - p) for p in range(1, N)]  # p = 1 .. N-1


def generate_full_tasklist(n_episodes: int = N_EPISODES_FULL,
                            alpha: float = ALPHA_DEFAULT,
                            seeds: List[int] = None,
                            criteria: List[str] = None) -> List[Dict[str, Any]]:
    """
    Build the full task list (defaults produce the full campaign). Pass
    smaller `seeds`/`criteria`/fewer episodes to build a cheap smoke-test
    list for local validation -- see run_local.py --quick.
    """
    seeds = SEEDS if seeds is None else seeds
    criteria = CRITERIA if criteria is None else criteria

    tasks: List[Dict[str, Any]] = []
    for N in range(N_MIN, N_MAX + 1):
        for (n_prod, n_cons) in compositions_for_N(N):
            for k in k_range_for_N(N):
                for criterion in criteria:
                    for seed in seeds:
                        tasks.append({
                            "n_producers": n_prod,
                            "n_consumers": n_cons,
                            "k": k,
                            "criterion": criterion,
                            "seed": seed,
                            "n_episodes": n_episodes,
                            "alpha": alpha,
                        })
    return tasks


def save_tasklist(tasks: List[Dict[str, Any]], path: str) -> None:
    """Persist the task list as JSON Lines: one task per line. This format
    lets run_hpc_task.py seek to task #i without parsing the whole file."""
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")


def load_task(path: str, index: int) -> Dict[str, Any]:
    """Read exactly task #index from a JSON-Lines task list (0-based),
    without loading the entire file into memory -- important once the list
    has thousands of entries."""
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Task index {index} out of range in {path}")


def count_tasks(path: str) -> int:
    with open(path, "r") as f:
        return sum(1 for _ in f)


if __name__ == "__main__":
    full_tasks = generate_full_tasklist()
    print(f"Full campaign: {len(full_tasks)} independent tasks.")

    save_tasklist(full_tasks, "tasklist_full.jsonl")
    print("Saved -> tasklist_full.jsonl")

    # A tiny local smoke-test list: the FIRST valid composition of each N,
    # k in {2,3}, 2 seeds, all 5 criteria, short episodes -- runs in seconds
    # on a laptop. Useful to validate correctness before scaling up.
    filtered: List[Dict[str, Any]] = []
    for N in range(N_MIN, N_MAX + 1):
        n_prod, n_cons = compositions_for_N(N)[0]   # first composition only
        for k in (2, 3):
            for criterion in CRITERIA:
                for seed in (0, 1):
                    filtered.append({
                        "n_producers": n_prod,
                        "n_consumers": n_cons,
                        "k": k,
                        "criterion": criterion,
                        "seed": seed,
                        "n_episodes": 20,
                        "alpha": ALPHA_DEFAULT,
                    })
    save_tasklist(filtered, "tasklist_quick.jsonl")
    print(f"Quick smoke-test list: {len(filtered)} tasks -> tasklist_quick.jsonl")
