# Fair-MADQN Bench

A microgrid multi-agent trading simulation and parallel sweep harness:
compares five joint-action selection mechanisms (Utilitarian, Nash
Bargaining, Maximin, α-fairness, OES) for a discrete cooperative-bargaining
layer (CBL) coordinating producer/consumer agents in a small energy
community, across every valid community composition, a range of
candidate-grid resolutions, multiple seeds, and a full episodic training
run per configuration.

**Only dependency: NumPy.** Everything else is Python standard library
(`multiprocessing`, `json`, `csv`, `argparse`).

## What this is

A self-consistent, fully-vectorized reference model:

- convex producer cost / concave consumer utility,
- a market-clearing price,
- an outside option (trading with the grid instead of the community) that
  grounds individual-rationality (IR) checks,
- IR infeasibility that varies hour-to-hour (a solar/demand availability
  profile makes some hours well-matched, others not),
- a ledger-based cumulative-IR mode (per-agent ledger, voluntary opt-out,
  ledger-adjusted disagreement point, two anti-abuse rules) layered on top
  of the naive per-hour IR rule,
- an OES equilibrium filter with a genuinely weighted (not equal-weighted)
  tie-break using exogenous per-agent weights, and no IR guarantee at all.

Two things worth knowing if you're calibrating this against other data:

- `market.GRID_SELL_PRICE` / `market.GRID_BUY_PRICE` (the feed-in/retail
  tariff spread) and `market.hourly_availability()` (the solar/demand
  day-profile) are the two knobs that control how often IR is infeasible.
  Tightening the tariff spread makes conflicts more common.
- This reference implementation is fully NumPy-vectorized per hour (no
  Python loop over the k^N joint actions), so it runs considerably faster
  than a naive per-combination implementation would. That's fine for
  validating experiment logic and the parallel infrastructure.

## Files

- `market.py` — agents, double-auction clearing, hourly availability
  profile, fully vectorized payoff/IR-gain tensors.
- `bargaining.py` — the 5 criteria + IR filter + vectorized Nash-equilibrium
  filter for OES.
- `fair_madqn_ledger.py` — the ledger-based cumulative-IR CBL variant
  (per-agent ledger, voluntary opt-out, ledger-adjusted disagreement point,
  two anti-abuse rules). This is the CBL mode the main sweep actually runs
  under — `simulate.py` calls `solve_cbl_with_ledger` from this module for
  all criteria (OES is fully excluded from ledger interaction — see that
  file's docstring).
- `rl_agents.py` — tabular Q-learning agent (per-agent, per-hour state).
- `simulate.py` — `run_single_task()`: one (seed, composition, k, criterion)
  → metrics dict. This is the one function both parallel runners call.
- `sweep.py` — generates the exhaustive task list (all valid compositions
  x k range x criteria x seeds).
- `run_local.py` — desktop parallel runner (`multiprocessing.Pool`).
- `run_hpc_task.py` — one cluster-array-task (or GREASY line) entry point.
- `make_greasy_tasklist.py` — converts the JSON-Lines task list into a
  GREASY-compatible plain-text command list. This is the primary,
  ready-to-submit HPC path — see "Scale to a cluster" below.
- `submit_hpc_greasy.sh` — SLURM job script that runs the whole sweep via
  GREASY.
- `submit_hpc_array.sh` — alternative plain SLURM array-job script, kept in
  case your queue policy prefers that over GREASY.
- `aggregate_results.py` — merges per-task result files into one CSV.

## 1. Validate on your desktop first

```bash
# Sanity-check a single run (few seconds):
python simulate.py

# Generate task lists:
python sweep.py
#   -> writes tasklist_full.jsonl and tasklist_quick.jsonl

# Run the tiny built-in smoke test across all your local cores:
python run_local.py --quick

# Run a slice of the REAL task list locally, e.g. to check how runtime
# scales with N/k before requesting cluster time:
python run_local.py --tasklist tasklist_full.jsonl --limit 200 --workers 8
```

Results stream into a CSV as they complete (safe to Ctrl-C and inspect
partial output).

## 2. Scale to a cluster

Two ways to submit, same underlying `run_hpc_task.py` either way:

### Option A — GREASY (recommended)

Given a highly skewed task-size profile (short median task, much longer
worst-case task), running thousands of separate array elements spends a
real share of walltime on per-element scheduling overhead. GREASY instead
runs the whole task list inside ONE job allocation and schedules the
individual (short, single-core) tasks across that allocation's cores
itself.

```bash
python sweep.py                                             # -> tasklist_full.jsonl
python make_greasy_tasklist.py --tasklist tasklist_full.jsonl \
    --out greasy_tasklist.txt                                # one command per line

# Edit submit_hpc_greasy.sh: fill in --account/--partition, and set
# --ntasks to your desired parallel width.
sbatch submit_hpc_greasy.sh
```

Consider splitting `greasy_tasklist.txt` into a short-task file (small
configurations) and a long-task file (largest configurations), and
submitting each as its own GREASY job with an appropriately sized `--time`,
rather than reserving the worst-case runtime for every worker.

### Option B — plain SLURM array job

```bash
python sweep.py

# Edit submit_hpc_array.sh: fill in --account/--partition, and set
#   --array=0-$(( $(wc -l < tasklist_full.jsonl) - 1 ))
sbatch submit_hpc_array.sh
```

This is the more "standard" SLURM mechanism and is fine at smaller task
counts, or if your queue policy favors array jobs — but expect more
scheduling overhead than GREASY at large scale.

### Either way

```bash
# After the job(s) complete (or periodically, to check progress):
python aggregate_results.py --results-dir results --tasklist tasklist_full.jsonl --out results_full.csv
```

## Extending this

- Swap the toy economic model in `market.py` for a different payoff model
  — `bargaining.py`, `simulate.py`, and both runners are agnostic to how
  `gains`/`payoffs` tensors are produced, only that they have shape `(k,)*N`.
- To sweep multiple α values instead of one fixed default, pass a list of
  alphas into `sweep.generate_full_tasklist(alpha=...)` in a small loop, or
  extend the task dict with several `alpha` entries.
- The naive per-hour IR rule (`bargaining.solve_cbl`) is still available if
  you want to reproduce that baseline behavior for comparison — call it
  directly instead of `fair_madqn_ledger.solve_cbl_with_ledger` in a copy
  of `simulate.py`.
- Ledger parameters (`ledger_decay`, `ledger_max_optout_rate`,
  `ledger_window_hours`) are exposed as keyword arguments on
  `simulate.run_single_task` if you want to sweep those too.
