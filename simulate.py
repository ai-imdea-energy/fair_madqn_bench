"""
simulate.py
===========

The atomic unit of work in the sweep: run ONE (seed, composition, k,
criterion) combination for `n_episodes` x 24 hourly CBL decisions, and
return a summary metrics dict.

CBL MODE
--------
All five criteria (Utilitarian, NBS, Maximin, alpha-fairness, OES) are run
through `fair_madqn_ledger.solve_cbl_with_ledger`, not
`bargaining.solve_cbl` directly. That function's own docstring has the
full mechanics; in short: the three IR-constrained criteria (nash, maximin,
alpha_fairness) use a ledger-adjusted, opt-out-aware feasibility check
instead of the naive per-hour IR rule, while utilitarian passes through
unaffected but is still ledger-tracked, and OES is fully excluded from all
ledger interaction (no IR guarantee to begin with).

bargaining.py itself is unmodified -- it's still what fair_madqn_ledger.py
calls through to for OES/Utilitarian and for the underlying per-criterion
objective math of nash/maximin/alpha_fairness. It also remains directly
usable on its own if you want to reproduce the naive per-hour baseline for
comparison.

Metrics recorded per task
--------------------------
    - welfare            : mean total in-community payoff per hour,
                            averaged over the final training episode
                            (opted-out agents contribute/receive 0 that
                            hour -- see below)
    - jain_fairness       : Jain's fairness index of the payoff allocation
    - ir_fallback_rate    : fraction of hourly epochs where even the ledger-
                            adjusted, opt-out-aware IR-feasible set was
                            empty (nash/maximin/alpha_fairness); always 0
                            for utilitarian (no IR filter to fail); for OES
                            it instead reflects the separate "no
                            pure-strategy equilibrium found" fallback from
                            bargaining.py, since OES has no ledger
                            interaction at all
    - opt_out_rate        : fraction of agent-hours where an agent
                            voluntarily sat out rather than accept a
                            below-floor trade (nash/maximin/alpha_fairness
                            only). Always 0 for utilitarian and OES,
                            neither of which consult any agent's ledger.
    - wall_clock_seconds  : actual measured wall-clock time for this task

This function is called identically by run_local.py (desktop
multiprocessing) and run_hpc_task.py (one array/GREASY task) -- the
parallelization layer never touches simulation internals, only
orchestrates many independent calls to `run_single_task`.
"""

from __future__ import annotations
import time
import numpy as np

from market import build_composition, evaluate_all_candidates
from fair_madqn_ledger import AgentLedger, solve_cbl_with_ledger
from rl_agents import TabularQAgent


def jains_fairness_index(x: np.ndarray) -> float:
    """
    Jain's fairness index: J(x) = (sum x_i)^2 / (n * sum x_i^2), in [1/n, 1].
    Classically defined for non-negative allocations; negative payoffs are
    clipped to 0 purely for the purpose of this index (documented choice --
    does not affect welfare or any bargaining computation elsewhere).
    """
    x = np.clip(x, 0.0, None)
    n = len(x)
    s1 = np.sum(x)
    s2 = np.sum(x ** 2)
    if s2 == 0:
        return 1.0  # everyone got exactly zero -> perfectly (trivially) equal
    return float((s1 ** 2) / (n * s2))


def run_single_task(n_producers: int,
                     n_consumers: int,
                     k: int,
                     criterion: str,
                     seed: int,
                     n_episodes: int = 300,
                     n_hours: int = 24,
                     alpha: float = 2.0,
                     ledger_decay: float = 0.98,
                     ledger_max_optout_rate: float = 0.2,
                     ledger_window_hours: int = 24 * 7) -> dict:
    """
    Run the full episodic training loop for one configuration and return a
    flat dict of results -- deliberately flat/JSON-serializable so it can be
    written straight to disk by run_hpc_task.py / run_local.py and later
    concatenated by aggregate_results.py.

    ledger_decay / ledger_max_optout_rate / ledger_window_hours configure
    the cumulative-IR ledger (see fair_madqn_ledger.AgentLedger); defaults
    match that module's own defaults. Each agent gets its own ledger,
    freshly initialized (balance=0) at the start of the run and carried
    CONTINUOUSLY across all `n_episodes` -- episodes here represent
    repeated days under identical conditions (used to let the tabular
    Q-learner converge, see rl_agents.py), and the ledger's cumulative
    accounting is meant to track standing across exactly that kind of
    repeated, ongoing participation, not reset every "day".
    """
    t_start = time.perf_counter()

    rng = np.random.default_rng(seed)
    agents = build_composition(n_producers, n_consumers, rng)
    n = len(agents)

    # Each agent's OES tie-break weight was assigned once, exogenously,
    # when the community was built -- see market.build_composition and
    # Agent.oes_weight. Extracted here once per run and passed through to
    # every OES decision this run makes.
    oes_weights = [a.oes_weight for a in agents]

    # Precompute ONE candidate-action tensor set per hour of the day (24
    # total), reflecting that hour's solar-availability / demand profile
    # (market.hourly_availability). Agent parameters are fixed for the whole
    # run, so each hour's tensors only need to be built once and are then
    # reused identically across all `n_episodes` repetitions of the same day
    # -- this is what makes IR-feasibility a genuine per-hour statistic
    # (some hours well-matched supply/demand, others not) rather than a
    # single fixed fact about the whole run.
    hourly_tensors = [evaluate_all_candidates(agents, k, hour=h) for h in range(n_hours)]

    q_agents = [TabularQAgent(k=k, n_hours=n_hours, n_episodes=n_episodes,
                               rng=np.random.default_rng(seed * 1000 + i))
                for i in range(n)]

    # One persistent ledger per agent for the cumulative IR guarantee -- see
    # fair_madqn_ledger.AgentLedger.
    ledgers = [AgentLedger(decay=ledger_decay,
                            max_optout_rate=ledger_max_optout_rate,
                            window=ledger_window_hours)
               for _ in range(n)]

    n_fallback_epochs = 0
    n_total_epochs = 0
    n_agent_hours = 0
    n_opt_outs = 0
    last_episode_welfare = []
    last_episode_payoffs = np.zeros(n)

    for episode in range(n_episodes):
        for hour in range(n_hours):
            gains, payoffs, grids = hourly_tensors[hour]

            # CBL decision for this hour. See fair_madqn_ledger.
            # solve_cbl_with_ledger for the full logic; utilitarian passes
            # through unaffected but still ledger-tracked; OES passes
            # through with its own arbitrary per-agent weights
            # (oes_weights) and NO ledger interaction at all.
            winning_idx, fallback, opted_out = solve_cbl_with_ledger(
                gains, criterion, ledgers, alpha=alpha, oes_weights=oes_weights)

            n_total_epochs += 1
            n_agent_hours += n
            n_opt_outs += sum(opted_out)
            if fallback:
                n_fallback_epochs += 1

            # An agent that opted out this hour truly sits out: it neither
            # trades nor realizes any gain/payoff, REGARDLESS of what
            # quantity the winning joint action happened to assign along
            # its axis (that axis value is only meaningful for agents that
            # are actually participating this hour).
            realized_payoffs = np.array([
                0.0 if opted_out[i] else float(payoffs[i][winning_idx])
                for i in range(n)
            ])
            realized_gains = np.array([
                0.0 if opted_out[i] else float(gains[i][winning_idx])
                for i in range(n)
            ])

            # Update each agent's tabular Q-function using its own realized
            # action-level and gain this hour (see rl_agents.py docstring).
            next_hour = (hour + 1) % n_hours
            for i in range(n):
                action_idx = winning_idx[i]
                reward = float(realized_gains[i])
                # choose_action() is invoked purely so the epsilon schedule /
                # Q-table gets exercised in a way comparable to a live policy;
                # the actual traded quantity this hour is fixed by the CBL.
                q_agents[i].choose_action(hour, episode)
                q_agents[i].update(hour, action_idx, reward, next_hour)

            if episode == n_episodes - 1:
                last_episode_welfare.append(float(np.sum(realized_payoffs)))
                last_episode_payoffs += realized_payoffs

    if n_episodes > 0:
        last_episode_payoffs /= n_hours  # average per-hour payoff, final episode

    welfare = float(np.mean(last_episode_welfare)) if last_episode_welfare else 0.0
    jain = jains_fairness_index(last_episode_payoffs)
    fallback_rate = n_fallback_epochs / max(n_total_epochs, 1)
    opt_out_rate = n_opt_outs / max(n_agent_hours, 1)

    wall_clock = time.perf_counter() - t_start

    return {
        "seed": seed,
        "n_producers": n_producers,
        "n_consumers": n_consumers,
        "N": n_producers + n_consumers,
        "k": k,
        "criterion": criterion,
        "alpha": alpha if criterion == "alpha_fairness" else None,
        "n_episodes": n_episodes,
        "welfare": welfare,
        "jain_fairness": jain,
        "ir_fallback_rate": fallback_rate,
        "opt_out_rate": opt_out_rate,
        "wall_clock_seconds": wall_clock,
    }


if __name__ == "__main__":
    # Quick manual sanity check (a handful of seconds on a laptop):
    #   python simulate.py
    result = run_single_task(n_producers=1, n_consumers=1, k=3, criterion="nash",
                              seed=0, n_episodes=20)
    print(result)
