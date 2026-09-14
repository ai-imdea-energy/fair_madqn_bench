"""
fair_madqn_ledger.py
=====================

Ledger-based cumulative individual-rationality (IR) layer on top of the
base CBL (bargaining.py): instead of requiring each agent's gain to be >= 0
in EVERY single hour (the naive per-epoch IR guarantee, which can be
unsatisfiable a large fraction of the time in small communities), this
module relaxes the guarantee to a CUMULATIVE one:

    - each agent has a running ledger balance (its cumulative, decayed
      surplus over the outside option),
    - an agent may accept a below-outside-option hour as long as its
      ledger is still in credit ("ledger-adjusted disagreement point"),
    - an agent whose ledger is exhausted may voluntarily opt out of a bad
      hour instead of forcing a fallback for the whole community,
    - two anti-abuse rules keep this from being gamed: the ledger decays
      over time (so old surpluses can't justify indefinitely bad treatment
      later), and opt-outs are capped at a rolling-window rate (so an agent
      can't cherry-pick only good rounds).

This is layered on top of bargaining.py's existing per-hour IR check
WITHOUT modifying bargaining.py itself: this module only swaps which
feasibility mask the three IR-constrained criteria consult.

This is the CBL mode simulate.py actually runs for the sweep -- see
`solve_cbl_with_ledger` below, which is the one entry point simulate.py
calls for every criterion.

OES IS FULLY EXCLUDED FROM THE LEDGER, BOTH READ AND WRITE. OES has no
individual-rationality restriction of any kind -- it's a baseline defined
precisely by the ABSENCE of any IR guarantee, so it has no ledger to relax
in the first place, and it never interacts with any agent's ledger.

Only dependency: NumPy + the Python standard library (collections.deque).
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import List

import bargaining  # reused, unmodified


class AgentLedger:
    """
    Per-agent running ledger of cumulative surplus over the outside option.

    balance          : cumulative (decayed) surplus over the walk-away point
                        (the grid outside option -- already netted into
                        market.py's `gain` tensors, so "walk-away point" is
                        simply gain == 0 here). Positive balance = agent has
                        been doing better than walking away, on average
                        recently; negative = has been doing worse.
    decay            : ANTI-ABUSE RULE #1. Balance decays geometrically each
                        hour (balance <- decay*balance + gain) rather than
                        accumulating forever. This bounds how far in the
                        past a "banked" surplus can still justify tolerating
                        a bad hour today -- an all-time, undecayed ledger
                        would let an agent coast indefinitely on a single
                        lucky early streak.
    max_optout_rate  : ANTI-ABUSE RULE #2. An agent may only opt out of at
                        most this FRACTION of hours within a rolling window
                        (`window` hours). Without this cap, an agent could
                        cherry-pick good rounds only -- participating
                        whenever a round favors it and sitting out every
                        unfavorable one, which would defeat the entire
                        point of pooling risk across time.
    """

    def __init__(self, decay: float = 0.98, max_optout_rate: float = 0.2,
                 window: int = 24 * 7):
        self.balance = 0.0
        self.decay = decay
        self.max_optout_rate = max_optout_rate
        self.optout_history: deque = deque(maxlen=window)

    def floor(self) -> float:
        """
        Ledger-adjusted disagreement point. Instead of demanding
        gain_i >= 0 THIS hour (bargaining.py's per-epoch rule), demand
        gain_i >= -balance: an agent in credit can absorb a
        worse-than-outside-option hour; an agent in deficit needs a
        BETTER-than-outside-option hour before the mechanism will accept
        a trade on its behalf again.
        """
        return -self.balance

    def can_opt_out(self) -> bool:
        """Anti-abuse rule #2 check: is this agent still under its
        rolling opt-out budget?"""
        if not self.optout_history:
            return True
        rate = sum(self.optout_history) / len(self.optout_history)
        return rate < self.max_optout_rate

    def record(self, gain: float, opted_out: bool) -> None:
        """Update the ledger after one hour. Opting out still decays the
        balance (it's a shelter from a bad round, not a way to freeze one's
        standing indefinitely)."""
        self.optout_history.append(1 if opted_out else 0)
        self.balance = self.decay * self.balance + (0.0 if opted_out else gain)


def ledger_ir_feasible_mask(gains: List[np.ndarray], ledgers: List[AgentLedger],
                             active: List[bool]) -> np.ndarray:
    """
    Cumulative-IR analogue of bargaining._ir_feasible_mask: agent i's
    requirement becomes gain_i >= ledger_i.floor() instead of gain_i >= 0,
    and agents that have voluntarily opted out this hour (active[i]==False)
    are excluded from the check entirely -- their non-participation cannot
    be blamed for making everyone else's IR infeasible.
    """
    floors = [l.floor() for l in ledgers]
    min_slack = None
    for i, (g, f) in enumerate(zip(gains, floors)):
        if not active[i]:
            continue
        slack = g - f
        min_slack = slack if min_slack is None else np.minimum(min_slack, slack)
    if min_slack is None:
        # everyone opted out -- vacuously "feasible" (nothing left to check)
        return np.ones_like(gains[0], dtype=bool)
    return min_slack >= 0.0


def solve_cbl_with_ledger(gains: List[np.ndarray], criterion: str,
                           ledgers: List[AgentLedger], alpha: float = 2.0,
                           oes_weights: List[float] = None):
    """
    Drop-in analogue of bargaining.solve_cbl for the three IR-constrained
    criteria (nash, maximin, alpha_fairness).

    OES: fully excluded from the ledger, both read and write. Calls
    bargaining.solve_cbl (passing through `oes_weights` -- see
    bargaining.select_oes and market.Agent.oes_weight) and returns
    immediately: it never reads any agent's ledger.floor()/can_opt_out(),
    and never WRITES to any ledger either. A run using OES therefore never
    touches its agents' ledgers at all; OES stays a pure, untouched
    baseline with zero interaction with the ledger bookkeeping used by the
    other criteria.

    Utilitarian: also unconstrained (no IR filter), so it's likewise
    passed straight through to bargaining.py's own argmax logic. Unlike
    OES, though, its realized gains ARE still recorded into each agent's
    ledger, since Utilitarian is the in-sweep comparison point the other
    three refinements are measured against.

    For nash / maximin / alpha_fairness:
    Step 1 (voluntary opt-out): for each agent, check whether EVERY
    candidate in the full grid would violate its ledger floor even when
    every other agent behaves optimally for it (i.e. no relief is possible
    this hour); if so AND the agent still has opt-out budget left
    (`can_opt_out()`), mark it inactive for this hour.
    Step 2: solve using ledger_ir_feasible_mask restricted to the ACTIVE
    agents only, falling back to utilitarian (same fallback convention as
    bargaining.py) if even that is empty.
    Step 3: update every agent's ledger with the realized outcome.

    Returns (winning_idx, fallback_flag, opted_out_mask). For OES,
    opted_out_mask is always all-False (opt-out is a ledger concept and
    OES has no ledger interaction at all).
    """
    if criterion == "oes":
        idx, fb = bargaining.solve_cbl(gains, "oes", oes_weights=oes_weights)
        return idx, fb, [False] * len(ledgers)

    if criterion == "utilitarian":
        idx, fb = bargaining.solve_cbl(gains, "utilitarian", alpha=alpha)
        for i, ledger in enumerate(ledgers):
            ledger.record(float(gains[i][idx]), opted_out=False)
        return idx, fb, [False] * len(ledgers)

    n = len(ledgers)
    active = [True] * n
    for i in range(n):
        best_possible = float(gains[i].max())  # best this agent could get under ANY joint action
        if best_possible < ledgers[i].floor() and ledgers[i].can_opt_out():
            active[i] = False

    feasible = ledger_ir_feasible_mask(gains, ledgers, active)
    if not feasible.any():
        idx, _ = bargaining.select_utilitarian(gains)
        fallback = True
    else:
        # Re-use bargaining.py's own per-criterion objective, just swap the mask.
        if criterion == "nash":
            eps = 1e-9
            objective = sum(np.log(np.clip(g, eps, None)) for g, act in zip(gains, active) if act)
        elif criterion == "maximin":
            objective = None
            for g, act in zip(gains, active):
                if not act:
                    continue
                objective = g if objective is None else np.minimum(objective, g)
        else:  # alpha_fairness
            eps = 1e-9
            if abs(alpha - 1.0) < 1e-9:
                objective = sum(np.log(np.clip(g, eps, None)) for g, act in zip(gains, active) if act)
            else:
                power = 1.0 - alpha
                objective = sum(np.clip(g, eps, None) ** power / power for g, act in zip(gains, active) if act)
        idx = bargaining._argmax_flat(objective, mask=feasible)
        fallback = False

    for i, ledger in enumerate(ledgers):
        ledger.record(float(gains[i][idx]), opted_out=not active[i])

    return idx, fallback, [not a for a in active]


if __name__ == "__main__":
    # Small runnable demo: naive per-hour rule (bargaining.solve_cbl alone)
    # against the ledger-adjusted mode (solve_cbl_with_ledger). Should show
    # the ledger mode sharply cutting the fallback rate.
    from market import build_composition, evaluate_all_candidates

    rng = np.random.default_rng(0)
    agents = build_composition(3, 3, rng)
    n_hours = 24
    hourly = [evaluate_all_candidates(agents, k=3, hour=h) for h in range(n_hours)]

    def run(use_ledger: bool, n_episodes: int = 15):
        ledgers = [AgentLedger() for _ in agents]
        n_fallback = 0
        n_total = 0
        welfare_last = []
        for ep in range(n_episodes):
            for h in range(n_hours):
                gains, payoffs, grids = hourly[h]
                if use_ledger:
                    idx, fb, _ = solve_cbl_with_ledger(gains, "nash", ledgers)
                else:
                    idx, fb = bargaining.solve_cbl(gains, "nash")
                n_total += 1
                n_fallback += int(fb)
                if ep == n_episodes - 1:
                    welfare_last.append(float(sum(payoffs[i][idx] for i in range(len(agents)))))
        return np.mean(welfare_last), n_fallback / n_total

    w_base, fb_base = run(use_ledger=False)
    w_ledger, fb_ledger = run(use_ledger=True)
    print(f"Naive per-hour rule : welfare={w_base:.3f}  fallback_rate={fb_base:.3f}")
    print(f"Ledger-adjusted mode: welfare={w_ledger:.3f}  fallback_rate={fb_ledger:.3f}")
