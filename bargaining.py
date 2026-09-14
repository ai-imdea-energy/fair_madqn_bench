"""
bargaining.py
=============

The Cooperative Bargaining Layer (CBL): given the per-agent "gain" tensors
produced by market.evaluate_all_candidates(), pick the winning joint action
under one of 5 criteria:

    utilitarian   - maximize sum(gain_i), NO individual-rationality (IR) filter
    nash          - maximize product(gain_i), subject to IR (gain_i >= 0 all i)
    maximin       - maximize min_i(gain_i), subject to IR
    alpha_fairness- maximize sum_i phi_alpha(gain_i), subject to IR
                     phi_alpha(x) = log(x)              if alpha == 1  (-> NBS)
                     phi_alpha(x) = x**(1-alpha)/(1-alpha) otherwise
                     alpha -> infinity recovers Maximin
    oes           - Optimal Equilibrium Selection: restrict to joint actions
                    that are pure-strategy Nash equilibria (no agent can
                    unilaterally deviate to a different one of its own k
                    candidate levels and improve its own gain), THEN pick the
                    equilibrium maximizing a WEIGHTED sum of gains, where the
                    per-agent weights are exogenous/arbitrary (see
                    market.Agent.oes_weight and select_oes below). NO
                    individual-rationality guarantee of any kind.

IR-infeasibility fallback
-------------------------
The base CBL's single-epoch IR guarantee can be frequently unsatisfiable in
small, low-liquidity communities. When the IR-feasible set is empty for
nash/maximin/alpha_fairness, this module falls back to the unconstrained
utilitarian-maximizing joint action and flags `fallback=True` for that
hour, so simulate.py can track a fallback rate as an outcome metric.

Everything below is fully vectorized NumPy over the (k,)*N gain tensors --
there is no explicit Python loop over the k**N joint actions.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


def _argmax_flat(tensor: np.ndarray, mask: np.ndarray = None) -> Tuple[int, ...]:
    """Return the multi-index of the maximum of `tensor`, optionally restricted
    to entries where `mask` is True (masked-out entries treated as -inf)."""
    if mask is not None:
        work = np.where(mask, tensor, -np.inf)
    else:
        work = tensor
    flat_idx = np.argmax(work)
    return np.unravel_index(flat_idx, tensor.shape)


def _ir_feasible_mask(gains: List[np.ndarray]) -> np.ndarray:
    """Boolean tensor: True where EVERY agent's gain is >= 0 (IR satisfied)."""
    min_gain = gains[0].copy()
    for g in gains[1:]:
        min_gain = np.minimum(min_gain, g)
    return min_gain >= 0.0


def select_utilitarian(gains: List[np.ndarray]) -> Tuple[Tuple[int, ...], bool]:
    total = sum(gains)
    idx = _argmax_flat(total)
    return idx, False  # never falls back; it has no IR constraint to fail


def select_nash(gains: List[np.ndarray]) -> Tuple[Tuple[int, ...], bool]:
    feasible = _ir_feasible_mask(gains)
    if not feasible.any():
        # IR-infeasible this hour -> fall back to the utilitarian pick.
        idx, _ = select_utilitarian(gains)
        return idx, True
    # Nash product; clip at a tiny epsilon so a zero-gain agent doesn't force
    # the product to exactly zero for every candidate (ties broken by argmax
    # naturally still favor larger positive gains elsewhere).
    eps = 1e-9
    log_product = sum(np.log(np.clip(g, eps, None)) for g in gains)
    idx = _argmax_flat(log_product, mask=feasible)
    return idx, False


def select_maximin(gains: List[np.ndarray]) -> Tuple[Tuple[int, ...], bool]:
    feasible = _ir_feasible_mask(gains)
    min_gain = gains[0].copy()
    for g in gains[1:]:
        min_gain = np.minimum(min_gain, g)
    if not feasible.any():
        idx, _ = select_utilitarian(gains)
        return idx, True
    idx = _argmax_flat(min_gain, mask=feasible)
    return idx, False


def select_alpha_fairness(gains: List[np.ndarray], alpha: float = 2.0) -> Tuple[Tuple[int, ...], bool]:
    feasible = _ir_feasible_mask(gains)
    if not feasible.any():
        idx, _ = select_utilitarian(gains)
        return idx, True
    eps = 1e-9
    if abs(alpha - 1.0) < 1e-9:
        # alpha == 1 recovers the Nash Bargaining Solution (log utility).
        total = sum(np.log(np.clip(g, eps, None)) for g in gains)
    else:
        # phi_alpha(x) = x^(1-alpha) / (1-alpha)
        power = 1.0 - alpha
        total = sum(np.clip(g, eps, None) ** power / power for g in gains)
    idx = _argmax_flat(total, mask=feasible)
    return idx, False


def select_oes(gains: List[np.ndarray], weights: List[float] = None) -> Tuple[Tuple[int, ...], bool]:
    """
    Optimal Equilibrium Selection: restrict to pure-strategy Nash equilibria,
    then pick the one maximizing a WEIGHTED sum of gains, where the weights
    are exogenous and arbitrary -- "whoever carries the most weight in a sum
    someone else set wins", NOT an equal-weighted (plain utilitarian) sum
    among equilibria. `weights` should be each agent's `Agent.oes_weight`
    (market.build_composition assigns these once per community,
    independently of the agent's own economics). If omitted, defaults to
    equal weights of 1.0 each.

    Vectorized equilibrium test
    ----------------------------
    For agent i, "no unilateral profitable deviation" means: holding every
    OTHER agent's action fixed, agent i's OWN gain tensor is already maximal
    along ITS OWN axis. Since gains[i] has one axis per agent (shape (k,)*N),
    this is simply:

        gains[i] == gains[i].max(axis=i, keepdims=True)

    A joint action is a pure Nash equilibrium iff this holds simultaneously
    for every agent i. No explicit enumeration of alternative actions is
    needed -- the whole check is N array reductions.
    """
    n = len(gains)
    if weights is None:
        weights = [1.0] * n

    is_ne = np.ones_like(gains[0], dtype=bool)
    for i in range(n):
        best_along_i = gains[i].max(axis=i, keepdims=True)
        is_ne &= (gains[i] >= best_along_i - 1e-12)

    weighted_total = sum(w * g for w, g in zip(weights, gains))
    if not is_ne.any():
        # No pure-strategy equilibrium exists on this discretized grid this
        # hour (can happen, especially at low k) -> fall back to the global
        # weighted-sum-maximizing joint action, flagged distinctly from the
        # IR fallback used by the other criteria.
        idx = _argmax_flat(weighted_total)
        return idx, True
    idx = _argmax_flat(weighted_total, mask=is_ne)
    return idx, False


CRITERIA = {
    "utilitarian": select_utilitarian,
    "nash": select_nash,
    "maximin": select_maximin,
    "alpha_fairness": select_alpha_fairness,
    "oes": select_oes,
}


def solve_cbl(gains: List[np.ndarray], criterion: str, alpha: float = 2.0,
              oes_weights: List[float] = None):
    """
    Dispatch to the requested criterion. Returns (winning_index, fallback_flag).
    winning_index is a tuple of per-agent level indices into the k-sized grid.

    oes_weights is only consulted when criterion == "oes" -- see
    select_oes's docstring; it should be each agent's Agent.oes_weight.
    """
    if criterion == "alpha_fairness":
        return select_alpha_fairness(gains, alpha=alpha)
    if criterion == "oes":
        return select_oes(gains, weights=oes_weights)
    fn = CRITERIA[criterion]
    return fn(gains)
