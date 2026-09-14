"""
market.py
=========

Reference implementation of a microgrid double-auction market:

    - Producer agents have an INCREASING marginal-cost function.
    - Consumer agents have a DECREASING marginal-utility function.
    - A market-clearing price + traded quantities determine each agent's payoff.
    - Each agent also has an "outside option": trading directly with the main
      grid instead of the community (this is the disagreement point used by
      every individual-rationality / bargaining criterion in bargaining.py).

This module implements a simple, transparent, self-consistent toy model
(quadratic cost/utility, uniform clearing price) with:

    * producers/consumers with convex cost / concave utility,
    * a single clearing price,
    * an outside option (grid feed-in tariff / grid retail price),
    * genuine, hour-varying individual-rationality (IR) infeasibility.

Everything here is written so the payoff model can be swapped out later
without touching bargaining.py, simulate.py, sweep.py, or the parallel
runners -- those only depend on the public functions below.

Only dependency: NumPy.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Grid economics (fixed reference tariffs -- the agents' "outside option")
# ---------------------------------------------------------------------------
# These represent what an agent would get/pay if it skipped the community
# market entirely and traded directly with the main grid. A wide spread
# (feed-in << retail) is realistic and is exactly what makes local trading
# valuable in the first place.
GRID_SELL_PRICE = 0.05   # EUR/kWh a producer gets by selling straight to the grid
GRID_BUY_PRICE = 0.25    # EUR/kWh a consumer pays by buying straight from the grid


@dataclass
class Agent:
    """
    One market participant (producer or consumer).

    role         : "producer" or "consumer"
    intercept    : c0 (producer marginal cost at q=0) or u0 (consumer marginal
                   utility at q=0)
    slope        : c1 (producer) or u1 (consumer) -- controls how fast the
                   marginal cost rises / marginal utility falls with quantity
    q_max        : maximum quantity (kWh) this agent can trade in one hour
    oes_weight   : this agent's weight in OES's weighted-sum tie-break among
                   equilibria. Genuinely arbitrary -- i.e. NOT proportional
                   to anything the agent controls (not its q_max, cost/
                   utility parameters, or role) -- drawn once per community
                   alongside the other parameters and then fixed for the
                   whole run, as if an external operator assigned priority
                   weights ahead of time.
    """
    role: str
    intercept: float
    slope: float
    q_max: float
    oes_weight: float = 1.0

    def cost_or_utility(self, q: np.ndarray) -> np.ndarray:
        """Total cost (producer) or total utility (consumer) at quantity q."""
        if self.role == "producer":
            # C(q) = c0*q + 0.5*c1*q^2  -> marginal cost c0 + c1*q (increasing)
            return self.intercept * q + 0.5 * self.slope * q ** 2
        else:
            # U(q) = u0*q - 0.5*u1*q^2  -> marginal utility u0 - u1*q (decreasing)
            return self.intercept * q - 0.5 * self.slope * q ** 2


def build_composition(n_producers: int, n_consumers: int, rng: np.random.Generator) -> List[Agent]:
    """
    Instantiate a random-but-reproducible community of agents.

    A "composition" here is a (n_producers, n_consumers) pair at a given
    community size N = n_producers + n_consumers. Valid compositions
    require BOTH roles present -- a pure-producer or pure-consumer
    "community" has nothing to trade internally -- so for a given N,
    n_producers ranges from 1 to N-1.
    """
    agents: List[Agent] = []
    for _ in range(n_producers):
        # e.g. solar/wind/dispatchable generation: cheap-ish, mildly convex cost
        c0 = rng.uniform(0.03, 0.12)
        c1 = rng.uniform(0.005, 0.03)
        qmax = rng.uniform(2.0, 6.0)
        # OES weight drawn independently of this agent's own economics --
        # arbitrary, not something the agent earns or influences.
        w = rng.uniform(0.5, 2.0)
        agents.append(Agent("producer", c0, c1, qmax, oes_weight=w))
    for _ in range(n_consumers):
        # e.g. residential/commercial/industrial load: values energy highly at
        # low quantities, saturates at higher quantities
        u0 = rng.uniform(0.15, 0.35)
        u1 = rng.uniform(0.01, 0.04)
        qmax = rng.uniform(2.0, 6.0)
        w = rng.uniform(0.5, 2.0)
        agents.append(Agent("consumer", u0, u1, qmax, oes_weight=w))
    return agents


def hourly_availability(role: str, hour: int) -> float:
    """
    Fraction (0-1] of an agent's q_max actually available at a given hour of
    day, representing the obvious real-world pattern that solar/production
    availability and consumption demand both vary hour-to-hour rather than
    being constant all day.

    Producers: bell-shaped curve peaking at midday (solar-like), near-zero
    overnight. Consumers: two-peak "morning + evening" residential/
    commercial demand curve. This is what makes IR-feasibility a genuine
    PER-HOUR statistic instead of a single yes/no fact about a whole 24h
    run: at some hours supply and demand are well matched (many candidates
    IR-feasible), at others one side is scarce and conflicts are common.

    Purely a reference profile -- tune freely, or replace with real
    generation/load traces if available.
    """
    if role == "producer":
        # Bell curve centered at hour 13, ~zero before 6 and after 20.
        return max(0.0, np.exp(-((hour - 13) ** 2) / (2 * 4.0 ** 2)))
    else:
        morning = np.exp(-((hour - 8) ** 2) / (2 * 2.5 ** 2))
        evening = np.exp(-((hour - 20) ** 2) / (2 * 2.5 ** 2))
        return float(max(0.15, morning, evening))  # small always-on baseload


def build_candidate_grids(agents: List[Agent], k: int, hour: int = None) -> List[np.ndarray]:
    """
    Discretize each agent's action space into k evenly spaced quantity
    levels. `k` is the CBL candidate-grid resolution.

    NOTE: the levels deliberately start at q_max/k rather than at 0.
    Including an exact "propose zero" level would make the
    all-agents-propose-zero joint action trivially individual-rational for
    EVERY criterion (gain = 0 >= 0 for everyone, always), which would mean
    the IR-feasible set could never actually be empty and the IR-fallback
    mechanism below could never trigger. Requiring every agent to propose a
    strictly positive quantity is both more realistic (an agent that shows
    up to an hourly market decision is assumed to want to trade something)
    and is what lets genuine IR infeasibility -- and hence the
    fallback-rate metric -- actually occur and be measured.
    """
    if hour is None:
        return [np.linspace(a.q_max / k, a.q_max, k) for a in agents]
    scaled = []
    for a in agents:
        avail = hourly_availability(a.role, hour) * a.q_max
        avail = max(avail, a.q_max * 1e-3)  # keep a tiny nonzero floor, avoid degenerate 0-width grid
        scaled.append(np.linspace(avail / k, avail, k))
    return scaled


def evaluate_all_candidates(agents: List[Agent], k: int, hour: int = None):
    """
    Build the FULL joint candidate-action tensor (shape (k,)*N, one axis per
    agent) and vectorically compute, for every joint action simultaneously:

        - each agent's in-community payoff tensor
        - each agent's outside-option (grid) payoff tensor
        - each agent's IR "gain" tensor = in-community payoff - outside option

    This is done with plain NumPy broadcasting (no Python-level loop over the
    k^N combinations), which is what makes even fairly large grids (k up to 9,
    N up to 6 -> up to 9**6 ~ 531k joint actions) tractable to evaluate.

    If `hour` is given (0-23), each agent's candidate grid is scaled by that
    hour's availability profile (see hourly_availability) before payoffs are
    computed -- so the returned tensors describe THAT hour's market
    conditions specifically. simulate.py precomputes one such tensor set per
    hour of the day (24 total) once per run, then reuses them across all
    training episodes (agents relive the same "day" repeatedly while their
    Q-tables adapt, a standard episodic-RL setup).

    Returns
    -------
    gains       : list of ndarrays, one per agent, each of shape (k,)*N
    payoffs     : list of ndarrays, each agent's in-community payoff tensor
    grids       : the per-agent 1-D candidate levels (for looking up realized
                  quantities once a winning joint action index is chosen)
    """
    n = len(agents)
    grids = build_candidate_grids(agents, k, hour=hour)

    # meshgrid with indexing='ij' gives, for each agent i, an (k,)*n tensor
    # holding agent i's chosen quantity, broadcast across all other agents'
    # choices -- exactly what we need for fully vectorized payoff evaluation.
    mesh = np.meshgrid(*grids, indexing="ij")  # list of n tensors, shape (k,)*n

    producer_idx = [i for i, a in enumerate(agents) if a.role == "producer"]
    consumer_idx = [i for i, a in enumerate(agents) if a.role == "consumer"]

    # Aggregate supply / demand tensors (shape (k,)*n), summing the relevant
    # per-agent quantity tensors.
    Q_supply = sum(mesh[i] for i in producer_idx) if producer_idx else np.zeros_like(mesh[0])
    Q_demand = sum(mesh[j] for j in consumer_idx) if consumer_idx else np.zeros_like(mesh[0])

    traded = np.minimum(Q_supply, Q_demand)
    # Pro-rata rationing: if one side offers more than the other wants, each
    # agent on the long side is scaled down proportionally.
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_prod = np.where(Q_supply > 0, traded / Q_supply, 0.0)
        scale_cons = np.where(Q_demand > 0, traded / Q_demand, 0.0)

    # Actual (post-rationing) traded quantity per agent.
    q_actual = [None] * n
    for i in producer_idx:
        q_actual[i] = mesh[i] * scale_prod
    for j in consumer_idx:
        q_actual[j] = mesh[j] * scale_cons

    # Uniform clearing price: average of the marginal cost of the marginal
    # producer unit and the marginal utility of the marginal consumer unit,
    # evaluated at the ACTUAL traded quantities. This is a simplification of
    # a full supply/demand-curve intersection, standard for a toy double
    # auction model; where nothing trades, price is undefined and irrelevant
    # (payoffs collapse to zero contribution from the community side).
    def marginal(agent: Agent, q: np.ndarray) -> np.ndarray:
        if agent.role == "producer":
            return agent.intercept + agent.slope * q       # c0 + c1*q
        else:
            return agent.intercept - agent.slope * q       # u0 - u1*q

    if producer_idx:
        mc_avg = sum(marginal(agents[i], q_actual[i]) for i in producer_idx) / len(producer_idx)
    else:
        mc_avg = np.zeros_like(traded)
    if consumer_idx:
        mu_avg = sum(marginal(agents[j], q_actual[j]) for j in consumer_idx) / len(consumer_idx)
    else:
        mu_avg = np.zeros_like(traded)

    p_clear = np.where(traded > 0, 0.5 * (mc_avg + mu_avg), 0.0)

    payoffs = [None] * n
    outside = [None] * n
    for i in producer_idx:
        cost_actual = agents[i].cost_or_utility(q_actual[i])
        payoffs[i] = p_clear * q_actual[i] - cost_actual
        # Outside option evaluated at the ORIGINALLY CHOSEN quantity (not the
        # rationed one) -- "what this agent would have earned selling that
        # same volume straight to the grid instead of proposing it locally".
        cost_full = agents[i].cost_or_utility(mesh[i])
        outside[i] = GRID_SELL_PRICE * mesh[i] - cost_full
    for j in consumer_idx:
        util_actual = agents[j].cost_or_utility(q_actual[j])
        payoffs[j] = util_actual - p_clear * q_actual[j]
        util_full = agents[j].cost_or_utility(mesh[j])
        outside[j] = util_full - GRID_BUY_PRICE * mesh[j]

    gains = [payoffs[i] - outside[i] for i in range(n)]

    return gains, payoffs, grids
