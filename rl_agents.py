"""
rl_agents.py
============

Minimal tabular Q-learning agent used by simulate.py.

Design note: the CBL (bargaining.py) is the one authoritative decision-maker
each hour -- it evaluates the FULL discretized grid and picks a winning
joint action under whichever criterion is being tested. Compute cost is
therefore driven purely by grid enumeration (N, k), not by whether any
agent's policy has converged.

Each agent nonetheless runs its own independent tabular Q-learner that:
  - observes the hour of day as state (0..23),
  - is updated using the REALIZED quantity level it ended up trading (i.e.
    its component of whichever joint action the CBL selected that hour) as
    the "action", and the IR-gain it received as the reward.

This gives a genuine per-agent learning process across the training
episodes without affecting the CBL's own enumeration cost.

This module has no effect on which joint action wins each hour -- it only
lets Q-tables be logged/inspected, e.g. to check learning curves as a
sanity check that the pipeline behaves sensibly. Swap in a different
training rule here if you need one that actually feeds back into the CBL.
"""

from __future__ import annotations
import numpy as np


class TabularQAgent:
    """
    Q-table shape: (24 hours, k quantity levels).
    """

    def __init__(self, k: int, n_hours: int = 24, lr: float = 0.1, gamma: float = 0.9,
                 eps_start: float = 0.3, eps_end: float = 0.02, n_episodes: int = 300,
                 rng: np.random.Generator = None):
        self.k = k
        self.n_hours = n_hours
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.n_episodes = max(n_episodes, 1)
        self.q = np.zeros((n_hours, k), dtype=np.float64)
        self.rng = rng if rng is not None else np.random.default_rng()

    def epsilon(self, episode: int) -> float:
        """Linearly anneal exploration rate across episodes."""
        frac = min(episode / self.n_episodes, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def choose_action(self, hour: int, episode: int) -> int:
        """Epsilon-greedy action selection (used only for logging/inspection;
        does not influence the CBL's own decision in this reference design)."""
        if self.rng.random() < self.epsilon(episode):
            return int(self.rng.integers(0, self.k))
        return int(np.argmax(self.q[hour]))

    def update(self, hour: int, action_idx: int, reward: float, next_hour: int) -> None:
        """Standard tabular Q-learning update."""
        best_next = np.max(self.q[next_hour])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[hour, action_idx]
        self.q[hour, action_idx] += self.lr * td_error
