"""Multi-Armed Bandit agents for gate library subset selection.

Both agents support sequential (batch_size=1) and batched operation via
select_action() / select_batch_actions() and update() / update_batch().
"""

import math
import random
import numpy as np


class EpsilonGreedyMAB:
    """Epsilon-greedy MAB that selects a subset of `sample_gate` arms per pull.

    Args:
        num_arms:    Total number of mutable gate arms.
        epsilon:     Exploration probability (0–1).
        sample_gate: Number of gates to select per action.
        batch_size:  Number of independent subsets per batch call.
    """

    def __init__(self, num_arms, epsilon, sample_gate, batch_size=1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.sample_gate = sample_gate
        self.batch_size = batch_size
        self.q_values = [0.0] * num_arms
        self.counts = [0] * num_arms

    def _select_one(self):
        selected = set()
        while len(selected) < self.sample_gate:
            if random.random() > self.epsilon:
                arm = int(np.argmax(self.q_values))
            else:
                arm = random.randint(0, self.num_arms - 1)
            selected.add(arm)
        return list(selected)

    def select_action(self):
        return self._select_one()

    def select_batch_actions(self):
        return [self._select_one() for _ in range(self.batch_size)]

    def update(self, selected_arms, reward):
        for arm in selected_arms:
            self.counts[arm] += 1
            self.q_values[arm] = (
                (self.q_values[arm] * (self.counts[arm] - 1) + reward) / self.counts[arm]
            )

    def update_batch(self, batch_actions, rewards):
        for arms, reward in zip(batch_actions, rewards):
            self.update(arms, reward)


class UCB_MAB:
    """UCB1 MAB that selects a subset of `sample_gate` arms per pull.

    Unvisited arms are always explored first; remaining slots use UCB scores.

    Args:
        num_arms:    Total number of mutable gate arms.
        c:           UCB exploration constant.
        sample_gate: Number of gates to select per action.
        batch_size:  Number of independent subsets per batch call.
    """

    def __init__(self, num_arms, c, sample_gate, batch_size=1):
        self.num_arms = num_arms
        self.c = c
        self.sample_gate = sample_gate
        self.batch_size = batch_size
        self.q_values = [0.0] * num_arms
        self.counts = [0] * num_arms

    def _select_one(self):
        selected = set()
        # Ensure unvisited arms get tried first
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                selected.add(arm)
                if len(selected) == self.sample_gate:
                    break

        if len(selected) < self.sample_gate:
            total = sum(self.counts)
            ucb = []
            for arm in range(self.num_arms):
                if arm in selected or self.counts[arm] == 0:
                    ucb.append(float("-inf"))
                else:
                    ucb.append(
                        self.q_values[arm] + self.c * math.sqrt(math.log(total) / self.counts[arm])
                    )
            while len(selected) < self.sample_gate:
                best = int(np.argmax(ucb))
                selected.add(best)
                ucb[best] = float("-inf")

        return list(selected)

    def select_action(self):
        return self._select_one()

    def select_batch_actions(self):
        return [self._select_one() for _ in range(self.batch_size)]

    def update(self, selected_arms, reward):
        for arm in selected_arms:
            self.counts[arm] += 1
            self.q_values[arm] = (
                (self.q_values[arm] * (self.counts[arm] - 1) + reward) / self.counts[arm]
            )

    def update_batch(self, batch_actions, rewards):
        for arms, reward in zip(batch_actions, rewards):
            self.update(arms, reward)
