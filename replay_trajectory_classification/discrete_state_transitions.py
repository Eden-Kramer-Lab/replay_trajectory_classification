from dataclasses import dataclass

import numpy as np


@dataclass
class DiagonalDiscrete:
    diagonal_value: float = 0.98

    def make_state_transition(self, n_states):
        strong_diagonal = np.identity(n_states) * self.diagonal_value
        is_off_diag = ~np.identity(n_states, dtype=bool)
        strong_diagonal[is_off_diag] = (
            (1 - self.diagonal_value) / (n_states - 1))

        self.state_transition_ = strong_diagonal
        return self.state_transition_


@dataclass
class UniformDiscrete:
    def make_state_transition(self, n_states):
        self.state_transition_ = np.ones((n_states, n_states)) / n_states

        return self.state_transition_


@dataclass
class RandomDiscrete:
    def make_state_transition(self, n_states):
        state_transition = np.random.random_sample(n_states, n_states)
        state_transition /= state_transition.sum(axis=1, keepdims=True)

        self.state_transition_ = state_transition
        return self.state_transition_
