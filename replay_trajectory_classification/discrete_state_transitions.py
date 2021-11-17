from dataclasses import dataclass

import numpy as np


@dataclass
class DiagonalDiscrete:
    """Transition matrix with `diagonal_value` on the value for n_states

    Off-diagonals are probability: (1 - `diagonal_value`) / (`n_states` - 1)

    Attributes
    ----------
    diagonal_value : float, optional

    """
    diagonal_value: float = 0.98

    def make_state_transition(self, n_states):
        """Makes discrete state transition matrix.

        Parameters
        ----------
        n_states : int

        Returns
        -------
        discrete_state_transition : np.ndarray, shape (n_states, n_states)

        """
        strong_diagonal = np.identity(n_states) * self.diagonal_value
        is_off_diag = ~np.identity(n_states, dtype=bool)
        strong_diagonal[is_off_diag] = (
            (1 - self.diagonal_value) / (n_states - 1))

        self.state_transition_ = strong_diagonal
        return self.state_transition_


@dataclass
class UniformDiscrete:
    """All transitions to states (including self transitions) are the same
    probability.
    """

    def make_state_transition(self, n_states):
        """Makes discrete state transition matrix.

        Parameters
        ----------
        n_states : int

        Returns
        -------
        discrete_state_transition : np.ndarray, shape (n_states, n_states)

        """
        self.state_transition_ = np.ones((n_states, n_states)) / n_states

        return self.state_transition_


@dataclass
class RandomDiscrete:
    """All state transitions are random"""

    def make_state_transition(self, n_states):
        """Makes discrete state transition matrix.

        Parameters
        ----------
        n_states : int

        Returns
        -------
        discrete_state_transition : np.ndarray, shape (n_states, n_states)

        """
        state_transition = np.random.random_sample(n_states, n_states)
        state_transition /= state_transition.sum(axis=1, keepdims=True)

        self.state_transition_ = state_transition
        return self.state_transition_


@dataclass
class UserDefinedDiscrete:
    """State transitions are provided by user.

    Attributes
    ----------
    state_transition : np.ndarray, shape (n_states, n_states)
    """
    state_transition_: np.ndarray

    def make_state_transition(self, n_states):
        """Makes discrete state transition matrix.

        Parameters
        ----------
        n_states : int

        Returns
        -------
        discrete_state_transition : np.ndarray, shape (n_states, n_states)

        """
        return self.state_transition_
