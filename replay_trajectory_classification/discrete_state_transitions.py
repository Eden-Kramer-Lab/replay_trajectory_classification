"""Classes to generate transitions between categories."""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


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
        strong_diagonal[is_off_diag] = (1 - self.diagonal_value) / (n_states - 1)

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


def expected_duration(discrete_state_transition, sampling_frequency=1):
    """The average duration of each discrete state if it follows
    a geometric distribution.

    Use `sampling_frequency` to convert duration to seconds. Time is in
    number of samples by default.

    Parameters
    ----------
    discrete_state_transition : np.ndarray, shape (n_states, n_states)
    sampling_frequency : int, optional

    Returns
    -------
    duration : np.ndarray, shape (n_states)

    """
    self_transitions = np.diag(discrete_state_transition)
    return (1 / (1 - self_transitions)) / sampling_frequency


def estimate_discrete_state_transition(classifier, results):

    likelihood = results.likelihood.sum("position").values
    causal_prob = results.causal_posterior.sum("position").values
    acausal_prob = results.acausal_posterior.sum("position").values
    transition_matrix = classifier.discrete_state_transition_

    n_time, n_states = causal_prob.shape

    # probability of state 1 in time t and state 2 in time t+1
    xi = np.zeros((n_time - 1, n_states, n_states))

    for from_state in range(n_states):
        for to_state in range(n_states):
            xi[:, from_state, to_state] = (
                causal_prob[:-1, from_state]
                * likelihood[1:, to_state]
                * acausal_prob[1:, to_state]
                * transition_matrix[from_state, to_state]
                / (causal_prob[1:, to_state] + np.spacing(1))
            )

    xi = xi / xi.sum(axis=(1, 2), keepdims=True)

    summed_xi = xi.sum(axis=0)

    new_transition_matrix = summed_xi / summed_xi.sum(axis=1, keepdims=True)

    return new_transition_matrix
