"""Classes for generating discrete state transitions.

This module provides discrete state transition models that define
transition probabilities between different trajectory categories
or behavioral states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import xarray as xr


@dataclass
class DiagonalDiscrete:
    """Discrete state transition matrix with dominant diagonal elements.

    Creates a transition matrix where diagonal elements have value `diagonal_value`
    and off-diagonal elements have probability (1 - `diagonal_value`) / (`n_states` - 1).

    Parameters
    ----------
    diagonal_value : float, optional
        Probability of staying in the same state, by default 0.98.

    """

    diagonal_value: float = 0.98

    def make_state_transition(self, n_states: int) -> NDArray[np.float64]:
        """Makes discrete state transition matrix.

        Parameters
        ----------
        n_states : int

        Returns
        -------
        discrete_state_transition : NDArray[np.float64], shape (n_states, n_states)

        """
        if n_states <= 0:
            raise ValueError("n_states must be >= 1")
        if n_states == 1:
            return np.array([[1.0]], dtype=float)
        if not (0.0 <= self.diagonal_value <= 1.0):
            raise ValueError("diagonal_value must be between 0 and 1")

        strong_diagonal = np.identity(n_states) * self.diagonal_value
        is_off_diag = ~np.identity(n_states, dtype=bool)
        strong_diagonal[is_off_diag] = (1 - self.diagonal_value) / (n_states - 1)

        self.state_transition_ = strong_diagonal
        return self.state_transition_


@dataclass
class UniformDiscrete:
    """Uniform discrete state transition matrix.

    All transitions between states (including self-transitions) have equal
    probability of 1/n_states.

    """

    def make_state_transition(self, n_states: int) -> NDArray[np.float64]:
        """Makes discrete state transition matrix.

        Parameters
        ----------
        n_states : int

        Returns
        -------
        discrete_state_transition : NDArray[np.float64], shape (n_states, n_states)

        """
        self.state_transition_ = np.ones((n_states, n_states)) / n_states

        return self.state_transition_


@dataclass
class RandomDiscrete:
    """All state transitions are random"""

    def make_state_transition(self, n_states: int) -> NDArray[np.float64]:
        """Makes discrete state transition matrix.

        Parameters
        ----------
        n_states : int

        Returns
        -------
        discrete_state_transition : NDArray[np.float64], shape (n_states, n_states)

        """
        if n_states <= 0:
            raise ValueError("n_states must be >= 1")
        state_transition = np.random.random_sample((n_states, n_states)).astype(float)

        # Guard: if any row accidentally sums to ~0, add a self-loop before normalization
        row_sums = state_transition.sum(axis=1, keepdims=True)
        dead = row_sums <= 0.0
        if np.any(dead):
            state_transition[dead, :] = 0.0
            state_transition[dead, np.arange(n_states)] = 1.0  # self-loop

        state_transition /= state_transition.sum(axis=1, keepdims=True)
        return state_transition


@dataclass
class UserDefinedDiscrete:
    """State transitions are provided by user.

    Attributes
    ----------
    state_transition : NDArray[np.float64], shape (n_states, n_states)
    """

    state_transition_: NDArray[np.float64]

    def make_state_transition(self, n_states: int) -> NDArray[np.float64]:
        """Makes discrete state transition matrix.

        Parameters
        ----------
        n_states : int

        Returns
        -------
        discrete_state_transition : NDArray[np.float64], shape (n_states, n_states)

        """
        return self.state_transition_


def expected_duration(
    discrete_state_transition: NDArray[np.float64], sampling_frequency: int = 1
):
    """The average duration of each discrete state if it follows
    a geometric distribution.

    Use `sampling_frequency` to convert duration to seconds. Time is in
    number of samples by default.

    Parameters
    ----------
    discrete_state_transition : NDArray[np.float64], shape (n_states, n_states)
    sampling_frequency : int, optional

    Returns
    -------
    duration : NDArray[np.float64], shape (n_states)

    """
    self_transitions = np.diag(discrete_state_transition)
    return (1 / (1 - self_transitions)) / sampling_frequency


def estimate_discrete_state_transition(
    classifier,
    results: xr.Dataset,
) -> NDArray[np.float64]:
    """Estimate a new discrete transition matrix given the old one and updated smoother results.

    Parameters
    ----------
    classifier : ClusterlessClassifier or SortedSpikesClassifier instance
    results : xr.Dataset

    Returns
    -------
    new_transition_matrix : NDArray[np.float64], shape (n_states, n_states)

    """
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
