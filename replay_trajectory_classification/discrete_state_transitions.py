"""Classes to generate transitions between categories."""
from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class DiagonalDiscrete:
    """Transition matrix with `diagonal_value` on the value for n_states

    Off-diagonals are probability: (1 - `diagonal_value`) / (`n_states` - 1)

    Attributes
    ----------
    diagonal_value : float, optional

    """

    diagonal_value: float = 0.98

    def make_state_transition(self, n_states: int) -> np.ndarray:
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

    def make_state_transition(self, n_states: int) -> np.ndarray:
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

    def make_state_transition(self, n_states: int) -> np.ndarray:
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

    def make_state_transition(self, n_states: int) -> np.ndarray:
        """Makes discrete state transition matrix.

        Parameters
        ----------
        n_states : int

        Returns
        -------
        discrete_state_transition : np.ndarray, shape (n_states, n_states)

        """
        return self.state_transition_


def expected_duration(
    discrete_state_transition: np.ndarray, sampling_frequency: int = 1
):
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


def estimate_discrete_state_transition(
    classifier,
    results: xr.Dataset,
) -> np.ndarray:
    """Estimate a new discrete transition matrix given the old one and updated smoother results.

    Parameters
    ----------
    classifier : ClusterlessClassifier or SortedSpikesClassifier instance
    results : xr.Dataset

    Returns
    -------
    new_transition_matrix : np.ndarray, shape (n_states, n_states)

    """

    causal_posterior = results.causal_posterior.groupby("state").sum().values
    acausal_posterior = results.acausal_posterior.groupby("state").sum().values
    predictive_distribution = classifier.predictive_distribution_
    transition_matrix = classifier.discrete_state_transition_

    relative_distribution = np.where(
        np.isclose(predictive_distribution[1:], 0.0),
        0.0,
        acausal_posterior[1:] / predictive_distribution[1:],
    )[:, np.newaxis]

    # p(I_t, I_{t+1} | O_{1:T})
    joint_distribution = (
        transition_matrix[np.newaxis]
        * causal_posterior[:-1, :, np.newaxis]
        * relative_distribution
    )

    new_transition_matrix = (
        joint_distribution.sum(axis=0)
        / acausal_posterior[:-1].sum(axis=0, keepdims=True).T
    )

    new_transition_matrix /= new_transition_matrix.sum(axis=1, keepdims=True)

    return new_transition_matrix
