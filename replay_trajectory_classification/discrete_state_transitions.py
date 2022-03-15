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


def expected_duration(discrete_state_transition, sampling_frequency=1):
    self_transitions = np.diag(discrete_state_transition)
    return (1 / (1 - self_transitions)) / sampling_frequency


def estimate_discrete_state_transition(classifier, results):
    '''Updates the discrete_state_transition based on the probability of each
    state.

    Parameters
    ----------
    classifier : ClusterlessClassifier or SortedSpikesClassifier
    results : xarray.Dataset

    Returns
    -------
    discrete_state_transition : numpy.ndarray, shape (n_states, n_states)

    Notes
    -----
    $$
    Pr(I_{k} = i, I_{k+1} = j \mid O_{1:T}) = \frac{Pr(I_{k+1} = j \mid I_{k} = i) Pr(I_{k} = i \mid O_{1:k})}{Pr(I_{k+1} = j \mid O_{1:k})} Pr(I_{k+1} = j \mid O_{1:T})
    $$
    '''
    EPS = 1e-32
    try:
        causal_prob = np.log(
            results.causal_posterior.sum('position').values + EPS)
        acausal_prob = np.log(
            results.acausal_posterior.sum('position').values + EPS)
    except ValueError:
        causal_prob = np.log(
            results.causal_posterior.sum(['x_position', 'y_position']).values
            + EPS)
        acausal_prob = np.log(
            results.acausal_posterior.sum(['x_position', 'y_position']).values
            + EPS)

    old_discrete_state_transition = np.log(
        classifier.discrete_state_transition_)
    n_states = old_discrete_state_transition.shape[0]

    new_log_discrete_state_transition = np.empty((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            new_log_discrete_state_transition[i, j] = logsumexp(
                old_discrete_state_transition[i, j] +
                causal_prob[:-1, i] +
                acausal_prob[1:, j] -
                causal_prob[1:, j])
            new_log_discrete_state_transition[i, j] -= logsumexp(
                acausal_prob[:-1, i])
    new_log_discrete_state_transition -= logsumexp(
        new_log_discrete_state_transition, axis=-1, keepdims=True)

    return np.exp(new_log_discrete_state_transition)
