"""Core algorithms for decoding."""
import numpy as np
from numba import njit


def atleast_2d(x: np.ndarray) -> np.ndarray:
    """Appends a dimension at the end if `x` is one dimensional

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    x : np.ndarray

    """
    return np.atleast_2d(x).T if x.ndim < 2 else x


def get_centers(bin_edges: np.ndarray) -> np.ndarray:
    """Given a set of bin edges, return the center of the bin.

    Parameters
    ----------
    bin_edges : np.ndarray, shape (n_edges,)

    Returns
    -------
    bin_centers : np.ndarray, shape (n_edges - 1,)

    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def make_transition_matrix(
    discrete_state_transitions: np.ndarray,
    continuous_state_transitions: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    discrete_state_transitions : np.ndarray, shape (n_states, n_states)
    continuous_state_transitions : np.ndarray, shape (n_bins, n_bins)
    state_ind : np.ndarray, shape (n_bins,)

    Returns
    -------
    transition_matrix : np.ndarray, shape (n_bins, n_bins)

    """
    return (
        discrete_state_transitions[np.ix_(state_ind, state_ind)]
        * continuous_state_transitions
    )


def forward(
    initial_conditions: np.ndarray,
    log_likelihood: np.ndarray,
    transition_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """_summary_

    Parameters
    ----------
    initial_conditions : np.ndarray, shape (n_states,)
    log_likelihood : np.ndarray, shape (n_time, n_states)
    transition_matrix : np.ndarray, shape (n_states, n_states)

    Returns
    -------
    causal_posterior : np.ndarray, shape (n_time, n_states)
    predictive_distribution : np.ndarray, shape (n_time, n_states)
    marginal_likelihood : float

    """
    n_time = log_likelihood.shape[0]

    predictive_distribution = np.zeros_like(log_likelihood)
    causal_posterior = np.zeros_like(log_likelihood)
    max_log_likelihood = np.max(log_likelihood, axis=1, keepdims=True)
    likelihood = np.exp(log_likelihood - max_log_likelihood)
    likelihood = np.clip(likelihood, a_min=1e-15, a_max=1.0)

    predictive_distribution[0] = initial_conditions
    causal_posterior[0] = initial_conditions * likelihood[0]
    norm = np.nansum(causal_posterior[0])
    marginal_likelihood = np.log(norm)
    causal_posterior[0] /= norm

    for t in range(1, n_time):
        # Predict
        predictive_distribution[t] = transition_matrix.T @ causal_posterior[t - 1]
        # Update
        causal_posterior[t] = predictive_distribution[t] * likelihood[t]
        # Normalize
        norm = np.nansum(causal_posterior[t])
        marginal_likelihood += np.log(norm)
        causal_posterior[t] /= norm

    marginal_likelihood += np.sum(max_log_likelihood)

    return causal_posterior, predictive_distribution, marginal_likelihood


def smoother(
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_states)
    predictive_distribution : np.ndarray, shape (n_time, n_states)
    transition_matrix : np.ndarray, shape (n_states, n_states)

    Returns
    -------
    acausal_posterior : np.ndarray, shape (n_time, n_states)

    """
    n_time = causal_posterior.shape[0]

    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1]

    for t in range(n_time - 2, -1, -1):
        # Handle divide by zero
        relative_distribution = np.where(
            np.isclose(predictive_distribution[t + 1], 0.0),
            0.0,
            acausal_posterior[t + 1] / predictive_distribution[t + 1],
        )
        acausal_posterior[t] = causal_posterior[t] * (
            transition_matrix @ relative_distribution
        )
        acausal_posterior[t] /= acausal_posterior[t].sum()

    return acausal_posterior


def viterbi(initial_conditions, log_likelihood, transition_matrix):

    EPS = 1e-15
    n_time, n_states = log_likelihood.shape

    log_state_transition = np.log(np.clip(transition_matrix, a_min=EPS, a_max=1.0))
    log_initial_conditions = np.log(np.clip(initial_conditions, a_min=EPS, a_max=1.0))

    path_log_prob = np.ones_like(log_likelihood)
    back_pointer = np.zeros_like(log_likelihood, dtype=int)

    path_log_prob[0] = log_initial_conditions + log_likelihood[0]

    for time_ind in range(1, n_time):
        prior = path_log_prob[time_ind - 1] + log_state_transition
        for state_ind in range(n_states):
            back_pointer[time_ind, state_ind] = np.argmax(prior[state_ind])
            path_log_prob[time_ind, state_ind] = (
                prior[state_ind, back_pointer[time_ind, state_ind]]
                + log_likelihood[time_ind, state_ind]
            )

    # Find the best accumulated path prob in the last time bin
    # and then trace back the best path
    best_path = np.zeros((n_time,), dtype=int)
    best_path[-1] = np.argmax(path_log_prob[-1])
    for time_ind in range(n_time - 2, -1, -1):
        best_path[time_ind] = back_pointer[time_ind + 1, best_path[time_ind + 1]]

    return best_path, np.exp(np.max(path_log_prob[-1]))


def check_converged(
    log_likelihood: np.ndarray,
    previous_log_likelihood: np.ndarray,
    tolerance: float = 1e-4,
) -> tuple[bool, bool]:
    """We have converged if the slope of the log-likelihood function falls below 'tolerance',

    i.e., |f(t) - f(t-1)| / avg < tolerance,
    where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.

    Parameters
    ----------
    log_likelihood : np.ndarray
        Current log likelihood
    previous_log_likelihood : np.ndarray
        Previous log likelihood
    tolerance : float, optional
        threshold for similarity, by default 1e-4

    Returns
    -------
    is_converged : bool
    is_increasing : bool

    """
    delta_log_likelihood = abs(log_likelihood - previous_log_likelihood)
    avg_log_likelihood = (
        abs(log_likelihood) + abs(previous_log_likelihood) + np.spacing(1)
    ) / 2

    is_increasing = log_likelihood - previous_log_likelihood >= -1e-3
    is_converged = (delta_log_likelihood / avg_log_likelihood) < tolerance

    return is_converged, is_increasing


try:
    import cupy as cp

    @cp.fuse()
    def forward_gpu(
        initial_conditions: cp.ndarray,
        log_likelihood: cp.ndarray,
        transition_matrix: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray, float]:
        n_time = log_likelihood.shape[0]

        predictive_distribution = cp.zeros_like(log_likelihood)
        causal_posterior = cp.zeros_like(log_likelihood)
        max_log_likelihood = cp.max(log_likelihood, axis=1, keepdims=True)
        likelihood = cp.exp(log_likelihood - max_log_likelihood)
        likelihood = cp.clip(likelihood, a_min=1e-15, a_max=1.0)

        predictive_distribution[0] = initial_conditions
        causal_posterior[0] = initial_conditions * likelihood[0]
        norm = cp.nansum(causal_posterior[0])
        marginal_likelihood = cp.log(norm)
        causal_posterior[0] /= norm

        for t in range(1, n_time):
            # Predict
            predictive_distribution[t] = transition_matrix.T @ causal_posterior[t - 1]
            # Update
            causal_posterior[t] = predictive_distribution[t] * likelihood[t]
            # Normalize
            norm = np.nansum(causal_posterior[t])
            marginal_likelihood += cp.log(norm)
            causal_posterior[t] /= norm

        marginal_likelihood += cp.sum(max_log_likelihood)

        return causal_posterior, predictive_distribution, marginal_likelihood

    @cp.fuse()
    def smoother_gpu(
        causal_posterior: cp.ndarray,
        predictive_distribution: cp.ndarray,
        transition_matrix: cp.ndarray,
    ) -> cp.ndarray:
        n_time = causal_posterior.shape[0]

        acausal_posterior = cp.zeros_like(causal_posterior)
        acausal_posterior[-1] = causal_posterior[-1]

        for t in range(n_time - 2, -1, -1):
            # Handle divide by zero
            relative_distribution = cp.where(
                cp.isclose(predictive_distribution[t + 1], 0.0),
                0.0,
                acausal_posterior[t + 1] / predictive_distribution[t + 1],
            )
            acausal_posterior[t] = causal_posterior[t] * (
                transition_matrix @ relative_distribution
            )
            acausal_posterior[t] /= acausal_posterior[t].sum()

        return acausal_posterior

    @cp.fuse()
    def viterbi_gpu(
        initial_conditions: cp.ndarray,
        log_likelihood: cp.ndarray,
        transition_matrix: cp.ndarray,
    ):

        EPS = 1e-15
        n_time, n_states = log_likelihood.shape

        log_state_transition = cp.log(cp.clip(transition_matrix, a_min=EPS, a_max=1.0))
        log_initial_conditions = cp.log(
            cp.clip(initial_conditions, a_min=EPS, a_max=1.0)
        )

        path_log_prob = cp.ones_like(log_likelihood)
        back_pointer = cp.zeros_like(log_likelihood, dtype=int)

        path_log_prob[0] = log_initial_conditions + log_likelihood[0]

        for time_ind in range(1, n_time):
            prior = path_log_prob[time_ind - 1] + log_state_transition
            for state_ind in range(n_states):
                back_pointer[time_ind, state_ind] = cp.argmax(prior[state_ind])
                path_log_prob[time_ind, state_ind] = (
                    prior[state_ind, back_pointer[time_ind, state_ind]]
                    + log_likelihood[time_ind, state_ind]
                )

        # Find the best accumulated path prob in the last time bin
        # and then trace back the best path
        best_path = cp.zeros((n_time,), dtype=int)
        best_path[-1] = cp.argmax(path_log_prob[-1])
        for time_ind in range(n_time - 2, -1, -1):
            best_path[time_ind] = back_pointer[time_ind + 1, best_path[time_ind + 1]]

        return best_path, cp.exp(cp.max(path_log_prob[-1]))

except ImportError:
    from logging import getLogger

    logger = getLogger(__name__)
    logger.warning(
        "Cupy is not installed or GPU is not detected."
        " Ignore this message if not using GPU"
    )

    def forward_gpu(*args, **kwargs):
        logger.error("No GPU detected...")

    def smoother_gpu(*args, **kwargs):
        logger.error("No GPU detected...")

    def viterbi_gpu(*args, **kwargs):
        logger.error("No GPU detected...")
