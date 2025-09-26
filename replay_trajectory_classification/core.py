"""Core algorithms for decoding.

This module contains the fundamental algorithms for Bayesian decoding including
causal and acausal state estimation and classification.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit


def atleast_2d(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Appends a dimension at the end if `x` is one dimensional

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array of any shape

    Returns
    -------
    NDArray[np.float64]
        2D array with shape (x.shape[0], x.shape[1:]) if x.ndim >= 2,
        otherwise (x.shape[0], 1)

    """
    return np.atleast_2d(x).T if x.ndim < 2 else x


def get_centers(bin_edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Given a set of bin edges, return the center of the bin.

    Parameters
    ----------
    bin_edges : NDArray[np.float64], shape (n_edges,)
        Edges of the bins

    Returns
    -------
    NDArray[np.float64], shape (n_edges - 1,)
        Centers of the bins

    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


@njit(parallel=True, error_model="numpy")
def normalize_to_probability(distribution: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ensure the distribution integrates to 1 so that it is a probability distribution.

    Parameters
    ----------
    distribution : NDArray[np.float64]
        Probability distribution values to normalize.

    Returns
    -------
    normalized_distribution : NDArray[np.float64]
        Normalized probability distribution that sums to 1.

    """
    return distribution / np.nansum(distribution)


@njit(nogil=True, error_model="numpy", cache=False)
def _causal_decode(
    initial_conditions: NDArray[np.float64], state_transition: NDArray[np.float64], likelihood: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], float]:
    """Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : NDArray[np.float64], shape (n_bins,)
        Initial probability distribution over state bins.
    state_transition : NDArray[np.float64], shape (n_bins, n_bins)
        Transition probability matrix between state bins.
    likelihood : NDArray[np.float64], shape (n_time, n_bins)
        Likelihood values for each time point and state bin.

    Returns
    -------
    posterior : NDArray[np.float64], shape (n_time, n_bins)
        Posterior probability distribution over time and state bins.
    log_data_likelihood : float
        Log-likelihood of the observed data.

    """

    n_time = likelihood.shape[0]
    posterior = np.zeros_like(likelihood)

    posterior[0] = initial_conditions.copy() * likelihood[0]
    norm = np.nansum(posterior[0])
    log_data_likelihood = np.log(norm)
    posterior[0] /= norm

    for k in np.arange(1, n_time):
        posterior[k] = state_transition.T @ posterior[k - 1] * likelihood[k]
        norm = np.nansum(posterior[k])
        log_data_likelihood += np.log(norm)
        posterior[k] /= norm

    return posterior, log_data_likelihood


@njit(nogil=True, error_model="numpy", cache=False)
def _acausal_decode(
    causal_posterior: NDArray[np.float64], state_transition: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Uses past and future information to estimate the state.

    Parameters
    ----------
    causal_posterior : NDArray[np.float64], shape (n_time, n_bins, 1)
        Causal (forward-pass) posterior probabilities.
    state_transition : NDArray[np.float64], shape (n_bins, n_bins)
        Transition probability matrix between state bins.

    Returns
    -------
    acausal_posterior : NDArray[np.float64], shape (n_time, n_bins, 1)
        Acausal (forward-backward) posterior probabilities.

    """
    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1].copy()
    n_time, n_bins = causal_posterior.shape[0], causal_posterior.shape[-2]
    weights = np.zeros((n_bins, 1))

    eps = np.spacing(1)

    for time_ind in np.arange(n_time - 2, -1, -1):
        acausal_prior = state_transition.T @ causal_posterior[time_ind]
        log_ratio = np.log(acausal_posterior[time_ind + 1, ..., 0] + eps) - np.log(
            acausal_prior[..., 0] + eps
        )
        weights[..., 0] = np.exp(log_ratio) @ state_transition

        acausal_posterior[time_ind] = normalize_to_probability(
            weights * causal_posterior[time_ind]
        )

    return acausal_posterior


@njit(nogil=True, error_model="numpy", cache=False)
def _causal_classify(
    initial_conditions: NDArray[np.float64],
    continuous_state_transition: NDArray[np.float64],
    discrete_state_transition: NDArray[np.float64],
    likelihood: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], float]:
    """Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : NDArray[np.float64], shape (n_states, n_bins, 1)
        Initial probability distribution for each state and spatial bin.
    continuous_state_transition : NDArray[np.float64], shape (n_states, n_states, n_bins, n_bins)
        Continuous-space transition probabilities between states and bins.
    discrete_state_transition : NDArray[np.float64], shape (n_states, n_states)
        Discrete state transition probabilities.
    likelihood : NDArray[np.float64], shape (n_time, n_states, n_bins, 1)
        Likelihood values for each time point, state, and spatial bin.

    Returns
    -------
    causal_posterior : NDArray[np.float64], shape (n_time, n_states, n_bins, 1)
        Causal posterior probabilities over time, states, and spatial bins.
    log_data_likelihood : float
        Log-likelihood of the observed data.

    """
    n_time, n_states, n_bins, _ = likelihood.shape
    posterior = np.zeros_like(likelihood)

    posterior[0] = initial_conditions.copy() * likelihood[0]
    norm = np.nansum(posterior[0])
    log_data_likelihood = np.log(norm)
    posterior[0] /= norm

    for k in np.arange(1, n_time):
        prior = np.zeros((n_states, n_bins, 1))
        for state_k in np.arange(n_states):
            for state_k_1 in np.arange(n_states):
                prior[state_k, :] += (
                    discrete_state_transition[state_k_1, state_k]
                    * continuous_state_transition[state_k_1, state_k].T
                    @ posterior[k - 1, state_k_1]
                )
        posterior[k] = prior * likelihood[k]
        norm = np.nansum(posterior[k])
        log_data_likelihood += np.log(norm)
        posterior[k] /= norm

    return posterior, log_data_likelihood


@njit(nogil=True, error_model="numpy", cache=False)
def _acausal_classify(
    causal_posterior: NDArray[np.float64],
    continuous_state_transition: NDArray[np.float64],
    discrete_state_transition: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Uses past and future information to estimate the state.

    Parameters
    ----------
    causal_posterior : NDArray[np.float64], shape (n_time, n_states, n_bins, 1)
        Causal (forward-pass) posterior probabilities.
    continuous_state_transition : NDArray[np.float64], shape (n_states, n_states, n_bins, n_bins)
        Continuous-space transition probabilities between states and bins.
    discrete_state_transition : NDArray[np.float64], shape (n_states, n_states)
        Discrete state transition probabilities.

    Returns
    -------
    acausal_posterior : NDArray[np.float64], shape (n_time, n_states, n_bins, 1)
        Acausal (forward-backward) posterior probabilities.

    """
    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1].copy()
    n_time, n_states, n_bins, _ = causal_posterior.shape
    eps = np.spacing(1)

    for k in np.arange(n_time - 2, -1, -1):
        # Prediction Step -- p(x_{k+1}, I_{k+1} | y_{1:k})
        prior = np.zeros((n_states, n_bins, 1))
        for state_k_1 in np.arange(n_states):
            for state_k in np.arange(n_states):
                prior[state_k_1, :] += (
                    discrete_state_transition[state_k, state_k_1]
                    * continuous_state_transition[state_k, state_k_1].T
                    @ causal_posterior[k, state_k]
                )

        # Backwards Update
        weights = np.zeros((n_states, n_bins, 1))
        ratio = np.exp(np.log(acausal_posterior[k + 1] + eps) - np.log(prior + eps))
        for state_k in np.arange(n_states):
            for state_k_1 in np.arange(n_states):
                weights[state_k] += (
                    discrete_state_transition[state_k, state_k_1]
                    * continuous_state_transition[state_k, state_k_1]
                    @ ratio[state_k_1]
                )

        acausal_posterior[k] = normalize_to_probability(weights * causal_posterior[k])

    return acausal_posterior


def scaled_likelihood(log_likelihood: NDArray[np.float64], axis: int = 1) -> NDArray[np.float64]:
    """Scale the likelihood so the maximum value is 1.

    Parameters
    ----------
    log_likelihood : NDArray[np.float64], shape (n_time, n_bins)
        Log-likelihood values to be scaled.
    axis : int, optional
        Axis along which to find the maximum, by default 1.

    Returns
    -------
    scaled_log_likelihood : NDArray[np.float64], shape (n_time, n_bins)
        Likelihood values scaled so that the maximum is 1.

    """
    max_log_likelihood = np.nanmax(log_likelihood, axis=axis, keepdims=True)
    # If maximum is infinity, set to zero
    if max_log_likelihood.ndim > 0:
        max_log_likelihood[~np.isfinite(max_log_likelihood)] = 0.0
    elif not np.isfinite(max_log_likelihood):
        max_log_likelihood = 0.0

    # Maximum likelihood is always 1
    likelihood = np.exp(log_likelihood - max_log_likelihood)
    # avoid zero likelihood
    likelihood += np.spacing(1, dtype=likelihood.dtype)
    return likelihood


def mask(value: NDArray[np.float64], is_track_interior: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Set bins that are not part of the track to NaN.

    Parameters
    ----------
    value : NDArray[np.float64], shape (..., n_bins)
        Input values to be masked.
    is_track_interior : NDArray[np.bool_], shape (n_bins,)
        Boolean array indicating which bins are part of the track interior.

    Returns
    -------
    masked_value : NDArray[np.float64], shape (..., n_bins)
        Values with non-track bins set to NaN.

    """
    try:
        value[..., ~is_track_interior] = np.nan
    except IndexError:
        value[..., ~is_track_interior, :] = np.nan
    return value


def check_converged(
    log_likelihood: NDArray[np.float64],
    previous_log_likelihood: NDArray[np.float64],
    tolerance: float = 1e-4,
) -> Tuple[bool, bool]:
    """Check if log-likelihood has converged.

    We have converged if the slope of the log-likelihood function falls below
    'tolerance', i.e., |f(t) - f(t-1)| / avg < tolerance, where
    avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log likelihood at iteration t.

    Parameters
    ----------
    log_likelihood : NDArray[np.float64]
        Current log likelihood values.
    previous_log_likelihood : NDArray[np.float64]
        Previous log likelihood values.
    tolerance : float, optional
        Threshold for similarity, by default 1e-4.

    Returns
    -------
    is_converged : bool
        Whether the log-likelihood has converged.
    is_increasing : bool
        Whether the log-likelihood is increasing.

    """
    delta_log_likelihood = abs(log_likelihood - previous_log_likelihood)
    avg_log_likelihood = (
        abs(log_likelihood) + abs(previous_log_likelihood) + np.spacing(1)
    ) / 2

    is_increasing = log_likelihood - previous_log_likelihood >= -1e-3
    is_converged = (delta_log_likelihood / avg_log_likelihood) < tolerance

    return bool(is_converged), bool(is_increasing)


try:
    import cupy as cp

    @cp.fuse()
    def _causal_decode_gpu(
        initial_conditions: NDArray[np.float64],
        state_transition: NDArray[np.float64],
        likelihood: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        """Adaptive filter to iteratively calculate the posterior probability
        of a state variable using past information.

        Parameters
        ----------
        initial_conditions : NDArray[np.float64], shape (n_bins,)
            Initial probability distribution over state bins.
        state_transition : NDArray[np.float64], shape (n_bins, n_bins)
            Transition probability matrix between state bins.
        likelihood : NDArray[np.float64], shape (n_time, n_bins)
            Likelihood values for each time point and state bin.

        Returns
        -------
        posterior : NDArray[np.float64], shape (n_time, n_bins)
            Posterior probability distribution over time and state bins.
        log_data_likelihood : float
            Log-likelihood of the observed data.

        """

        initial_conditions = cp.asarray(initial_conditions, dtype=cp.float32)
        state_transition = cp.asarray(state_transition, dtype=cp.float32)
        likelihood = cp.asarray(likelihood, dtype=cp.float32)

        n_time = likelihood.shape[0]
        posterior = cp.zeros_like(likelihood)

        posterior[0] = initial_conditions * likelihood[0]
        norm = cp.nansum(posterior[0])
        log_data_likelihood = cp.log(norm)
        posterior[0] /= norm

        for k in np.arange(1, n_time):
            posterior[k] = state_transition.T @ posterior[k - 1] * likelihood[k]
            norm = np.nansum(posterior[k])
            log_data_likelihood += cp.log(norm)
            posterior[k] /= norm

        return cp.asnumpy(posterior), cp.asnumpy(log_data_likelihood)

    @cp.fuse()
    def _acausal_decode_gpu(
        causal_posterior: NDArray[np.float64], state_transition: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Uses past and future information to estimate the state.

        Parameters
        ----------
        causal_posterior : NDArray[np.float64], shape (n_time, n_bins, 1)
            Causal (forward-pass) posterior probabilities.
        state_transition : NDArray[np.float64], shape (n_bins, n_bins)
            Transition probability matrix between state bins.

        Returns
        -------
        acausal_posterior : NDArray[np.float64], shape (n_time, n_bins, 1)
            Acausal (forward-backward) posterior probabilities.

        """
        causal_posterior = cp.asarray(causal_posterior, dtype=cp.float32)
        state_transition = cp.asarray(state_transition, dtype=cp.float32)

        acausal_posterior = cp.zeros_like(causal_posterior)
        acausal_posterior[-1] = causal_posterior[-1]
        n_time, n_bins = causal_posterior.shape[0], causal_posterior.shape[-2]
        weights = cp.zeros((n_bins, 1))
        eps = np.spacing(1, dtype=np.float32)

        for time_ind in np.arange(n_time - 2, -1, -1):
            acausal_prior = state_transition.T @ causal_posterior[time_ind]
            log_ratio = cp.log(acausal_posterior[time_ind + 1, ..., 0] + eps) - cp.log(
                acausal_prior[..., 0] + eps
            )
            weights[..., 0] = cp.exp(log_ratio) @ state_transition

            acausal_posterior[time_ind] = weights * causal_posterior[time_ind]
            acausal_posterior[time_ind] /= np.nansum(acausal_posterior[time_ind])

        return cp.asnumpy(acausal_posterior)

    @cp.fuse()
    def _causal_classify_gpu(
        initial_conditions: NDArray[np.float64],
        continuous_state_transition: NDArray[np.float64],
        discrete_state_transition: NDArray[np.float64],
        likelihood: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        """Adaptive filter to iteratively calculate the posterior probability
        of a state variable using past information.

        Parameters
        ----------
        initial_conditions : NDArray[np.float64], shape (n_states, n_bins, 1)
            Initial probability distribution for each state and spatial bin.
        continuous_state_transition : NDArray[np.float64], shape (n_states, n_states, n_bins, n_bins)
            Continuous-space transition probabilities between states and bins.
        discrete_state_transition : NDArray[np.float64], shape (n_states, n_states)
            Discrete state transition probabilities.
        likelihood : NDArray[np.float64], shape (n_time, n_states, n_bins, 1)
            Likelihood values for each time point, state, and spatial bin.

        Returns
        -------
        causal_posterior : NDArray[np.float64], shape (n_time, n_states, n_bins, 1)
            Causal posterior probabilities over time, states, and spatial bins.
        log_data_likelihood : float
            Log-likelihood of the observed data.

        """
        initial_conditions = cp.asarray(initial_conditions, dtype=cp.float32)
        continuous_state_transition = cp.asarray(
            continuous_state_transition, dtype=cp.float32
        )
        discrete_state_transition = cp.asarray(
            discrete_state_transition, dtype=cp.float32
        )
        likelihood = cp.asarray(likelihood, dtype=cp.float32)

        n_time, n_states, n_bins, _ = likelihood.shape
        posterior = cp.zeros_like(likelihood)

        posterior[0] = initial_conditions * likelihood[0]
        norm = cp.nansum(posterior[0])
        log_data_likelihood = cp.log(norm)
        posterior[0] /= norm

        for k in np.arange(1, n_time):
            for state_k in np.arange(n_states):
                for state_k_1 in np.arange(n_states):
                    posterior[k, state_k] += (
                        discrete_state_transition[state_k_1, state_k]
                        * continuous_state_transition[state_k_1, state_k].T
                        @ posterior[k - 1, state_k_1]
                    )
            posterior[k] *= likelihood[k]
            norm = cp.nansum(posterior[k])
            log_data_likelihood += cp.log(norm)
            posterior[k] /= norm

        return cp.asnumpy(posterior), cp.asnumpy(log_data_likelihood)

    @cp.fuse()
    def _acausal_classify_gpu(
        causal_posterior: NDArray[np.float64],
        continuous_state_transition: NDArray[np.float64],
        discrete_state_transition: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Uses past and future information to estimate the state.

        Parameters
        ----------
        causal_posterior : NDArray[np.float64], shape (n_time, n_states, n_bins, 1)
            Causal (forward-pass) posterior probabilities.
        continuous_state_transition : NDArray[np.float64], shape (n_states, n_states, n_bins, n_bins)
            Continuous-space transition probabilities between states and bins.
        discrete_state_transition : NDArray[np.float64], shape (n_states, n_states)
            Discrete state transition probabilities.

        Returns
        -------
        acausal_posterior : NDArray[np.float64], shape (n_time, n_states, n_bins, 1)
            Acausal (forward-backward) posterior probabilities.

        """
        causal_posterior = cp.asarray(causal_posterior, dtype=cp.float32)
        continuous_state_transition = cp.asarray(
            continuous_state_transition, dtype=cp.float32
        )
        discrete_state_transition = cp.asarray(
            discrete_state_transition, dtype=cp.float32
        )

        acausal_posterior = cp.zeros_like(causal_posterior)
        acausal_posterior[-1] = causal_posterior[-1]
        n_time, n_states, n_bins, _ = causal_posterior.shape
        eps = np.spacing(1, dtype=np.float32)

        for k in np.arange(n_time - 2, -1, -1):
            # Prediction Step -- p(x_{k+1}, I_{k+1} | y_{1:k})
            prior = cp.zeros((n_states, n_bins, 1))
            for state_k_1 in np.arange(n_states):
                for state_k in np.arange(n_states):
                    prior[state_k_1, :] += (
                        discrete_state_transition[state_k, state_k_1]
                        * continuous_state_transition[state_k, state_k_1].T
                        @ causal_posterior[k, state_k]
                    )

            # Backwards Update
            weights = cp.zeros((n_states, n_bins, 1))
            ratio = cp.exp(cp.log(acausal_posterior[k + 1] + eps) - cp.log(prior + eps))
            for state_k in np.arange(n_states):
                for state_k_1 in np.arange(n_states):
                    weights[state_k] += (
                        discrete_state_transition[state_k, state_k_1]
                        * continuous_state_transition[state_k, state_k_1]
                        @ ratio[state_k_1]
                    )

            acausal_posterior[k] = weights * causal_posterior[k]
            acausal_posterior[k] /= cp.nansum(acausal_posterior[k])

        return cp.asnumpy(acausal_posterior)

except ImportError:
    from logging import getLogger

    logger = getLogger(__name__)
    logger.warning(
        "Cupy is not installed or GPU is not detected."
        " Ignore this message if not using GPU"
    )

    def _causal_decode_gpu(*args, **kwargs):
        logger.error("No GPU detected...")

    def _acausal_decode_gpu(*args, **kwargs):
        logger.error("No GPU detected...")

    def _causal_classify_gpu(*args, **kwargs):
        logger.error("No GPU detected...")

    def _acausal_classify_gpu(*args, **kwargs):
        logger.error("No GPU detected...")
