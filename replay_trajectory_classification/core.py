import numpy as np
from numba import njit


@njit(nogil=True, fastmath=True)
def logsumexp(a):
    a_max = np.max(a)
    out = np.log(np.sum(np.exp(a - a_max)))
    out += a_max
    return out


@njit(parallel=True, error_model='numpy')
def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / np.nansum(distribution)


@njit(nogil=True, error_model='numpy', cache=False)
def _causal_decode(initial_conditions, state_transition, likelihood):
    '''Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : ndarray, shape (n_bins,)
    state_transition : ndarray, shape (n_bins, n_bins)
    likelihood : ndarray, shape (n_time, n_bins)

    Returns
    -------
    posterior : ndarray, shape (n_time, n_bins)

    '''

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


@njit(nogil=True, error_model='numpy', cache=False)
def _acausal_decode(causal_posterior, state_transition):
    '''Uses past and future information to estimate the state.

    Parameters
    ----------
    causal_posterior : ndarray, shape (n_time, n_bins, 1)
    state_transition : ndarray, shape (n_bins, n_bins)

    Return
    ------
    acausal_posterior : ndarray, shape (n_time, n_bins, 1)

    '''
    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1].copy()
    n_time, n_bins = causal_posterior.shape[0], causal_posterior.shape[-2]
    weights = np.zeros((n_bins, 1))

    for time_ind in np.arange(n_time - 2, -1, -1):
        acausal_prior = (
            state_transition.T @ causal_posterior[time_ind])
        log_ratio = (
            np.log(acausal_posterior[time_ind + 1, ..., 0] + np.spacing(1)) -
            np.log(acausal_prior[..., 0] + np.spacing(1)))
        weights[..., 0] = np.exp(log_ratio) @ state_transition

        acausal_posterior[time_ind] = normalize_to_probability(
            weights * causal_posterior[time_ind])

    return acausal_posterior


@njit(nogil=True, error_model='numpy', cache=False)
def _causal_classify(initial_conditions, continuous_state_transition,
                     discrete_state_transition, likelihood):
    '''Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : ndarray, shape (n_states, n_bins, 1)
    continuous_state_transition : ndarray, shape (n_states, n_states,
                                                  n_bins, n_bins)
    discrete_state_transition : ndarray, shape (n_states, n_states)
    likelihood : ndarray, shape (n_time, n_states, n_bins, 1)

    Returns
    -------
    causal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)

    '''
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
                    discrete_state_transition[state_k_1, state_k] *
                    continuous_state_transition[state_k_1, state_k].T @
                    posterior[k - 1, state_k_1])
        posterior[k] = prior * likelihood[k]
        norm = np.nansum(posterior[k])
        log_data_likelihood += np.log(norm)
        posterior[k] /= norm

    return posterior, log_data_likelihood


@njit(nogil=True, error_model='numpy', cache=False)
def _acausal_classify(causal_posterior, continuous_state_transition,
                      discrete_state_transition):
    '''Uses past and future information to estimate the state.

    Parameters
    ----------
    causal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)
    continuous_state_transition : ndarray, shape (n_states, n_states,
                                                  n_bins, n_bins)
    discrete_state_transition : ndarray, shape (n_states, n_states)

    Return
    ------
    acausal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)

    '''
    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1].copy()
    n_time, n_states, n_bins, _ = causal_posterior.shape

    for k in np.arange(n_time - 2, -1, -1):
        # Prediction Step -- p(x_{k+1}, I_{k+1} | y_{1:k})
        prior = np.zeros((n_states, n_bins, 1))
        for state_k_1 in np.arange(n_states):
            for state_k in np.arange(n_states):
                prior[state_k_1, :] += (
                    discrete_state_transition[state_k, state_k_1] *
                    continuous_state_transition[state_k, state_k_1].T @
                    causal_posterior[k, state_k])

        # Backwards Update
        weights = np.zeros((n_states, n_bins, 1))
        ratio = np.exp(
            np.log(acausal_posterior[k + 1] + np.spacing(1)) -
            np.log(prior + np.spacing(1)))
        for state_k in np.arange(n_states):
            for state_k_1 in np.arange(n_states):
                weights[state_k] += (
                    discrete_state_transition[state_k, state_k_1] *
                    continuous_state_transition[state_k, state_k_1] @
                    ratio[state_k_1])

        acausal_posterior[k] = normalize_to_probability(
            weights * causal_posterior[k])

    return acausal_posterior


def scaled_likelihood(log_likelihood, axis=1):
    '''
    Parameters
    ----------
    log_likelihood : ndarray, shape (n_time, n_bins)

    Returns
    -------
    scaled_log_likelihood : ndarray, shape (n_time, n_bins)

    '''
    max_log_likelihood = np.nanmax(log_likelihood, axis=axis, keepdims=True)
    # If maximum is infinity, set to zero
    if max_log_likelihood.ndim > 0:
        max_log_likelihood[~np.isfinite(max_log_likelihood)] = 0.0
    elif not np.isfinite(max_log_likelihood):
        max_log_likelihood = 0.0

    # Maximum likelihood is always 1
    likelihood = np.exp(log_likelihood - max_log_likelihood)
    likelihood += np.spacing(1)  # avoid zero likelihood
    return likelihood


def mask(value, is_track_interior):
    try:
        value[..., ~is_track_interior] = np.nan
    except IndexError:
        value[..., ~is_track_interior, :] = np.nan
    return value


def check_converged(loglik, previous_loglik, tolerance=1e-4):
    '''
    We have converged if the slope of the log-likelihood function falls below 'threshold',
    i.e., |f(t) - f(t-1)| / avg < threshold,
    where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.
    'threshold' defaults to 1e-4.

    This stopping criterion is from Numerical Recipes in C p423
    '''
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.spacing(1)) / 2

    is_increasing = loglik - previous_loglik >= -1e-3
    is_converged = (delta_loglik / avg_loglik) < tolerance

    return is_converged, is_increasing


try:
    import cupy as cp

    @cp.fuse()
    def _causal_decode_gpu(initial_conditions, state_transition, likelihood):
        '''Adaptive filter to iteratively calculate the posterior probability
        of a state variable using past information.

        Parameters
        ----------
        initial_conditions : ndarray, shape (n_bins,)
        state_transition : ndarray, shape (n_bins, n_bins)
        likelihood : ndarray, shape (n_time, n_bins)

        Returns
        -------
        posterior : ndarray, shape (n_time, n_bins)

        '''

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
            posterior[k] = state_transition.T @ posterior[k - 1] * \
                likelihood[k]
            norm = np.nansum(posterior[k])
            log_data_likelihood += cp.log(norm)
            posterior[k] /= norm

        return cp.asnumpy(posterior), cp.asnumpy(log_data_likelihood)

    @cp.fuse()
    def _acausal_decode_gpu(causal_posterior, state_transition):
        '''Uses past and future information to estimate the state.

        Parameters
        ----------
        causal_posterior : ndarray, shape (n_time, n_bins, 1)
        state_transition : ndarray, shape (n_bins, n_bins)

        Return
        ------
        acausal_posterior : ndarray, shape (n_time, n_bins, 1)

        '''
        causal_posterior = cp.asarray(causal_posterior, dtype=cp.float32)
        state_transition = cp.asarray(state_transition, dtype=cp.float32)

        acausal_posterior = cp.zeros_like(causal_posterior)
        acausal_posterior[-1] = causal_posterior[-1]
        n_time, n_bins = causal_posterior.shape[0], causal_posterior.shape[-2]
        weights = cp.zeros((n_bins, 1))

        for time_ind in np.arange(n_time - 2, -1, -1):
            acausal_prior = (
                state_transition.T @ causal_posterior[time_ind])
            log_ratio = (
                cp.log(acausal_posterior[time_ind + 1, ..., 0] + np.spacing(1)) -
                cp.log(acausal_prior[..., 0] + np.spacing(1)))
            weights[..., 0] = cp.exp(log_ratio) @ state_transition

            acausal_posterior[time_ind] = weights * causal_posterior[time_ind]
            acausal_posterior[time_ind] /= np.nansum(
                acausal_posterior[time_ind])

        return cp.asnumpy(acausal_posterior)

    @cp.fuse()
    def _causal_classify_gpu(initial_conditions, continuous_state_transition,
                             discrete_state_transition, likelihood):
        '''Adaptive filter to iteratively calculate the posterior probability
        of a state variable using past information.

        Parameters
        ----------
        initial_conditions : ndarray, shape (n_states, n_bins, 1)
        continuous_state_transition : ndarray, shape (n_states, n_states,
                                                      n_bins, n_bins)
        discrete_state_transition : ndarray, shape (n_states, n_states)
        likelihood : ndarray, shape (n_time, n_states, n_bins, 1)

        Returns
        -------
        causal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)

        '''
        initial_conditions = cp.asarray(initial_conditions, dtype=cp.float32)
        continuous_state_transition = cp.asarray(
            continuous_state_transition, dtype=cp.float32)
        discrete_state_transition = cp.asarray(
            discrete_state_transition, dtype=cp.float32)
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
                        discrete_state_transition[state_k_1, state_k] *
                        continuous_state_transition[state_k_1, state_k].T @
                        posterior[k - 1, state_k_1])
            posterior[k] *= likelihood[k]
            norm = cp.nansum(posterior[k])
            log_data_likelihood += cp.log(norm)
            posterior[k] /= norm

        return cp.asnumpy(posterior), cp.asnumpy(log_data_likelihood)

    @cp.fuse()
    def _acausal_classify_gpu(causal_posterior, continuous_state_transition,
                              discrete_state_transition):
        '''Uses past and future information to estimate the state.

        Parameters
        ----------
        causal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)
        continuous_state_transition : ndarray, shape (n_states, n_states,
                                                      n_bins, n_bins)
        discrete_state_transition : ndarray, shape (n_states, n_states)

        Return
        ------
        acausal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)

        '''
        causal_posterior = cp.asarray(causal_posterior, dtype=cp.float32)
        continuous_state_transition = cp.asarray(
            continuous_state_transition, dtype=cp.float32)
        discrete_state_transition = cp.asarray(
            discrete_state_transition, dtype=cp.float32)

        acausal_posterior = cp.zeros_like(causal_posterior)
        acausal_posterior[-1] = causal_posterior[-1]
        n_time, n_states, n_bins, _ = causal_posterior.shape

        for k in np.arange(n_time - 2, -1, -1):
            # Prediction Step -- p(x_{k+1}, I_{k+1} | y_{1:k})
            prior = cp.zeros((n_states, n_bins, 1))
            for state_k_1 in np.arange(n_states):
                for state_k in np.arange(n_states):
                    prior[state_k_1, :] += (
                        discrete_state_transition[state_k, state_k_1] *
                        continuous_state_transition[state_k, state_k_1].T @
                        causal_posterior[k, state_k])

            # Backwards Update
            weights = cp.zeros((n_states, n_bins, 1))
            ratio = cp.exp(
                cp.log(acausal_posterior[k + 1] + np.spacing(1)) -
                cp.log(prior + np.spacing(1)))
            for state_k in np.arange(n_states):
                for state_k_1 in np.arange(n_states):
                    weights[state_k] += (
                        discrete_state_transition[state_k, state_k_1] *
                        continuous_state_transition[state_k, state_k_1] @
                        ratio[state_k_1])

            acausal_posterior[k] = (weights * causal_posterior[k])
            acausal_posterior[k] /= cp.nansum(acausal_posterior[k])

        return cp.asnumpy(acausal_posterior)
except ImportError:
    from logging import getLogger
    logger = getLogger(__name__)
    logger.warning('Cupy is not installed or GPU is not detected.'
                   ' Ignore this message if not using GPU')

    def _causal_decode_gpu(*args, **kwargs):
        logger.error('No GPU detected...')

    def _acausal_decode_gpu(*args, **kwargs):
        logger.error('No GPU detected...')

    def _causal_classify_gpu(*args, **kwargs):
        logger.error('No GPU detected...')

    def _acausal_classify_gpu(*args, **kwargs):
        logger.error('No GPU detected...')
