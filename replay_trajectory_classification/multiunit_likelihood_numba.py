import math as math

import numba
import numpy as np

SQRT_2PI = np.sqrt(2.0 * np.pi)


@numba.vectorize(['float64(float64, float64, float64)'], nopython=True)
def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with
    given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)


@numba.njit(parallel=True, fastmath=True, nogil=True)
def product_kde2(eval_points, samples, bandwidths):
    n_samples, n_bandwidths = samples.shape
    n_eval = eval_points.shape[0]

    assert(eval_points.shape[1] == samples.shape[1])
    assert(eval_points.shape[1] == bandwidths.shape[0])

    results = np.zeros((n_eval,))

    for eval_ind in numba.prange(n_eval):
        eval_point = eval_points[eval_ind]
        for sample_ind in range(n_samples):
            product_kde = 1.0
            sample = samples[sample_ind]
            for bandwidth_ind in range(n_bandwidths):
                bandwidth = bandwidths[bandwidth_ind]
                product_kde *= (
                    np.exp(-0.5 *
                           ((eval_point[bandwidth_ind] - sample[bandwidth_ind])
                            / bandwidth)**2) / (bandwidth * SQRT_2PI)
                    / bandwidth)
            results[eval_ind] += product_kde
        results[eval_ind] /= n_samples

    return results


@numba.njit(parallel=True, fastmath=True)
def product_kde(eval_points, samples, bandwidths):
    result = 0
    n_samples = len(samples)

    for sample_ind in numba.prange(n_samples):
        result += np.prod(
            gaussian_pdf(eval_points, samples[sample_ind], bandwidths)
        )

    return result / (n_samples * np.prod(bandwidths))


@numba.njit(parallel=True, nogil=True)
def gauss(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with
    given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)


@numba.njit(parallel=True, nogil=True)
def kde(test_mark, test_position, train_marks, train_position,
        place_bandwidth, mark_bandwidth, n_marks, n_position_dims,
        n_samples):
    joint_density = 0.0

    for sample_ind in range(n_samples):
        product_kde = 1.0
        for x, mean in zip(test_mark, train_marks[sample_ind]):
            product_kde *= gauss(x, mean, mark_bandwidth) / mark_bandwidth

        for x, mean in zip(test_position, train_position[sample_ind]):
            product_kde *= gauss(x, mean, mark_bandwidth) / mark_bandwidth
        joint_density += product_kde

    return joint_density / n_samples


@numba.njit(parallel=True, nogil=True)
def get_log_likelihood(marks, place_bin_centers, occupancy,
                       ground_intensity, train_marks, train_positions,
                       mean_rates, place_bandwidth=2.0, mark_bandwidth=20.0):
    '''

    Parameters
    ----------
    marks : ndarray, shape (n_time, n_marks, n_tetrodes, )
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    bandwidths : ndarray, shape (n_marks + n_position_dims, )
    occupancy : ndarray, shape (n_bins,)
    ground_intensity : ndarray, shape (n_tetrodes, n_bins)
    train_data : list of ndarray, shape (n_samples, n_marks +
                                            n_position_dims)
    mean_rates : ndarray, shape (n_tetrodes,)

    Returns
    -------
    log_likelihood : ndarray, shape (n_time, n_bins)

    '''
    n_time, n_marks, n_tetrodes = marks.shape
    n_bins, n_position_dims = place_bin_centers.shape
    log_likelihood = np.zeros((n_time, n_bins))

    for time_ind in numba.prange(n_time):
        for tetrode_ind in numba.prange(n_tetrodes):
            is_spike = np.any(~np.isnan(marks[time_ind, :, tetrode_ind]))
            if is_spike:
                test_mark = marks[time_ind, :, tetrode_ind]
                train_mark = train_marks[tetrode_ind]
                mean_rate = np.log(mean_rates[tetrode_ind])
                n_train = train_marks[tetrode_ind].shape[0]

                for bin_ind in numba.prange(n_bins):
                    position_bin = place_bin_centers[bin_ind, :]
                    log_likelihood[time_ind, bin_ind] += np.log(kde(
                        test_mark, position_bin, train_mark,
                        train_positions[tetrode_ind], place_bandwidth,
                        mark_bandwidth, n_marks, n_position_dims, n_train))

                    log_likelihood[time_ind, bin_ind] += mean_rate
                    log_likelihood[time_ind, bin_ind] -= np.log(
                        occupancy[bin_ind] + np.spacing(1))
                    log_likelihood[time_ind, bin_ind] -= (
                        ground_intensity[tetrode_ind][bin_ind])

            else:
                for bin_ind in range(n_bins):
                    log_likelihood[time_ind, bin_ind] -= (
                        ground_intensity[tetrode_ind][bin_ind])

    return log_likelihood
