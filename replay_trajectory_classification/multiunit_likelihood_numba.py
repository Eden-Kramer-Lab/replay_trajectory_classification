import math as math

import numba
import numpy as np

SQRT_2PI = np.float64(math.sqrt(2.0 * math.pi))


@numba.vectorize(['float64(float64, float64, float64)'], nopython=True)
def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with
    given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)


def product_kde(eval_points, samples, bandwidths):
    result = 0
    n_samples = len(samples)

    for sample_ind in range(n_samples):
        result += np.prod(
            gaussian_pdf(eval_points, samples[sample_ind], bandwidths)
        )

    return result / (n_samples * np.prod(bandwidths))


def get_likelihood(marks, place_bin_centers, bandwidths, occupancy,
                   ground_intensity, training_data, mean_rates):
    '''

    Parameters
    ----------
    marks : ndarray, shape (n_time, n_tetrodes, n_marks)
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    bandwidths : ndarray, shape (n_marks + n_position_dims, )
    occupancy : ndarray, shape (n_bins,)
    ground_intensity : ndarray, shape (n_tetrodes, n_bins)
    training_data : list of ndarray, shape (n_samples, n_marks +
                                            n_position_dims)
    mean_rates : ndarray, shape (n_tetrodes,)

    Returns
    -------
    likelihood : ndarray, shape (n_time, n_bins)

    '''
    n_time, n_tetrodes, _ = marks.shape
    n_bins = place_bin_centers.shape[0]
    likelihood = np.zeros((n_time, n_bins))

    for time_ind in range(n_time):
        for bin_ind in range(n_bins):
            for tetrode_ind in range(n_tetrodes):
                if ~np.all(np.isnan(marks[time_ind, tetrode_ind])):
                    eval_points = np.concatenate(
                        (marks[time_ind, tetrode_ind],
                         place_bin_centers[bin_ind]))
                    joint_mark_pdf = product_kde(
                        eval_points, training_data[tetrode_ind], bandwidths)
                else:
                    joint_mark_pdf = 1.0

                likelihood[time_ind, bin_ind] *= (
                    (mean_rates[tetrode_ind] * joint_mark_pdf /
                     occupancy[bin_ind]) *
                    math.exp(-ground_intensity[tetrode_ind, bin_ind])
                )
    return likelihood
