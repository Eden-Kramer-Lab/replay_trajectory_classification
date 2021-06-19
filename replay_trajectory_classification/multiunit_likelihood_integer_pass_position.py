import math as math

import dask
import dask.array as da
import numba
import numpy as np
from replay_trajectory_classification.bins import atleast_2d

SQRT_2PI = np.sqrt(2.0 * np.pi)


@numba.vectorize(['float64(float64, float64, float64)'], nopython=True,
                 cache=True)
def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with
    given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)


def estimate_position_distance(place_bin_centers, positions, position_std):
    return np.prod(
        gaussian_pdf(
            np.expand_dims(place_bin_centers, axis=0),
            np.expand_dims(positions, axis=1),
            position_std),
        axis=-1
    )


def estimate_position_density(place_bin_centers, positions, position_std):
    return np.mean(estimate_position_distance(
        place_bin_centers, positions, position_std), axis=0)


def estimate_log_intensity(density, occupancy, mean_rate):
    return np.log(mean_rate) + np.log(density) - np.log(occupancy)


def estimate_intensity(density, occupancy, mean_rate):
    '''

    Parameters
    ----------
    density : ndarray, shape (n_bins,)
    occupancy : ndarray, shape (n_bins,)
    mean_rate : float

    Returns
    -------
    intensity : ndarray, shape (n_bins,)

    '''
    return np.exp(estimate_log_intensity(density, occupancy, mean_rate))


def normal_pdf_integer_lookup(x, mean, std=20, max_value=3000):
    """Fast density evaluation for integers by precomputing a hash table of
    values.

    Parameters
    ----------
    x : int
    mean : int
    std : float
    max_value : int

    Returns
    -------
    probability_density : int

    """
    normal_density = gaussian_pdf(np.arange(-max_value, max_value), 0, std)

    return normal_density[(x - mean) + max_value]


def estimate_log_joint_mark_intensity(decoding_marks,
                                      encoding_marks,
                                      mark_std,
                                      place_bin_centers,
                                      encoding_positions,
                                      position_std,
                                      occupancy,
                                      mean_rate,
                                      max_mark_value=6000,
                                      set_diag_zero=False,
                                      position_distance=None):
    """

    Parameters
    ----------
    decoding_marks : ndarray, shape (n_decoding_spikes, n_features)
    encoding_marks : ndarray, shape (n_encoding_spikes, n_features)
    mark_std : float or ndarray, shape (n_features,)
    place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)
    encoding_positions : ndarray, shape (n_decoding_spikes, n_position_dims)
    position_std : float or ndarray, shape (n_position_dims,)
    occupancy : ndarray, shape (n_position_bins,)
    mean_rate : float
    is_track_interior : None or ndarray, shape (n_position_bins,)
    max_mark_value : int
    set_diag_zero : bool

    Returns
    -------
    log_joint_mark_intensity : ndarray, shape (n_decoding_spikes, n_position_bins)

    """
    # mark_distance: ndarray, shape (n_decoding_spikes, n_encoding_spikes)
    mark_distance = np.prod(
        normal_pdf_integer_lookup(
            np.expand_dims(decoding_marks, axis=1),
            np.expand_dims(encoding_marks, axis=0),
            std=mark_std,
            max_value=max_mark_value
        ),
        axis=-1
    )

    if set_diag_zero:
        diag_ind = np.diag_indices_from(mark_distance)
        mark_distance[diag_ind] = 0.0

    n_encoding_spikes = encoding_marks.shape[0]

    if position_distance is None:
        position_distance = estimate_position_distance(
            place_bin_centers, encoding_positions, position_std)

    return estimate_log_intensity(
        mark_distance @ position_distance / n_encoding_spikes,
        occupancy,
        mean_rate)


def fit_multiunit_likelihood_integer_pass_position(position,
                                                   multiunits,
                                                   place_bin_centers,
                                                   mark_std,
                                                   position_std,
                                                   is_track_interior=None,
                                                   **kwargs):
    '''

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    multiunits : ndarray, shape (n_time, n_marks, n_electrodes)
    place_bin_centers : ndarray, shape ( n_bins, n_position_dims)
    model : sklearn model
    model_kwargs : dict
    occupancy_model : sklearn model
    occupancy_kwargs : dict
    is_track_interior : None or ndarray, shape (n_bins,)

    Returns
    -------
    joint_pdf_models : list of sklearn models, shape (n_electrodes,)
    ground_process_intensities : list of ndarray, shape (n_electrodes,)
    occupancy : ndarray, (n_bins, n_position_dims)
    mean_rates : ndarray, (n_electrodes,)

    '''
    if is_track_interior is None:
        is_track_interior = np.ones((place_bin_centers.shape[0],),
                                    dtype=np.bool)
    position = atleast_2d(position)
    place_bin_centers = atleast_2d(place_bin_centers)

    not_nan_position = np.all(~np.isnan(position), axis=1)

    occupancy = np.zeros((place_bin_centers.shape[0],))
    occupancy[is_track_interior] = estimate_position_density(
        place_bin_centers[is_track_interior],
        position[not_nan_position],
        position_std)

    mean_rates = []
    ground_process_intensities = []
    encoding_marks = []
    encoding_positions = []
    position_at_spike_distances = []

    for multiunit in np.moveaxis(multiunits, -1, 0):

        # ground process intensity
        is_spike = np.any(~np.isnan(multiunit), axis=1)
        mean_rates.append(is_spike.mean())
        marginal_density = np.zeros((place_bin_centers.shape[0],))
        position_at_spike_distance = estimate_position_distance(
            place_bin_centers[is_track_interior],
            position[is_spike & not_nan_position],
            position_std)

        if is_spike.sum() > 0:
            marginal_density[is_track_interior] = np.mean(
                position_at_spike_distance, axis=0)

        position_at_spike_distances.append(position_at_spike_distance)
        ground_process_intensities.append(
            estimate_intensity(marginal_density, occupancy, mean_rates[-1])
            + np.spacing(1))

        encoding_marks.append(
            multiunit[is_spike & not_nan_position].astype(int))
        encoding_positions.append(position[is_spike & not_nan_position])

    summed_ground_process_intensity = np.sum(
        np.stack(ground_process_intensities, axis=0), axis=0, keepdims=True)

    return {
        'encoding_marks': encoding_marks,
        'encoding_positions': encoding_positions,
        'summed_ground_process_intensity': summed_ground_process_intensity,
        'position_at_spike_distances': position_at_spike_distances,
        'occupancy': occupancy,
        'mean_rates': mean_rates,
        'mark_std': mark_std,
        'position_std': position_std,
        **kwargs,
    }


def estimate_multiunit_likelihood_integer_pass_position(multiunits,
                                                        encoding_marks,
                                                        mark_std,
                                                        place_bin_centers,
                                                        encoding_positions,
                                                        position_std,
                                                        position_at_spike_distances,
                                                        occupancy,
                                                        mean_rates,
                                                        summed_ground_process_intensity,
                                                        max_mark_value=6000,
                                                        set_diag_zero=False,
                                                        is_track_interior=None,
                                                        time_bin_size=1,
                                                        chunks=None):
    '''

    Parameters
    ----------
    multiunits : ndarray, shape (n_time, n_marks, n_electrodes)
    place_bin_centers : ndarray, (n_bins, n_position_dims)
    joint_pdf_models : list of sklearn models, shape (n_electrodes,)
    ground_process_intensities : list of ndarray, shape (n_electrodes,)
    occupancy : ndarray, (n_bins, n_position_dims)
    mean_rates : ndarray, (n_electrodes,)

    Returns
    -------
    log_likelihood : (n_time, n_bins)

    '''

    if is_track_interior is None:
        is_track_interior = np.ones((place_bin_centers.shape[0],),
                                    dtype=np.bool)

    n_time = multiunits.shape[0]
    log_likelihood = (-time_bin_size * summed_ground_process_intensity *
                      np.ones((n_time, 1)))

    multiunits = np.moveaxis(multiunits, -1, 0)
    log_joint_mark_intensities = []

    for multiunit, enc_marks, enc_pos, mean_rate, pos_at_spike_dist in zip(
            multiunits, encoding_marks, encoding_positions, mean_rates,
            position_at_spike_distances):
        is_spike = np.any(~np.isnan(multiunit), axis=1)
        decoding_marks = da.from_array(
            multiunit[is_spike].astype(np.int))
        log_joint_mark_intensities.append(
            decoding_marks.map_blocks(
                estimate_log_joint_mark_intensity,
                enc_marks,
                mark_std,
                place_bin_centers[is_track_interior],
                enc_pos,
                position_std,
                occupancy[is_track_interior],
                mean_rate,
                position_distance=da.from_array(pos_at_spike_dist),
                max_mark_value=max_mark_value,
                set_diag_zero=set_diag_zero,
                chunks=chunks
            ))

    for log_joint_mark_intensity, multiunit in zip(
            dask.compute(*log_joint_mark_intensities), multiunits):
        is_spike = np.any(~np.isnan(multiunit), axis=1)
        log_likelihood[np.ix_(is_spike, is_track_interior)] += (
            log_joint_mark_intensity + np.spacing(1))

    log_likelihood[:, ~is_track_interior] = np.nan

    return log_likelihood
