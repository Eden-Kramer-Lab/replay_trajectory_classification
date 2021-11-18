import itertools
import warnings

import dask
import dask.array as da
import numpy as np
from replay_trajectory_classification.bins import atleast_2d

SQRT_2PI = np.sqrt(2.0 * np.pi)


def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with
    given mean and sigma.'''
    return np.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)


def estimate_position_distance(place_bin_centers, positions, position_std):
    return da.prod(
        gaussian_pdf(
            place_bin_centers[np.newaxis],
            positions[:, np.newaxis],
            position_std),
        axis=-1
    )


def estimate_position_density(place_bin_centers, positions, position_std):
    return da.mean(estimate_position_distance(
        place_bin_centers, positions, position_std), axis=0)


def estimate_log_intensity(density, occupancy, mean_rate):
    return da.log(mean_rate) + da.log(density) - da.log(occupancy)


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
    return da.exp(estimate_log_intensity(density, occupancy, mean_rate))


def normal_pdf_integer_lookup(x, mean, std=20.0, max_value=3000):
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
    normal_density = da.from_array(
        gaussian_pdf(np.arange(-max_value, max_value), 0.0, std))
    diff = (x - mean) + max_value
    # Look up each diff value, flatten and reshape because
    # dask doesn't allow integer indexing in more than one dimension
    return normal_density[diff.reshape(-1)].reshape(diff.shape)


def estimate_log_joint_mark_intensity(decoding_marks,
                                      encoding_marks,
                                      mark_std,
                                      place_bin_centers,
                                      encoding_positions,
                                      position_std,
                                      occupancy,
                                      mean_rate,
                                      max_mark_value=6000,
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
    n_encoding_spikes = encoding_marks.shape[0]
    n_decoding_spikes = decoding_marks.shape[0]

    if (n_decoding_spikes > 0) & (n_encoding_spikes > 0):
        # mark_distance: ndarray, shape (n_decoding_spikes, n_encoding_spikes)
        pdf = normal_pdf_integer_lookup(
            decoding_marks[:, np.newaxis],
            encoding_marks[np.newaxis],
            std=mark_std,
            max_value=max_mark_value)
        mark_distance = da.prod(pdf, axis=-1)

        if decoding_marks is encoding_marks:
            np.fill_diagonal(mark_distance, 0.0)

        if position_distance is None:
            position_distance = estimate_position_distance(
                place_bin_centers, encoding_positions, position_std)

        return estimate_log_intensity(
            mark_distance @ position_distance / n_encoding_spikes,
            occupancy,
            mean_rate)
    else:
        n_position_bins = place_bin_centers.shape[0]
        return np.empty((n_decoding_spikes, n_position_bins))

def fit_multiunit_likelihood_integer(position,
                                     multiunits,
                                     place_bin_centers,
                                     mark_std,
                                     position_std,
                                     is_track_interior=None,
                                     dtype=np.int64,
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

    for multiunit in np.moveaxis(multiunits, -1, 0):

        # ground process intensity
        nan_multiunit = np.isnan(multiunit)
        is_spike = np.any(~nan_multiunit, axis=1)
        nan_mark_dims = np.all(nan_multiunit, axis=0)

        mean_rates.append(is_spike.mean())
        marginal_density = np.zeros((place_bin_centers.shape[0],))

        if is_spike.sum() > 0:
            marginal_density[is_track_interior] = estimate_position_density(
                place_bin_centers[is_track_interior],
                position[is_spike & not_nan_position], position_std)

        ground_process_intensities.append(
            estimate_intensity(marginal_density, occupancy, mean_rates[-1])
            + np.spacing(1))

        encoding_marks.append(
            multiunit[np.ix_(is_spike & not_nan_position, ~nan_mark_dims)
                      ].astype(dtype))
        encoding_positions.append(position[is_spike & not_nan_position])

    summed_ground_process_intensity = np.sum(
        np.stack(ground_process_intensities, axis=0), axis=0, keepdims=True)

    return {
        'encoding_marks': encoding_marks,
        'encoding_positions': encoding_positions,
        'summed_ground_process_intensity': summed_ground_process_intensity,
        'occupancy': occupancy,
        'mean_rates': mean_rates,
        'mark_std': mark_std,
        'position_std': position_std,
        **kwargs,
    }


def estimate_multiunit_likelihood_integer(multiunits,
                                          encoding_marks,
                                          mark_std,
                                          place_bin_centers,
                                          encoding_positions,
                                          position_std,
                                          occupancy,
                                          mean_rates,
                                          summed_ground_process_intensity,
                                          max_mark_value=6000,
                                          is_track_interior=None,
                                          time_bin_size=1,
                                          chunks=1000,
                                          dtype=np.int64):
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

    for multiunit, enc_marks, enc_pos, mean_rate in zip(
            multiunits, encoding_marks, encoding_positions, mean_rates):
        nan_multiunit = np.isnan(multiunit)
        is_spike = np.any(~nan_multiunit, axis=1)
        nan_mark_dims = np.all(nan_multiunit, axis=0)
        multiunit = multiunit[np.ix_(is_spike, ~nan_mark_dims)]

        log_joint_mark_intensities.append(
            estimate_log_joint_mark_intensity(
                da.from_array(multiunit.astype(dtype), chunks=chunks),
                da.from_array(enc_marks.astype(dtype), chunks=chunks),
                mark_std,
                da.from_array(place_bin_centers[is_track_interior]),
                da.from_array(enc_pos),
                position_std,
                da.from_array(occupancy[is_track_interior]),
                mean_rate,
                max_mark_value=max_mark_value,
            ))

    with dask.config.set(array_plugins=[warn_on_large_chunks]):
        for log_joint_mark_intensity, multiunit in zip(
                dask.compute(*log_joint_mark_intensities), multiunits):
            is_spike = np.any(~np.isnan(multiunit), axis=1)
            if (is_spike.sum() > 0) & (log_joint_mark_intensity is not None):
                # TODO: combine spikes in the same timebin using np.bincount
                log_likelihood[np.ix_(is_spike, is_track_interior)] += (
                    np.asarray(log_joint_mark_intensity) + np.spacing(1))

    log_likelihood[:, ~is_track_interior] = np.nan

    return log_likelihood


def warn_on_large_chunks(x):
    shapes = list(itertools.product(*x.chunks))
    nbytes = [x.dtype.itemsize * np.prod(shape) for shape in shapes]
    if any(nb > 1e9 for nb in nbytes):
        warnings.warn("Array contains very large chunks")
