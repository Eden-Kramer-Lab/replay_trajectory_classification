import math

import numpy as np
from numba import cuda
from numba.types import float32
from replay_trajectory_classification.bins import atleast_2d

# Precompute this constant as a float32.  Numba will inline it at compile time.
SQRT_2PI = np.float32((2 * math.pi)**0.5)


@cuda.jit(device=True)
def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with
    given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)


@cuda.jit
def kde1(eval_points, samples, bandwidths, out):
    """

    Parameters
    ----------
    eval_points : ndarray, shape (n_eval_points, n_bandwidths)
    samples : ndarray, shape (n_samples, n_bandwidths)
    bandwidths : ndarray, shape (n_bandwidths,)
    out : ndarray, shape (n_eval_points,)

    """
    n_bandwidths = len(bandwidths)
    n_samples = len(samples)
    n_eval_points = len(eval_points)

    for thread_id in range(cuda.grid(1), n_eval_points, cuda.gridsize(1)):
        sum_kernel = float32(0.0)
        for sample_ind in range(n_samples):
            product_kernel = float32(1.0)
            for bandwidth_ind in range(n_bandwidths):
                product_kernel *= gaussian_pdf(
                    eval_points[thread_id, bandwidth_ind],
                    samples[sample_ind, bandwidth_ind],
                    bandwidths[bandwidth_ind]
                )
            sum_kernel += product_kernel

        out[thread_id] = sum_kernel / n_samples


def get_marks_by_place_bin_centers(marks, place_bin_centers):
    """

    Parameters
    ----------
    marks : ndarray, shape (n_spikes, n_features)
    place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)

    Returns
    -------
    marks_by_place_bin_centers : ndarray, shape (n_spikes * n_position_bins,
                                                 n_features + n_position_dims)

    """
    n_spikes = marks.shape[0]
    n_place_bin_centers = place_bin_centers.shape[0]
    return np.concatenate(
        (np.tile(marks, reps=(n_place_bin_centers, 1)),
         np.repeat(place_bin_centers, n_spikes, axis=0)), axis=1)


def estimate_log_intensity(density, occupancy, mean_rate):
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
    return np.log(mean_rate) + np.log(density) - np.log(occupancy)


def estimate_pdf(
    decoding_marks, encoding_marks, mark_std,
    place_bin_centers, encoding_positions, position_std, stream=0
):
    '''

    Parameters
    ----------
    decoding_marks : ndarray, shape (n_decoding_spikes, n_marks)
    encoding_marks : ndarray, shape (n_encoding_spikes, n_marks)
    mark_std : float
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    encoding_positions : ndarray, shape (n_encoding_spikes, n_position_dims)
    position_std : float
    stream : numba.cuda.stream, optional

    Returns
    -------
    joint_mark_intensity : ndarray, shape (n_decoding_spikes, n_bins)

    '''
    decoding_marks = np.atleast_2d(decoding_marks)

    # Copy the arrays to the GPU
    eval_points = cuda.to_device(
        get_marks_by_place_bin_centers(decoding_marks, place_bin_centers)
        .astype(np.float32),
        stream=stream)
    encoding_samples = cuda.to_device(
        np.concatenate((encoding_marks, encoding_positions), axis=1)
        .astype(np.float32), stream=stream)

    n_marks = decoding_marks.shape[1]
    n_position_dims = place_bin_centers.shape[1]
    bandwidths = cuda.to_device(
        np.concatenate(
            ([mark_std] * n_marks,
             [position_std] * n_position_dims,
             )
        ).astype(np.float32), stream=stream)
    n_eval_points = len(eval_points)
    # Allocate memory on the GPU for the result
    pdf = cuda.device_array(
        shape=(n_eval_points,), dtype=np.float32, stream=stream)

    # Run KDE
    kde1.forall(n_eval_points, stream=stream)(
        eval_points, encoding_samples, bandwidths, pdf)

    return pdf


def estimate_multiunit_likelihood_gpu(multiunits,
                                      encoding_marks,
                                      mark_std,
                                      place_bin_centers,
                                      encoding_positions,
                                      position_std,
                                      occupancy,
                                      mean_rates,
                                      summed_ground_process_intensity,
                                      is_track_interior=None,
                                      time_bin_size=1,
                                      n_streams=16):
    '''

    Parameters
    ----------
    multiunits : ndarray, shape (n_time, n_marks, n_electrodes)
    encoding_marks : ndarray, shape (n_encoding_spikes, n_marks)
    mark_std : float
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    encoding_positions : ndarray, shape (n_encoding_spikes, n_position_dims)
    position_std : float
    occupancy : ndarray, (n_bins,)
    mean_rates : ndarray, (n_electrodes,)
    summed_ground_process_intensity : ndarray, shape (n_bins,)

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
    n_electrodes = multiunits.shape[-1]
    multiunits = np.moveaxis(multiunits, -1, 0)
    streams = [cuda.stream() for _ in range(min(n_streams, n_electrodes))]
    pdfs = []
    is_spikes = []

    with cuda.defer_cleanup():
        for elec_ind, (multiunit, enc_marks, enc_pos) in enumerate(zip(
                multiunits, encoding_marks, encoding_positions)):
            is_spike = np.any(~np.isnan(multiunit), axis=1)
            is_spikes.append(is_spike)
            pdfs.append(estimate_pdf(
                multiunit[is_spike],
                enc_marks,
                mark_std,
                place_bin_centers[is_track_interior],
                enc_pos,
                position_std,
                stream=streams[elec_ind % n_streams]
            ))

        n_interior_place_bins = is_track_interior.sum()
        for elec_ind, (pdf, mean_rate, is_spike) in enumerate(
                zip(pdfs, mean_rates, is_spikes)):
            n_spikes = is_spike.sum()
            # Copy results from GPU to CPU and
            # reshape to (n_decoding_spikes, n_interior_place_bins)
            pdf = (pdf
                   .copy_to_host(stream=streams[elec_ind % n_streams])
                   .reshape((n_spikes, n_interior_place_bins), order='F'))
            log_likelihood[np.ix_(is_spike, is_track_interior)] += (
                estimate_log_intensity(pdf, occupancy[is_track_interior],
                                       mean_rate) +
                np.finfo(np.float32).eps)

    log_likelihood[:, ~is_track_interior] = np.nan

    return log_likelihood


def estimate_position_density(place_bin_centers, positions, position_std):
    # Copy the arrays to the device
    eval_points = cuda.to_device(place_bin_centers.astype(np.float32))
    samples = cuda.to_device(positions.astype(np.float32))
    n_position_dims = positions.shape[1]
    bandwidths = cuda.to_device(
        np.asarray([position_std] * n_position_dims).astype(np.float32))

    # Allocate memory on the device for the result
    n_eval_points = len(eval_points)
    out = cuda.device_array((n_eval_points,), dtype=np.float32)

    kde1.forall(n_eval_points)(
        eval_points, samples, bandwidths, out)

    return out.copy_to_host()


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


def fit_multiunit_likelihood_gpu(position,
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
    mark_std : float
    position_std : float
    is_track_interior : None or ndarray, shape (n_bins,)

    Returns
    -------
    encoding_model : dict

    '''
    if is_track_interior is None:
        is_track_interior = np.ones((place_bin_centers.shape[0],),
                                    dtype=np.bool)
    position = atleast_2d(position)
    place_bin_centers = atleast_2d(place_bin_centers)

    not_nan_position = np.all(~np.isnan(position), axis=1)

    occupancy = np.zeros((place_bin_centers.shape[0],), dtype=np.float32)
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
        is_spike = np.any(~np.isnan(multiunit), axis=1)
        mean_rates.append(is_spike.mean())
        marginal_density = np.zeros(
            (place_bin_centers.shape[0],), dtype=np.float32)

        if is_spike.sum() > 0:
            marginal_density[is_track_interior] = estimate_position_density(
                place_bin_centers[is_track_interior],
                position[is_spike & not_nan_position], position_std)

        ground_process_intensities.append(
            estimate_intensity(marginal_density, occupancy, mean_rates[-1])
            + np.finfo(np.float32).eps)

        encoding_marks.append(
            multiunit[is_spike & not_nan_position].astype(int))
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
