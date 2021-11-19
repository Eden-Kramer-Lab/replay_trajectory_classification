import numpy as np
from replay_trajectory_classification.core import scaled_likelihood
from replay_trajectory_classification.likelihoods.multiunit_likelihood import (
    estimate_intensity, poisson_mark_log_likelihood)
from scipy.special import cotdg
from scipy.stats import rv_histogram
from skimage.transform import radon
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


def predict_mark_likelihood(
    start_time,
    end_time,
    place_bin_centers,
    occupancy,
    joint_pdf_models,
    multiunit_dfs,
    ground_process_intensities,
    mean_rates,
    is_track_interior,
    dt=0.020,
):
    n_time_bins = np.ceil((end_time - start_time) / dt).astype(int)
    time_bin_edges = start_time + np.arange(n_time_bins + 1) * dt
    n_place_bins = len(place_bin_centers)

    log_likelihood = np.zeros((n_time_bins, n_place_bins))
    interior_bin_inds = np.nonzero(is_track_interior)[0]

    for joint_model, multiunit_df, gpi, mean_rate in zip(
        joint_pdf_models, multiunit_dfs, ground_process_intensities, mean_rates
    ):
        time_index = np.searchsorted(
            time_bin_edges, multiunit_df.index.total_seconds())
        in_time_bins = np.nonzero(
            ~np.isin(time_index, [0, len(time_bin_edges)]))[0]
        time_index = time_index[in_time_bins] - 1
        multiunit_df = multiunit_df.iloc[in_time_bins, :4]
        multiunit_df["time_bin_ind"] = time_index

        n_spikes = multiunit_df.shape[0]
        joint_mark_intensity = np.ones((n_spikes, n_place_bins))

        if n_spikes > 0:
            zipped = zip(
                interior_bin_inds,
                place_bin_centers[interior_bin_inds],
                occupancy[interior_bin_inds],
            )
            for bin_ind, bin, bin_occupancy in zipped:
                marks_pos = np.asarray(multiunit_df.iloc[:, :4])
                marks_pos = np.concatenate(
                    (marks_pos, bin * np.ones((n_spikes, 1))), axis=1
                )
                joint_mark_intensity[:, bin_ind] = estimate_intensity(
                    np.exp(joint_model.score_samples(marks_pos)),
                    bin_occupancy,
                    mean_rate,
                )

            tetrode_likelihood = poisson_mark_log_likelihood(
                joint_mark_intensity, np.atleast_2d(gpi)
            )
            for time_bin_ind in np.unique(time_index):
                log_likelihood[time_bin_ind] += np.sum(
                    tetrode_likelihood[time_index == time_bin_ind], axis=0
                )

    mask = np.ones_like(is_track_interior, dtype=np.float)
    mask[~is_track_interior] = np.nan

    log_likelihood = log_likelihood * mask

    time = np.arange(n_time_bins) * dt

    return scaled_likelihood(log_likelihood), time


def predict_poisson_likelihood(start_time, end_time, spike_times, place_fields,
                               is_track_interior, dt=0.020):
    place_fields = np.asarray(place_fields)
    n_time_bins = np.ceil((end_time - start_time) / dt).astype(int)
    time_bin_edges = start_time + np.arange(n_time_bins + 1) * dt
    time_bin_centers = time_bin_edges[:-1] + np.diff(time_bin_edges) / 2

    spike_time_ind, neuron_ind = [], []
    for ind, times in enumerate(spike_times):
        is_valid_time = (times >= start_time) & (times <= end_time)
        inds = np.digitize(times[is_valid_time], time_bin_edges[1:-1])
        spike_time_ind.append(inds)
        neuron_ind.append(np.ones_like(inds) * ind)

    neuron_ind = np.concatenate(neuron_ind)
    spike_time_ind = np.concatenate(spike_time_ind)

    log_likelihood = np.stack(
        [np.sum(np.log(place_fields[:, neuron_ind[spike_time_ind == time_bin]] +
                       np.spacing(1)), axis=1)
         for time_bin in np.arange(n_time_bins)])
    log_likelihood -= dt * np.sum(place_fields, axis=1)

    mask = np.ones_like(is_track_interior, dtype=np.float)
    mask[~is_track_interior] = np.nan

    return scaled_likelihood(log_likelihood) * mask, time_bin_centers


def normalize_to_posterior(likelihood, prior=None):
    if prior is None:
        n_position_bins = likelihood.shape[1]
        prior = np.ones_like(likelihood) / n_position_bins
    posterior = likelihood * prior
    return posterior / np.nansum(posterior, axis=1, keepdims=True)


def convert_polar_to_slope_intercept(
    n_pixels_from_center, projection_angle, center_pixel
):
    slope = -cotdg(-projection_angle)
    intercept = (
        n_pixels_from_center / np.sin(-np.deg2rad(projection_angle))
        - slope * center_pixel[0]
        + center_pixel[1]
    )
    return intercept, slope


def detect_line_with_radon(
    posterior,
    dt,  # s
    dp,  # cm
    projection_angles=np.arange(-90, 90, 0.5),  # degrees
    filter_invalid_positions=True,
    incorporate_nearby_positions=True,
    nearby_positions_max=15,  # cm
):

    if incorporate_nearby_positions:
        n_nearby_bins = int(nearby_positions_max / 2 // dp)
        filt = np.ones(2 * n_nearby_bins + 1)
        posterior = np.apply_along_axis(
            lambda time_bin: np.convolve(time_bin, filt, mode="same"),
            axis=1, arr=posterior
        )
    else:
        n_nearby_bins = 1
    # Sinogram is shape (pixels_from_center, projection_angles)
    sinogram = radon(
        posterior.T, theta=projection_angles, circle=False,
        preserve_range=False
    )
    n_time, n_position_bins = posterior.shape
    center_pixel = np.asarray((n_time // 2, n_position_bins // 2))
    pixels_from_center = np.arange(
        -sinogram.shape[0] // 2 + 1, sinogram.shape[0] // 2 + 1)

    if filter_invalid_positions:
        start_positions, velocities = convert_polar_to_slope_intercept(
            pixels_from_center[:, np.newaxis],
            projection_angles[np.newaxis, :],
            center_pixel,
        )
        end_positions = start_positions + velocities * (n_time - 1)
        sinogram[(start_positions < 0) |
                 (start_positions > n_position_bins - 1)] = 0.0
        sinogram[(end_positions < 0) |
                 (end_positions > n_position_bins - 1)] = 0.0
        sinogram[:, np.isinf(velocities.squeeze())] = 0.0

    # Find the maximum of the sinogram
    n_pixels_from_center_ind, projection_angle_ind = np.unravel_index(
        indices=np.argmax(sinogram), shape=sinogram.shape
    )
    projection_angle = projection_angles[projection_angle_ind]
    n_pixels_from_center = pixels_from_center[n_pixels_from_center_ind]

    # Normalized score based on the integrated projection
    score = np.max(sinogram) / (n_time * n_nearby_bins)

    # Convert from polar form to slope-intercept form
    start_position, velocity = convert_polar_to_slope_intercept(
        n_pixels_from_center, projection_angle, center_pixel
    )

    # Convert from pixels to position units
    start_position *= dp
    velocity *= dp / dt

    # Estimate position for the posterior
    time = np.arange(n_time) * dt
    radon_position = start_position + velocity * time

    return start_position, velocity, radon_position, score


def map_estimate(posterior, place_bin_centers):
    posterior[np.isnan(posterior)] = 0.0
    return place_bin_centers[posterior.argmax(axis=1)].squeeze()


def _m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def _cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - _m(x, w)) * (y - _m(y, w))) / np.sum(w)


def _corr(x, y, w):
    """Weighted Correlation"""
    return _cov(x, y, w) / np.sqrt(_cov(x, x, w) * _cov(y, y, w))


def weighted_correlation(posterior, time, place_bin_centers):
    place_bin_centers = place_bin_centers.squeeze()
    posterior[np.isnan(posterior)] = 0.0

    return _corr(time[:, np.newaxis],
                 place_bin_centers[np.newaxis, :], posterior)


def isotonic_regression(posterior, time, place_bin_centers):
    place_bin_centers = place_bin_centers.squeeze()
    posterior[np.isnan(posterior)] = 0.0

    map = map_estimate(posterior, place_bin_centers)
    map_probabilities = np.max(posterior, axis=1)

    regression = IsotonicRegression(increasing='auto').fit(
        X=time,
        y=map,
        sample_weight=map_probabilities,
    )

    score = regression.score(
        X=time,
        y=map,
        sample_weight=map_probabilities,
    )

    prediction = regression.predict(time)

    return prediction, score


def _sample_posterior(posterior, place_bin_edges, n_samples=1000):
    """Samples the posterior positions.

    Parameters
    ----------
    posterior : np.array, shape (n_time, n_position_bins)

    Returns
    -------
    posterior_samples : numpy.ndarray, shape (n_time, n_samples)

    """

    place_bin_edges = place_bin_edges.squeeze()
    n_time = posterior.shape[0]

    posterior_samples = [
        rv_histogram((posterior[time_ind], place_bin_edges)).rvs(
            size=n_samples)
        for time_ind in range(n_time)
    ]

    return np.asarray(posterior_samples)


def linear_regression(posterior, place_bin_edges, time, n_samples=1000):
    posterior[np.isnan(posterior)] = 0.0
    samples = _sample_posterior(
        posterior, place_bin_edges, n_samples=n_samples
    )
    design_matrix = np.tile(time, n_samples)[:, np.newaxis]
    response = samples.ravel(order="F")
    regression = LinearRegression().fit(X=design_matrix, y=response)

    r2 = regression.score(X=design_matrix, y=response)
    slope = regression.coef_[0]
    intercept = regression.intercept_
    prediction = regression.predict(time[:, np.newaxis])

    return intercept, slope, r2, prediction
