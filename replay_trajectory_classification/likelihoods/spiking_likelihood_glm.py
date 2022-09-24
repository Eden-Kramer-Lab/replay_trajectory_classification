"""Estimates a Poisson likelihood using place fields estimated with a GLM
with a spline basis"""
import logging

import dask
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr
from dask.distributed import Client, get_client
from patsy import build_design_matrices, dmatrix
from regularized_glm import penalized_IRLS
from replay_trajectory_classification.environments import get_n_bins
from statsmodels.api import families


def make_spline_design_matrix(position, place_bin_edges, knot_spacing=10):
    """Creates a design matrix for regression with a position spline basis.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dims)
    place_bin_edges : np.ndarray, shape (n_bins, n_position_dims)

    Returns
    -------
    design_matrix : statsmodels.DesignMatrix

    """
    inner_knots = []
    for pos, edges in zip(position.T, place_bin_edges.T):
        n_points = get_n_bins(edges, bin_size=knot_spacing)
        knots = np.linspace(edges.min(), edges.max(), n_points)[1:-1]
        knots = knots[(knots > pos.min()) & (knots < pos.max())]
        inner_knots.append(knots)

    inner_knots = np.meshgrid(*inner_knots)

    n_position_dims = position.shape[1]
    data = {}
    formula = "1 + te("
    for ind in range(n_position_dims):
        formula += f"cr(x{ind}, knots=inner_knots[{ind}])"
        formula += ", "
        data[f"x{ind}"] = position[:, ind]

    formula += 'constraints="center")'

    return dmatrix(formula, data)


def make_spline_predict_matrix(design_info, place_bin_centers):
    """Make a design matrix for position bins"""
    predict_data = {}
    for ind in range(place_bin_centers.shape[1]):
        predict_data[f"x{ind}"] = place_bin_centers[:, ind]
    return build_design_matrices([design_info], predict_data)[0]


def get_firing_rate(design_matrix, results, sampling_frequency=1):
    """Predicts the firing rate given fitted model coefficents."""
    if np.any(np.isnan(results.coefficients)):
        n_time = design_matrix.shape[0]
        rate = np.zeros((n_time,))
    else:
        rate = np.exp(design_matrix @ results.coefficients) * sampling_frequency

    return rate


@dask.delayed
def fit_glm(response, design_matrix, penalty=None, tolerance=1e-5):
    """Fits a L2-penalized GLM.

    Parameters
    ----------
    response : np.ndarray, shape (n_time,)
        Calcium activity trace
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
    penalty : None or float
        L2 penalty on regression. If None, penalty is smallest possible.
    tolerance : float
        Smallest difference between iterations to consider model fitting
        converged.

    Returns
    -------
    results : tuple

    """
    if penalty is not None:
        penalty = np.ones((design_matrix.shape[1],)) * penalty
        penalty[0] = 0.0  # don't penalize the intercept
    else:
        penalty = np.finfo(np.float).eps
    return penalized_IRLS(
        design_matrix,
        response.squeeze(),
        family=families.Poisson(),
        penalty=penalty,
        tolerance=tolerance,
    )


def poisson_log_likelihood(spikes, conditional_intensity):
    """Probability of parameters given spiking at a particular time.

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time,)
        Indicator of spike or no spike at current time.
    conditional_intensity : np.ndarray, shape (n_place_bins,)
        Instantaneous probability of observing a spike

    Returns
    -------
    poisson_log_likelihood : array_like, shape (n_time, n_place_bins)

    """
    # Logarithm of the absolute value of the gamma function is always 0 when
    # spikes are 0 or 1
    return scipy.stats.poisson.logpmf(
        spikes[:, np.newaxis], conditional_intensity[np.newaxis, :] + np.spacing(1)
    )


def combined_likelihood(spikes, conditional_intensity):
    """Combines the likelihoods of all the cells.

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time, n_neurons)
    conditional_intensity : np.ndarray, shape (n_bins, n_neurons)

    """
    n_time = spikes.shape[0]
    n_bins = conditional_intensity.shape[0]
    log_likelihood = np.zeros((n_time, n_bins))

    for is_spike, ci in zip(spikes.T, conditional_intensity.T):
        log_likelihood += poisson_log_likelihood(is_spike, ci)

    return log_likelihood


def estimate_spiking_likelihood(spikes, conditional_intensity, is_track_interior=None):
    """

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time, n_neurons)
    conditional_intensity : np.ndarray, shape (n_bins, n_neurons)
    is_track_interior : None or np.ndarray, optional, shape (n_x_position_bins,
                                                             n_y_position_bins)
    Returns
    -------
    likelihood : np.ndarray, shape (n_time, n_bins)
    """
    if is_track_interior is not None:
        is_track_interior = is_track_interior.ravel(order="F")
    else:
        n_bins = conditional_intensity.shape[0]
        is_track_interior = np.ones((n_bins,), dtype=np.bool)

    log_likelihood = combined_likelihood(spikes, conditional_intensity)

    mask = np.ones_like(is_track_interior, dtype=np.float)
    mask[~is_track_interior] = np.nan

    return log_likelihood * mask


def estimate_place_fields(
    position,
    spikes,
    place_bin_centers,
    place_bin_edges,
    edges=None,
    is_track_boundary=None,
    is_track_interior=None,
    penalty=1e-1,
    knot_spacing=10,
):
    """Gives the conditional intensity of the neurons' spiking with respect to
    position.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dims)
    spikes : np.ndarray, shape (n_time, n_neurons)
    place_bin_centers : np.ndarray, shape (n_bins, n_position_dims)
    place_bin_edges : np.ndarray, shape (n_bins + 1, n_position_dims)
    is_track_boundary : None or np.ndarray
    is_track_interior : None or np.ndarray
    penalty : None or float, optional
        L2 penalty on regression. If None, penalty is smallest possible.
    knot_spacing : int, optional
        Spacing of position knots. Controls how smooth the firing rate is.

    Returns
    -------
    conditional_intensity : np.ndarray, shape (n_bins, n_neurons)

    """
    if np.any(np.ptp(place_bin_edges, axis=0) <= knot_spacing):
        logging.warning("Range of position is smaller than knot spacing.")
    design_matrix = make_spline_design_matrix(position, place_bin_edges, knot_spacing)
    design_info = design_matrix.design_info
    try:
        client = get_client()
    except ValueError:
        client = Client()
    design_matrix = client.scatter(np.asarray(design_matrix), broadcast=True)
    results = [fit_glm(is_spike, design_matrix, penalty) for is_spike in spikes.T]
    results = dask.compute(*results)

    predict_matrix = make_spline_predict_matrix(design_info, place_bin_centers)
    place_fields = np.stack(
        [get_firing_rate(predict_matrix, result) for result in results], axis=1
    )

    DIMS = ["position", "neuron"]
    if position.shape[1] == 1:
        names = ["position"]
        coords = {"position": place_bin_centers.squeeze()}
    elif position.shape[1] == 2:
        names = ["x_position", "y_position"]
        coords = {
            "position": pd.MultiIndex.from_arrays(
                place_bin_centers.T.tolist(), names=names
            )
        }

    return xr.DataArray(data=place_fields, coords=coords, dims=DIMS)
