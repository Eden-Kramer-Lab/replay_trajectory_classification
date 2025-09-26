"""Estimate Poisson likelihood using GLM with spline basis for place fields.

This module provides functions for estimating spike likelihoods using
Generalized Linear Models (GLM) with spline basis functions to model
spatial firing patterns.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import dask
import pandas as pd
import scipy.stats
import xarray as xr
from dask.distributed import Client, get_client
from patsy import DesignInfo, DesignMatrix, build_design_matrices, dmatrix
from regularized_glm import penalized_IRLS
from statsmodels.api import families

from replay_trajectory_classification.environments import get_n_bins


def make_spline_design_matrix(
    position: NDArray[np.float64], place_bin_edges: NDArray[np.float64], knot_spacing: float = 10.0
) -> DesignMatrix:
    """Creates a design matrix for regression with a position spline basis.

    Parameters
    ----------
    position : NDArray[np.float64], shape (n_time, n_position_dims)
    place_bin_edges : NDArray[np.float64], shape (n_bins, n_position_dims)

    Returns
    -------
    design_matrix : patsy.DesignMatrix

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


def make_spline_predict_matrix(
    design_info: DesignInfo, place_bin_centers: NDArray[np.float64]
) -> DesignMatrix:
    """Make a design matrix for position bins.

    Parameters
    ----------
    design_info : patsy.DesignInfo
        Design information from fitted model.
    place_bin_centers : NDArray[np.float64], shape (n_bins, n_position_dims)
        Center coordinates of spatial bins.

    Returns
    -------
    DesignMatrix
        Design matrix for predicting at position bin centers.

    """
    predict_data = {}
    for ind in range(place_bin_centers.shape[1]):
        predict_data[f"x{ind}"] = place_bin_centers[:, ind]
    return build_design_matrices([design_info], predict_data)[0]


def get_firing_rate(
    design_matrix: DesignMatrix, results: tuple, sampling_frequency: int = 1
) -> NDArray[np.float64]:
    """Predict the firing rate given fitted model coefficients.

    Parameters
    ----------
    design_matrix : patsy.DesignMatrix
        Design matrix for prediction locations.
    results : tuple
        Results tuple from fitted GLM containing coefficients.
    sampling_frequency : int, optional
        Sampling rate to scale the firing rate, by default 1.

    Returns
    -------
    NDArray[np.float64], shape (n_time,)
        Predicted firing rates.

    """
    if np.any(np.isnan(results.coefficients)):
        n_time = design_matrix.shape[0]
        rate = np.zeros((n_time,))
    else:
        rate = np.exp(design_matrix @ results.coefficients) * sampling_frequency

    return rate


@dask.delayed
def fit_glm(
    response: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
    penalty: Optional[float] = None,
    tolerance: float = 1e-5,
) -> tuple:
    """Fit a L2-penalized GLM.

    Parameters
    ----------
    response : NDArray[np.float64], shape (n_time,)
        Response variable (e.g., spike counts).
    design_matrix : NDArray[np.float64], shape (n_time, n_coefficients)
        Design matrix with predictor variables.
    penalty : float, optional
        L2 penalty on regression coefficients. If None, uses smallest possible
        penalty, by default None.
    tolerance : float, optional
        Convergence tolerance for iterative fitting, by default 1e-5.

    Returns
    -------
    results : tuple
        GLM fitting results containing coefficients and other fit statistics.

    """
    if penalty is not None:
        penalty = np.ones((design_matrix.shape[1],)) * penalty
        penalty[0] = 0.0  # don't penalize the intercept
    else:
        penalty = np.finfo(float).eps
    return penalized_IRLS(
        design_matrix,
        response.squeeze(),
        family=families.Poisson(),
        penalty=penalty,
        tolerance=tolerance,
    )


def poisson_log_likelihood(
    spikes: NDArray[np.float64], conditional_intensity: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Calculate Poisson log-likelihood of spikes given conditional intensity.

    Parameters
    ----------
    spikes : NDArray[np.float64], shape (n_time,)
        Spike counts at each time point.
    conditional_intensity : NDArray[np.float64], shape (n_place_bins,)
        Instantaneous firing rate for each spatial bin.

    Returns
    -------
    poisson_log_likelihood : NDArray[np.float64], shape (n_time, n_place_bins)
        Log-likelihood of observed spikes given firing rates.

    """
    # Logarithm of the absolute value of the gamma function is always 0 when
    # spikes are 0 or 1
    return scipy.stats.poisson.logpmf(
        spikes[:, np.newaxis], conditional_intensity[np.newaxis, :] + np.spacing(1)
    )


def combined_likelihood(
    spikes: NDArray[np.float64], conditional_intensity: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Combines the likelihoods of all the cells.

    Parameters
    ----------
    spikes : NDArray[np.float64], shape (n_time, n_neurons)
    conditional_intensity : NDArray[np.float64], shape (n_bins, n_neurons)

    """
    n_time = spikes.shape[0]
    n_bins = conditional_intensity.shape[0]
    log_likelihood = np.zeros((n_time, n_bins))

    for is_spike, ci in zip(spikes.T, conditional_intensity.T):
        log_likelihood += poisson_log_likelihood(is_spike, ci)

    return log_likelihood


def estimate_spiking_likelihood(
    spikes: NDArray[np.float64],
    conditional_intensity: NDArray[np.float64],
    is_track_interior: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.float64]:
    """

    Parameters
    ----------
    spikes : NDArray[np.float64], shape (n_time, n_neurons)
    conditional_intensity : NDArray[np.float64], shape (n_bins, n_neurons)
    is_track_interior : None or NDArray[np.bool_], optional, shape (n_x_position_bins,
                                                                   n_y_position_bins)
    Returns
    -------
    likelihood : NDArray[np.float64], shape (n_time, n_bins)

    """
    if is_track_interior is not None:
        is_track_interior = is_track_interior.ravel(order="F")
    else:
        n_bins = conditional_intensity.shape[0]
        is_track_interior = np.ones((n_bins,), dtype=bool)

    log_likelihood = combined_likelihood(spikes, conditional_intensity)

    mask = np.ones_like(is_track_interior, dtype=float)
    mask[~is_track_interior] = np.nan

    return log_likelihood * mask


def estimate_place_fields(
    position: NDArray[np.float64],
    spikes: NDArray[np.float64],
    place_bin_centers: NDArray[np.float64],
    place_bin_edges: NDArray[np.float64],
    edges: Optional[NDArray[np.float64]] = None,
    is_track_boundary: Optional[NDArray[np.bool_]] = None,
    is_track_interior: Optional[NDArray[np.bool_]] = None,
    penalty: float = 1e-1,
    knot_spacing: int = 10,
) -> xr.DataArray:
    """Gives the conditional intensity of the neurons' spiking with respect to
    position.

    Parameters
    ----------
    position : NDArray[np.float64], shape (n_time, n_position_dims)
    spikes : NDArray[np.float64], shape (n_time, n_neurons)
    place_bin_centers : NDArray[np.float64], shape (n_bins, n_position_dims)
    place_bin_edges : NDArray[np.float64], shape (n_bins + 1, n_position_dims)
    is_track_boundary : None or NDArray[np.bool_]
    is_track_interior : None or NDArray[np.bool_]
    penalty : None or float, optional
        L2 penalty on regression. If None, penalty is smallest possible.
    knot_spacing : int, optional
        Spacing of position knots. Controls how smooth the firing rate is.

    Returns
    -------
    conditional_intensity : xr.DataArray, shape (n_bins, n_neurons)

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
    scattered_spikes = [client.scatter(is_spike) for is_spike in spikes.T]
    results = [fit_glm(is_spike, design_matrix, penalty) for is_spike in scattered_spikes]
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
