import logging

import dask
import numpy as np
import pandas as pd
import scipy
import statsmodels as sm
import xarray as xr
from dask.distributed import Client, get_client
from patsy import build_design_matrices, dmatrix
from replay_trajectory_classification.bins import get_n_bins


def make_spline_design_matrix(position, place_bin_edges, knot_spacing=10):
    # TODO: Add history dependence
    inner_knots = []
    for pos, edges in zip(position.T, place_bin_edges.T):
        n_points = get_n_bins(edges, bin_size=knot_spacing)
        knots = np.linspace(edges.min(), edges.max(), n_points)[1:-1]
        knots = knots[(knots > pos.min()) & (knots < pos.max())]
        inner_knots.append(knots)

    inner_knots = np.meshgrid(*inner_knots)

    n_position_dims = position.shape[1]
    data = {}
    formula = '1 + te('
    for ind in range(n_position_dims):
        formula += f'cr(x{ind}, knots=inner_knots[{ind}])'
        formula += ', '
        data[f'x{ind}'] = position[:, ind]

    formula += ')'

    return dmatrix(formula, data)


def make_spline_predict_matrix(design_info, place_bin_centers):
    predict_data = {}
    for ind in range(place_bin_centers.shape[1]):
        predict_data[f'x{ind}'] = place_bin_centers[:, ind]
    return build_design_matrices(
        [design_info], predict_data)[0]


def get_activity_rate(design_matrix, results):
    rate = (design_matrix @ results.coefficients)
    rate = np.max(rate, 0.1)
    return rate


@dask.delayed
def fit_glm(response, design_matrix):
    fit = sm.GLM(response, design_matrix, family=sm.families.Gamma(
        sm.families.links.identity)).fit()
    mu = fit.mu
    v = 1 / fit.scale  # shape, dispersion
    coefficients = fit.params

    return mu, v, coefficients


def gamma_log_likelihood(mu, v):
    """Probability of parameters given spiking at a particular time.

    Parameters
    ----------
    mu : ndarray, shape (n_place_bins,)
    v : float

    Returns
    -------
    gamma_log_likelihood : array_like, shape (n_time, n_place_bins)

    """
    # return scipy.stats.gamma.logpdf(v * calcium_activity / mu, v)
    # return scipy.stats.gamma.logpdf(mu, v)
    # return (-scipy.special.loggamma(v) +
    #         v * np.log(v * calcium_activity / mu) -
    #         v * calcium_activity / mu -
    #         np.log(calcium_activity))
    pass


def combined_likelihood(calcium_activity, conditional_intensity):
    '''

    Parameters
    ----------
    calcium_activity : ndarray, shape (n_time, n_neurons)
        Deconvolved activity rate estimated from the fluorescence level.
    conditional_intensity : ndarray, shape (n_bins, n_neurons)

    '''
    n_time = calcium_activity.shape[0]
    n_bins = conditional_intensity.shape[0]
    log_likelihood = np.zeros((n_time, n_bins))

    for activity, ci in zip(calcium_activity.T, conditional_intensity.T):
        log_likelihood += gamma_log_likelihood(activity, ci)

    return log_likelihood


def estimate_calcium_likelihood(calcium_activity, conditional_intensity,
                                is_track_interior=None):
    '''

    Parameters
    ----------
    calcium_activity : ndarray, shape (n_time, n_neurons)
        Deconvolved activity rate estimated from the fluorescence level.
    conditional_intensity : ndarray, shape (n_bins, n_neurons)
    is_track_interior : None or ndarray, optional, shape (n_x_position_bins,
                                                          n_y_position_bins)
    Returns
    -------
    likelihood : ndarray, shape (n_time, n_bins)
    '''
    if is_track_interior is not None:
        is_track_interior = is_track_interior.ravel(order='F')
    else:
        n_bins = conditional_intensity.shape[0]
        is_track_interior = np.ones((n_bins,), dtype=np.bool)

    log_likelihood = combined_likelihood(
        calcium_activity, conditional_intensity)

    mask = np.ones_like(is_track_interior, dtype=np.float)
    mask[~is_track_interior] = np.nan

    return log_likelihood * mask


def estimate_place_fields(position,
                          calcium_activity,
                          place_bin_centers,
                          place_bin_edges,
                          knot_spacing=10):
    '''Gives the conditional intensity of the neurons' spiking with respect to
    position.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    calcium_activity : ndarray, shape (n_time, n_neurons)
        Deconvolved activity rate estimated from the fluorescence level.
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    place_bin_edges : ndarray, shape (n_bins + 1, n_position_dims)
    penalty : float, optional
    knot_spacing : int, optional

    Returns
    -------
    conditional_intensity : ndarray, shape (n_bins, n_neurons)

    '''
    if np.any(np.ptp(place_bin_edges, axis=0) <= knot_spacing):
        logging.warning("Range of position is smaller than knot spacing.")
    design_matrix = make_spline_design_matrix(
        position, place_bin_edges, knot_spacing)
    design_info = design_matrix.design_info
    try:
        client = get_client()
    except ValueError:
        client = Client()
    design_matrix = client.scatter(np.asarray(design_matrix), broadcast=True)
    results = [fit_glm(activity, design_matrix)
               for activity in calcium_activity.T]
    results = dask.compute(*results)

    predict_matrix = make_spline_predict_matrix(design_info, place_bin_centers)
    place_fields = np.stack([get_activity_rate(predict_matrix, result)
                             for result in results], axis=1)

    DIMS = ['position', 'neuron']
    if position.shape[1] == 1:
        names = ['position']
        coords = {
            'position': place_bin_centers.squeeze()
        }
    elif position.shape[1] == 2:
        names = ['x_position', 'y_position']
        coords = {
            'position': pd.MultiIndex.from_arrays(
                place_bin_centers.T.tolist(), names=names)
        }

    return xr.DataArray(data=place_fields, coords=coords, dims=DIMS)
