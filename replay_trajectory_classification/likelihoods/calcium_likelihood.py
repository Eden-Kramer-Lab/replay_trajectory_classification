import logging

import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, get_client
from patsy import build_design_matrices, dmatrix
from regularized_glm import penalized_IRLS
from replay_trajectory_classification.bins import get_n_bins
from statsmodels.api import families


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

    formula += 'constraints="center")'

    return dmatrix(formula, data)


def make_spline_predict_matrix(design_info, place_bin_centers):
    predict_data = {}
    for ind in range(place_bin_centers.shape[1]):
        predict_data[f'x{ind}'] = place_bin_centers[:, ind]
    return build_design_matrices(
        [design_info], predict_data)[0]


def get_activity_rate(design_matrix, results):
    rate = (design_matrix @ results.coefficients)
    rate[rate < 0.1] = 0.1
    return rate


@dask.delayed
def fit_glm(response, design_matrix, penalty=None, tolerance=1E-5):
    if penalty is not None:
        penalty = np.ones((design_matrix.shape[1],)) * penalty
        penalty[0] = 0.0  # don't penalize the intercept
    else:
        penalty = np.finfo(np.float).eps
    return penalized_IRLS(
        design_matrix, response.squeeze(), family=families.Gamma(
            families.links.identity()),
        penalty=penalty, tolerance=tolerance)


def gamma_log_likelihood(calcium_activity, place_field, scale):
    """Probability of parameters given spiking at a particular time.

    Parameters
    ----------
    calcium_activity : np.ndarray, shape (n_time,)
    place_field : np.ndaarray, shape (n_place_bins,)
    scale : float

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
    gamma = families.Gamma(families.links.identity())
    return gamma.loglike_obs(
        endog=calcium_activity[:, np.newaxis],
        mu=place_field[np.newaxis, :],
        scale=scale)


def combined_likelihood(calcium_activity, place_fields, scales):
    '''

    Parameters
    ----------
    calcium_activity : np.ndarray, shape (n_time, n_neurons)
        Deconvolved activity rate estimated from the fluorescence level.
    place_fields : np.ndarray, shape (n_bins, n_neurons)
    scales : np.ndarray, shape (n_neurons,)

    '''
    n_time = calcium_activity.shape[0]
    n_bins = place_fields.shape[0]
    log_likelihood = np.zeros((n_time, n_bins))

    for activity, place_field, scale in zip(
            calcium_activity.T, place_fields.T, scales):
        log_likelihood += gamma_log_likelihood(activity, place_field, scale)

    return log_likelihood


def estimate_calcium_likelihood(calcium_activity, place_fields, scales,
                                is_track_interior=None):
    '''

    Parameters
    ----------
    calcium_activity : ndarray, shape (n_time, n_neurons)
        Deconvolved activity rate estimated from the fluorescence level.
    place_fields : ndarray, shape (n_bins, n_neurons)
    is_track_interior : None or ndarray, optional, shape (n_x_position_bins,
                                                          n_y_position_bins)
    Returns
    -------
    likelihood : ndarray, shape (n_time, n_bins)
    '''
    if is_track_interior is not None:
        is_track_interior = is_track_interior.ravel(order='F')
    else:
        n_bins = place_fields.shape[0]
        is_track_interior = np.ones((n_bins,), dtype=np.bool)

    log_likelihood = combined_likelihood(
        calcium_activity, place_fields, scales)

    mask = np.ones_like(is_track_interior, dtype=np.float)
    mask[~is_track_interior] = np.nan

    return log_likelihood * mask


def estimate_calcium_place_fields(position,
                                  calcium_activity,
                                  place_bin_centers,
                                  place_bin_edges,
                                  penalty=1E-1,
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
    results = [fit_glm(activity, design_matrix, penalty)
               for activity in calcium_activity.T]
    results = dask.compute(*results)

    predict_matrix = make_spline_predict_matrix(design_info, place_bin_centers)
    place_fields = np.stack([get_activity_rate(predict_matrix, result)
                             for result in results], axis=1)
    scales = np.asarray([result.scale for result in results])

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

    return xr.DataArray(data=place_fields, coords=coords, dims=DIMS), scales
