import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, get_client
from patsy import build_design_matrices, dmatrix
from regularized_glm import penalized_IRLS
from statsmodels.api import families

from .core import get_n_bins


def make_spline_design_matrix(position, bin_size=10):
    inner_knots = []
    for pos in position.T:
        n_points = get_n_bins(pos, bin_size=bin_size)
        inner_knots.append(np.linspace(
            pos.min(), pos.max(), n_points)[1:-1])

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


def get_firing_rate(design_matrix, results, sampling_frequency=1):
    if np.any(np.isnan(results.coefficients)):
        n_time = design_matrix.shape[0]
        rate = np.zeros((n_time,))
    else:
        rate = np.exp(
            design_matrix @ results.coefficients) * sampling_frequency

    return rate


@dask.delayed
def fit_glm(response, design_matrix, penalty=None, tolerance=1E-5):
    if penalty is not None:
        penalty = np.ones((design_matrix.shape[1],)) * penalty
        penalty[0] = 0.0  # don't penalize the intercept
    else:
        penalty = np.finfo(np.float).eps
    return penalized_IRLS(
        design_matrix, response.squeeze(), family=families.Poisson(),
        penalty=penalty, tolerance=tolerance)


def poisson_log_likelihood(spikes, conditional_intensity):
    """Probability of parameters given spiking at a particular time.

    Parameters
    ----------
    spikes : ndarray, shape (n_time,)
        Indicator of spike or no spike at current time.
    conditional_intensity : ndarray, shape (n_place_bins,)
        Instantaneous probability of observing a spike

    Returns
    -------
    poisson_log_likelihood : array_like, shape (n_time, n_place_bins)

    """
    return (np.log(conditional_intensity[np.newaxis, :] + np.spacing(1)) *
            spikes[:, np.newaxis] - conditional_intensity[np.newaxis, :])


def combined_likelihood(spikes, conditional_intensity):
    '''

    Parameters
    ----------
    spikes : ndarray, shape (n_time, n_neurons)
    conditional_intensity : ndarray, shape (n_bins, n_neurons)

    '''
    n_time = spikes.shape[0]
    n_bins = conditional_intensity.shape[0]
    log_likelihood = np.zeros((n_time, n_bins))

    for is_spike, ci in zip(spikes.T, conditional_intensity.T):
        log_likelihood += poisson_log_likelihood(is_spike, ci)

    return log_likelihood


def scaled_likelihood(log_likelihood):
    '''
    Parameters
    ----------
    log_likelihood : ndarray, shape (n_time, n_bins)

    Returns
    -------
    scaled_log_likelihood : ndarray, shape (n_time, n_bins)
    '''
    return np.exp(log_likelihood -
                  np.max(log_likelihood, axis=1, keepdims=True))


def estimate_spiking_likelihood(spikes, conditional_intensity,
                                is_track_interior=None):
    '''

    Parameters
    ----------
    spikes : ndarray, shape (n_time, n_neurons)
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

    likelihood = scaled_likelihood(
        combined_likelihood(spikes, conditional_intensity))
    likelihood[:, ~is_track_interior] = 0.0
    return likelihood


def estimate_place_fields(position, spikes, place_bin_centers, penalty=1E-1,
                          knot_spacing=10):
    '''Gives the conditional intensity of the neurons' spiking with respect to
    position.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    spikes : ndarray, shape (n_time, n_neurons)
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    penalty : float, optional
    knot_spacing : int, optional

    Returns
    -------
    conditional_intensity : ndarray, shape (n_bins, n_neurons)

    '''
    design_matrix = make_spline_design_matrix(position, knot_spacing)
    design_info = design_matrix.design_info
    try:
        client = get_client()
    except ValueError:
        client = Client()
    design_matrix = client.scatter(np.asarray(design_matrix), broadcast=True)
    results = [fit_glm(is_spike, design_matrix, penalty)
               for is_spike in spikes.T]
    results = dask.compute(*results)

    predict_matrix = make_spline_predict_matrix(design_info, place_bin_centers)
    place_fields = np.stack([get_firing_rate(predict_matrix, result)
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
