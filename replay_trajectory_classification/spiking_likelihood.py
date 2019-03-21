import dask
import numpy as np
import pandas as pd
import xarray as xr
from patsy import build_design_matrices, dmatrix
from sklearn.gaussian_process.kernels import RBF
from sklearn.mixture import GaussianMixture
from statsmodels.api import families

from regularized_glm import penalized_IRLS

from .core import atleast_2d, get_n_bins


def make_spline_design_matrix(position, bin_size=10):
    n_x_points = get_n_bins(position[:, 0], bin_size=bin_size)
    x_inner_knots = np.linspace(
        position[:, 0].min(), position[:, 0].max(), n_x_points)[1:-1]
    n_y_points = get_n_bins(position[:, 1], bin_size=bin_size)
    y_inner_knots = np.linspace(
        position[:, 1].min(), position[:, 1].max(), n_y_points)[1:-1]
    x_inner_knots, y_inner_knots = np.meshgrid(x_inner_knots, y_inner_knots)

    formula = ('1 + te(cr(x_position, knots=x_inner_knots), '
               'cr(y_position, knots=y_inner_knots), constraints="center")')

    return dmatrix(
        formula, dict(x_position=position[:, 0], y_position=position[:, 1]))


def make_spline_predict_matrix(design_info, place_bin_centers):
    predict_data = {'x_position': place_bin_centers[:, 0],
                    'y_position': place_bin_centers[:, 1]}
    return build_design_matrices(
        [design_info], predict_data)[0]


def get_rbf_parameters(position_info, n_components=120):
    position = position_info.loc[:, ['x_position', 'y_position']].values
    model_kwargs = dict(n_components=n_components, tol=1E-6, max_iter=200)
    occupancy_model = GaussianMixture(**model_kwargs).fit(atleast_2d(position))
    return occupancy_model.means_, occupancy_model.covariances_


def make_rbf_design_matrix(position_info, means, covariances):
    position = position_info.loc[:, ['x_position', 'y_position']].values

    radial_basis = [RBF(length_scale=np.diag(cov))
                    for cov in covariances]
    design_matrix = [rbf(mean, position).T
                     for rbf, mean in zip(radial_basis, means)]
    design_matrix.insert(0, np.ones((position.shape[0], 1)))
    design_matrix = np.concatenate(design_matrix, axis=1)

    return design_matrix, radial_basis


def make_rbf_predict_matrix(place_bin_centers, radial_basis, mixture_means):
    predict_design_matrix = [rbf(mean, place_bin_centers).T
                             for rbf, mean in zip(radial_basis, mixture_means)]
    predict_design_matrix.insert(0, np.ones((place_bin_centers.shape[0], 1)))
    predict_design_matrix = np.concatenate(predict_design_matrix, axis=1)

    return predict_design_matrix


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


def estimate_spiking_likelihood(spikes, conditional_intensity):
    '''

    Parameters
    ----------
    spikes : ndarray, shape (n_time, n_neurons)
    conditional_intensity : ndarray, shape (n_bins, n_neurons)

    Returns
    -------
    likelihood : ndarray, shape (n_time, n_bins)
    '''
    return scaled_likelihood(
        combined_likelihood(spikes, conditional_intensity))


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
    design_matrix = dask.delayed(np.array(design_matrix))
    results = [fit_glm(is_spike, design_matrix, penalty)
               for is_spike in spikes.T]
    results = dask.compute(*results)

    predict_matrix = make_spline_predict_matrix(design_info, place_bin_centers)
    place_fields = np.stack([get_firing_rate(predict_matrix, result)
                             for result in results], axis=1)

    DIMS = ['position', 'neuron']
    NAMES = ['x_position', 'y_position']
    coords = {
        'position': pd.MultiIndex.from_arrays(
            place_bin_centers.T.tolist(), names=NAMES)
    }

    return xr.DataArray(data=place_fields, coords=coords, dims=DIMS)
