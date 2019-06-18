import networkx as nx
import numpy as np
from numba import njit
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors


def get_n_bins(position, bin_size=2.5, position_range=None):
    '''Get number of bins need to span a range given a bin size.

    Parameters
    ----------
    position : ndarray, shape (n_time,)
    bin_size : float, optional

    Returns
    -------
    n_bins : int

    '''
    if position_range is not None:
        extent = np.diff(position_range, axis=1).squeeze()
    else:
        extent = np.ptp(position, axis=0)
    return np.ceil(extent / bin_size).astype(np.int)


def atleast_2d(x):
    return np.atleast_2d(x).T if x.ndim < 2 else x


def get_centers(edge):
    return edge[:-1] + np.diff(edge) / 2


def add_zero_end_bins(hist, edges):
    new_edges = []

    for edge_ind, edge in enumerate(edges):
        bin_size = np.diff(edge)[0]
        try:
            if hist.sum(axis=edge_ind)[0] != 0:
                edge = np.insert(edge, 0, edge[0] - bin_size)
            if hist.sum(axis=edge_ind)[-1] != 0:
                edge = np.append(edge, edge[-1] + bin_size)
        except IndexError:
            if hist[0] != 0:
                edge = np.insert(edge, 0, edge[0] - bin_size)
            if hist[-1] != 0:
                edge = np.append(edge, edge[-1] + bin_size)
        new_edges.append(edge)

    return new_edges


def get_grid(position, bin_size=2.5, position_range=None,
             infer_track_interior=True):
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    n_bins = get_n_bins(position, bin_size, position_range)
    hist, edges = np.histogramdd(position, bins=n_bins, range=position_range)
    if infer_track_interior:
        edges = add_zero_end_bins(hist, edges)
    mesh_edges = np.meshgrid(*edges)
    place_bin_edges = np.stack([edge.ravel() for edge in mesh_edges], axis=1)

    mesh_centers = np.meshgrid(
        *[get_centers(edge) for edge in edges])
    place_bin_centers = np.stack(
        [center.ravel() for center in mesh_centers], axis=1)
    centers_shape = mesh_centers[0].shape

    return edges, place_bin_edges, place_bin_centers, centers_shape


def order_border(border):
    '''
    https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    '''
    n_points = border.shape[0]
    clf = NearestNeighbors(2).fit(border)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)

    paths = [list(nx.dfs_preorder_nodes(T, i))
             for i in range(n_points)]
    min_idx, min_dist = 0, np.inf

    for idx, path in enumerate(paths):
        ordered = border[path]    # ordered nodes
        cost = np.sum(np.diff(ordered) ** 2)
        if cost < min_dist:
            min_idx, min_dist = idx, cost

    opt_order = paths[min_idx]
    return border[opt_order][:-1]


@njit(nogil=True, fastmath=True)
def logsumexp(a):
    a_max = np.max(a)
    out = np.log(np.sum(np.exp(a - a_max)))
    out += a_max
    return out


@njit(parallel=True)
def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / np.sum(distribution)


@njit(nogil=True)
def _causal_decode(initial_conditions, state_transition, likelihood):
    '''Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : ndarray, shape (n_bins,)
    state_transition : ndarray, shape (n_bins, n_bins)
    likelihood : ndarray, shape (n_time, n_bins)

    Returns
    -------
    posterior : ndarray, shape (n_time, n_bins)

    '''

    n_time = likelihood.shape[0]
    posterior = np.zeros_like(likelihood)

    posterior[0] = normalize_to_probability(
        initial_conditions.copy() * likelihood[0])

    for time_ind in np.arange(1, n_time):
        prior = state_transition @ posterior[time_ind - 1]
        posterior[time_ind] = normalize_to_probability(
            prior * likelihood[time_ind])

    return posterior


@njit(nogil=True)
def _acausal_decode(causal_posterior, state_transition):
    '''Uses past and future information to estimate the state.

    Parameters
    ----------
    causal_posterior : ndarray, shape (n_time, n_bins, 1)
    state_transition : ndarray, shape (n_bins, n_bins)

    Return
    ------
    acausal_posterior : ndarray, shape (n_time, n_bins, 1)
    acausal_prior : ndarray, shape (n_time, n_bins, 1)

    '''
    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1].copy()
    acausal_prior = np.zeros_like(causal_posterior)
    n_time, n_bins = causal_posterior.shape[0], causal_posterior.shape[-2]
    weights = np.zeros((n_bins, 1))

    for time_ind in np.arange(n_time - 2, -1, -1):
        acausal_prior[time_ind] = (
            state_transition @ causal_posterior[time_ind])
        log_ratio = (
            np.log(acausal_posterior[time_ind + 1, ..., 0] + np.spacing(1)) -
            np.log(acausal_prior[time_ind, ..., 0] + np.spacing(1)))
        weights[..., 0] = np.exp(log_ratio) @ state_transition

        acausal_posterior[time_ind] = normalize_to_probability(
            weights * causal_posterior[time_ind])

    return acausal_posterior, acausal_prior


@njit(nogil=True)
def _causal_classify(initial_conditions, continuous_state_transition,
                     discrete_state_transition, likelihood):
    '''Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : ndarray, shape (n_states, n_bins, 1)
    continuous_state_transition : ndarray, shape (n_states, n_states,
                                                  n_bins, n_bins)
    discrete_state_transition : ndarray, shape (n_states, n_states)
    likelihood : ndarray, shape (n_time, n_states, n_bins, 1)

    Returns
    -------
    causal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)

    '''
    n_time, n_states, n_bins, _ = likelihood.shape
    posterior = np.zeros_like(likelihood)

    posterior[0] = normalize_to_probability(
        initial_conditions.copy() * likelihood[0])

    for k in np.arange(1, n_time):
        prior = np.zeros((n_states, n_bins, 1))
        for state_k in np.arange(n_states):
            for state_k_1 in np.arange(n_states):
                prior[state_k, :] += (
                    discrete_state_transition[state_k_1, state_k] *
                    continuous_state_transition[state_k_1, state_k] @
                    posterior[k - 1, state_k_1])
        posterior[k] = normalize_to_probability(prior * likelihood[k])

    return posterior


@njit(nogil=True)
def _acausal_classify(causal_posterior, continuous_state_transition,
                      discrete_state_transition):
    '''Uses past and future information to estimate the state.

    Parameters
    ----------
    causal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)
    continuous_state_transition : ndarray, shape (n_states, n_states,
                                                  n_bins, n_bins)
    discrete_state_transition : ndarray, shape (n_states, n_states)

    Return
    ------
    acausal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)

    '''
    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1].copy()
    n_time, n_states, n_bins, _ = causal_posterior.shape

    for k in np.arange(n_time - 2, -1, -1):
        # Prediction Step -- p(x_{k+1}, I_{k+1} | y_{1:k})
        prior = np.zeros((n_states, n_bins, 1))
        for state_k_1 in np.arange(n_states):
            for state_k in np.arange(n_states):
                prior[state_k_1, :] += (
                    discrete_state_transition[state_k, state_k_1] *
                    continuous_state_transition[state_k, state_k_1] @
                    causal_posterior[k, state_k])

        # Backwards Update
        weights = np.zeros((n_states, n_bins, 1))
        ratio = np.exp(
            np.log(acausal_posterior[k + 1] + np.spacing(1)) -
            np.log(prior + np.spacing(1)))
        for state_k in np.arange(n_states):
            for state_k_1 in np.arange(n_states):
                weights[state_k] += (
                    discrete_state_transition[state_k, state_k_1] *
                    continuous_state_transition[state_k, state_k_1] @
                    ratio[state_k_1])

        acausal_posterior[k] = normalize_to_probability(
            weights * causal_posterior[k])

    return acausal_posterior


def get_track_interior(position, bins):
    '''

    position : ndarray, shape (n_time, n_position_dims)
    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    '''
    bin_counts, edges = np.histogramdd(position, bins=bins)
    is_maze = (bin_counts > 0).astype(int)
    n_position_dims = position.shape[1]
    structure = np.ones([1] * n_position_dims)
    is_maze = ndimage.binary_closing(is_maze, structure=structure)
    is_maze = ndimage.binary_fill_holes(is_maze)
    return ndimage.binary_dilation(is_maze, structure=structure)


def get_track_border(is_maze, edges):
    '''

    Parameters
    ----------
    is_maze : ndarray, shape (n_x_bins, n_y_bins)
    edges : list of ndarray

    '''
    structure = ndimage.generate_binary_structure(2, 2)
    border = ndimage.binary_dilation(is_maze, structure=structure) ^ is_maze

    inds = np.nonzero(border)
    centers = [get_centers(x) for x in edges]
    border = np.stack([center[ind] for center, ind in zip(centers, inds)],
                      axis=1)
    return order_border(border)
