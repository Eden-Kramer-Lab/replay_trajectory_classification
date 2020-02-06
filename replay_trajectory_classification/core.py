import networkx as nx
import numpy as np
import pandas as pd
from numba import njit
from scipy import ndimage
from scipy.interpolate import interp1d
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


def get_grid(position, bin_size=2.5, position_range=None,
             infer_track_interior=True):
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    n_bins = get_n_bins(position, bin_size, position_range)
    hist, edges = np.histogramdd(position, bins=n_bins, range=position_range)
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
    return distribution / np.nansum(distribution)


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

    '''
    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1].copy()
    n_time, n_bins = causal_posterior.shape[0], causal_posterior.shape[-2]
    weights = np.zeros((n_bins, 1))

    for time_ind in np.arange(n_time - 2, -1, -1):
        acausal_prior = (
            state_transition @ causal_posterior[time_ind])
        log_ratio = (
            np.log(acausal_posterior[time_ind + 1, ..., 0] + np.spacing(1)) -
            np.log(acausal_prior[..., 0] + np.spacing(1)))
        weights[..., 0] = np.exp(log_ratio) @ state_transition

        acausal_posterior[time_ind] = normalize_to_probability(
            weights * causal_posterior[time_ind])

    return acausal_posterior


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
    if n_position_dims > 1:
        structure = np.ones([1] * n_position_dims)
        is_maze = ndimage.binary_closing(is_maze, structure=structure)
        is_maze = ndimage.binary_fill_holes(is_maze)
        is_maze = ndimage.binary_dilation(is_maze, structure=structure)
    return is_maze.astype(np.bool)


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


def scaled_likelihood(log_likelihood):
    '''
    Parameters
    ----------
    log_likelihood : ndarray, shape (n_time, n_bins)

    Returns
    -------
    scaled_log_likelihood : ndarray, shape (n_time, n_bins)

    '''
    max_log_likelihood = np.nanmax(log_likelihood, axis=1, keepdims=True)
    likelihood = np.exp(log_likelihood - max_log_likelihood)
    likelihood[np.isnan(likelihood)] = 0.0
    return likelihood


def mask(value, is_track_interior):
    try:
        value[..., ~is_track_interior] = np.nan
    except IndexError:
        value[..., ~is_track_interior, :] = np.nan
    return value


def convert_linear_distance_to_linear_position(
        linear_distance, edge_id, edge_order, edge_spacing=30):
    linear_position = linear_distance.copy()
    n_edges = len(edge_order)
    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        edge_spacing = [edge_spacing, ] * (n_edges - 1)

    for prev_edge, cur_edge, space in zip(
            edge_order[:-1], edge_order[1:], edge_spacing):
        is_cur_edge = (edge_id == cur_edge)
        is_prev_edge = (edge_id == prev_edge)

        cur_distance = linear_position[is_cur_edge]
        cur_distance -= cur_distance.min()
        cur_distance += linear_position[is_prev_edge].max() + space
        linear_position[is_cur_edge] = cur_distance

    return linear_position


def linear_position_to_2D_projection(linear_position, node_linear_position,
                                     edge_dist, node_2D_position):
    try:
        is_node = np.isclose(linear_position, node_linear_position)
        edge_ind = np.where((
            (linear_position >= node_linear_position[:, 0]) | is_node[:, 0]) &
            ((linear_position <= node_linear_position[:, 1]) | is_node[:, 1])
        )[0][0]
    except IndexError:
        return np.full((2,), np.nan), -1
    pct_dist = (linear_position -
                node_linear_position[edge_ind][0]) / edge_dist[edge_ind]
    segment_diff = np.diff(node_2D_position, axis=1).squeeze()
    position_2D = node_2D_position[edge_ind, 0] + (
        segment_diff[edge_ind] * pct_dist)
    return position_2D, edge_ind


def get_graph_1D_2D_relationships(track_graph, edge_order, edge_spacing,
                                  center_well_id):
    '''

    Parameters
    ----------
    track_graph : networkx.Graph
    edge_order : array-like, shape (n_edges,)
    edge_spacing : float or array-like, shape (n_edges,)
    center_well_id : int

    Returns
    -------
    node_linear_position : numpy.ndarray, shape (n_edges, n_position_dims)
    edges : numpy.ndarray, shape (n_edges, 2)
    node_2D_position : numpy.ndarray, shape (n_edges, 2, n_position_dims)
    edge_dist : numpy.ndarray, shape (n_edges,)

    '''
    linear_distance = []
    edge_id = []
    node_2D_position = []
    edge_dist = []

    dist = dict(
        nx.all_pairs_dijkstra_path_length(track_graph, weight="distance")
    )
    n_edges = len(track_graph.edges)

    for ind, (node1, node2) in enumerate(track_graph.edges):
        linear_distance.append(dist[center_well_id][node1])
        linear_distance.append(dist[center_well_id][node2])
        edge_id.append(ind)
        edge_id.append(ind)
        node_2D_position.append(track_graph.nodes[node1]['pos'])
        node_2D_position.append(track_graph.nodes[node2]['pos'])

    linear_distance = np.array(linear_distance)
    edge_id = np.array(edge_id)
    node_2D_position = np.array(node_2D_position).reshape((n_edges, 2, 2))[
        edge_order]  # shape (n_edges, n_nodes, 2)
    edge_dist = np.linalg.norm(
        np.diff(node_2D_position, axis=1), axis=2).squeeze()

    node_linear_position = convert_linear_distance_to_linear_position(
        linear_distance, edge_id, edge_order, edge_spacing=edge_spacing
    )

    node_linear_position = node_linear_position.reshape((n_edges, 2))[
        edge_order]
    edges = np.array(track_graph.edges)[edge_order]

    return node_linear_position, edges, node_2D_position, edge_dist


def get_track_grid(
        track_graph, center_well_id, edge_order, edge_spacing, place_bin_size):
    track_graph1 = track_graph.copy()
    n_nodes = len(track_graph.nodes)

    for edge_ind, (node1, node2) in enumerate(track_graph.edges):
        node1_x_pos, node1_y_pos = track_graph.nodes[node1]["pos"]
        node2_x_pos, node2_y_pos = track_graph.nodes[node2]["pos"]

        edge_size = np.linalg.norm(
            [(node2_x_pos - node1_x_pos), (node2_y_pos - node1_y_pos)]
        )
        n_bins = 2 * np.ceil(edge_size / place_bin_size).astype(np.int) + 1
        if ~np.isclose(node1_x_pos, node2_x_pos):
            f = interp1d((node1_x_pos, node2_x_pos),
                         (node1_y_pos, node2_y_pos))
            xnew = np.linspace(node1_x_pos, node2_x_pos,
                               num=n_bins, endpoint=True)
            xy = np.stack((xnew, f(xnew)), axis=1)
        else:
            ynew = np.linspace(node1_y_pos, node2_y_pos,
                               num=n_bins, endpoint=True)
            xnew = np.ones_like(ynew) * node1_x_pos
            xy = np.stack((xnew, ynew), axis=1)
        dist_between_nodes = np.linalg.norm(np.diff(xy, axis=0), axis=1)

        new_node_ids = n_nodes + np.arange(len(dist_between_nodes) + 1)
        nx.add_path(track_graph1, [*new_node_ids],
                    distance=dist_between_nodes[0])
        nx.add_path(track_graph1, [node1, new_node_ids[0]], distance=0)
        nx.add_path(track_graph1, [node2, new_node_ids[-1]], distance=0)
        track_graph1.remove_edge(node1, node2)
        for ind, (node_id, pos) in enumerate(zip(new_node_ids, xy)):
            track_graph1.nodes[node_id]["pos"] = pos
            track_graph1.nodes[node_id]["edge_id"] = edge_ind
            if ind % 2:
                track_graph1.nodes[node_id]["is_bin_edge"] = False
            else:
                track_graph1.nodes[node_id]["is_bin_edge"] = True
        track_graph1.nodes[node1]["edge_id"] = edge_ind
        track_graph1.nodes[node2]["edge_id"] = edge_ind
        track_graph1.nodes[node1]["is_bin_edge"] = True
        track_graph1.nodes[node2]["is_bin_edge"] = True
        n_nodes = len(track_graph1.nodes)

    distance_between_nodes = dict(
        nx.all_pairs_dijkstra_path_length(track_graph1, weight="distance"))

    node_ids, linear_distance = list(
        zip(*distance_between_nodes[center_well_id].items())
    )
    linear_distance = np.array(linear_distance)

    edge_ids = nx.get_node_attributes(track_graph1, "edge_id")
    edge_id = np.array([edge_ids[node_id] for node_id in node_ids])

    is_bin_edges = nx.get_node_attributes(track_graph1, "is_bin_edge")
    is_bin_edge = np.array([is_bin_edges[node_id] for node_id in node_ids])

    node_linear_position = convert_linear_distance_to_linear_position(
        linear_distance, edge_id, edge_order, edge_spacing=edge_spacing
    )

    nodes_df = (pd.DataFrame(
        dict(node_ids=node_ids, edge_id=edge_id, is_bin_edge=is_bin_edge,
             linear_position=node_linear_position))
        .sort_values(by=['linear_position', 'edge_id'], axis='rows'))

    place_bin_edges, unique_ind = np.unique(
        node_linear_position[is_bin_edge], return_index=True)
    place_bin_centers = get_centers(place_bin_edges)
    place_bin_edge_ind_to_edge_id = edge_id[is_bin_edge][unique_ind]
    change_edge_ind = np.nonzero(np.diff(place_bin_edge_ind_to_edge_id))[0]
    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        n_edges = len(edge_order)
        edge_spacing = [edge_spacing, ] * (n_edges - 1)

    is_track_interior = np.ones_like(place_bin_centers, dtype=np.bool)
    not_track = change_edge_ind[np.array(edge_spacing) > 0]
    is_track_interior[not_track] = False

    place_bin_center_ind_to_edge_id = place_bin_edge_ind_to_edge_id[1:].copy()
    place_bin_center_ind_to_edge_id[~is_track_interior] = -1

    closest_node_ind = np.argmin(
        np.abs(node_linear_position - place_bin_centers[:, np.newaxis]),
        axis=1)
    place_bin_center_ind_to_node = np.array(
        [node_ids[ind] for ind in closest_node_ind])

    (node_linear_position, edges,
     node_2D_position, edge_dist) = get_graph_1D_2D_relationships(
        track_graph, edge_order, edge_spacing, center_well_id)
    place_bin_center_2D_position = np.stack([
        linear_position_to_2D_projection(
            center, node_linear_position, edge_dist, node_2D_position)[0]
        for center in place_bin_centers])
    place_bin_edges_2D_position = np.stack([
        linear_position_to_2D_projection(
            edge, node_linear_position, edge_dist, node_2D_position)[0]
        for edge in place_bin_edges])
    edges = [place_bin_edges]
    centers_shape = (place_bin_centers.size,)

    return (
        place_bin_centers[:, np.newaxis],
        place_bin_edges[:, np.newaxis],
        is_track_interior,
        distance_between_nodes,
        place_bin_center_ind_to_node,
        place_bin_center_2D_position,
        place_bin_edges_2D_position,
        centers_shape,
        edges,
        track_graph1,
        place_bin_center_ind_to_edge_id,
        nodes_df,
    )
