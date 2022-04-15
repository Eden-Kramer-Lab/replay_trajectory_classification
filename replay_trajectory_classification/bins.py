import networkx as nx
import numpy as np
import pandas as pd
from numba import njit
from scipy import ndimage
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors


def atleast_2d(x):
    return np.atleast_2d(x).T if x.ndim < 2 else x


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

    return np.ceil(extent / bin_size).astype(np.int32)


def get_centers(edge):
    return edge[:-1] + np.diff(edge) / 2


def get_grid(position, bin_size=2.5, position_range=None,
             infer_track_interior=True):
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    n_bins = get_n_bins(position, bin_size, position_range)
    _, edges = np.histogramdd(position, bins=n_bins, range=position_range)
    if len(edges) > 1:
        edges = [np.insert(edge, (0, len(edge)), (edge[0] - np.diff(edge)[0],
                                                  edge[-1] + np.diff(edge)[0]))
                 for edge in edges]
    mesh_edges = np.meshgrid(*edges)
    place_bin_edges = np.stack([edge.ravel() for edge in mesh_edges], axis=1)

    mesh_centers = np.meshgrid(
        *[get_centers(edge) for edge in edges])
    place_bin_centers = np.stack(
        [center.ravel() for center in mesh_centers], axis=1)
    centers_shape = mesh_centers[0].T.shape

    return edges, place_bin_edges, place_bin_centers, centers_shape


def get_track_interior(position, bins):
    '''

    position : ndarray, shape (n_time, n_position_dims)
    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    '''
    bin_counts, _ = np.histogramdd(position, bins=bins)
    is_maze = (bin_counts > 0).astype(int)
    n_position_dims = position.shape[1]
    if n_position_dims > 1:
        structure = np.ones([1] * n_position_dims)
        is_maze = ndimage.binary_closing(is_maze, structure=structure)
        is_maze = ndimage.binary_fill_holes(is_maze)
        is_maze = ndimage.binary_dilation(is_maze, structure=structure)

        # adjust for boundary edges in 2D
        is_maze[-1] = False
        is_maze[:, -1] = False
    return is_maze.astype(np.bool)


def get_track_segments_from_graph(track_graph):
    '''

    Parameters
    ----------
    track_graph : networkx Graph

    Returns
    -------
    track_segments : ndarray, shape (n_segments, n_nodes, n_space)

    '''
    node_positions = nx.get_node_attributes(track_graph, 'pos')
    return np.asarray([(node_positions[node1], node_positions[node2])
                       for node1, node2 in track_graph.edges()])


def project_points_to_segment(track_segments, position):
    '''Finds the closet point on a track segment in terms of Euclidean distance

    Parameters
    ----------
    track_segments : ndarray, shape (n_segments, n_nodes, 2)
    position : ndarray, shape (n_time, 2)

    Returns
    -------
    projected_positions : ndarray, shape (n_time, n_segments, n_space)

    '''
    segment_diff = np.diff(track_segments, axis=1).squeeze(axis=1)
    sum_squares = np.sum(segment_diff ** 2, axis=1)
    node1 = track_segments[:, 0, :]
    nx = (np.sum(segment_diff *
                 (position[:, np.newaxis, :] - node1), axis=2) /
          sum_squares)
    nx[np.where(nx < 0)] = 0.0
    nx[np.where(nx > 1)] = 1.0
    return node1[np.newaxis, ...] + (
        nx[:, :, np.newaxis] * segment_diff[np.newaxis, ...])


def _calculate_linear_position(track_graph, position, track_segment_id,
                               edge_order, edge_spacing):
    is_nan = np.isnan(track_segment_id)
    track_segment_id[is_nan] = 0  # need to check
    track_segment_id = track_segment_id.astype(int)

    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_positions = project_points_to_segment(
        track_segments, position)
    n_time = projected_track_positions.shape[0]
    projected_track_positions = projected_track_positions[(
        np.arange(n_time), track_segment_id)]

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        edge_spacing = [edge_spacing, ] * (n_edges - 1)

    counter = 0.0
    start_node_linear_position = []

    for ind, edge in enumerate(edge_order):
        start_node_linear_position.append(counter)

        try:
            counter += track_graph.edges[edge]["distance"] + edge_spacing[ind]
        except IndexError:
            pass

    start_node_linear_position = np.asarray(start_node_linear_position)

    track_segment_id_to_start_node_linear_position = {
        track_graph.edges[e]["edge_id"]: snlp
        for e, snlp in zip(edge_order, start_node_linear_position)}

    start_node_linear_position = np.asarray([
        track_segment_id_to_start_node_linear_position[edge_id]
        for edge_id in track_segment_id
    ])

    track_segment_id_to_edge = {
        track_graph.edges[e]["edge_id"]: e for e in edge_order}
    start_node_id = np.asarray([track_segment_id_to_edge[edge_id][0]
                                for edge_id in track_segment_id])
    start_node_2D_position = np.asarray(
        [track_graph.nodes[node]["pos"] for node in start_node_id])

    linear_position = start_node_linear_position + (
        np.linalg.norm(start_node_2D_position -
                       projected_track_positions, axis=1))
    linear_position[is_nan] = np.nan

    return (linear_position,
            projected_track_positions[:, 0],
            projected_track_positions[:, 1])


def make_track_graph_with_bin_centers_edges(track_graph, place_bin_size):
    """Insert the bin center and bin edge positions as nodes in the track graph.
    """
    track_graph_with_bin_centers_edges = track_graph.copy()
    n_nodes = len(track_graph.nodes)

    for edge_ind, (node1, node2) in enumerate(track_graph.edges):
        node1_x_pos, node1_y_pos = track_graph.nodes[node1]["pos"]
        node2_x_pos, node2_y_pos = track_graph.nodes[node2]["pos"]

        edge_size = np.linalg.norm(
            [(node2_x_pos - node1_x_pos), (node2_y_pos - node1_y_pos)]
        )
        n_bins = 2 * np.ceil(edge_size / place_bin_size).astype(np.int32) + 1
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
        nx.add_path(track_graph_with_bin_centers_edges, [*new_node_ids],
                    distance=dist_between_nodes[0])
        nx.add_path(track_graph_with_bin_centers_edges, [
                    node1, new_node_ids[0]], distance=0)
        nx.add_path(track_graph_with_bin_centers_edges, [
                    node2, new_node_ids[-1]], distance=0)
        track_graph_with_bin_centers_edges.remove_edge(node1, node2)
        for ind, (node_id, pos) in enumerate(zip(new_node_ids, xy)):
            track_graph_with_bin_centers_edges.nodes[
                node_id]["pos"] = pos
            track_graph_with_bin_centers_edges.nodes[
                node_id]["edge_id"] = edge_ind
            if ind % 2:
                track_graph_with_bin_centers_edges.nodes[
                    node_id]["is_bin_edge"] = False
            else:
                track_graph_with_bin_centers_edges.nodes[
                    node_id]["is_bin_edge"] = True
        track_graph_with_bin_centers_edges.nodes[node1]["edge_id"] = edge_ind
        track_graph_with_bin_centers_edges.nodes[node2]["edge_id"] = edge_ind
        track_graph_with_bin_centers_edges.nodes[node1]["is_bin_edge"] = True
        track_graph_with_bin_centers_edges.nodes[node2]["is_bin_edge"] = True
        n_nodes = len(track_graph_with_bin_centers_edges.nodes)

    return track_graph_with_bin_centers_edges


def extract_bin_info_from_track_graph(
        track_graph, track_graph_with_bin_centers_edges, edge_order,
        edge_spacing):
    """For each node, find edge_id, is_bin_edge, x_position, y_position, and
    linear_position"""
    nodes_df = (pd.DataFrame.from_dict(
        dict(track_graph_with_bin_centers_edges.nodes(data=True)),
        orient="index")
        .assign(x_position=lambda df: np.asarray(list(df.pos))[:, 0])
        .assign(y_position=lambda df: np.asarray(list(df.pos))[:, 1])
        .drop(columns="pos")
    )
    node_linear_position, _, _ = _calculate_linear_position(
        track_graph, np.asarray(nodes_df.loc[:, ["x_position", "y_position"]]),
        nodes_df.edge_id, edge_order, edge_spacing)
    nodes_df["linear_position"] = node_linear_position
    nodes_df = nodes_df.rename_axis(index="node_id")
    edge_avg_linear_position = nodes_df.groupby(
        "edge_id").linear_position.mean().rename("edge_avg_linear_position")

    nodes_df = (pd.merge(nodes_df.reset_index(), edge_avg_linear_position,
                         on="edge_id")
                .sort_values(
        by=['edge_avg_linear_position', 'linear_position'],
        axis='rows')
        .set_index("node_id")
        .drop(columns="edge_avg_linear_position")
    )

    return nodes_df


def get_track_grid(track_graph, edge_order, edge_spacing, place_bin_size):
    track_graph_with_bin_centers_edges = make_track_graph_with_bin_centers_edges(
        track_graph, place_bin_size)
    nodes_df = extract_bin_info_from_track_graph(
        track_graph, track_graph_with_bin_centers_edges, edge_order,
        edge_spacing)

    # Dataframe with nodes from track graph only
    original_nodes = list(track_graph.nodes)
    original_nodes_df = nodes_df.loc[original_nodes].reset_index()

    # Dataframe with only added edge nodes
    place_bin_edges_nodes_df = nodes_df.loc[~nodes_df.index.isin(
        original_nodes) & nodes_df.is_bin_edge].reset_index()

    # Dataframe with only added center nodes
    place_bin_centers_nodes_df = (nodes_df
                                  .loc[~nodes_df.is_bin_edge]
                                  .reset_index())

    # Determine place bin edges and centers.
    # Make sure to remove duplicate nodes from bins with no gaps
    is_duplicate_edge = np.isclose(
        np.diff(np.asarray(place_bin_edges_nodes_df.linear_position)), 0.0)
    is_duplicate_edge = np.append(is_duplicate_edge, False)
    no_duplicate_place_bin_edges_nodes_df = (
        place_bin_edges_nodes_df.iloc[~is_duplicate_edge])
    place_bin_edges = np.asarray(
        no_duplicate_place_bin_edges_nodes_df.linear_position)
    place_bin_centers = get_centers(place_bin_edges)

    # Compute distance between nodes
    distance_between_nodes = dict(
        nx.all_pairs_dijkstra_path_length(
            track_graph_with_bin_centers_edges, weight="distance"))

    # Figure out which points are on the track and not just gaps
    change_edge_ind = np.nonzero(np.diff(
        no_duplicate_place_bin_edges_nodes_df.edge_id
    ))[0]

    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        n_edges = len(edge_order)
        edge_spacing = [edge_spacing, ] * (n_edges - 1)

    is_track_interior = np.ones_like(place_bin_centers, dtype=np.bool)
    not_track = change_edge_ind[np.asarray(edge_spacing) > 0]
    is_track_interior[not_track] = False

    # Add information about bin centers not on track
    place_bin_centers_nodes_df = (
        pd.concat(
            (place_bin_centers_nodes_df,
             pd.DataFrame({
                 "linear_position": place_bin_centers[~is_track_interior],
                 "node_id": -1,
                 "edge_id": -1,
                 "is_bin_edge": False,
             })))
        .sort_values(by=['linear_position'], axis='rows')
    ).reset_index(drop=True)

    # Other needed information
    edges = [place_bin_edges]
    centers_shape = (place_bin_centers.size,)

    return (
        place_bin_centers[:, np.newaxis],
        place_bin_edges[:, np.newaxis],
        is_track_interior,
        distance_between_nodes,
        centers_shape,
        edges,
        track_graph_with_bin_centers_edges,
        original_nodes_df,
        place_bin_edges_nodes_df,
        place_bin_centers_nodes_df,
        nodes_df.reset_index(),
    )


def get_track_boundary(is_track, n_dims=2, connectivity=1):
    structure = ndimage.generate_binary_structure(
        rank=n_dims, connectivity=connectivity)
    return ndimage.binary_dilation(is_track, structure=structure) ^ is_track


def order_border(border):
    '''
    https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    '''
    n_points = border.shape[0]
    clf = NearestNeighbors(n_neighbors=2).fit(border)
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


def get_track_border_points(is_maze, edges, n_dims=2, connectivity=1):
    '''

    Parameters
    ----------
    is_maze : ndarray, shape (n_x_bins, n_y_bins)
    edges : list of ndarray

    '''
    border = get_track_boundary(
        is_maze, n_dims=n_dims, connectivity=connectivity)

    inds = np.nonzero(border)
    centers = [get_centers(x) for x in edges]
    border = np.stack([center[ind] for center, ind in zip(centers, inds)],
                      axis=1)
    return order_border(border)


@njit
def diffuse(
    position_grid: np.ndarray,
    Fx: float,
    Fy: float,
    is_track_interior: np.ndarray,
    is_track_boundary: np.ndarray
):
    """
    Parameters
    ----------
    position_grid : np.ndarray, shape (n_bins_x, n_bins_y)
        Function to diffusion
    Fx : float
        Diffusion coefficient x
    Fy : float
        Diffusion coefficient y
    is_track_interior : np.ndarray, shape (n_bins_x, n_bins_y)
    is_track_boundary : np.ndarray, shape (n_bins_x, n_bins_y)

    Returns
    -------
    diffused_grid : np.ndarray, shape (n_bins_x, n_bins_y)

    """
    # interior points
    for x_ind, y_ind in zip(*np.nonzero(is_track_interior)):
        # no flux boundary condition
        if is_track_boundary[x_ind - 1, y_ind]:
            position_grid[x_ind - 1, y_ind] = position_grid[x_ind, y_ind]

        if is_track_boundary[x_ind + 1, y_ind]:
            position_grid[x_ind + 1, y_ind] = position_grid[x_ind, y_ind]

        if is_track_boundary[x_ind, y_ind - 1]:
            position_grid[x_ind, y_ind - 1] = position_grid[x_ind, y_ind]

        if is_track_boundary[x_ind, y_ind + 1]:
            position_grid[x_ind, y_ind + 1] = position_grid[x_ind, y_ind]

        position_grid[x_ind, y_ind] += (
            Fx * (position_grid[x_ind - 1, y_ind] -
                  2.0 * position_grid[x_ind, y_ind] +
                  position_grid[x_ind + 1, y_ind]) +
            Fy * (position_grid[x_ind, y_ind - 1] -
                  2.0 * position_grid[x_ind, y_ind] +
                  position_grid[x_ind, y_ind + 1]))

    return position_grid


@njit
def run_diffusion(
    position_grid: np.ndarray,
    is_track_interior: np.ndarray,
    is_track_boundary: np.ndarray,
    dx: float,
    dy: float,
    std=6.0,
    alpha=0.5,
    dt=0.250,
):
    Fx = alpha * (dt / dx**2)
    Fy = alpha * (dt / dy**2)

    T = std**2 / (2 * alpha)
    n_time = int((T // dt) + 1)

    for _ in range(n_time):
        position_grid = diffuse(position_grid, Fx, Fy,
                                is_track_interior, is_track_boundary)

    return position_grid


@njit
def diffuse_each_bin(
    is_track_interior: np.ndarray,
    is_track_boundary: np.ndarray,
    dx: float,
    dy: float,
    std=6.0,
    alpha=0.5,
):
    '''
    For each position bin in the grid, diffuse by `std`.

    Parameters
    ----------
    is_track_interior : np.ndarray, shape (n_bins_x, n_bins_y)
        Boolean that denotes which bins that are on the track
    is_track_boundary : np.ndarray, shape (n_bins_x, n_bins_y)
        Boolean that denotes which bins that are just outside the track
    dx : float
        Size of grid bins in x-direction
    dy : float
        Size of grid bins in y-direction
    std : float
        Standard deviation of the diffusion if it were Gaussian
    alpha : float
        Diffusion constant. Should be 0.5 if Gaussian diffusion.

    Returns
    -------
    diffused_grid : np.ndarray, shape (n_bins_x * n_bins_y, n_bins_x * n_bins_y)
        For each bin in the grid, the diffusion of that bin

    '''
    x_inds, y_inds = np.nonzero(is_track_interior)
    n_interior_bins = len(x_inds)
    n_bins = is_track_interior.shape[0] * is_track_interior.shape[1]
    bins_shape = is_track_interior.shape
    diffused_grid = np.zeros((n_bins, *bins_shape))

    dt = 0.25 / (alpha / dx**2 + alpha / dy**2)

    for ind in range(n_interior_bins):
        # initial conditions
        position_grid = np.zeros(bins_shape)
        position_grid[x_inds[ind], y_inds[ind]] = 1.0

        diffused_grid[x_inds[ind] + y_inds[ind] * bins_shape[0]] = run_diffusion(
            position_grid,
            is_track_interior,
            is_track_boundary,
            dx,
            dy,
            std=std,
            alpha=alpha,
            dt=dt
        )

    return diffused_grid


def get_bin_ind(sample, edges):
    '''Extracted from histogramdd to get bin inds'''

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    return Ncount
