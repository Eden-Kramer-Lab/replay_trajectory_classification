"""Classes for constructing discrete grids of spatial environments in 1D and 2D.

This module provides the Environment class and related functions for creating
spatial discretizations used in trajectory decoding and classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import joblib
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numba import njit
from numpy.typing import NDArray
from scipy import ndimage
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from track_linearization import plot_graph_as_1D

from replay_trajectory_classification.core import atleast_2d, get_centers

try:
    from networkx.convert_matrix import from_scipy_sparse_array
except ImportError:  # pragma: no cover
    # Networkx < 2.6
    from networkx import from_scipy_sparse_matrix as from_scipy_sparse_array


@dataclass
class Environment:
    """Represent the spatial environment with a discrete grid.

    Parameters
    ----------
    environment_name : str, optional
        Name identifier for the environment, by default "".
    place_bin_size : float, optional
        Approximate size of the position bins, by default 2.0.
    track_graph : networkx.Graph, optional
        Graph representing the 1D spatial topology, by default None.
    edge_order : tuple of 2-tuples, optional
        The order of the edges in 1D space, by default None.
    edge_spacing : None or int or tuple of len n_edges-1, optional
        Any gaps between the edges in 1D space, by default None.
    is_track_interior : NDArray[np.bool_] or None, optional
        If given, this will be used to define the valid areas of the track.
        Must be of type boolean, by default None.
    position_range : sequence, optional
        A sequence of `n_position_dims`, each an optional (lower, upper)
        tuple giving the outer bin edges for position.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of
        `n_position_dims` None values, by default None.
    infer_track_interior : bool, optional
        If True, then use the given positions to figure out the valid track
        areas, by default True.
    fill_holes : bool, optional
        Fill holes when inferring the track, by default False.
    dilate : bool, optional
        Inflate the available track area with binary dilation, by default False.
    bin_count_threshold : int, optional
        Greater than this number of samples should be in the bin for it to
        be considered on the track, by default 0.

    """

    environment_name: str = ""
    place_bin_size: float = 2.0
    track_graph: Optional[nx.Graph] = None
    edge_order: Optional[tuple] = None
    edge_spacing: Optional[tuple] = None
    is_track_interior: Optional[NDArray[np.bool_]] = None
    position_range: Optional[list[NDArray[np.float64]]] = None
    infer_track_interior: bool = True
    fill_holes: bool = False
    dilate: bool = False
    bin_count_threshold: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Environment, str)):
            return NotImplemented
        if isinstance(other, str):
            return self.environment_name == other
        return self.environment_name == other.environment_name

    def fit_place_grid(
        self,
        position: Optional[NDArray[np.float64]] = None,
        infer_track_interior: bool = True,
    ):
        """Fit a discrete grid of the spatial environment.

        Parameters
        ----------
        position : NDArray[np.float64], optional
            Position of the animal with shape (n_time, n_position_dims), by default None.
        infer_track_interior : bool, optional
            Whether to infer the spatial geometry of track from position, by default True.

        Returns
        -------
        Environment
            Returns self for method chaining.

        """
        if self.track_graph is None:
            if position is None:
                raise ValueError("Must provide position if no track graph given.")
            (
                self.edges_,
                self.place_bin_edges_,
                self.place_bin_centers_,
                self.centers_shape_,
            ) = get_grid(
                position,
                self.place_bin_size,
                self.position_range,
            )

            self.infer_track_interior = infer_track_interior

            if self.is_track_interior is None and self.infer_track_interior:
                self.is_track_interior_ = get_track_interior(
                    position,
                    self.edges_,
                    self.fill_holes,
                    self.dilate,
                    self.bin_count_threshold,
                )
            elif self.is_track_interior is None and not self.infer_track_interior:
                self.is_track_interior_ = np.ones(self.centers_shape_, dtype=bool)

            if len(self.edges_) > 1:
                self.is_track_boundary_ = get_track_boundary(
                    self.is_track_interior_, connectivity=1
                )
            else:
                self.is_track_boundary_ = None

        else:
            (
                self.place_bin_centers_,
                self.place_bin_edges_,
                self.is_track_interior_,
                self.distance_between_nodes_,
                self.centers_shape_,
                self.edges_,
                self.track_graph_with_bin_centers_edges_,
                self.original_nodes_df_,
                self.place_bin_edges_nodes_df_,
                self.place_bin_centers_nodes_df_,
                self.nodes_df_,
            ) = get_track_grid(
                self.track_graph,
                self.edge_order,
                self.edge_spacing,
                self.place_bin_size,
            )
            self.is_track_boundary_ = None

        return self

    def plot_grid(self, ax: Optional[matplotlib.axes.Axes] = None):
        """Plot the fitted spatial grid of the environment.

        Parameters
        ----------
        ax : plt.axes, optional
            Plot on this axis if given, by default None

        """
        if self.track_graph is not None:
            if ax is None:
                _, ax = plt.subplots(figsize=(15, 2))

            plot_graph_as_1D(
                self.track_graph, self.edge_order, self.edge_spacing, ax=ax
            )
            for edge in self.edges_[0]:
                ax.axvline(edge.squeeze(), linewidth=0.5, color="black")
            ax.set_ylim((0, 0.1))
        else:
            if ax is None:
                _, ax = plt.subplots(figsize=(6, 7))
            ax.pcolormesh(
                self.edges_[0], self.edges_[1], self.is_track_interior_.T, cmap="bone_r"
            )
            ax.set_xticks(self.edges_[0], minor=True)
            ax.set_yticks(self.edges_[1], minor=True)
            ax.grid(visible=True, which="minor")
            ax.grid(visible=False, which="major")

    def save_environment(self, filename: str = "environment.pkl"):
        """Saves the environment as a pickled file.

        Parameters
        ----------
        filename : str, optional
            File name to pickle the environment to, by default "environment.pkl"

        """
        joblib.dump(self, filename)

    @staticmethod
    def load_environment(filename: str = "environment.pkl"):
        """Load the environment from a file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to load the environment from, by default 'environment.pkl'.

        Returns
        -------
        Environment
            Loaded environment instance.

        """
        return joblib.load(filename)


def get_n_bins(
    position: NDArray[np.float64],
    bin_size: float = 2.5,
    position_range: Optional[list[NDArray[np.float64]]] = None,
) -> NDArray[np.int32]:
    """Get number of bins needed to span a range given a bin size.

    Parameters
    ----------
    position : NDArray[np.float64], shape (n_time, n_position_dims)
        Position data to determine extent from.
    bin_size : float, optional
        Size of each position bin, by default 2.5.
    position_range : None or list of NDArray[np.float64], optional
        Use this to define the extent instead of position.

    Returns
    -------
    n_bins : NDArray[np.int32], shape (n_position_dims,)
        Number of bins needed for each position dimension.

    """
    if position_range is not None:
        extent = np.diff(position_range, axis=1).squeeze()
    else:
        extent = np.ptp(position, axis=0)

    return np.ceil(extent / bin_size).astype(np.int32)


def get_grid(
    position: NDArray[np.float64],
    bin_size: float = 2.5,
    position_range: Optional[list[NDArray[np.float64]]] = None,
) -> tuple[tuple, NDArray[np.float64], NDArray[np.float64], tuple]:
    """Get the spatial grid of bins.

    Parameters
    ----------
    position : NDArray[np.float64], shape (n_time, n_position_dims)
        Position data to create grid from.
    bin_size : float, optional
        Maximum size of each position bin, by default 2.5.
    position_range : None or list of NDArray[np.float64], optional
        Use this to define the extent instead of position.

    Returns
    -------
    edges : tuple of NDArray[np.float64], len n_position_dims
        Bin edges for each dimension.
    place_bin_edges : NDArray[np.float64], shape (n_bins, n_position_dims)
        Edges of each position bin.
    place_bin_centers : NDArray[np.float64], shape (n_bins, n_position_dims)
        Center of each position bin.
    centers_shape : tuple of int
        Shape of the position grid.

    """
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    n_bins = get_n_bins(position, bin_size, position_range)
    _, edges = np.histogramdd(position, bins=n_bins, range=position_range)
    if len(edges) > 1:
        edges = [
            np.insert(
                edge,
                (0, len(edge)),
                (edge[0] - np.diff(edge)[0], edge[-1] + np.diff(edge)[0]),
            )
            for edge in edges
        ]
    mesh_edges = np.meshgrid(*edges)
    place_bin_edges = np.stack([edge.ravel() for edge in mesh_edges], axis=1)

    mesh_centers = np.meshgrid(*[get_centers(edge) for edge in edges])
    place_bin_centers = np.stack([center.ravel() for center in mesh_centers], axis=1)
    centers_shape = mesh_centers[0].T.shape

    return edges, place_bin_edges, place_bin_centers, centers_shape


def get_track_interior(
    position: NDArray[np.float64],
    bins: int | NDArray[np.float64],
    fill_holes: bool = False,
    dilate: bool = False,
    bin_count_threshold: int = 0,
) -> NDArray[np.bool_]:
    """Infer the interior bins of the track given positions.

    Parameters
    ----------
    position : NDArray[np.float64], shape (n_time, n_position_dims)
        Position data to infer track interior from.
    bins : sequence or int
        The bin specification:

        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).
    fill_holes : bool, optional
        Fill any holes in the extracted track interior bins, by default False.
    dilate : bool, optional
        Inflate the extracted track interior bins, by default False.
    bin_count_threshold : int, optional
        Greater than this number of samples should be in the bin for it to
        be considered on the track, by default 0.

    Returns
    -------
    is_track_interior : NDArray[np.bool_], shape (n_bins,)
        Boolean array indicating the interior bins of the track.

    """
    bin_counts, _ = np.histogramdd(position, bins=bins)
    is_track_interior = (bin_counts > bin_count_threshold).astype(int)
    n_position_dims = position.shape[1]
    if n_position_dims > 1:
        structure = ndimage.generate_binary_structure(n_position_dims, 1)
        is_track_interior = ndimage.binary_closing(
            is_track_interior, structure=structure
        )
        if fill_holes:
            is_track_interior = ndimage.binary_fill_holes(is_track_interior)

        if dilate:
            is_track_interior = ndimage.binary_dilation(is_track_interior)
        # adjust for boundary edges in 2D
        is_track_interior[-1] = False
        is_track_interior[:, -1] = False
    return is_track_interior.astype(bool)


def get_track_segments_from_graph(track_graph: nx.Graph) -> NDArray[np.float64]:
    """Return a 2D array of node positions corresponding to each edge.

    Parameters
    ----------
    track_graph : networkx.Graph
        Graph representing the track structure with node positions.

    Returns
    -------
    track_segments : NDArray[np.float64], shape (n_segments, 2, n_position_dims)
        Array of node position pairs for each edge segment.

    """
    node_positions = nx.get_node_attributes(track_graph, "pos")
    return np.asarray(
        [
            (node_positions[node1], node_positions[node2])
            for node1, node2 in track_graph.edges()
        ]
    )


def project_points_to_segment(
    track_segments: NDArray[np.float64], position: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Find the closest point on a track segment in terms of Euclidean distance.

    Parameters
    ----------
    track_segments : NDArray[np.float64], shape (n_segments, 2, n_position_dims)
        Array of track segment node positions.
    position : NDArray[np.float64], shape (n_time, n_position_dims)
        Position data to project onto segments.

    Returns
    -------
    projected_positions : NDArray[np.float64], shape (n_time, n_segments, n_position_dims)
        Projected positions on track segments.

    """
    segment_diff = np.diff(track_segments, axis=1).squeeze(axis=1)
    sum_squares = np.sum(segment_diff**2, axis=1)
    node1 = track_segments[:, 0, :]
    nx = (
        np.sum(segment_diff * (position[:, np.newaxis, :] - node1), axis=2)
        / sum_squares
    )
    nx[np.where(nx < 0)] = 0.0
    nx[np.where(nx > 1)] = 1.0
    return node1[np.newaxis, ...] + (
        nx[:, :, np.newaxis] * segment_diff[np.newaxis, ...]
    )


def _calculate_linear_position(
    track_graph: nx.Graph,
    position: NDArray[np.float64],
    track_segment_id: NDArray[np.int64],
    edge_order: list[tuple],
    edge_spacing: Union[float, list],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Determines the linear position given a 2D position and a track graph.

    Parameters
    ----------
    track_graph : nx.Graph
    position : NDArray[np.float64], shape (n_time, n_position_dims)
    track_segment_id : NDArray[np.int64], shape (n_time,)
    edge_order : list of 2-tuples
    edge_spacing : float or list, len n_edges - 1

    Returns
    -------
    linear_position : NDArray[np.float64], shape (n_time,)
    projected_track_positions_x : NDArray[np.float64], shape (n_time,)
    projected_track_positions_y : NDArray[np.float64], shape (n_time,)

    """
    is_nan = np.isnan(track_segment_id)
    track_segment_id[is_nan] = 0  # need to check
    track_segment_id = track_segment_id.astype(int)

    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_positions = project_points_to_segment(track_segments, position)
    n_time = projected_track_positions.shape[0]
    projected_track_positions = projected_track_positions[
        (np.arange(n_time), track_segment_id)
    ]

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        edge_spacing = [
            edge_spacing,
        ] * (n_edges - 1)

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
        for e, snlp in zip(edge_order, start_node_linear_position)
    }

    start_node_linear_position = np.asarray(
        [
            track_segment_id_to_start_node_linear_position[edge_id]
            for edge_id in track_segment_id
        ]
    )

    track_segment_id_to_edge = {track_graph.edges[e]["edge_id"]: e for e in edge_order}
    start_node_id = np.asarray(
        [track_segment_id_to_edge[edge_id][0] for edge_id in track_segment_id]
    )
    start_node_2D_position = np.asarray(
        [track_graph.nodes[node]["pos"] for node in start_node_id]
    )

    linear_position = start_node_linear_position + (
        np.linalg.norm(start_node_2D_position - projected_track_positions, axis=1)
    )
    linear_position[is_nan] = np.nan

    return (
        linear_position,
        projected_track_positions[:, 0],
        projected_track_positions[:, 1],
    )


def make_track_graph_with_bin_centers_edges(
    track_graph: nx.Graph, place_bin_size: float
) -> nx.Graph:
    """Insert the bin center and bin edge positions as nodes in the track graph.

    Parameters
    ----------
    track_graph : nx.Graph
    place_bin_size : float

    Returns
    -------
    track_graph_with_bin_centers_edges : nx.Graph

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
            f = interp1d((node1_x_pos, node2_x_pos), (node1_y_pos, node2_y_pos))
            xnew = np.linspace(node1_x_pos, node2_x_pos, num=n_bins, endpoint=True)
            xy = np.stack((xnew, f(xnew)), axis=1)
        else:
            ynew = np.linspace(node1_y_pos, node2_y_pos, num=n_bins, endpoint=True)
            xnew = np.ones_like(ynew) * node1_x_pos
            xy = np.stack((xnew, ynew), axis=1)
        dist_between_nodes = np.linalg.norm(np.diff(xy, axis=0), axis=1)

        new_node_ids = n_nodes + np.arange(len(dist_between_nodes) + 1)
        nx.add_path(
            track_graph_with_bin_centers_edges,
            [*new_node_ids],
            distance=dist_between_nodes[0],
        )
        nx.add_path(
            track_graph_with_bin_centers_edges, [node1, new_node_ids[0]], distance=0
        )
        nx.add_path(
            track_graph_with_bin_centers_edges, [node2, new_node_ids[-1]], distance=0
        )
        track_graph_with_bin_centers_edges.remove_edge(node1, node2)
        for ind, (node_id, pos) in enumerate(zip(new_node_ids, xy)):
            track_graph_with_bin_centers_edges.nodes[node_id]["pos"] = pos
            track_graph_with_bin_centers_edges.nodes[node_id]["edge_id"] = edge_ind
            if ind % 2:
                track_graph_with_bin_centers_edges.nodes[node_id]["is_bin_edge"] = False
            else:
                track_graph_with_bin_centers_edges.nodes[node_id]["is_bin_edge"] = True
        track_graph_with_bin_centers_edges.nodes[node1]["edge_id"] = edge_ind
        track_graph_with_bin_centers_edges.nodes[node2]["edge_id"] = edge_ind
        track_graph_with_bin_centers_edges.nodes[node1]["is_bin_edge"] = True
        track_graph_with_bin_centers_edges.nodes[node2]["is_bin_edge"] = True
        n_nodes = len(track_graph_with_bin_centers_edges.nodes)

    return track_graph_with_bin_centers_edges


def extract_bin_info_from_track_graph(
    track_graph: nx.Graph,
    track_graph_with_bin_centers_edges: nx.Graph,
    edge_order: list[tuple],
    edge_spacing: Union[float, list],
) -> pd.DataFrame:
    """For each node, find edge_id, is_bin_edge, x_position, y_position, and
    linear_position.

    Parameters
    ----------
    track_graph : nx.Graph
    track_graph_with_bin_centers_edges : nx.Graph
    edge_order : list of 2-tuples
    edge_spacing : list, len n_edges - 1

    Returns
    -------
    nodes_df : pd.DataFrame
        Collect information about each bin

    """
    nodes_df = (
        pd.DataFrame.from_dict(
            dict(track_graph_with_bin_centers_edges.nodes(data=True)), orient="index"
        )
        .assign(x_position=lambda df: np.asarray(list(df.pos))[:, 0])
        .assign(y_position=lambda df: np.asarray(list(df.pos))[:, 1])
        .drop(columns="pos")
    )
    node_linear_position, _, _ = _calculate_linear_position(
        track_graph,
        np.asarray(nodes_df.loc[:, ["x_position", "y_position"]]),
        np.asarray(nodes_df.edge_id),
        edge_order,
        edge_spacing,
    )
    nodes_df["linear_position"] = node_linear_position
    nodes_df = nodes_df.rename_axis(index="node_id")
    edge_avg_linear_position = (
        nodes_df.groupby("edge_id")
        .linear_position.mean()
        .rename("edge_avg_linear_position")
    )

    nodes_df = (
        pd.merge(nodes_df.reset_index(), edge_avg_linear_position, on="edge_id")
        .sort_values(by=["edge_avg_linear_position", "linear_position"], axis="rows")
        .set_index("node_id")
        .drop(columns="edge_avg_linear_position")
    )

    return nodes_df


def get_track_grid(
    track_graph: nx.Graph,
    edge_order: list[tuple],
    edge_spacing: Union[float, list],
    place_bin_size: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    dict,
    tuple,
    tuple,
    nx.Graph,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Figures out 1D spatial bins given a track graph.

    Parameters
    ----------
    track_graph : nx.Graph
    edge_order : list of 2-tuples
    edge_spacing : list, len n_edges - 1
    place_bin_size : float

    Returns
    -------
    place_bin_centers : NDArray[np.float64], shape (n_bins, n_position_dims)
    place_bin_edges : NDArray[np.float64], shape (n_bins + n_position_dims, n_position_dims)
    is_track_interior : NDArray[np.bool_], shape (n_bins, n_position_dim)
    distance_between_nodes : dict
    centers_shape : tuple
    edges : tuple of NDArray[np.float64]
    track_graph_with_bin_centers_edges : nx.Graph
    original_nodes_df : pd.DataFrame
        Table of information about the original nodes in the track graph
    place_bin_edges_nodes_df : pd.DataFrame
        Table of information with bin edges and centers
    place_bin_centers_nodes_df : pd.DataFrame
        Table of information about bin centers
    nodes_df : pd.DataFrame
        Table of information with information about the original nodes,
        bin edges, and bin centers

    """
    track_graph_with_bin_centers_edges = make_track_graph_with_bin_centers_edges(
        track_graph, place_bin_size
    )
    nodes_df = extract_bin_info_from_track_graph(
        track_graph, track_graph_with_bin_centers_edges, edge_order, edge_spacing
    )

    # Dataframe with nodes from track graph only
    original_nodes = list(track_graph.nodes)
    original_nodes_df = nodes_df.loc[original_nodes].reset_index()

    # Dataframe with only added edge nodes
    place_bin_edges_nodes_df = nodes_df.loc[
        ~nodes_df.index.isin(original_nodes) & nodes_df.is_bin_edge
    ].reset_index()

    # Dataframe with only added center nodes
    place_bin_centers_nodes_df = nodes_df.loc[~nodes_df.is_bin_edge].reset_index()

    # Determine place bin edges and centers.
    # Make sure to remove duplicate nodes from bins with no gaps
    is_duplicate_edge = np.isclose(
        np.diff(np.asarray(place_bin_edges_nodes_df.linear_position)), 0.0
    )
    is_duplicate_edge = np.append(is_duplicate_edge, False)
    no_duplicate_place_bin_edges_nodes_df = place_bin_edges_nodes_df.iloc[
        ~is_duplicate_edge
    ]
    place_bin_edges = np.asarray(no_duplicate_place_bin_edges_nodes_df.linear_position)
    place_bin_centers = get_centers(place_bin_edges)

    # Compute distance between nodes
    distance_between_nodes = dict(
        nx.all_pairs_dijkstra_path_length(
            track_graph_with_bin_centers_edges, weight="distance"
        )
    )

    # Figure out which points are on the track and not just gaps
    change_edge_ind = np.nonzero(
        np.diff(no_duplicate_place_bin_edges_nodes_df.edge_id)
    )[0]

    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        n_edges = len(edge_order)
        edge_spacing = [
            edge_spacing,
        ] * (n_edges - 1)

    is_track_interior = np.ones_like(place_bin_centers, dtype=bool)
    not_track = change_edge_ind[np.asarray(edge_spacing) > 0]
    is_track_interior[not_track] = False

    # Add information about bin centers not on track
    place_bin_centers_nodes_df = (
        pd.concat(
            (
                place_bin_centers_nodes_df,
                pd.DataFrame(
                    {
                        "linear_position": place_bin_centers[~is_track_interior],
                        "node_id": -1,
                        "edge_id": -1,
                        "is_bin_edge": False,
                    }
                ),
            )
        ).sort_values(by=["linear_position"], axis="rows")
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


def get_track_boundary(
    is_track_interior: NDArray[np.bool_],
    n_position_dims: int = 2,
    connectivity: int = 1,
) -> NDArray[np.bool_]:
    """Determines the boundary of the valid interior track bins. The boundary
    are not bins on the track but surround it.

    Parameters
    ----------
    is_track_interior : NDArray[np.bool_], shape (n_bins_x, n_bins_y)
    n_position_dims : int
    connectivity : int
         `connectivity` determines which elements of the output array belong
         to the structure, i.e., are considered as neighbors of the central
         element. Elements up to a squared distance of `connectivity` from
         the center are considered neighbors. `connectivity` may range from 1
         (no diagonal elements are neighbors) to `rank` (all elements are
         neighbors).

    Returns
    -------
    is_track_boundary : NDArray[np.bool_], shape (n_bins,)

    """
    structure = ndimage.generate_binary_structure(
        rank=n_position_dims, connectivity=connectivity
    )
    return (
        ndimage.binary_dilation(is_track_interior, structure=structure)
        ^ is_track_interior
    )


def order_boundary(boundary: NDArray[np.float64]) -> NDArray[np.float64]:
    """Given boundary bin centers, orders them in a way to make a continuous line.

    https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line

    Parameters
    ----------
    boundary : NDArray[np.float64], shape (n_boundary_points, n_position_dims)

    Returns
    -------
    ordered_boundary : NDArray[np.float64], shape (n_boundary_points, n_position_dims)

    """
    n_points = boundary.shape[0]
    clf = NearestNeighbors(n_neighbors=2).fit(boundary)
    G = clf.kneighbors_graph()
    T = from_scipy_sparse_array(G)

    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(n_points)]
    min_idx, min_dist = 0, np.inf

    for idx, path in enumerate(paths):
        ordered = boundary[path]  # ordered nodes
        cost = np.sum(np.diff(ordered) ** 2)
        if cost < min_dist:
            min_idx, min_dist = idx, cost

    opt_order = paths[min_idx]
    return boundary[opt_order][:-1]


def get_track_boundary_points(
    is_track_interior: NDArray[np.bool_],
    edges: list[NDArray[np.float64]],
    connectivity: int = 1,
) -> NDArray[np.float64]:
    """

    Parameters
    ----------
    is_track_interior : NDArray[np.bool_], shape (n_x_bins, n_y_bins)
    edges : list of NDArray[np.float64]

    Returns
    -------
    boundary_points : NDArray[np.float64], shape (n_boundary_points, n_position_dims)

    """
    n_position_dims = len(edges)
    boundary = get_track_boundary(
        is_track_interior, n_position_dims=n_position_dims, connectivity=connectivity
    )

    inds = np.nonzero(boundary)
    centers = [get_centers(x) for x in edges]
    boundary = np.stack([center[ind] for center, ind in zip(centers, inds)], axis=1)
    return order_boundary(boundary)


@njit
def diffuse(
    position_grid: NDArray[np.float64],
    Fx: float,
    Fy: float,
    is_track_interior: NDArray[np.bool_],
    is_track_boundary: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Calculates diffusion for a single time step given a track.

    Parameters
    ----------
    position_grid : NDArray[np.float64], shape (n_bins_x, n_bins_y)
        Function to diffusion
    Fx : float
        Diffusion coefficient x
    Fy : float
        Diffusion coefficient y
    is_track_interior : NDArray[np.bool_], shape (n_bins_x, n_bins_y)
    is_track_boundary : NDArray[np.bool_], shape (n_bins_x, n_bins_y)

    Returns
    -------
    diffused_grid : NDArray[np.float64], shape (n_bins_x, n_bins_y)

    """
    # interior points
    for x_ind, y_ind in zip(*np.nonzero(is_track_interior)):
        # check if current point is on the boundary
        if not is_track_boundary[x_ind, y_ind]:
            # no flux boundary condition
            if is_track_boundary[x_ind - 1, y_ind]:
                position_grid[x_ind - 1, y_ind] = position_grid[x_ind, y_ind]

            if is_track_boundary[x_ind + 1, y_ind]:
                position_grid[x_ind + 1, y_ind] = position_grid[x_ind, y_ind]

            if is_track_boundary[x_ind, y_ind - 1]:
                position_grid[x_ind, y_ind - 1] = position_grid[x_ind, y_ind]

            if is_track_boundary[x_ind, y_ind + 1]:
                position_grid[x_ind, y_ind + 1] = position_grid[x_ind, y_ind]

        position_grid[x_ind, y_ind] += Fx * (
            position_grid[x_ind - 1, y_ind]
            - 2.0 * position_grid[x_ind, y_ind]
            + position_grid[x_ind + 1, y_ind]
        ) + Fy * (
            position_grid[x_ind, y_ind - 1]
            - 2.0 * position_grid[x_ind, y_ind]
            + position_grid[x_ind, y_ind + 1]
        )

    return position_grid


@njit
def run_diffusion(
    position_grid: NDArray[np.float64],
    is_track_interior: NDArray[np.bool_],
    is_track_boundary: NDArray[np.bool_],
    dx: float,
    dy: float,
    std: float = 6.0,
    alpha: float = 0.5,
    dt: float = 0.250,
) -> NDArray[np.float64]:
    """Calculates diffusion of a single point over time up until it matches a
    Gaussian with standard deviation `std`.

    Parameters
    ----------
    position_grid : NDArray[np.float64], shape (n_bins_x, n_bins_y)
        Function to diffusion
    is_track_interior : NDArray[np.bool_], shape (n_bins_x, n_bins_y)
        Boolean that denotes which bins that are on the track
    is_track_boundary : NDArray[np.bool_], shape (n_bins_x, n_bins_y)
        Boolean that denotes which bins that are just outside the track
    dx : float
        Size of grid bins in x-direction
    dy : float
        Size of grid bins in y-direction
    std : float
        Standard deviation of the diffusion if it were Gaussian
    alpha : float
        Diffusion constant. Should be 0.5 if Gaussian diffusion.
    dt : float
        Time step size

    Returns
    -------
    diffused_grid : NDArray[np.float64], shape (n_bins_x, n_bins_y)

    """
    Fx = alpha * (dt / dx**2)
    Fy = alpha * (dt / dy**2)

    T = std**2 / (2.0 * alpha)
    n_time = int((T // dt) + 1)

    for _ in range(n_time):
        position_grid = diffuse(
            position_grid, Fx, Fy, is_track_interior, is_track_boundary
        )

    return position_grid


@njit
def diffuse_each_bin(
    is_track_interior: NDArray[np.bool_],
    is_track_boundary: NDArray[np.bool_],
    dx: float,
    dy: float,
    std: float = 6.0,
    alpha: float = 0.5,
) -> NDArray[np.float64]:
    """
    For each position bin in the grid, diffuse by `std`.

    Parameters
    ----------
    is_track_interior : NDArray[np.bool_], shape (n_bins_x, n_bins_y)
        Boolean that denotes which bins that are on the track
    is_track_boundary : NDArray[np.bool_], shape (n_bins_x, n_bins_y)
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
    diffused_grid : NDArray[np.float64], shape (n_bins_x * n_bins_y, n_bins_x * n_bins_y)
        For each bin in the grid, the diffusion of that bin

    """
    x_inds, y_inds = np.nonzero(is_track_interior)
    n_interior_bins = len(x_inds)
    n_bins = is_track_interior.size
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
            dt=dt,
        )

    return diffused_grid


def get_bin_ind(
    sample: NDArray[np.float64], edges: list[NDArray[np.float64]]
) -> NDArray[np.int64]:
    """Figure out which bin a given sample falls into.

    Extracted from np.histogramdd.

    Parameters
    ----------
    sample : NDArray[np.float64]
    edges : list of NDArray[np.float64]

    Returns
    -------
    Ncount : NDArray[np.int64]

    """

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
        np.searchsorted(edges[i], sample[:, i], side="right")
        for i in range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(D):
        # Find which points are on the rightmost edge.
        on_edge = sample[:, i] == edges[i][-1]
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    return Ncount
