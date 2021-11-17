from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from replay_trajectory_classification.bins import (get_grid, get_track_grid,
                                                   get_track_interior)
from track_linearization import plot_graph_as_1D


@dataclass
class Environment:
    """
    environment_name : str, optional
    place_bin_size : float, optional
        Approximate size of the position bins.
    track_graph : networkx.Graph
        Graph representing the 1D spatial topology
    edge_order : tuple of 2-tuples
        The order of the edges in 1D space
    edge_spacing : None or int or tuples of len n_edges-1
        Any gapes between the edges in 1D space
    is_track_interior : np.ndarray
    position_range : sequence, optional
        A sequence of `n_position_dims`, each an optional (lower, upper)
        tuple giving the outer bin edges for position.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of
        `n_position_dims` None values.
    infer_track_interior : bool, optional

    """
    environment_name: str = ''
    place_bin_size: float = 2.0
    track_graph: nx.Graph = None
    edge_order: tuple = None
    edge_spacing: tuple = None
    is_track_interior: np.ndarray = None
    position_range: np.ndarray = None
    infer_track_interior: bool = True

    def __eq__(self, other):
        return self.environment_name == other

    def fit_place_grid(self, position=None, infer_track_interior=True):
        if self.track_graph is None:
            (self.edges_,
             self.place_bin_edges_,
             self.place_bin_centers_,
             self.centers_shape_
             ) = get_grid(position, self.place_bin_size, self.position_range,
                          self.infer_track_interior)

            self.infer_track_interior = infer_track_interior

            if self.is_track_interior is None and self.infer_track_interior:
                self.is_track_interior_ = get_track_interior(
                    position, self.edges_)
            elif self.is_track_interior is None and not self.infer_track_interior:
                self.is_track_interior_ = np.ones(
                    self.centers_shape_, dtype=np.bool)
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
                self.nodes_df_
            ) = get_track_grid(self.track_graph, self.edge_order,
                               self.edge_spacing, self.place_bin_size)

        return self

    def plot_grid(self, ax=None):
        if self.track_graph is not None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(15, 2))

            plot_graph_as_1D(self.track_graph, self.edge_order,
                             self.edge_spacing, ax=ax)
            for edge in self.edges_[0]:
                ax.axvline(edge.squeeze(), linewidth=0.5, color='black')
            ax.set_ylim((0, 0.1))
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 7))
            ax.pcolormesh(self.edges_[0],
                          self.edges_[1],
                          self.is_track_interior_.T,
                          cmap='bone_r')
            ax.set_xticks(self.edges_[0], minor=True)
            ax.set_yticks(self.edges_[1], minor=True)
            ax.grid(True, which='both')
