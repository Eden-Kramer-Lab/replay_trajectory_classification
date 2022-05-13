import numpy as np
from replay_trajectory_classification.environments import (diffuse_each_bin,
                                                           get_bin_ind)


def estimate_diffusion_position_distance(
        positions,
        edges,
        is_track_interior=None,
        is_track_boundary=None,
        position_std=3.0,
        bin_distances=None,
):
    '''

    Parameters
    ----------
    positions : ndarray, shape (n_time, n_position_dims)
    position_std : float

    Returns
    -------
    position_distance : ndarray, shape (n_time, n_position_bins)

    '''
    if bin_distances is None:
        n_time = positions.shape[0]

        dx = edges[0][1] - edges[0][0]
        dy = edges[1][1] - edges[1][0]

        bin_distances = diffuse_each_bin(
            is_track_interior,
            is_track_boundary,
            dx,
            dy,
            std=position_std,
        ).reshape((n_time, -1), order='F')

    bin_ind = get_bin_ind(positions, [edge[1:-1] for edge in edges])
    linear_ind = np.ravel_multi_index(
        bin_ind, [len(edge) - 1 for edge in edges], order='F')
    return bin_distances[linear_ind]


def estimate_diffusion_position_density(
        positions,
        edges,
        is_track_interior=None,
        is_track_boundary=None,
        position_std=3.0,
        bin_distances=None,
        block_size=100):
    '''

    Parameters
    ----------
    place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)
    positions : ndarray, shape (n_time, n_position_dims)
    position_std : float

    Returns
    -------
    position_density : ndarray, shape (n_position_bins,)

    '''
    n_time = positions.shape[0]

    if block_size is None:
        block_size = n_time

    return np.mean(estimate_diffusion_position_distance(
        positions,
        edges,
        is_track_interior=is_track_interior,
        is_track_boundary=is_track_boundary,
        position_std=position_std,
        bin_distances=bin_distances,
    ), axis=0)
