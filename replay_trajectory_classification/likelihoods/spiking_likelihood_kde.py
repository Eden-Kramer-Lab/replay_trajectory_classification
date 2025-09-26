"""Estimate Poisson likelihood using kernel density estimate for place fields.

This module provides functions for estimating spike likelihoods using
kernel density estimation (KDE) to model spatial firing patterns
without parametric assumptions.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import xarray as xr
from scipy.special import xlogy
from tqdm.auto import tqdm

from replay_trajectory_classification.core import atleast_2d
from replay_trajectory_classification.likelihoods.diffusion import (
    diffuse_each_bin,
    estimate_diffusion_position_density,
)


def gaussian_pdf(x: NDArray[np.float64], mean: NDArray[np.float64], sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the Gaussian probability density function.

    Parameters
    ----------
    x : NDArray[np.float64]
        Points at which to evaluate the PDF.
    mean : NDArray[np.float64]
        Mean of the Gaussian distribution.
    sigma : NDArray[np.float64]
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    NDArray[np.float64]
        Gaussian PDF values at the given points.

    """
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def estimate_position_distance(
    place_bin_centers: NDArray[np.float64], positions: NDArray[np.float64], position_std: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Estimate the Gaussian-weighted distance between positions and position bins.

    Parameters
    ----------
    place_bin_centers : NDArray[np.float64], shape (n_position_bins, n_position_dims)
        Center coordinates of spatial bins.
    positions : NDArray[np.float64], shape (n_time, n_position_dims)
        Position coordinates over time.
    position_std : array_like, shape (n_position_dims,)
        Standard deviation for Gaussian weighting in each dimension.

    Returns
    -------
    position_distance : NDArray[np.float64], shape (n_time, n_position_bins)
        Gaussian-weighted distance matrix.

    """
    n_time, n_position_dims = positions.shape
    n_position_bins = place_bin_centers.shape[0]

    position_distance = np.ones((n_time, n_position_bins), dtype=np.float32)

    if isinstance(position_std, (int, float)):
        position_std = [position_std] * n_position_dims

    for position_ind, std in enumerate(position_std):
        position_distance *= gaussian_pdf(
            np.expand_dims(place_bin_centers[:, position_ind], axis=0),
            np.expand_dims(positions[:, position_ind], axis=1),
            std,
        )

    return position_distance


def estimate_position_density(
    place_bin_centers: NDArray[np.float64],
    positions: NDArray[np.float64],
    position_std: Union[float, NDArray[np.float64]],
    block_size: int = 100,
) -> NDArray[np.float64]:
    """Estimate a kernel density estimate over position bins.

    Parameters
    ----------
    place_bin_centers : NDArray[np.float64], shape (n_position_bins, n_position_dims)
        Center coordinates of spatial bins.
    positions : NDArray[np.float64], shape (n_time, n_position_dims)
        Position coordinates over time.
    position_std : float or array_like, shape (n_position_dims,)
        Standard deviation for kernel density estimation.
    block_size : int, optional
        Number of bins to process in each block for memory efficiency, by default 100.

    Returns
    -------
    position_density : NDArray[np.float64], shape (n_position_bins,)
        Kernel density estimate at each position bin.

    """
    n_time = positions.shape[0]
    n_position_bins = place_bin_centers.shape[0]

    if block_size is None:
        block_size = n_time

    position_density = np.empty((n_position_bins,))
    for start_ind in range(0, n_position_bins, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        position_density[block_inds] = np.mean(
            estimate_position_distance(
                place_bin_centers[block_inds], positions, position_std
            ),
            axis=0,
        )
    return position_density


def get_firing_rate(
    is_spike: NDArray[np.bool_],
    position: NDArray[np.float64],
    place_bin_centers: NDArray[np.float64],
    is_track_interior: NDArray[np.bool_],
    not_nan_position: NDArray[np.bool_],
    occupancy: NDArray[np.float64],
    position_std: NDArray[np.float64],
    block_size: Optional[int] = None,
) -> NDArray[np.float64]:
    if is_spike.sum() > 0:
        mean_rate = is_spike.mean()
        marginal_density = np.zeros((place_bin_centers.shape[0],), dtype=np.float32)

        marginal_density[is_track_interior] = estimate_position_density(
            place_bin_centers[is_track_interior],
            np.asarray(position[is_spike & not_nan_position], dtype=np.float32),
            position_std,
            block_size=block_size,
        )
        return np.exp(np.log(mean_rate) + np.log(marginal_density) - np.log(occupancy))
    else:
        return np.zeros_like(occupancy)


def get_diffusion_firing_rate(
    is_spike: NDArray[np.bool_],
    position: NDArray[np.float64],
    edges: list[NDArray[np.float64]],
    bin_diffusion_distances: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    not_nan_position: NDArray[np.bool_],
) -> NDArray[np.float64]:
    if is_spike.sum() > 0:
        mean_rate = is_spike.mean()
        marginal_density = estimate_diffusion_position_density(
            position[is_spike & not_nan_position],
            edges,
            bin_distances=bin_diffusion_distances,
        ).astype(np.float32)

        return np.exp(np.log(mean_rate) + np.log(marginal_density) - np.log(occupancy))
    else:
        return np.zeros_like(occupancy)


def estimate_place_fields_kde(
    position: NDArray[np.float64],
    spikes: NDArray[np.float64],
    place_bin_centers: NDArray[np.float64],
    position_std: NDArray[np.float64],
    is_track_boundary: Optional[NDArray[np.bool_]] = None,
    is_track_interior: Optional[NDArray[np.bool_]] = None,
    edges: Optional[list[NDArray[np.float64]]] = None,
    place_bin_edges: Optional[NDArray[np.float64]] = None,
    use_diffusion: bool = False,
    block_size: Optional[int] = None,
) -> xr.DataArray:
    """Gives the conditional intensity of the neurons' spiking with respect to
    position.

    Parameters
    ----------
    position : NDArray[np.float64], shape (n_time, n_position_dims)
    spikes : NDArray[np.float64], shape (n_time,)
        Indicator of spike or no spike at current time.
    place_bin_centers : NDArray[np.float64], shape (n_bins, n_position_dims)
    position_std : float or array_like, shape (n_position_dims,)
        Amount of smoothing for position.  Standard deviation of kernel.
    is_track_boundary : None or NDArray[np.bool_], shape (n_bins,)
    is_track_interior : None or NDArray[np.bool_], shape (n_bins,)
    edges : None or list of NDArray[np.float64]
    place_bin_edges : NDArray[np.float64], shape (n_bins + 1, n_position_dims)
    use_diffusion : bool
        Respect track geometry by using diffusion distances
    block_size : int
        Size of data to process in chunks

    Returns
    -------
    conditional_intensity : NDArray[np.float64], shape (n_bins, n_neurons)

    """

    position = atleast_2d(position).astype(np.float32)
    place_bin_centers = atleast_2d(place_bin_centers).astype(np.float32)
    not_nan_position = np.all(~np.isnan(position), axis=1)

    if use_diffusion & (position.shape[1] > 1):
        n_total_bins = np.prod(is_track_interior.shape)
        bin_diffusion_distances = diffuse_each_bin(
            is_track_interior,
            is_track_boundary,
            dx=edges[0][1] - edges[0][0],
            dy=edges[1][1] - edges[1][0],
            std=position_std,
        ).reshape((n_total_bins, -1), order="F")

        occupancy = estimate_diffusion_position_density(
            position[not_nan_position],
            edges,
            bin_distances=bin_diffusion_distances,
        ).astype(np.float32)

        place_fields = np.stack(
            [
                get_diffusion_firing_rate(
                    is_spike,
                    position,
                    edges,
                    bin_diffusion_distances,
                    occupancy,
                    not_nan_position,
                )
                for is_spike in np.asarray(spikes, dtype=bool).T
            ],
            axis=1,
        )
    else:
        occupancy = np.zeros((place_bin_centers.shape[0],), dtype=np.float32)
        occupancy[is_track_interior.ravel(order="F")] = estimate_position_density(
            place_bin_centers[is_track_interior.ravel(order="F")],
            position[not_nan_position],
            position_std,
            block_size=block_size,
        )
        place_fields = np.stack(
            [
                get_firing_rate(
                    is_spike,
                    position,
                    place_bin_centers,
                    is_track_interior.ravel(order="F"),
                    not_nan_position,
                    occupancy,
                    position_std,
                )
                for is_spike in np.asarray(spikes, dtype=bool).T
            ],
            axis=1,
        )

    DIMS = ["position", "neuron"]
    if position.shape[1] == 1:
        names = ["position"]
        coords = {"position": place_bin_centers.squeeze()}
    elif position.shape[1] == 2:
        names = ["x_position", "y_position"]
        coords = {
            "position": pd.MultiIndex.from_arrays(
                place_bin_centers.T.tolist(), names=names
            )
        }

    return xr.DataArray(data=place_fields, coords=coords, dims=DIMS)


def poisson_log_likelihood(
    spikes: NDArray[np.float64], conditional_intensity: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Probability of parameters given spiking at a particular time.

    Parameters
    ----------
    spikes : NDArray[np.float64], shape (n_time,)
        Indicator of spike or no spike at current time.
    conditional_intensity : NDArray[np.float64], shape (n_place_bins,)
        Instantaneous probability of observing a spike

    Returns
    -------
    poisson_log_likelihood : NDArray[np.float64], shape (n_time, n_place_bins)

    """
    return (
        xlogy(spikes[:, np.newaxis], conditional_intensity[np.newaxis, :])
        - conditional_intensity[np.newaxis, :]
    )


def combined_likelihood(
    spikes: NDArray[np.float64], conditional_intensity: NDArray[np.float64]
) -> NDArray[np.float64]:
    """

    Parameters
    ----------
    spikes : NDArray[np.float64], shape (n_time, n_neurons)
    conditional_intensity : NDArray[np.float64], shape (n_bins, n_neurons)

    """
    n_time = spikes.shape[0]
    n_bins = conditional_intensity.shape[0]
    log_likelihood = np.zeros((n_time, n_bins))

    conditional_intensity = np.clip(conditional_intensity, a_min=1e-15, a_max=None)

    for is_spike, ci in zip(tqdm(spikes.T), conditional_intensity.T):
        log_likelihood += poisson_log_likelihood(is_spike, ci)

    return log_likelihood


def estimate_spiking_likelihood_kde(
    spikes: NDArray[np.float64],
    conditional_intensity: NDArray[np.float64],
    is_track_interior: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.float64]:
    """

    Parameters
    ----------
    spikes : NDArray[np.float64], shape (n_time, n_neurons)
    conditional_intensity : NDArray[np.float64], shape (n_bins, n_neurons)
    is_track_interior : None or NDArray[np.bool_], optional, shape (n_x_position_bins,
                                                                   n_y_position_bins)
    Returns
    -------
    likelihood : NDArray[np.float64], shape (n_time, n_bins)

    """
    spikes = np.asarray(spikes, dtype=np.float32)

    if is_track_interior is not None:
        is_track_interior = is_track_interior.ravel(order="F")
    else:
        n_bins = conditional_intensity.shape[0]
        is_track_interior = np.ones((n_bins,), dtype=bool)

    log_likelihood = combined_likelihood(spikes, conditional_intensity)

    mask = np.ones_like(is_track_interior, dtype=float)
    mask[~is_track_interior] = np.nan

    return log_likelihood * mask
