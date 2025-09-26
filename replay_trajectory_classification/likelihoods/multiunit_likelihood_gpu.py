"""Estimate marked point process likelihood using GPU acceleration.

This module provides GPU-accelerated functions for computing likelihoods
from clusterless spike data where marks are float32 waveform features.
Requires CUDA-compatible hardware and CuPy.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm

from replay_trajectory_classification.core import atleast_2d
from replay_trajectory_classification.likelihoods.diffusion import (
    diffuse_each_bin,
    estimate_diffusion_position_density,
    estimate_diffusion_position_distance,
)

try:
    import cupy as cp

    @cp.fuse
    def gaussian_pdf(x: NDArray[np.float64], mean: NDArray[np.float64], sigma: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the value of a Gaussian probability density function at x with
        given mean and sigma."""
        return cp.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * cp.sqrt(2.0 * cp.pi))

    def estimate_position_distance(
        place_bin_centers: NDArray[np.float64],
        positions: NDArray[np.float64],
        position_std: Union[float, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Estimates the Euclidean distance between positions and position bins.

        Parameters
        ----------
        place_bin_centers : NDArray[np.float64], shape (n_position_bins, n_position_dims)
        positions : NDArray[np.float64], shape (n_time, n_position_dims)
        position_std : NDArray[np.float64], shape (n_position_dims,)

        Returns
        -------
        position_distance : NDArray[np.float64], shape (n_time, n_position_bins)

        """
        n_time, n_position_dims = positions.shape
        n_position_bins = place_bin_centers.shape[0]

        if isinstance(position_std, (int, float)):
            position_std = [position_std] * n_position_dims

        position_distance = cp.ones((n_time, n_position_bins), dtype=cp.float32)

        for position_ind, std in enumerate(position_std):
            position_distance *= gaussian_pdf(
                cp.expand_dims(place_bin_centers[:, position_ind], axis=0),
                cp.expand_dims(positions[:, position_ind], axis=1),
                std,
            )

        return position_distance

    def estimate_position_density(
        place_bin_centers: NDArray[np.float64],
        positions: NDArray[np.float64],
        position_std: Union[float, NDArray[np.float64]],
        block_size: int = 100,
    ) -> NDArray[np.float64]:
        """Estimates a kernel density estimate over position bins using
        Euclidean distances.

        Parameters
        ----------
        place_bin_centers : NDArray[np.float64], shape (n_position_bins, n_position_dims)
        positions : NDArray[np.float64], shape (n_time, n_position_dims)
        position_std : float or NDArray[np.float64], shape (n_position_dims,)

        Returns
        -------
        position_density : NDArray[np.float64], shape (n_position_bins,)

        """
        n_time = positions.shape[0]
        n_position_bins = place_bin_centers.shape[0]

        if block_size is None:
            block_size = n_time

        position_density = cp.empty((n_position_bins,))
        for start_ind in range(0, n_position_bins, block_size):
            block_inds = slice(start_ind, start_ind + block_size)
            position_density[block_inds] = cp.mean(
                estimate_position_distance(
                    place_bin_centers[block_inds], positions, position_std
                ),
                axis=0,
            )
        return position_density

    def estimate_log_intensity(
        density: NDArray[np.float64], occupancy: NDArray[np.float64], mean_rate: float
    ) -> NDArray[np.float64]:
        """Calculates intensity in log space."""
        return cp.log(mean_rate) + cp.log(density) - cp.log(occupancy)

    def estimate_intensity(
        density: NDArray[np.float64], occupancy: NDArray[np.float64], mean_rate: float
    ) -> NDArray[np.float64]:
        """Calculates intensity.

        Parameters
        ----------
        density : NDArray[np.float64], shape (n_bins,)
        occupancy : NDArray[np.float64], shape (n_bins,)
        mean_rate : float

        Returns
        -------
        intensity : NDArray[np.float64], shape (n_bins,)

        """
        return cp.clip(
            cp.exp(estimate_log_intensity(density, occupancy, mean_rate)), a_min=1e-8
        )

    def estimate_log_joint_mark_intensity(
        decoding_marks: NDArray[np.float64],
        encoding_marks: NDArray[np.float64],
        mark_std: Union[float, NDArray[np.float64]],
        occupancy: NDArray[np.float64],
        mean_rate: float,
        place_bin_centers: Optional[NDArray[np.float64]] = None,
        encoding_positions: Optional[NDArray[np.float64]] = None,
        position_std: Union[float, NDArray[np.float64], None] = None,
        max_mark_diff: int = 6000,
        set_diag_zero: bool = False,
        position_distance: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Finds the joint intensity of the marks and positions in log space.

        Parameters
        ----------
        decoding_marks : NDArray[np.float64], shape (n_decoding_spikes, n_features)
        encoding_marks : NDArray[np.float64], shape (n_encoding_spikes, n_features)
        mark_std : float or NDArray[np.float64], shape (n_features,)
        occupancy : NDArray[np.float64], shape (n_position_bins,)
        mean_rate : float
        place_bin_centers : None or NDArray[np.float64], shape (n_position_bins, n_position_dims)
            If None, position distance must be not None
        encoding_positions : None or NDArray[np.float64], shape (n_decoding_spikes, n_position_dims)
            If None, position distance must be not None
        position_std : None or float or array_like, shape (n_position_dims,)
            If None, position distance must be not None
        max_mark_diff : int
            Maximum distance between integer marks.
        set_diag_zero : bool
        position_distance : NDArray[np.float64], shape (n_encoding_spikes, n_position_bins)
            Precalculated distance between position and position bins.

        Returns
        -------
        log_joint_mark_intensity : NDArray[np.float64], shape (n_decoding_spikes, n_position_bins)

        """
        n_encoding_spikes, n_marks = encoding_marks.shape

        n_decoding_spikes = decoding_marks.shape[0]
        mark_distance = cp.ones(
            (n_decoding_spikes, n_encoding_spikes), dtype=cp.float32
        )

        for mark_ind in range(n_marks):
            mark_distance *= gaussian_pdf(
                cp.expand_dims(decoding_marks[:, mark_ind], axis=1),
                cp.expand_dims(encoding_marks[:, mark_ind], axis=0),
                mark_std[mark_ind],
            )

        if set_diag_zero:
            diag_ind = (cp.arange(n_decoding_spikes), cp.arange(n_decoding_spikes))
            mark_distance[diag_ind] = 0.0

        if position_distance is None:
            position_distance = estimate_position_distance(
                place_bin_centers, encoding_positions, position_std
            ).astype(cp.float32)

        return cp.asnumpy(
            estimate_log_intensity(
                mark_distance @ position_distance / n_encoding_spikes,
                occupancy,
                mean_rate,
            )
        )

    def fit_multiunit_likelihood_gpu(
        position: NDArray[np.float64],
        multiunits: NDArray[np.float64],
        place_bin_centers: NDArray[np.float64],
        mark_std: Union[float, NDArray[np.float64]],
        position_std: Union[float, NDArray[np.float64]],
        is_track_boundary: Optional[NDArray[np.bool_]] = None,
        is_track_interior: Optional[NDArray[np.bool_]] = None,
        edges: Optional[list[NDArray[np.float64]]] = None,
        block_size: int = 100,
        use_diffusion: bool = False,
        **kwargs,
    ) -> dict:
        """Fits the clusterless place field model.

        Parameters
        ----------
        position : NDArray[np.float64], shape (n_time, n_position_dims)
        multiunits : NDArray[np.float64], shape (n_time, n_marks, n_electrodes)
        place_bin_centers : NDArray[np.float64], shape (n_bins, n_position_dims)
        mark_std : float or NDArray[np.float64], shape (n_marks,)
            Amount of smoothing for the mark features.  Standard deviation of kernel.
        position_std : float or NDArray[np.float64], shape (n_position_dims,)
            Amount of smoothing for position.  Standard deviation of kernel.
        is_track_boundary : None or NDArray[np.bool_], shape (n_bins,)
        is_track_interior : None or NDArray[np.bool_], shape (n_bins,)
        edges : None or list of NDArray[np.float64]
        block_size : int
            Size of data to process in chunks
        use_diffusion : bool
            Use diffusion to respect the track geometry.

        Returns
        -------
        encoding_model : dict

        """
        if is_track_interior is None:
            is_track_interior = np.ones((place_bin_centers.shape[0],), dtype=bool)

        position = atleast_2d(position)
        place_bin_centers = atleast_2d(place_bin_centers)
        interior_place_bin_centers = cp.asarray(
            place_bin_centers[is_track_interior.ravel(order="F")], dtype=cp.float32
        )
        gpu_is_track_interior = cp.asarray(is_track_interior.ravel(order="F"))

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
        else:
            bin_diffusion_distances = None

        if use_diffusion & (position.shape[1] > 1):
            occupancy = cp.asarray(
                estimate_diffusion_position_density(
                    position[not_nan_position],
                    edges,
                    bin_distances=bin_diffusion_distances,
                ),
                dtype=cp.float32,
            )
        else:
            occupancy = cp.zeros((place_bin_centers.shape[0],), dtype=cp.float32)
            occupancy[gpu_is_track_interior] = estimate_position_density(
                interior_place_bin_centers,
                cp.asarray(position[not_nan_position], dtype=cp.float32),
                position_std,
                block_size=block_size,
            )

        mean_rates = []
        summed_ground_process_intensity = cp.zeros(
            (place_bin_centers.shape[0],), dtype=cp.float32
        )
        encoding_marks = []
        encoding_positions = []

        n_marks = multiunits.shape[1]
        if isinstance(mark_std, (int, float)):
            mark_std = np.asarray([mark_std] * n_marks)
        else:
            mark_std = np.asarray(mark_std)

        for multiunit in np.moveaxis(multiunits, -1, 0):
            # ground process intensity
            is_spike = np.any(~np.isnan(multiunit), axis=1)
            mean_rates.append(is_spike.mean())

            if is_spike.sum() > 0:
                if use_diffusion & (position.shape[1] > 1):
                    marginal_density = cp.asarray(
                        estimate_diffusion_position_density(
                            position[is_spike & not_nan_position],
                            edges,
                            bin_distances=bin_diffusion_distances,
                        ),
                        dtype=cp.float32,
                    )
                else:
                    marginal_density = cp.zeros(
                        (place_bin_centers.shape[0],), dtype=cp.float32
                    )
                    marginal_density[gpu_is_track_interior] = estimate_position_density(
                        interior_place_bin_centers,
                        cp.asarray(
                            position[is_spike & not_nan_position], dtype=cp.float32
                        ),
                        position_std,
                        block_size=block_size,
                    )
            else:
                marginal_density = cp.zeros(
                    (place_bin_centers.shape[0],), dtype=cp.float32
                )

            summed_ground_process_intensity += estimate_intensity(
                marginal_density, occupancy, mean_rates[-1]
            )

            is_mark_features = np.any(~np.isnan(multiunit), axis=0)
            encoding_marks.append(
                cp.asarray(
                    multiunit[np.ix_(is_spike & not_nan_position, is_mark_features)],
                    dtype=cp.float32,
                )
            )
            encoding_positions.append(position[is_spike & not_nan_position])

        summed_ground_process_intensity = cp.asnumpy(
            summed_ground_process_intensity
        ) + np.spacing(1)

        return {
            "encoding_marks": encoding_marks,
            "encoding_positions": encoding_positions,
            "summed_ground_process_intensity": summed_ground_process_intensity,
            "occupancy": occupancy,
            "mean_rates": mean_rates,
            "mark_std": mark_std,
            "position_std": position_std,
            "block_size": block_size,
            "bin_diffusion_distances": bin_diffusion_distances,
            "use_diffusion": use_diffusion,
            "edges": edges,
            **kwargs,
        }

    def estimate_multiunit_likelihood_gpu(
        multiunits: NDArray[np.float64],
        encoding_marks: NDArray[np.float64],
        mark_std: NDArray[np.float64],
        place_bin_centers: NDArray[np.float64],
        encoding_positions: NDArray[np.float64],
        position_std: NDArray[np.float64],
        occupancy: NDArray[np.float64],
        mean_rates: NDArray[np.float64],
        summed_ground_process_intensity: NDArray[np.float64],
        bin_diffusion_distances: NDArray[np.float64],
        edges: list[NDArray[np.float64]],
        max_mark_diff: int = 6000,
        set_diag_zero: bool = False,
        is_track_interior: Optional[NDArray[np.bool_]] = None,
        time_bin_size: int = 1,
        block_size: int = 100,
        ignore_no_spike: bool = False,
        disable_progress_bar: bool = False,
        use_diffusion: bool = False,
    ) -> NDArray[np.float64]:
        """Estimates the likelihood of position bins given multiunit marks.

        Parameters
        ----------
        multiunits : NDArray[np.float64], shape (n_decoding_time, n_marks, n_electrodes)
        encoding_marks : NDArray[np.float64], shape (n_encoding_spikes, n_marks, n_electrodes)
        mark_std : list, shape (n_marks,)
            Amount of smoothing for mark features
        place_bin_centers : NDArray[np.float64], shape (n_bins, n_position_dims)
        encoding_positions : NDArray[np.float64], shape (n_encoding_spikes, n_position_dims)
        position_std : float or array_like, shape (n_position_dims,)
            Amount of smoothing for position
        occupancy : NDArray[np.float64], (n_bins,)
        mean_rates : list, len (n_electrodes,)
        summed_ground_process_intensity : NDArray[np.float64], shape (n_bins,)
        bin_diffusion_distances : NDArray[np.float64], shape (n_bins, n_bins)
        edges : list of NDArray[np.float64]
        max_mark_diff : int
            Maximum difference between mark features
        set_diag_zero : bool
            Remove influence of the same mark in encoding and decoding.
        is_track_interior : None or NDArray[np.bool_], shape (n_bins_x, n_bins_y)
        time_bin_size : float
            Size of time steps
        block_size : int
            Size of data to process in chunks
        ignore_no_spike : bool
            Set contribution of no spikes to zero
        disable_progress_bar : bool
            If False, a progress bar will be displayed.
        use_diffusion : bool
            Respect track geometry by using diffusion distances

        Returns
        -------
        log_likelihood : (n_time, n_bins)

        """

        if is_track_interior is None:
            is_track_interior = np.ones((place_bin_centers.shape[0],), dtype=bool)
        else:
            is_track_interior = is_track_interior.ravel(order="F")

        n_time = multiunits.shape[0]
        if ignore_no_spike:
            log_likelihood = (
                -time_bin_size
                * summed_ground_process_intensity
                * np.zeros((n_time, 1), dtype=np.float32)
            )
        else:
            log_likelihood = (
                -time_bin_size
                * summed_ground_process_intensity
                * np.ones((n_time, 1), dtype=np.float32)
            )

        multiunits = np.moveaxis(multiunits, -1, 0)
        n_position_bins = is_track_interior.sum()
        interior_place_bin_centers = cp.asarray(
            place_bin_centers[is_track_interior], dtype=cp.float32
        )
        gpu_is_track_interior = cp.asarray(is_track_interior)
        interior_occupancy = occupancy[gpu_is_track_interior]

        for multiunit, enc_marks, enc_pos, mean_rate in zip(
            tqdm(multiunits, desc="n_electrodes", disable=disable_progress_bar),
            encoding_marks,
            encoding_positions,
            mean_rates,
        ):
            is_spike = np.any(~np.isnan(multiunit), axis=1)
            is_mark_features = np.any(~np.isnan(multiunit), axis=0)
            decoding_marks = cp.asarray(
                multiunit[np.ix_(is_spike, is_mark_features)], dtype=cp.float32
            )
            n_decoding_marks = decoding_marks.shape[0]
            log_joint_mark_intensity = np.zeros(
                (n_decoding_marks, n_position_bins), dtype=np.float32
            )

            if block_size is None:
                block_size = n_decoding_marks

            if use_diffusion & (place_bin_centers.shape[1] > 1):
                position_distance = cp.asarray(
                    estimate_diffusion_position_distance(
                        enc_pos,
                        edges,
                        bin_distances=bin_diffusion_distances,
                    )[:, is_track_interior],
                    dtype=cp.float32,
                )
            else:
                position_distance = estimate_position_distance(
                    interior_place_bin_centers,
                    cp.asarray(enc_pos, dtype=cp.float32),
                    position_std,
                ).astype(cp.float32)

            for start_ind in range(0, n_decoding_marks, block_size):
                block_inds = slice(start_ind, start_ind + block_size)
                log_joint_mark_intensity[block_inds] = (
                    estimate_log_joint_mark_intensity(
                        decoding_marks[block_inds],
                        enc_marks,
                        mark_std[is_mark_features],
                        interior_occupancy,
                        mean_rate,
                        max_mark_diff=max_mark_diff,
                        set_diag_zero=set_diag_zero,
                        position_distance=position_distance,
                    )
                )
            log_likelihood[np.ix_(is_spike, is_track_interior)] += np.nan_to_num(
                log_joint_mark_intensity
            )

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        log_likelihood[:, ~is_track_interior] = np.nan

        return log_likelihood

except ImportError:

    def estimate_multiunit_likelihood_gpu(*args, **kwargs):
        print("Cupy is not installed or no GPU detected...")

    def fit_multiunit_likelihood_gpu(*args, **kwargs):
        print("Cupy is not installed or no GPU detected...")
