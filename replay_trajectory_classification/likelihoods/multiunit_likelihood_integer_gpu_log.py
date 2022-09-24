"""Estimates a marked point process likelihood where the marks are
 features of the spike waveform using GPUs. Mark features are int16.

 This algorithm is more numerically stable KDE done in log space but slower
 right now because of lack of matrix multiplication.

 Marks are converted to integers and KDE uses hash tables to compute Gaussian
 kernels."""


import math

import numpy as np
from numba import cuda
from replay_trajectory_classification.core import atleast_2d
from replay_trajectory_classification.likelihoods.diffusion import (
    diffuse_each_bin,
    estimate_diffusion_position_density,
    estimate_diffusion_position_distance,
)
from tqdm.autonotebook import tqdm

try:
    import cupy as cp

    def logsumexp(a, axis):
        a_max = cp.amax(a, axis=axis, keepdims=True)

        if a_max.ndim > 0:
            a_max[~cp.isfinite(a_max)] = 0
        elif not np.isfinite(a_max):
            a_max = 0

        return cp.log(cp.sum(cp.exp(a - a_max), axis=axis, keepdims=True)) + a_max

    def log_mean(x, axis):
        return cp.squeeze(logsumexp(x, axis=axis) - cp.log(x.shape[axis]))

    @cp.fuse()
    def log_gaussian_pdf(x, sigma):
        return -cp.log(sigma) - 0.5 * cp.log(2 * cp.pi) - 0.5 * (x / sigma) ** 2

    @cp.fuse()
    def estimate_log_intensity(log_density, log_occupancy, log_mean_rate):
        return log_mean_rate + log_density - log_occupancy

    @cp.fuse()
    def estimate_intensity(log_density, log_occupancy, log_mean_rate):
        return cp.exp(estimate_log_intensity(log_density, log_occupancy, log_mean_rate))

    def estimate_log_position_distance(place_bin_centers, positions, position_std):
        """Estimates the Euclidean distance between positions and position bins.

        Parameters
        ----------
        place_bin_centers : cp.ndarray, shape (n_position_bins, n_position_dims)
        positions : cp.ndarray, shape (n_time, n_position_dims)
        position_std : array_like, shape (n_position_dims,)

        Returns
        -------
        log_position_distance : np.ndarray, shape (n_time, n_position_bins)

        """
        n_time, n_position_dims = positions.shape
        n_position_bins = place_bin_centers.shape[0]

        if isinstance(position_std, (int, float)):
            position_std = [position_std] * n_position_dims

        log_position_distance = cp.zeros((n_time, n_position_bins), dtype=cp.float32)

        for position_ind, std in enumerate(position_std):
            log_position_distance += log_gaussian_pdf(
                cp.expand_dims(place_bin_centers[:, position_ind], axis=0)
                - cp.expand_dims(positions[:, position_ind], axis=1),
                std,
            )

        return log_position_distance

    def estimate_log_position_density(
        place_bin_centers, positions, position_std, block_size=100
    ):
        """Estimates a kernel density estimate over position bins using
        Euclidean distances.

        Parameters
        ----------
        place_bin_centers : cp.ndarray, shape (n_position_bins, n_position_dims)
        positions : cp.ndarray, shape (n_time, n_position_dims)
        position_std : float or array_like, shape (n_position_dims,)
        block_size : int

        Returns
        -------
        log_position_density : cp.ndarray, shape (n_position_bins,)

        """
        n_time = positions.shape[0]
        n_position_bins = place_bin_centers.shape[0]

        if block_size is None:
            block_size = n_time

        log_position_density = cp.empty((n_position_bins,))
        for start_ind in range(0, n_position_bins, block_size):
            block_inds = slice(start_ind, start_ind + block_size)
            log_position_density[block_inds] = log_mean(
                estimate_log_position_distance(
                    place_bin_centers[block_inds], positions, position_std
                ),
                axis=0,
            )

        return log_position_density

    @cuda.jit()
    def log_mean_over_bins(log_mark_distances, log_position_distances, output):
        """

        Parameters
        ----------
        log_mark_distances : cp.ndarray, shape (n_decoding_spikes, n_encoding_spikes)
        log_position_distances : cp.ndarray, shape (n_encoding_spikes, n_position_bins)

        Returns
        -------
        output : cp.ndarray, shape (n_decoding_spikes, n_position_bins)

        """
        thread_id = cuda.grid(1)

        n_decoding_spikes, n_position_bins = output.shape
        n_encoding_spikes = log_position_distances.shape[0]

        decoding_ind = thread_id // n_position_bins
        pos_bin_ind = thread_id % n_position_bins

        if (decoding_ind < n_decoding_spikes) and (pos_bin_ind < n_position_bins):

            # find maximum
            max_exp = (
                log_mark_distances[decoding_ind, 0]
                + log_position_distances[0, pos_bin_ind]
            )

            for encoding_ind in range(1, n_encoding_spikes):
                candidate_max = (
                    log_mark_distances[decoding_ind, encoding_ind]
                    + log_position_distances[encoding_ind, pos_bin_ind]
                )
                if candidate_max > max_exp:
                    max_exp = candidate_max

            # logsumexp
            tmp = 0.0
            for encoding_ind in range(n_encoding_spikes):
                tmp += math.exp(
                    log_mark_distances[decoding_ind, encoding_ind]
                    + log_position_distances[encoding_ind, pos_bin_ind]
                    - max_exp
                )

            output[decoding_ind, pos_bin_ind] = math.log(tmp) + max_exp

            # divide by n_spikes to get the mean
            output[decoding_ind, pos_bin_ind] -= math.log(n_encoding_spikes)

    def estimate_log_joint_mark_intensity(
        decoding_marks,
        encoding_marks,
        mark_std,
        log_position_distances,
        log_occupancy,
        log_mean_rate,
        max_mark_diff=6000,
        set_diag_zero=False,
    ):
        """Finds the joint intensity of the marks and positions in log space.

        Parameters
        ----------
        decoding_marks : cp.ndarray, shape (n_decoding_spikes, n_features)
        encoding_marks : cp.ndarray, shape (n_encoding_spikes, n_features)
        mark_std : float
        log_position_distance : cp.ndarray, shape (n_encoding_spikes, n_position_bins)
            Precalculated distance between position and position bins.
        log_occupancy : cp.ndarray, shape (n_position_bins,)
        log_mean_rate : float
        max_mark_diff : int
            Maximum distance between integer marks.
        set_diag_zero : bool

        Returns
        -------
        log_joint_mark_intensity : np.ndarray, shape (n_decoding_spikes, n_position_bins)

        """
        n_encoding_spikes, n_marks = encoding_marks.shape
        n_decoding_spikes = decoding_marks.shape[0]
        log_mark_distances = cp.zeros(
            (n_decoding_spikes, n_encoding_spikes), dtype=cp.float32
        )

        log_normal_pdf_lookup = cp.asarray(
            log_gaussian_pdf(cp.arange(-max_mark_diff, max_mark_diff), mark_std),
            dtype=cp.float32,
        )

        for mark_ind in range(n_marks):
            log_mark_distances += log_normal_pdf_lookup[
                (
                    cp.expand_dims(decoding_marks[:, mark_ind], axis=1)
                    - cp.expand_dims(encoding_marks[:, mark_ind], axis=0)
                )
                + max_mark_diff
            ]

        if set_diag_zero:
            diag_ind = (cp.arange(n_decoding_spikes), cp.arange(n_decoding_spikes))
            log_mark_distances[diag_ind] = cp.nan_to_num(cp.log(0).astype(cp.float32))

        n_position_bins = log_position_distances.shape[1]
        pdf = cp.empty((n_decoding_spikes, n_position_bins), dtype=cp.float32)

        log_mean_over_bins.forall(pdf.size)(
            log_mark_distances, log_position_distances, pdf
        )

        return cp.asnumpy(estimate_log_intensity(pdf, log_occupancy, log_mean_rate))

    def fit_multiunit_likelihood_integer_gpu_log(
        position,
        multiunits,
        place_bin_centers,
        mark_std,
        position_std,
        is_track_boundary=None,
        is_track_interior=None,
        edges=None,
        block_size=100,
        use_diffusion=False,
        **kwargs
    ):
        """Fits the clusterless place field model.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
        multiunits : np.ndarray, shape (n_time, n_marks, n_electrodes)
        place_bin_centers : np.ndarray, shape (n_bins, n_position_dims)
        mark_std : float
            Amount of smoothing for the mark features.  Standard deviation of kernel.
        position_std : float or array_like, shape (n_position_dims,)
            Amount of smoothing for position.  Standard deviation of kernel.
        is_track_boundary : None or np.ndarray, shape (n_bins,)
        is_track_interior : None or np.ndarray, shape (n_bins,)
        edges : None or list of np.ndarray
        block_size : int
            Size of data to process in chunks
        use_diffusion : bool
            Use diffusion to respect the track geometry.

        Returns
        -------
        encoding_model : dict

        """
        if is_track_interior is None:
            is_track_interior = np.ones((place_bin_centers.shape[0],), dtype=np.bool)

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
            log_occupancy = cp.log(
                cp.asarray(
                    estimate_diffusion_position_density(
                        position[not_nan_position],
                        edges,
                        bin_distances=bin_diffusion_distances,
                    ),
                    dtype=cp.float32,
                )
            )
        else:
            log_occupancy = cp.zeros((place_bin_centers.shape[0],), dtype=cp.float32)
            log_occupancy[gpu_is_track_interior] = estimate_log_position_density(
                interior_place_bin_centers,
                cp.asarray(position[not_nan_position], dtype=cp.float32),
                position_std,
                block_size=block_size,
            )

        log_mean_rates = []
        encoding_marks = []
        encoding_positions = []
        summed_ground_process_intensity = cp.zeros(
            (place_bin_centers.shape[0],), dtype=cp.float32
        )

        for multiunit in np.moveaxis(multiunits, -1, 0):

            # ground process intensity
            is_spike = np.any(~np.isnan(multiunit), axis=1)
            log_mean_rates.append(np.log(is_spike.mean()))

            if is_spike.sum() > 0:
                if use_diffusion & (position.shape[1] > 1):
                    log_marginal_density = cp.log(
                        cp.asarray(
                            estimate_diffusion_position_density(
                                position[is_spike & not_nan_position],
                                edges,
                                bin_distances=bin_diffusion_distances,
                            ),
                            dtype=cp.float32,
                        )
                    )
                else:
                    log_marginal_density = cp.zeros(
                        (place_bin_centers.shape[0],), dtype=cp.float32
                    )
                    log_marginal_density[
                        gpu_is_track_interior
                    ] = estimate_log_position_density(
                        interior_place_bin_centers,
                        cp.asarray(
                            position[is_spike & not_nan_position], dtype=cp.float32
                        ),
                        position_std,
                        block_size=block_size,
                    )

            summed_ground_process_intensity += estimate_intensity(
                log_marginal_density, log_occupancy, log_mean_rates[-1]
            )

            is_mark_features = np.any(~np.isnan(multiunit), axis=0)
            encoding_marks.append(
                cp.asarray(
                    multiunit[np.ix_(is_spike & not_nan_position, is_mark_features)],
                    dtype=cp.int16,
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
            "log_occupancy": log_occupancy,
            "log_mean_rates": log_mean_rates,
            "mark_std": mark_std,
            "position_std": position_std,
            "block_size": block_size,
            "bin_diffusion_distances": bin_diffusion_distances,
            "use_diffusion": use_diffusion,
            "edges": edges,
            **kwargs,
        }

    def estimate_multiunit_likelihood_integer_gpu_log(
        multiunits,
        encoding_marks,
        mark_std,
        place_bin_centers,
        encoding_positions,
        position_std,
        log_occupancy,
        log_mean_rates,
        summed_ground_process_intensity,
        bin_diffusion_distances,
        edges,
        max_mark_diff=6000,
        set_diag_zero=False,
        is_track_interior=None,
        time_bin_size=1,
        block_size=100,
        ignore_no_spike=False,
        disable_progress_bar=False,
        use_diffusion=False,
    ):
        """Estimates the likelihood of position bins given multiunit marks.

        Parameters
        ----------
        multiunits : np.ndarray, shape (n_decoding_time, n_marks, n_electrodes)
        encoding_marks : cp.ndarray, shape (n_encoding_spikes, n_marks, n_electrodes)
        mark_std : float
            Amount of smoothing for mark features
        place_bin_centers : cp.ndarray, shape (n_bins, n_position_dims)
        encoding_positions : cp.ndarray, shape (n_encoding_spikes, n_position_dims)
        position_std : float or array_like, shape (n_position_dims,)
            Amount of smoothing for position
        log_occupancy : cp.ndarray, (n_bins,)
        log_mean_rates : list, len (n_electrodes,)
        summed_ground_process_intensity : np.ndarray, shape (n_bins,)
        bin_diffusion_distances : np.ndarray, shape (n_bins, n_bins)
        edges : list of np.ndarray
        max_mark_diff : int
            Maximum difference between mark features
        set_diag_zero : bool
            Remove influence of the same mark in encoding and decoding.
        is_track_interior : None or np.ndarray, shape (n_bins_x, n_bins_y)
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
            is_track_interior = np.ones((place_bin_centers.shape[0],), dtype=np.bool)
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
        interior_log_occupancy = log_occupancy[gpu_is_track_interior]

        for multiunit, enc_marks, enc_pos, log_mean_rate in zip(
            tqdm(multiunits, desc="n_electrodes", disable=disable_progress_bar),
            encoding_marks,
            encoding_positions,
            log_mean_rates,
        ):
            is_spike = np.any(~np.isnan(multiunit), axis=1)
            is_mark_features = np.any(~np.isnan(multiunit), axis=0)
            decoding_marks = cp.asarray(
                multiunit[np.ix_(is_spike, is_mark_features)], dtype=cp.int16
            )
            n_decoding_marks = decoding_marks.shape[0]
            log_joint_mark_intensity = np.zeros(
                (n_decoding_marks, n_position_bins), dtype=np.float32
            )

            if block_size is None:
                block_size = n_decoding_marks

            if use_diffusion & (place_bin_centers.shape[1] > 1):
                log_position_distances = cp.log(
                    cp.asarray(
                        estimate_diffusion_position_distance(
                            enc_pos,
                            edges,
                            bin_distances=bin_diffusion_distances,
                        )[:, is_track_interior],
                        dtype=cp.float32,
                    )
                )
            else:
                log_position_distances = estimate_log_position_distance(
                    interior_place_bin_centers,
                    cp.asarray(enc_pos, dtype=cp.float32),
                    position_std,
                ).astype(cp.float32)

            for start_ind in range(0, n_decoding_marks, block_size):
                block_inds = slice(start_ind, start_ind + block_size)
                log_joint_mark_intensity[
                    block_inds
                ] = estimate_log_joint_mark_intensity(
                    decoding_marks[block_inds],
                    enc_marks,
                    mark_std,
                    log_position_distances,
                    interior_log_occupancy,
                    log_mean_rate,
                    max_mark_diff=max_mark_diff,
                    set_diag_zero=set_diag_zero,
                )
            log_likelihood[np.ix_(is_spike, is_track_interior)] += np.nan_to_num(
                log_joint_mark_intensity
            )

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        log_likelihood[:, ~is_track_interior] = np.nan

        return log_likelihood

except ImportError:

    def estimate_multiunit_likelihood_integer_gpu_log(*args, **kwargs):
        print("Cupy is not installed or no GPU detected...")

    def fit_multiunit_likelihood_integer_gpu_log(*args, **kwargs):
        print("Cupy is not installed or no GPU detected...")
