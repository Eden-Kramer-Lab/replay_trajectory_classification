"""Estimates a marked point process likelihood where the marks are
 features of the spike waveform using GPUs. Mark features are int16.

 Marks are converted to integers and KDE uses hash tables to compute Gaussian
 kernels."""


import numpy as np
from replay_trajectory_classification.core import atleast_2d
from replay_trajectory_classification.likelihoods.diffusion import (
    diffuse_each_bin,
    estimate_diffusion_position_density,
    estimate_diffusion_position_distance,
)
from tqdm.autonotebook import tqdm

try:
    import cupy as cp

    @cp.fuse
    def gaussian_pdf(x, mean, sigma):
        """Compute the value of a Gaussian probability density function at x with
        given mean and sigma."""
        return cp.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * cp.sqrt(2.0 * cp.pi))

    def estimate_position_distance(place_bin_centers, positions, position_std):
        """Estimates the Euclidean distance between positions and position bins.

        Parameters
        ----------
        place_bin_centers : np.ndarray, shape (n_position_bins, n_position_dims)
        positions : np.ndarray, shape (n_time, n_position_dims)
        position_std : array_like, shape (n_position_dims,)

        Returns
        -------
        position_distance : np.ndarray, shape (n_time, n_position_bins)

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
        place_bin_centers, positions, position_std, block_size=100
    ):
        """Estimates a kernel density estimate over position bins using
        Euclidean distances.

        Parameters
        ----------
        place_bin_centers : np.ndarray, shape (n_position_bins, n_position_dims)
        positions : np.ndarray, shape (n_time, n_position_dims)
        position_std : float or array_like, shape (n_position_dims,)

        Returns
        -------
        position_density : np.ndarray, shape (n_position_bins,)

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

    def estimate_log_intensity(density, occupancy, mean_rate):
        """Calculates intensity in log space."""
        return cp.log(mean_rate) + cp.log(density) - cp.log(occupancy)

    def estimate_intensity(density, occupancy, mean_rate):
        """Calculates intensity.

        Parameters
        ----------
        density : np.ndarray, shape (n_bins,)
        occupancy : np.ndarray, shape (n_bins,)
        mean_rate : float

        Returns
        -------
        intensity : np.ndarray, shape (n_bins,)

        """
        return cp.exp(estimate_log_intensity(density, occupancy, mean_rate))

    def normal_pdf_integer_lookup(x, mean, std=20, max_value=6000):
        """Fast density evaluation for integers by precomputing a hash table of
        values.

        Parameters
        ----------
        x : int
        mean : int
        std : float
        max_value : int

        Returns
        -------
        probability_density : int

        """
        normal_density = gaussian_pdf(cp.arange(-max_value, max_value), 0, std).astype(
            cp.float32
        )

        return normal_density[(x - mean) + max_value]

    def estimate_log_joint_mark_intensity(
        decoding_marks,
        encoding_marks,
        mark_std,
        occupancy,
        mean_rate,
        place_bin_centers=None,
        encoding_positions=None,
        position_std=None,
        max_mark_diff=6000,
        set_diag_zero=False,
        position_distance=None,
    ):
        """Finds the joint intensity of the marks and positions in log space.

        Parameters
        ----------
        decoding_marks : np.ndarray, shape (n_decoding_spikes, n_features)
        encoding_marks : np.ndarray, shape (n_encoding_spikes, n_features)
        mark_std : float
        occupancy : np.ndarray, shape (n_position_bins,)
        mean_rate : float
        place_bin_centers : None or np.ndarray, shape (n_position_bins, n_position_dims)
            If None, position distance must be not None
        encoding_positions : None or np.ndarray, shape (n_decoding_spikes, n_position_dims)
            If None, position distance must be not None
        position_std : None or float or array_like, shape (n_position_dims,)
            If None, position distance must be not None
        max_mark_diff : int
            Maximum distance between integer marks.
        set_diag_zero : bool
        position_distance : np.ndarray, shape (n_encoding_spikes, n_position_bins)
            Precalculated distance between position and position bins.

        Returns
        -------
        log_joint_mark_intensity : np.ndarray, shape (n_decoding_spikes, n_position_bins)

        """
        n_encoding_spikes, n_marks = encoding_marks.shape
        n_decoding_spikes = decoding_marks.shape[0]
        mark_distance = cp.ones(
            (n_decoding_spikes, n_encoding_spikes), dtype=cp.float32
        )

        for mark_ind in range(n_marks):
            mark_distance *= normal_pdf_integer_lookup(
                cp.expand_dims(decoding_marks[:, mark_ind], axis=1),
                cp.expand_dims(encoding_marks[:, mark_ind], axis=0),
                std=mark_std,
                max_value=max_mark_diff,
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

    def fit_multiunit_likelihood_integer_gpu(
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
            Amount of smoothing for the mark features. Standard deviation of kernel.
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

            summed_ground_process_intensity += estimate_intensity(
                marginal_density, occupancy, mean_rates[-1]
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

    def estimate_multiunit_likelihood_integer_gpu(
        multiunits,
        encoding_marks,
        mark_std,
        place_bin_centers,
        encoding_positions,
        position_std,
        occupancy,
        mean_rates,
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
            Amount of smoothing for mark features.  Standard deviation of kernel.
        place_bin_centers : cp.ndarray, shape (n_bins, n_position_dims)
        encoding_positions : cp.ndarray, shape (n_encoding_spikes, n_position_dims)
        position_std : float or array_like, shape (n_position_dims,)
            Amount of smoothing for position.  Standard deviation of kernel.
        occupancy : cp.ndarray, (n_bins,)
        mean_rates : list, len (n_electrodes,)
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
                multiunit[np.ix_(is_spike, is_mark_features)], dtype=cp.int16
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
                log_joint_mark_intensity[
                    block_inds
                ] = estimate_log_joint_mark_intensity(
                    decoding_marks[block_inds],
                    enc_marks,
                    mark_std,
                    interior_occupancy,
                    mean_rate,
                    max_mark_diff=max_mark_diff,
                    set_diag_zero=set_diag_zero,
                    position_distance=position_distance,
                )
            log_likelihood[np.ix_(is_spike, is_track_interior)] += np.nan_to_num(
                log_joint_mark_intensity
            )

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        log_likelihood[:, ~is_track_interior] = np.nan

        return log_likelihood

except ImportError:

    def estimate_multiunit_likelihood_integer_gpu(*args, **kwargs):
        print("Cupy is not installed or no GPU detected...")

    def fit_multiunit_likelihood_integer_gpu(*args, **kwargs):
        print("Cupy is not installed or no GPU detected...")
