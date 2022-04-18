import math

import numpy as np
from numba import cuda
from replay_trajectory_classification.bins import (atleast_2d,
                                                   diffuse_each_bin,
                                                   get_bin_ind)
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
    def log_gaussian_pdf(x, mean, sigma):
        return -cp.log(sigma) - 0.5 * cp.log(2 * cp.pi) - 0.5 * ((x - mean) / sigma)**2

    @cp.fuse()
    def estimate_log_intensity(log_density, log_occupancy, log_mean_rate):
        return log_mean_rate + log_density - log_occupancy

    @cp.fuse()
    def estimate_intensity(log_density, log_occupancy, log_mean_rate):
        return cp.exp(
            estimate_log_intensity(log_density, log_occupancy, log_mean_rate))

    def estimate_log_position_distance(place_bin_centers, positions, position_std):
        '''

        Parameters
        ----------
        place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)
        positions : ndarray, shape (n_time, n_position_dims)
        position_std : float

        Returns
        -------
        position_distance : ndarray, shape (n_time, n_position_bins)

        '''
        n_time, n_position_dims = positions.shape
        n_position_bins = place_bin_centers.shape[0]

        log_position_distance = cp.zeros(
            (n_time, n_position_bins), dtype=cp.float32)

        for position_ind in range(n_position_dims):
            log_position_distance += log_gaussian_pdf(
                cp.expand_dims(place_bin_centers[:, position_ind], axis=0),
                cp.expand_dims(positions[:, position_ind], axis=1),
                position_std)

        return log_position_distance

    def estimate_log_position_density(place_bin_centers, positions, position_std,
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
        n_position_bins = place_bin_centers.shape[0]

        if block_size is None:
            block_size = n_time

        position_density = cp.empty((n_position_bins,))
        for start_ind in range(0, n_position_bins, block_size):
            block_inds = slice(start_ind, start_ind + block_size)
            position_density[block_inds] = log_mean(estimate_log_position_distance(
                place_bin_centers[block_inds], positions, position_std), axis=0)

        return position_density

    @cuda.jit()
    def log_mean_over_bins(log_mark_distances, log_position_distances, output):
        """

        Parameters
        ----------
        log_mark_distances : cupy.ndarray, shape (n_decoding_spikes, n_encoding_spikes)
        log_position_distances : cupy.ndarray, shape (n_encoding_spikes, n_position_bins)

        Returns
        -------
        output : cupy.ndarray, shape (n_decoding_spikes, n_position_bins)

        """
        thread_id = cuda.grid(1)

        n_decoding_spikes, n_position_bins = output.shape
        n_encoding_spikes = log_position_distances.shape[0]

        decoding_ind = thread_id // n_position_bins
        pos_bin_ind = thread_id % n_position_bins

        if (decoding_ind < n_decoding_spikes) and (pos_bin_ind < n_position_bins):

            # find maximum
            max_exp = (log_mark_distances[decoding_ind, 0] +
                       log_position_distances[0, pos_bin_ind])

            for encoding_ind in range(1, n_encoding_spikes):
                candidate_max = (log_mark_distances[decoding_ind, encoding_ind] +
                                 log_position_distances[encoding_ind, pos_bin_ind])
                if candidate_max > max_exp:
                    max_exp = candidate_max

            # logsumexp
            tmp = 0.0
            for encoding_ind in range(n_encoding_spikes):
                tmp += math.exp(log_mark_distances[decoding_ind, encoding_ind] +
                                log_position_distances[encoding_ind, pos_bin_ind] -
                                max_exp)

            output[decoding_ind, pos_bin_ind] = math.log(tmp) + max_exp

            # divide by n_spikes to get the mean
            output[decoding_ind, pos_bin_ind] -= math.log(n_encoding_spikes)

    def estimate_log_joint_mark_intensity(decoding_marks,
                                          encoding_marks,
                                          mark_std,
                                          log_position_distances,
                                          log_occupancy,
                                          log_mean_rate,
                                          max_mark_diff_value=6000,
                                          threads_per_block=(32, 32)):

        n_encoding_spikes, n_marks = encoding_marks.shape
        n_decoding_spikes = decoding_marks.shape[0]
        n_position_bins = log_position_distances.shape[1]

        log_normal_pdf_lookup = (
            -cp.log(mark_std) -
            0.5 * cp.log(2 * cp.pi) -
            0.5 * (cp.arange(-max_mark_diff_value, max_mark_diff_value, dtype=cp.float32) /
                   mark_std)**2
        )

        pdf = cp.empty((n_decoding_spikes, n_position_bins), dtype=cp.float32)

        prod_kde.forall(pdf.size)(
            decoding_marks, encoding_marks, log_normal_pdf_lookup,
            log_position_distances, max_mark_diff_value, pdf)

        return cp.asnumpy(estimate_log_intensity(
            pdf,
            log_occupancy,
            log_mean_rate
        ))

    def fit_multiunit_likelihood_integer_gpu3(position,
                                              multiunits,
                                              place_bin_centers,
                                              mark_std,
                                              position_std,
                                              is_track_boundary=None,
                                              is_track_interior=None,
                                              edges=None,
                                              block_size=100,
                                              use_diffusion_distance=False,
                                              **kwargs):
        '''

        Parameters
        ----------
        position : ndarray, shape (n_time, n_position_dims)
        multiunits : ndarray, shape (n_time, n_marks, n_electrodes)
        place_bin_centers : ndarray, shape ( n_bins, n_position_dims)
        model : sklearn model
        model_kwargs : dict
        occupancy_model : sklearn model
        occupancy_kwargs : dict
        is_track_interior : None or ndarray, shape (n_bins,)

        Returns
        -------
        joint_pdf_models : list of sklearn models, shape (n_electrodes,)
        ground_process_intensities : list of ndarray, shape (n_electrodes,)
        occupancy : ndarray, (n_bins, n_position_dims)
        mean_rates : ndarray, (n_electrodes,)

        '''
        if is_track_interior is None:
            is_track_interior = np.ones((place_bin_centers.shape[0],),
                                        dtype=np.bool)

        position = atleast_2d(position)
        place_bin_centers = atleast_2d(place_bin_centers)
        interior_place_bin_centers = cp.asarray(
            place_bin_centers[is_track_interior.ravel(order='F')],
            dtype=cp.float32)
        gpu_is_track_interior = cp.asarray(is_track_interior.ravel(order='F'))

        not_nan_position = np.all(~np.isnan(position), axis=1)

        if use_diffusion_distance:
            n_total_bins = np.prod(is_track_interior.shape)
            bin_diffusion_distances = diffuse_each_bin(
                is_track_interior,
                is_track_boundary,
                dx=edges[0][1] - edges[0][0],
                dy=edges[1][1] - edges[1][0],
                std=position_std,
            ).reshape((n_total_bins, -1), order='F')
        else:
            bin_diffusion_distances = None

        if use_diffusion_distance:
            log_occupancy = cp.log(cp.asarray(
                estimate_diffusion_position_density(
                    position,
                    edges,
                    bin_distances=bin_diffusion_distances,
                ), dtype=cp.float32))
        else:
            log_occupancy = cp.zeros(
                (place_bin_centers.shape[0],), dtype=cp.float32)
            log_occupancy[gpu_is_track_interior] = estimate_log_position_density(
                interior_place_bin_centers,
                cp.asarray(position[not_nan_position], dtype=cp.float32),
                position_std, block_size=block_size)

        log_mean_rates = []
        encoding_marks = []
        encoding_positions = []
        summed_ground_process_intensity = cp.zeros((place_bin_centers.shape[0],),
                                                   dtype=cp.float32)

        for multiunit in np.moveaxis(multiunits, -1, 0):

            # ground process intensity
            is_spike = np.any(~np.isnan(multiunit), axis=1)
            log_mean_rates.append(np.log(is_spike.mean()))

            if is_spike.sum() > 0:
                if use_diffusion_distance:
                    log_marginal_density = cp.log(cp.asarray(
                        estimate_diffusion_position_density(
                            position[is_spike & not_nan_position],
                            edges,
                            bin_distances=bin_diffusion_distances
                        ), dtype=cp.float32))
                else:
                    log_marginal_density = cp.zeros(
                        (place_bin_centers.shape[0],), dtype=cp.float32)
                    log_marginal_density[gpu_is_track_interior] = estimate_log_position_density(
                        interior_place_bin_centers,
                        cp.asarray(position[is_spike & not_nan_position],
                                   dtype=cp.float32),
                        position_std,
                        block_size=block_size)

            summed_ground_process_intensity += estimate_intensity(
                log_marginal_density, log_occupancy, log_mean_rates[-1])

            is_mark_features = np.any(~np.isnan(multiunit), axis=0)
            encoding_marks.append(
                cp.asarray(multiunit[
                    np.ix_(is_spike & not_nan_position, is_mark_features)],
                    dtype=cp.int16))
            encoding_positions.append(position[is_spike & not_nan_position])

        summed_ground_process_intensity = cp.asnumpy(
            summed_ground_process_intensity) + np.spacing(1)

        return {
            'encoding_marks': encoding_marks,
            'encoding_positions': encoding_positions,
            'summed_ground_process_intensity': summed_ground_process_intensity,
            'log_occupancy': log_occupancy,
            'log_mean_rates': log_mean_rates,
            'mark_std': mark_std,
            'position_std': position_std,
            'block_size': block_size,
            'bin_diffusion_distances': bin_diffusion_distances,
            'use_diffusion_distance': use_diffusion_distance,
            'edges': edges,
            **kwargs,
        }

    def estimate_multiunit_likelihood_integer_gpu3(multiunits,
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
                                                   max_mark_diff_value=6000,
                                                   set_diag_zero=False,
                                                   is_track_interior=None,
                                                   time_bin_size=1,
                                                   block_size=100,
                                                   ignore_no_spike=False,
                                                   disable_progress_bar=True,
                                                   use_diffusion_distance=False,
                                                   threads_per_block=(32, 32)):
        '''

        Parameters
        ----------
        multiunits : ndarray, shape (n_time, n_marks, n_electrodes)
        place_bin_centers : ndarray, (n_bins, n_position_dims)
        joint_pdf_models : list of sklearn models, shape (n_electrodes,)
        ground_process_intensities : list of ndarray, shape (n_electrodes,)
        occupancy : ndarray, (n_bins, n_position_dims)
        mean_rates : ndarray, (n_electrodes,)

        Returns
        -------
        log_likelihood : (n_time, n_bins)

        '''

        if is_track_interior is None:
            is_track_interior = np.ones((place_bin_centers.shape[0],),
                                        dtype=np.bool)
        else:
            is_track_interior = is_track_interior.ravel(order='F')

        n_time = multiunits.shape[0]
        if ignore_no_spike:
            log_likelihood = (-time_bin_size * summed_ground_process_intensity *
                              np.zeros((n_time, 1), dtype=np.float32))
        else:
            log_likelihood = (-time_bin_size * summed_ground_process_intensity *
                              np.ones((n_time, 1), dtype=np.float32))

        multiunits = np.moveaxis(multiunits, -1, 0)
        n_position_bins = is_track_interior.sum()
        interior_place_bin_centers = cp.asarray(
            place_bin_centers[is_track_interior], dtype=cp.float32)
        gpu_is_track_interior = cp.asarray(is_track_interior)
        interior_log_occupancy = log_occupancy[gpu_is_track_interior]

        for multiunit, enc_marks, enc_pos, log_mean_rate in zip(
                tqdm(multiunits, desc='n_electrodes',
                     disable=disable_progress_bar),
                encoding_marks, encoding_positions, log_mean_rates):
            is_spike = np.any(~np.isnan(multiunit), axis=1)
            is_mark_features = np.any(~np.isnan(multiunit), axis=0)
            decoding_marks = cp.asarray(
                multiunit[np.ix_(is_spike, is_mark_features)], dtype=cp.int16)
            n_decoding_marks = decoding_marks.shape[0]
            log_joint_mark_intensity = np.zeros(
                (n_decoding_marks, n_position_bins), dtype=np.float32)

            if block_size is None:
                block_size = n_decoding_marks

            if use_diffusion_distance:
                log_position_distances = cp.log(cp.asarray(
                    estimate_diffusion_position_distance(
                        enc_pos,
                        edges,
                        bin_distances=bin_diffusion_distances,
                    )[:, is_track_interior], dtype=cp.float32))
            else:
                log_position_distances = estimate_log_position_distance(
                    interior_place_bin_centers,
                    cp.asarray(enc_pos, dtype=cp.float32),
                    position_std
                ).astype(cp.float32)

            for start_ind in range(0, n_decoding_marks, block_size):
                block_inds = slice(start_ind, start_ind + block_size)
                log_joint_mark_intensity[block_inds] = estimate_log_joint_mark_intensity(
                    decoding_marks[block_inds],
                    enc_marks,
                    mark_std,
                    log_position_distances,
                    interior_log_occupancy,
                    log_mean_rate,
                    max_mark_diff_value=max_mark_diff_value,
                    threads_per_block=threads_per_block,
                )
            log_likelihood[np.ix_(is_spike, is_track_interior)] += np.nan_to_num(
                log_joint_mark_intensity)

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        log_likelihood[:, ~is_track_interior] = np.nan

        return log_likelihood

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

    @cuda.jit
    def prod_kde(
        decoding_marks,
        encoding_marks,
        log_normal_pdf_lookup,
        log_position_distances,
        max_mark_diff,
        output
    ):
        """

        Parameters
        ----------
        decoding_marks : cupy.ndarray, shape (n_decoding_spikes, n_mark_features)
        encoding_marks : cupy.ndarray, shape (n_encoding_spikes, n_mark_features)
        log_normal_pdf_lookup : cupy.ndarray, shape (n_samples,)
        log_position_distances : cupy.ndarray, shape (n_encoding_spikes, n_position_bins)
        max_mark_diff : int

        Returns
        -------
        output : cupy.ndarray, shape (n_decoding_spikes, n_position_bins)

        """
        thread_id = cuda.grid(1)

        n_decoding_spikes, n_position_bins = output.shape
        n_encoding_spikes, n_mark_features = encoding_marks.shape

        decoding_ind = thread_id // n_position_bins
        pos_bin_ind = thread_id % n_position_bins

        if (decoding_ind < n_decoding_spikes) and (pos_bin_ind < n_position_bins):

            # Compute distance for the first encoding spike
            max_dist = 0.0
            for feature_ind in range(n_mark_features):
                max_dist += log_normal_pdf_lookup[
                    decoding_marks[decoding_ind, feature_ind] -
                    encoding_marks[0, feature_ind] +
                    max_mark_diff
                ]

            max_dist += log_position_distances[0, pos_bin_ind]

            # Compute distances for the rest of the encoding spikes
            for enc_ind in range(1, n_encoding_spikes):
                candidate_max = 0.0
                for feature_ind in range(n_mark_features):
                    candidate_max += log_normal_pdf_lookup[
                        decoding_marks[decoding_ind, feature_ind] -
                        encoding_marks[enc_ind, feature_ind] +
                        max_mark_diff
                    ]
                candidate_max += log_position_distances[enc_ind, pos_bin_ind]

                # find max
                if candidate_max > max_dist:
                    max_dist = candidate_max

            # Compute log sum exp
            log_sum_dist = 0.0
            for enc_ind in range(0, n_encoding_spikes):
                dist = 0.0
                for feature_ind in range(n_mark_features):
                    dist += log_normal_pdf_lookup[
                        decoding_marks[decoding_ind, feature_ind] -
                        encoding_marks[enc_ind, feature_ind] +
                        max_mark_diff
                    ]
                dist += log_position_distances[enc_ind, pos_bin_ind]

                log_sum_dist += math.exp(dist - max_dist)

            output[decoding_ind, pos_bin_ind] = (
                math.log(log_sum_dist) +
                max_dist -
                math.log(n_encoding_spikes)
            )
except ImportError:
    def estimate_multiunit_likelihood_integer_gpu3(*args, **kwargs):
        print('Cupy is not installed or no GPU detected...')

    def fit_multiunit_likelihood_integer_gpu3(*args, **kwargs):
        print('Cupy is not installed or no GPU detected...')
