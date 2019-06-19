import math

import numpy as np
from numba import cuda
from numba.types import float32

SQRT_2PI = np.float32(math.sqrt(2.0 * math.pi))
TILE_SIZE = 512
N_BANDWIDTHS = 6


@cuda.jit(device=True)
def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with
    given mean and sigma.

    Parameters
    ----------
    x : float
    mean : float
    sigma : float

    Returns
    -------
    pdf : float

    '''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)


@cuda.jit
def numba_kde_cuda(eval_points, samples, bandwidths, out):
    '''

    Parameters
    ----------
    eval_points : ndarray, shape (1,)
    samples : ndarray, shape (n_samples, n_bandwidths)
    bandwidths : ndarray, shape (n_bandwidths,)
    out : ndarray, shape (1,)

    '''
    thread_id = cuda.grid(1)
    stride = cuda.gridsize(1)

    n_samples = samples.shape[0]

    for sample_ind in range(thread_id, n_samples, stride):
        product_kernel = 1.0
        for bandwidth_ind in range(bandwidths.shape[0]):
            product_kernel *= (gaussian_pdf(eval_points[0, bandwidth_ind],
                                            samples[sample_ind, bandwidth_ind],
                                            bandwidths[bandwidth_ind])
                               / bandwidths[bandwidth_ind])
        product_kernel /= n_samples
        cuda.atomic.add(out, 0, product_kernel)


@cuda.jit
def numba_kde_cuda2(eval_points, samples, bandwidths, out):
    '''

    Parameters
    ----------
    eval_points : ndarray, shape (n_eval, n_bandwidths)
    samples : ndarray, shape (n_samples, n_bandwidths)
    out : ndarray, shape (n_eval,)

    '''
    thread_id1, thread_id2 = cuda.grid(2)
    stride1, stride2 = cuda.gridsize(2)

    (n_eval, n_bandwidths), n_samples = eval_points.shape, samples.shape[0]

    for eval_ind in range(thread_id1, n_eval, stride1):
        for sample_ind in range(thread_id2, n_samples, stride2):
            product_kernel = 1.0
            for bandwidth_ind in range(n_bandwidths):
                product_kernel *= (
                    gaussian_pdf(eval_points[eval_ind, bandwidth_ind],
                                 samples[sample_ind, bandwidth_ind],
                                 bandwidths[bandwidth_ind])
                    / bandwidths[bandwidth_ind])
            product_kernel /= n_samples
            cuda.atomic.add(out, eval_ind, product_kernel)


@cuda.jit
def numba_kde_cuda3(covariate_bins, marks, samples, bandwidths, out):
    '''

    Parameters
    ----------
    covariate_bins : ndarray, shape (n_bins, n_cov)
    marks : ndarray, shape (n_test, n_features)
    samples : ndarray, shape (n_test, n_cov + n_features)
    bandwidths : ndarray, shape (n_cov + n_features,)
    out : ndarray, shape (n_test, n_bins)

    '''
    thread_id1, thread_id2, thread_id3 = cuda.grid(3)
    stride1, stride2, stride3 = cuda.gridsize(3)

    n_bins, n_cov = covariate_bins.shape
    n_test, n_features = marks.shape
    n_samples = samples.shape[0]

    for test_ind in range(thread_id1, n_test, stride1):
        for bin_ind in range(thread_id2, n_bins, stride2):
            for sample_ind in range(thread_id3, n_samples, stride3):
                product_kernel = 1.0

                for cov_ind in range(n_cov):
                    product_kernel *= (
                        gaussian_pdf(covariate_bins[bin_ind, cov_ind],
                                     samples[sample_ind, cov_ind],
                                     bandwidths[cov_ind])
                        / bandwidths[cov_ind])

                for feature_ind in range(n_features):
                    product_kernel *= (
                        gaussian_pdf(marks[test_ind, feature_ind],
                                     samples[sample_ind, n_cov + feature_ind],
                                     bandwidths[n_cov + feature_ind])
                        / bandwidths[n_cov + feature_ind])

                product_kernel /= n_samples

                cuda.atomic.add(out, (test_ind, bin_ind), product_kernel)


@cuda.jit
def numba_kde_cuda2a(eval_points, samples, bandwidths, out):
    n_samples, n_bandwidths = samples.shape
    thread_id = cuda.grid(1)
    sum_kernel = 0.0

    for sample_ind in range(n_samples):
        product_kernel = 1.0
        for bandwidth_ind in range(n_bandwidths):
            product_kernel *= (
                gaussian_pdf(eval_points[thread_id, bandwidth_ind],
                             samples[sample_ind, bandwidth_ind],
                             bandwidths[bandwidth_ind])
                / bandwidths[bandwidth_ind])
        sum_kernel += product_kernel

    out[thread_id] = sum_kernel / n_samples


@cuda.jit
def numba_kde_cuda2b(eval_points, samples, bandwidths, out):
    '''

    Parameters
    ----------
    eval_points : ndarray, shape (n_eval, n_bandwidths)
    samples : ndarray, shape (n_samples, n_bandwidths)
    out : ndarray, shape (n_eval,)

    '''
    n_eval, n_bandwidths = eval_points.shape
    n_samples = samples.shape[0]

    thread_id = cuda.grid(1)
    if thread_id < n_eval:
        relative_thread_id = cuda.threadIdx.x
        n_threads = cuda.blockDim.x

        samples_tile = cuda.shared.array((TILE_SIZE, N_BANDWIDTHS), float32)

        sum_kernel = 0.0

        for tile_ind in range(0, n_samples, TILE_SIZE):
            tile_index = tile_ind + relative_thread_id
            if tile_index < n_samples:
                for bandwidth_ind in range(n_bandwidths):
                    samples_tile[relative_thread_id, bandwidth_ind] = samples[
                        tile_index, bandwidth_ind]

            cuda.syncthreads()

            for i in range(n_threads):
                if tile_ind + i < n_samples:
                    product_kernel = 1.0
                    for bandwidth_ind in range(n_bandwidths):
                        product_kernel *= (
                            gaussian_pdf(eval_points[thread_id, bandwidth_ind],
                                         samples_tile[i, bandwidth_ind],
                                         bandwidths[bandwidth_ind])
                            / bandwidths[bandwidth_ind])
                    sum_kernel += product_kernel

            cuda.syncthreads()

        out[thread_id] = sum_kernel / n_samples
