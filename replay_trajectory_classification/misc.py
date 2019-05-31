import math

import numba
import numpy as np
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

SQRT_2PI = np.float64(math.sqrt(2.0 * math.pi))


class WhitenedKDE(BaseEstimator, DensityMixin):
    def __init__(self, **kwargs):
        self.kde = KernelDensity(**kwargs)
        self.pre_whiten = PCA(whiten=True)

    def fit(self, X, y=None, sample_weight=None):
        self.kde.fit(self.pre_whiten.fit_transform(X))
        return self

    def score_samples(self, X):
        return self.kde.score_samples(self.pre_whiten.transform(X))


@numba.vectorize(['float64(float64, float64, float64)'], nopython=True)
def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x
    with given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)


@numba.njit(parallel=True, fastmath=True)
def numba_kde(eval_points, samples, bandwidths):
    n_eval_points = len(eval_points)
    result = np.zeros((n_eval_points,))
    n_samples = len(samples)
    denom = n_samples * np.prod(bandwidths)

    for i in numba.prange(n_eval_points):
        for j in range(n_samples):
            result[i] += np.prod(
                gaussian_pdf(eval_points[i], samples[j], bandwidths))
        result[i] /= denom

    return result


class NumbaKDE(BaseEstimator, DensityMixin):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y=None, sample_weight=None):
        self.training_data = X
        return self

    def score_samples(self, X):
        return np.log(numba_kde(X, self.training_data,
                                self.bandwidth[-X.shape[1]:]))
