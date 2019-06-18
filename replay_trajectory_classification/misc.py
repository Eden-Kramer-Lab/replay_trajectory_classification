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


@numba.njit(nogil=True, cache=True)
def numba_kde(eval_points, samples, bandwidths):
    n_eval_points, n_bandwidths = eval_points.shape
    result = np.zeros((n_eval_points,))
    n_samples = len(samples)

    for i in range(n_eval_points):
        for j in range(n_samples):
            product_kernel = 1.0
            for k in range(n_bandwidths):
                bandwidth = bandwidths[k]
                eval_point = eval_points[i, k]
                sample = samples[j, k]
                product_kernel *= (np.exp(
                    -0.5 * ((eval_point - sample) / bandwidth)**2) /
                    (bandwidth * SQRT_2PI)) / bandwidth
            result[i] += product_kernel
        result[i] /= n_samples

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
