import numpy as np

from .core import normalize_to_probability


def uniform(place_bin_centers):
    return normalize_to_probability(np.ones((place_bin_centers.shape[0],)))


def uniform_on_track(place_bin_centers, is_track_interior):
    uniform = np.ones((place_bin_centers.shape[0],))
    is_track_interior = is_track_interior.ravel(order='F')
    uniform[~is_track_interior] = 0.0
    return normalize_to_probability(uniform)
