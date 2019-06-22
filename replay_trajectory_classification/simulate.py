import numpy as np
from scipy.stats import multivariate_normal

from .core import atleast_2d


def simulate_time(n_samples, sampling_frequency):
    return np.arange(n_samples) / sampling_frequency


def simulate_linear_distance(time, track_height, running_speed=15):
    half_height = (track_height / 2)
    freq = 1 / (2 * track_height / running_speed)
    return (half_height * np.sin(freq * 2 * np.pi * time - np.pi / 2)
            + half_height)


def simulate_linear_distance_with_pauses(time, track_height, running_speed=15,
                                         pause=0.5, sampling_frequency=1):
    linear_distance = simulate_linear_distance(
        time, track_height, running_speed)
    peaks = np.nonzero(np.isclose(linear_distance, track_height))[0]
    n_pause_samples = int(pause * sampling_frequency)
    pause_linear_distance = np.zeros(
        (time.size + n_pause_samples * peaks.size,))
    pause_ind = (peaks[:, np.newaxis] + np.arange(n_pause_samples))
    pause_ind += np.arange(peaks.size)[:, np.newaxis] * n_pause_samples

    pause_linear_distance[pause_ind.ravel()] = track_height
    pause_linear_distance[pause_linear_distance == 0] = linear_distance
    return pause_linear_distance[:time.size]


def simulate_poisson_spikes(rate, sampling_frequency):
    '''Given a rate, returns a time series of spikes.

    Parameters
    ----------
    rate : ndarray, shape (n_time,)
    sampling_frequency : float

    Returns
    -------
    spikes : ndarray, shape (n_time,)

    '''
    return 1.0 * (np.random.poisson(rate / sampling_frequency) > 0)


def simulate_place_field_firing_rate(means, position, max_rate=15,
                                     variance=10):
    '''Simulates the firing rate of a neuron with a place field at `means`.

    Parameters
    ----------
    means : ndarray, shape (n_position_dims,)
    position : ndarray, shape (n_time, n_position_dims)
    max_rate : float, optional
    variance : float, optional

    Returns
    -------
    firing_rate : ndarray, shape (n_time,)

    '''
    position = atleast_2d(position)
    firing_rate = multivariate_normal(means, variance).pdf(position)
    firing_rate /= firing_rate.max()
    firing_rate *= max_rate
    return firing_rate


def simulate_neuron_with_place_field(means, position, max_rate=15, variance=10,
                                     sampling_frequency=500):
    '''Simulates the spiking of a neuron with a place field at `means`.

    Parameters
    ----------
    means : ndarray, shape (n_position_dims,)
    position : ndarray, shape (n_time, n_position_dims)
    max_rate : float, optional
    variance : float, optional
    sampling_frequency : float, optional

    Returns
    -------
    spikes : ndarray, shape (n_time,)

    '''
    firing_rate = simulate_place_field_firing_rate(
        means, position, max_rate, variance)
    return simulate_poisson_spikes(firing_rate, sampling_frequency)


def simulate_multiunit_with_place_fields(place_means, position, mark_spacing=5,
                                         n_mark_dims=4,
                                         sampling_frequency=1000):
    '''Simulates a multiunit with neurons at `place_means`

    Parameters
    ----------
    place_means : ndarray, shape (n_neurons, n_position_dims)
    position : ndarray, shape (n_time, n_position_dims)
    mark_spacing : int, optional
    n_mark_dims : int, optional

    Returns
    -------
    multiunit : ndarray, shape (n_time, n_mark_dims)

    '''
    n_neurons = place_means.shape[0]
    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)
    n_time = position.shape[0]
    marks = np.full((n_time, n_mark_dims), np.nan)
    for mean, mark_center in zip(place_means, mark_centers):
        is_spike = simulate_neuron_with_place_field(
            mean, position, max_rate=100, variance=12.5**2,
            sampling_frequency=sampling_frequency) > 0
        n_spikes = int(is_spike.sum())
        marks[is_spike] = multivariate_normal(
            mean=[mark_center] * n_mark_dims).rvs(size=n_spikes)
    return marks
