"""Main code for simulating position and sorted spikes or clusterless spikes and waveforms."""
import numpy as np
from replay_trajectory_classification.core import atleast_2d
from scipy.stats import multivariate_normal

from typing import Optional


def simulate_time(n_samples: int, sampling_frequency: float) -> np.ndarray:
    """Simulate a time in seconds.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    sampling_frequency : float
        Samples per second.

    Returns
    -------
    time : ndarray, shape (n_samples,)
        Time in seconds

    """
    return np.arange(n_samples) / sampling_frequency


def simulate_position(
    time: np.ndarray, track_height: float, running_speed: float = 15
) -> np.ndarray:
    """Simulate animal moving through linear space.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        Time in seconds.
    track_height : float
        The height of the simulated track.
    running_speed : float, optional
        The running speed of the simulated animal (default is 15).

    Returns
    -------
    position : ndarray, shape (n_time,)
       The simulated position of the animal.

    """
    half_height = track_height / 2
    freq = 1 / (2 * track_height / running_speed)
    return half_height * np.sin(freq * 2 * np.pi * time - np.pi / 2) + half_height


def simulate_position_with_pauses(
    time: np.ndarray,
    track_height: float,
    running_speed: float = 15,
    pause: float = 0.5,
    sampling_frequency: float = 1,
) -> np.ndarray:
    """Simulate an animal moving with pauses.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The time vector.
    track_height : float
        The height of the track.
    running_speed : float, optional
        The running speed (default is 15).
    pause : float, optional
        The pause duration (default is 0.5).
    sampling_frequency : float, optional
        The sampling frequency (default is 1).

    Returns
    -------
    position : ndarray, shape (n_time,)
        The simulated position of the animal with pauses.

    """
    position = simulate_position(time, track_height, running_speed)
    peaks = np.nonzero(np.isclose(position, track_height))[0]
    n_pause_samples = int(pause * sampling_frequency)
    pause_position = np.zeros((time.size + n_pause_samples * peaks.size,))
    pause_ind = peaks[:, np.newaxis] + np.arange(n_pause_samples)
    pause_ind += np.arange(peaks.size)[:, np.newaxis] * n_pause_samples

    pause_position[pause_ind.ravel()] = track_height
    pause_position[pause_position == 0] = position

    return pause_position[: time.size]


def simulate_poisson_spikes(rate: np.ndarray, sampling_frequency: int) -> np.ndarray:
    """Given a rate, returns a time series of spikes.

    Parameters
    ----------
    rate : np.ndarray, shape (n_time,)
    sampling_frequency : int

    Returns
    -------
    spikes : np.ndarray, shape (n_time,)

    """
    return 1.0 * (np.random.poisson(rate / sampling_frequency) > 0)


def simulate_place_field_firing_rate(
    means: np.ndarray,
    position: np.ndarray,
    max_rate: float = 15.0,
    variance: float = 10.0,
    is_condition: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulates the firing rate of a neuron with a place field at `means`.

    Parameters
    ----------
    means : np.ndarray, shape (n_position_dims,)
    position : np.ndarray, shape (n_time, n_position_dims)
    max_rate : float, optional
    variance : float, optional
    is_condition : None or np.ndarray, (n_time,)

    Returns
    -------
    firing_rate : np.ndarray, shape (n_time,)

    """
    if is_condition is None:
        is_condition = np.ones(position.shape[0], dtype=bool)
    position = atleast_2d(position)
    firing_rate = multivariate_normal(means, variance).pdf(position)
    firing_rate /= firing_rate.max()
    firing_rate *= max_rate
    firing_rate[~is_condition] = 0.0

    return firing_rate


def simulate_neuron_with_place_field(
    means: np.ndarray,
    position: np.ndarray,
    max_rate: float = 15.0,
    variance: float = 36.0,
    sampling_frequency: int = 500,
    is_condition: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulates the spiking of a neuron with a place field at `means`.

    Parameters
    ----------
    means : np.ndarray, shape (n_position_dims,)
    position : np.ndarray, shape (n_time, n_position_dims)
    max_rate : float, optional
    variance : float, optional
    sampling_frequency : float, optional
    is_condition : None or np.ndarray, (n_time,)

    Returns
    -------
    spikes : np.ndarray, shape (n_time,)

    """
    firing_rate = simulate_place_field_firing_rate(
        means, position, max_rate, variance, is_condition
    )
    return simulate_poisson_spikes(firing_rate, sampling_frequency)


def simulate_multiunit_with_place_fields(
    place_means: np.ndarray,
    position: np.ndarray,
    mark_spacing: int = 5,
    n_mark_dims: int = 4,
    place_variance: float = 36.0,
    mark_variance: float = 1.0,
    max_rate: float = 100.0,
    sampling_frequency: int = 1000,
    is_condition: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulates a multiunit with neurons at `place_means`

    Parameters
    ----------
    place_means : np.ndarray, shape (n_neurons, n_position_dims)
    position : np.ndarray, shape (n_time, n_position_dims)
    mark_spacing : int, optional
    n_mark_dims : int, optional
    place_variance : float
    max_rate : float
    sampling_frequency : int
    is_condition : np.ndarray or None

    Returns
    -------
    multiunit : np.ndarray, shape (n_time, n_mark_dims)

    """
    n_neurons = place_means.shape[0]
    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)
    n_time = position.shape[0]
    marks = np.full((n_time, n_mark_dims), np.nan)
    for mean, mark_center in zip(place_means, mark_centers):
        is_spike = (
            simulate_neuron_with_place_field(
                mean,
                position,
                max_rate=max_rate,
                variance=place_variance,
                sampling_frequency=sampling_frequency,
                is_condition=is_condition,
            )
            > 0
        )
        n_spikes = int(is_spike.sum())
        marks[is_spike] = multivariate_normal(
            mean=[mark_center] * n_mark_dims, cov=mark_variance
        ).rvs(size=n_spikes)
    return marks


def get_trajectory_direction(position: np.ndarray) -> np.ndarray:
    """Find if the trajectory is inbound or outbound.

    Parameters
    ----------
    position : np.ndarray, shape (n_time,)

    Returns
    -------
    is_inbound : np.ndarray, shape (n_time,)

    """
    is_inbound = np.insert(np.diff(position) < 0, 0, False)
    return np.where(is_inbound, "Inbound", "Outbound")
