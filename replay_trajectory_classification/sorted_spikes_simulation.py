"""Functions for generating sorted spike data.

This module provides simulation functions specifically for sorted (clustered)
spike data used in neural decoding validation and testing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from replay_trajectory_classification.simulate import (
    get_trajectory_direction,
    simulate_neuron_with_place_field,
    simulate_place_field_firing_rate,
    simulate_position,
    simulate_time,
)

SAMPLING_FREQUENCY = 1000
TRACK_HEIGHT = 180
RUNNING_SPEED = 15
PLACE_FIELD_VARIANCE = 6.0**2
PLACE_FIELD_MEANS = np.arange(0, TRACK_HEIGHT + 10, 10, dtype=np.float64)
N_RUNS = 15
REPLAY_SPEEDUP = 120

# Figure Parameters
MM_TO_INCHES = 1.0 / 25.4
ONE_COLUMN = 89.0 * MM_TO_INCHES
ONE_AND_HALF_COLUMN = 140.0 * MM_TO_INCHES
TWO_COLUMN = 178.0 * MM_TO_INCHES
PAGE_HEIGHT = 247.0 * MM_TO_INCHES
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0


def make_simulated_run_data(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    place_field_variance: float = PLACE_FIELD_VARIANCE,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    make_inbound_outbound_neurons: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64]]:
    """Make simulated data of a rat running back and forth
    on a linear maze with sorted spikes.

    Parameters
    ----------
    sampling_frequency : float, optional
        Samples per second
    track_height : float, optional
        Height of the simulated track
    running_speed : float, optional
        Speed of the simulated animal
    n_runs : int, optional
        Number of runs across the track the simulated animal will perform
    place_field_variance : float, optional
        Spatial variance of the place field
    place_field_means : NDArray[np.float64], shape (n_neurons,), optional
        Location of the center of the Gaussian place fields.
    make_inbound_outbound_neurons : bool
        Makes neurons direction selective.

    Returns
    -------
    time : NDArray[np.float64], shape (n_time,)
    position : NDArray[np.float64], shape (n_time,)
        Position of the simualted animal
    sampling_frequency : float
        Samples per second
    spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.
    place_fields : NDArray[np.float64], shape (n_time, n_neurons)

    """
    n_samples = int(n_runs * sampling_frequency * 2 * track_height / running_speed)

    time = simulate_time(n_samples, sampling_frequency)
    position = simulate_position(time, track_height, running_speed)

    if not make_inbound_outbound_neurons:
        place_fields = np.stack(
            [
                simulate_place_field_firing_rate(
                    place_field_mean, position, variance=place_field_variance
                )
                for place_field_mean in place_field_means
            ],
            axis=1,
        )
        spikes = np.stack(
            [
                simulate_neuron_with_place_field(
                    place_field_mean,
                    position,
                    max_rate=15,
                    variance=place_field_variance,
                    sampling_frequency=sampling_frequency,
                )
                for place_field_mean in place_field_means.T
            ],
            axis=1,
        )
    else:
        trajectory_direction = get_trajectory_direction(position)
        place_fields_list: list[NDArray[np.float64]] = []
        spikes_list: list[NDArray[np.float64]] = []
        for direction in np.unique(trajectory_direction):
            is_condition = trajectory_direction == direction
            for place_field_mean in place_field_means:
                place_fields_list.append(
                    simulate_place_field_firing_rate(
                        place_field_mean,
                        position,
                        variance=place_field_variance,
                        is_condition=is_condition,
                    )
                )
                spikes_list.append(
                    simulate_neuron_with_place_field(
                        place_field_mean,
                        position,
                        max_rate=15,
                        variance=place_field_variance,
                        sampling_frequency=sampling_frequency,
                        is_condition=is_condition,
                    )
                )

        place_fields = np.stack(place_fields_list, axis=1)
        spikes = np.stack(spikes_list, axis=1)

    return time, position, sampling_frequency, spikes, place_fields


def make_continuous_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    replay_speedup: int = REPLAY_SPEEDUP,
    is_outbound: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a simulated continuous replay.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    track_height : float, optional
        Height of the simulated track
    running_speed : float, optional
        Speed of the simulated animal
    place_field_means : NDArray[np.float64], optional
        Location of the center of the Gaussian place fields.
    replay_speedup : int, optional
        _description_, by default REPLAY_SPEEDUP
    is_outbound : bool, optional
        _description_, by default True

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    replay_speed = running_speed * replay_speedup
    n_samples = int(np.ceil(2 * sampling_frequency * track_height / replay_speed))
    replay_time = simulate_time(n_samples, sampling_frequency)
    true_replay_position = simulate_position(replay_time, track_height, replay_speed)

    # Make inbound or outbound
    replay_time = replay_time[: n_samples // 2]
    if is_outbound:
        true_replay_position = true_replay_position[: n_samples // 2]
    else:
        true_replay_position = true_replay_position[n_samples // 2 :]

    min_times_ind = np.argmin(
        np.abs(true_replay_position[:, np.newaxis] - place_field_means), axis=0
    )

    n_neurons = place_field_means.shape[0]
    test_spikes = np.zeros((replay_time.size, n_neurons))
    test_spikes[
        (
            min_times_ind,
            np.arange(n_neurons),
        )
    ] = 1.0

    return replay_time, test_spikes


def make_hover_replay(
    hover_neuron_ind: Optional[int] = None,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a simulated stationary replay.

    Parameters
    ----------
    hover_neuron_ind : int, optional
        _description_, by default None
    place_field_means : NDArray[np.float64], optional
        _description_, by default PLACE_FIELD_MEANS
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    n_neurons = place_field_means.shape[0]
    if hover_neuron_ind is None:
        hover_neuron_ind = n_neurons // 2

    N_TIME = 50
    replay_time = np.arange(N_TIME) / sampling_frequency

    spike_time_ind = np.arange(0, N_TIME, 2)

    test_spikes = np.zeros((N_TIME, n_neurons))
    neuron_ind = np.ones((N_TIME // 2,), dtype=int) * hover_neuron_ind

    test_spikes[(spike_time_ind, neuron_ind)] = 1.0

    return replay_time, test_spikes


def make_fragmented_replay(
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a simulated fragmented replay.

    Parameters
    ----------
    place_field_means : NDArray[np.float64], optional
        _description_, by default PLACE_FIELD_MEANS
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    N_TIME = 10
    replay_time = np.arange(N_TIME) / sampling_frequency
    ind = ([1, 3, 5, 7, 9], [1, -1, 10, -5, 8])
    n_neurons = place_field_means.shape[0]
    test_spikes = np.zeros((N_TIME, n_neurons))
    test_spikes[ind] = 1.0

    return replay_time, test_spikes


def make_hover_continuous_hover_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a replay that starts stationary, then is continuous, then is stationary again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    _, test_spikes1 = make_hover_replay(hover_neuron_ind=0)
    _, test_spikes2 = make_continuous_replay()
    _, test_spikes3 = make_hover_replay(hover_neuron_ind=-1)

    test_spikes = np.concatenate((test_spikes1, test_spikes2, test_spikes3))
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes


def make_fragmented_hover_fragmented_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a replay that starts fragmented, then is stationary, and then is fragmented again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    _, test_spikes1 = make_fragmented_replay()
    _, test_spikes2 = make_hover_replay(hover_neuron_ind=6)
    _, test_spikes3 = make_fragmented_replay()

    test_spikes = np.concatenate((test_spikes1, test_spikes2, test_spikes3))
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes


def make_fragmented_continuous_fragmented_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a replay that is fragmented, then is continuous, then is fragmented again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    _, test_spikes1 = make_fragmented_replay()
    _, test_spikes2 = make_continuous_replay()
    _, test_spikes3 = make_fragmented_replay()

    test_spikes = np.concatenate((test_spikes1, test_spikes2, test_spikes3))
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes


def make_theta_sweep(
    sampling_frequency: int = SAMPLING_FREQUENCY, n_sweeps: int = 5
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate theta sweeping

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    n_sweeps : int, optional
        Number of sweeps to simulate, by default 5

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_spikes : NDArray[np.float64], shape (n_time, n_neurons)
        Binned spike indicator. 1 means spike occured. 0 means no spike occured.

    """
    _, test_spikes1 = make_continuous_replay(is_outbound=False, replay_speedup=145)
    _, test_spikes2 = make_continuous_replay(is_outbound=True, replay_speedup=145)

    test_spikes = np.concatenate(
        [test_spikes1[test_spikes1.shape[0] // 2 :], test_spikes2] * n_sweeps
    )
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes
