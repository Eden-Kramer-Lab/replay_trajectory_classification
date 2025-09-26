"""Simulate clusterless spikes and associated spike waveform features.

This module provides functions for simulating clusterless (unsorted) spike
data with associated waveform features used in multiunit decoding approaches.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from replay_trajectory_classification.simulate import (
    get_trajectory_direction,
    simulate_multiunit_with_place_fields,
    simulate_position,
    simulate_time,
)

SAMPLING_FREQUENCY = 1000
TRACK_HEIGHT = 175
RUNNING_SPEED = 15
PLACE_FIELD_VARIANCE = 6.0**2
PLACE_FIELD_MEANS = np.arange(0, 200, 10)
N_RUNS = 15
REPLAY_SPEEDUP = 120.0
N_TETRODES = 5
N_FEATURES = 4
MARK_SPACING = 5


def make_simulated_run_data(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    place_field_variance: float = PLACE_FIELD_VARIANCE,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    n_tetrodes: int = N_TETRODES,
    make_inbound_outbound_neurons: bool = False,
):
    """Make simulated data of a rat running back and forth
    on a linear maze with unclustered spikes.

    Parameters
    ----------
    sampling_frequency : int, optional
    track_height : float, optional
        Height of the simulated track
    running_speed : float, optional
        Speed of the simulated animal
    n_runs : int, optional
        Number of runs across the track the simulated animal will perform
    place_field_variance : float, optional
        Spatial extent of place fields
    place_field_means : NDArray[np.float64], shape (n_neurons,), optional
        Location of the center of the Gaussian place fields.
    n_tetrodes : int, optional
        Total number of tetrodes to simulate
    make_inbound_outbound_neurons : bool, optional
        Create neurons with directional place fields

    Returns
    -------
    time : NDArray[np.float64], shape (n_time,)
    position : NDArray[np.float64], shape (n_time,)
    sampling_frequency : float
    multiunits : NDArray[np.float64], shape (n_time, n_features, n_electrodes)
    multiunits_spikes : NDArray[np.float64] (n_time, n_electrodes)
    place_field_means : NDArray[np.float64] (n_tetrodes, n_place_fields)

    """
    n_samples = int(n_runs * sampling_frequency * 2 * track_height / running_speed)

    time = simulate_time(n_samples, sampling_frequency)
    position = simulate_position(time, track_height, running_speed)

    multiunits = []
    if not make_inbound_outbound_neurons:
        for place_means in place_field_means.reshape(((n_tetrodes, -1))):
            multiunits.append(
                simulate_multiunit_with_place_fields(
                    place_means,
                    position,
                    mark_spacing=MARK_SPACING,
                    n_mark_dims=4,
                    sampling_frequency=sampling_frequency,
                )
            )
    else:
        trajectory_direction = get_trajectory_direction(position)
        for direction in np.unique(trajectory_direction):
            is_condition = trajectory_direction == direction
            for place_means in place_field_means.reshape(((n_tetrodes, -1))):
                multiunits.append(
                    simulate_multiunit_with_place_fields(
                        place_means,
                        position,
                        mark_spacing=10,
                        n_mark_dims=4,
                        sampling_frequency=sampling_frequency,
                        is_condition=is_condition,
                    )
                )
    multiunits = np.stack(multiunits, axis=-1)
    multiunits_spikes = np.any(~np.isnan(multiunits), axis=1)

    return (time, position, sampling_frequency, multiunits, multiunits_spikes)


def make_continuous_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    replay_speedup: int = REPLAY_SPEEDUP,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Creates a simulated continuous replay.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    track_height : float, optional
        Height of the simulated track
    running_speed : float, optional
        Simualted speed of the animal
    place_field_means : NDArray[np.float64], optional
        Location of the center of the Gaussian place fields.
    replay_speedup : int, optional
        Number of times faster the replay event is faster than the running speed
    n_tetrodes : int, optional
        Number of simulated tetrodes
    n_features : int, optional
        Number of simulated features
    mark_spacing : float, optional
        Spacing between Gaussian mark features

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_multiunits : NDArray[np.float64], shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.
    """

    replay_speed = running_speed * replay_speedup
    n_samples = int(0.5 * sampling_frequency * 2 * track_height / replay_speed)
    replay_time = simulate_time(n_samples, sampling_frequency)
    true_replay_position = simulate_position(replay_time, track_height, replay_speed)
    place_field_means = place_field_means.reshape(((n_tetrodes, -1)))

    min_times_ind = np.argmin(
        np.abs(true_replay_position[:, np.newaxis] - place_field_means.ravel()), axis=0
    )
    tetrode_ind = (
        np.ones_like(place_field_means) * np.arange(5)[:, np.newaxis]
    ).ravel()

    test_multiunits = np.full((replay_time.size, n_features, n_tetrodes), np.nan)
    n_neurons = place_field_means.shape[1]

    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)
    mark_ind = (np.ones_like(place_field_means) * np.arange(4)).ravel()

    for i in range(n_features):
        test_multiunits[(min_times_ind, i, tetrode_ind)] = mark_centers[mark_ind]

    return replay_time, test_multiunits


def make_hover_replay(
    hover_neuron_ind: int = None,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    sampling_frequency: int = SAMPLING_FREQUENCY,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Creates a simulated stationary replay.

    Parameters
    ----------
    hover_neuron_ind : int, optional
        Index of which neuron is the stationary neuron.
    place_field_means : NDArray[np.float64], optional
        Location of the center of the Gaussian place fields.
    sampling_frequency : int, optional
        Samples per second
    n_tetrodes : int, optional
        Number of simulated tetrodes
    n_features : int, optional
        Number of simulated features
    mark_spacing : float, optional
        Spacing between Gaussian mark features

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_multiunits : NDArray[np.float64], shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.
    """

    place_field_means = place_field_means.reshape(((n_tetrodes, -1)))

    if hover_neuron_ind is None:
        hover_neuron_ind = place_field_means.size // 2
    tetrode_ind, neuron_ind = np.unravel_index(
        hover_neuron_ind, place_field_means.shape
    )

    N_TIME = 50
    replay_time = np.arange(N_TIME) / sampling_frequency

    spike_time_ind = np.arange(0, N_TIME, 2)
    test_multiunits = np.full((replay_time.size, n_features, n_tetrodes), np.nan)
    n_neurons = place_field_means.shape[1]
    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)
    test_multiunits[spike_time_ind, :, tetrode_ind] = mark_centers[neuron_ind]

    return replay_time, test_multiunits


def make_fragmented_replay(
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
    sampling_frequency: int = SAMPLING_FREQUENCY,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Creates a simulated fragmented replay.

    Parameters
    ----------
    place_field_means : NDArray[np.float64], optional
        Location of the center of the Gaussian place fields.
    sampling_frequency : int, optional
        Samples per second
    n_tetrodes : int, optional
        Number of simulated tetrodes
    n_features : int, optional
        Number of simulated features
    mark_spacing : float, optional
        Spacing between Gaussian mark features

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_multiunits : NDArray[np.float64], shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.
    """

    N_TIME = 10

    place_field_means = place_field_means.reshape(((n_tetrodes, -1)))
    replay_time = np.arange(N_TIME) / sampling_frequency

    n_total_neurons = place_field_means.size
    neuron_inds = [1, n_total_neurons - 1, 10, n_total_neurons - 5, 8]
    neuron_inds = np.unravel_index(neuron_inds, place_field_means.shape)
    spike_time_ind = [1, 3, 5, 7, 9]
    test_multiunits = np.full((replay_time.size, n_features, n_tetrodes), np.nan)
    n_neurons = place_field_means.shape[1]
    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)

    for t_ind, tetrode_ind, neuron_ind in zip(spike_time_ind, *neuron_inds):
        test_multiunits[t_ind, :, tetrode_ind] = mark_centers[neuron_ind]

    return replay_time, test_multiunits


def make_hover_continuous_hover_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    place_field_means: NDArray[np.float64] = PLACE_FIELD_MEANS,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Make a simulated replay that first is stationary, then is continuous, then is stationary again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    place_field_means : NDArray[np.float64], optional
        Location of the center of the Gaussian place fields.

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_multiunits : NDArray[np.float64], shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.
    """
    _, test_multiunits1 = make_hover_replay(hover_neuron_ind=0)
    _, test_multiunits2 = make_continuous_replay()
    n_total_neurons = place_field_means.size
    _, test_multiunits3 = make_hover_replay(hover_neuron_ind=n_total_neurons - 1)

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3)
    )
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits


def make_fragmented_hover_fragmented_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Makes a simulated replay that first is fragmented, then is stationary, then is fragmented again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_multiunits : NDArray[np.float64], shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.
    """
    _, test_multiunits1 = make_fragmented_replay()
    _, test_multiunits2 = make_hover_replay(hover_neuron_ind=6)
    _, test_multiunits3 = make_fragmented_replay()

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3)
    )
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits


def make_fragmented_continuous_fragmented_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Makes a simulated replay that first is fragmented, then is continuous, then is fragmented again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : NDArray[np.float64], shape (n_time,)
        Time in seconds.
    test_multiunits : NDArray[np.float64], shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.

    """
    _, test_multiunits1 = make_fragmented_replay()
    _, test_multiunits2 = make_continuous_replay()
    _, test_multiunits3 = make_fragmented_replay()

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3)
    )
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits
