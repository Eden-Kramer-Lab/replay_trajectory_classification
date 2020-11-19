import numpy as np
from replay_trajectory_classification.simulate import (
    get_trajectory_direction, simulate_linear_distance,
    simulate_multiunit_with_place_fields, simulate_time)

SAMPLING_FREQUENCY = 1000
TRACK_HEIGHT = 175
RUNNING_SPEED = 15
PLACE_FIELD_VARIANCE = 6.0 ** 2
PLACE_FIELD_MEANS = np.arange(0, 200, 10)
N_RUNS = 15
REPLAY_SPEEDUP = 120.0
N_TETRODES = 5
N_FEATURES = 4
MARK_SPACING = 5


def make_simulated_run_data(sampling_frequency=SAMPLING_FREQUENCY,
                            track_height=TRACK_HEIGHT,
                            running_speed=RUNNING_SPEED, n_runs=N_RUNS,
                            place_field_variance=PLACE_FIELD_VARIANCE,
                            place_field_means=PLACE_FIELD_MEANS,
                            n_tetrodes=N_TETRODES,
                            make_inbound_outbound_neurons=False):
    '''Make simulated data of a rat running back and forth
    on a linear maze with unclustered spikes.

    Parameters
    ----------
    sampling_frequency : float, optional
    track_height : float, optional
    running_speed : float, optional
    n_runs : int, optional
    place_field_variance : float, optional
    place_field_means : ndarray, shape (n_neurons,), optional

    Returns
    -------
    time : ndarray, shape (n_time,)
    linear_distance : ndarray, shape (n_time,)
    sampling_frequency : float
    multiunits : ndarray, shape (n_time, n_features, n_electrodes)
    multiunits_spikes : ndarray (n_time, n_electrodes)
    place_field_means : ndarray (n_tetrodes, n_place_fields)
    '''
    n_samples = int(n_runs * sampling_frequency *
                    2 * track_height / running_speed)

    time = simulate_time(n_samples, sampling_frequency)
    linear_distance = simulate_linear_distance(
        time, track_height, running_speed)

    multiunits = []
    if not make_inbound_outbound_neurons:
        for place_means in place_field_means.reshape(((n_tetrodes, -1))):
            multiunits.append(
                simulate_multiunit_with_place_fields(
                    place_means, linear_distance, mark_spacing=10,
                    n_mark_dims=4,
                    sampling_frequency=sampling_frequency))
    else:
        trajectory_direction = get_trajectory_direction(linear_distance)
        for direction in np.unique(trajectory_direction):
            is_condition = trajectory_direction == direction
            for place_means in place_field_means.reshape(((n_tetrodes, -1))):
                multiunits.append(
                    simulate_multiunit_with_place_fields(
                        place_means, linear_distance, mark_spacing=10,
                        n_mark_dims=4,
                        sampling_frequency=sampling_frequency,
                        is_condition=is_condition))
    multiunits = np.stack(multiunits, axis=-1)
    multiunits_spikes = np.any(~np.isnan(multiunits), axis=1)

    return (time, linear_distance, sampling_frequency,
            multiunits, multiunits_spikes)


def make_continuous_replay(sampling_frequency=SAMPLING_FREQUENCY,
                           track_height=TRACK_HEIGHT,
                           running_speed=RUNNING_SPEED,
                           place_field_means=PLACE_FIELD_MEANS,
                           replay_speedup=REPLAY_SPEEDUP,
                           n_tetrodes=N_TETRODES,
                           n_features=N_FEATURES,
                           mark_spacing=MARK_SPACING):
    replay_speed = running_speed * replay_speedup
    n_samples = int(0.5 * sampling_frequency *
                    2 * track_height / replay_speed)
    replay_time = simulate_time(n_samples, sampling_frequency)
    true_replay_position = simulate_linear_distance(
        replay_time, track_height, replay_speed)
    place_field_means = place_field_means.reshape(((n_tetrodes, -1)))

    min_times_ind = np.argmin(
        np.abs(true_replay_position[:, np.newaxis] -
               place_field_means.ravel()), axis=0)
    tetrode_ind = (np.ones_like(place_field_means) *
                   np.arange(5)[:, np.newaxis]).ravel()

    test_multiunits = np.full(
        (replay_time.size, n_features, n_tetrodes), np.nan)
    n_neurons = place_field_means.shape[1]

    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)
    mark_ind = (np.ones_like(place_field_means) * np.arange(4)).ravel()

    for i in range(n_features):
        test_multiunits[(min_times_ind, i, tetrode_ind)
                        ] = mark_centers[mark_ind]

    return replay_time, test_multiunits


def make_hover_replay(hover_neuron_ind=None,
                      place_field_means=PLACE_FIELD_MEANS,
                      sampling_frequency=SAMPLING_FREQUENCY,
                      n_tetrodes=N_TETRODES,
                      n_features=N_FEATURES,
                      mark_spacing=MARK_SPACING):

    place_field_means = place_field_means.reshape(((n_tetrodes, -1)))

    if hover_neuron_ind is None:
        hover_neuron_ind = place_field_means.size // 2
    tetrode_ind, neuron_ind = np.unravel_index(
        hover_neuron_ind, place_field_means.shape)

    N_TIME = 50
    replay_time = np.arange(N_TIME) / sampling_frequency

    spike_time_ind = np.arange(0, N_TIME, 2)
    test_multiunits = np.full(
        (replay_time.size, n_features, n_tetrodes), np.nan)
    n_neurons = place_field_means.shape[1]
    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)
    test_multiunits[spike_time_ind, :, tetrode_ind] = mark_centers[neuron_ind]

    return replay_time, test_multiunits


def make_fragmented_replay(place_field_means=PLACE_FIELD_MEANS,
                           sampling_frequency=SAMPLING_FREQUENCY,
                           n_tetrodes=N_TETRODES,
                           n_features=N_FEATURES,
                           mark_spacing=MARK_SPACING):
    N_TIME = 10

    place_field_means = place_field_means.reshape(((n_tetrodes, -1)))
    replay_time = np.arange(N_TIME) / sampling_frequency

    n_total_neurons = place_field_means.size
    neuron_inds = [1, n_total_neurons - 1, 10, n_total_neurons - 5, 8]
    neuron_inds = np.unravel_index(neuron_inds, place_field_means.shape)
    spike_time_ind = [1, 3, 5, 7, 9]
    test_multiunits = np.full(
        (replay_time.size, n_features, n_tetrodes), np.nan)
    n_neurons = place_field_means.shape[1]
    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)

    for t_ind, tetrode_ind, neuron_ind in zip(spike_time_ind, *neuron_inds):
        test_multiunits[t_ind, :, tetrode_ind] = mark_centers[neuron_ind]

    return replay_time, test_multiunits


def make_hover_continuous_hover_replay(sampling_frequency=SAMPLING_FREQUENCY,
                                       place_field_means=PLACE_FIELD_MEANS):
    _, test_multiunits1 = make_hover_replay(hover_neuron_ind=0)
    _, test_multiunits2 = make_continuous_replay()
    n_total_neurons = place_field_means.size
    _, test_multiunits3 = make_hover_replay(
        hover_neuron_ind=n_total_neurons - 1)

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3))
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits


def make_fragmented_hover_fragmented_replay(
        sampling_frequency=SAMPLING_FREQUENCY):
    _, test_multiunits1 = make_fragmented_replay()
    _, test_multiunits2 = make_hover_replay(hover_neuron_ind=6)
    _, test_multiunits3 = make_fragmented_replay()

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3))
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits


def make_fragmented_continuous_fragmented_replay(
        sampling_frequency=SAMPLING_FREQUENCY):
    _, test_multiunits1 = make_fragmented_replay()
    _, test_multiunits2 = make_continuous_replay()
    _, test_multiunits3 = make_fragmented_replay()

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3))
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits
