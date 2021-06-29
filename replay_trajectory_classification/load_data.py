import os
from logging import getLogger

import numpy as np
import pandas as pd
from loren_frank_data_processing import (Animal, get_all_multiunit_indicators,
                                         get_all_spike_indicators, get_LFPs,
                                         get_trial_time, make_epochs_dataframe,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe)
from loren_frank_data_processing.core import get_data_structure
from loren_frank_data_processing.DIO import get_DIO, get_DIO_indicator
from loren_frank_data_processing.position import get_well_locations
from loren_frank_data_processing.well_traversal_classification import (
    score_inbound_outbound, segment_path)
from ripple_detection import (Kay_ripple_detector, filter_ripple_band,
                              get_multiunit_population_firing_rate,
                              multiunit_HSE_detector)
from ripple_detection.core import gaussian_smooth, get_envelope
from scipy.io import loadmat
from scipy.stats import zscore
from spectral_connectivity import Connectivity, Multitaper
from track_linearization import get_linearized_position
from track_linearization import make_track_graph as _make_track_graph

logger = getLogger(__name__)

# LFP sampling frequency
SAMPLING_FREQUENCY = 500

ANIMALS = {
    'Jaq': Animal(directory='/stelmo/abhilasha/animals/Jaq/filterframework',
                  short_name='Jaq'),
    'Roqui': Animal(directory='/stelmo/abhilasha/animals/Roqui/filterframework',
                    short_name='Roqui'),
    'Peanut': Animal(directory='/stelmo/abhilasha/animals/Peanut/filterframework',
                     short_name='Peanut'),
}

WTRACK_EDGE_ORDER = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]
WTRACK_EDGE_SPACING = [15, 0, 15, 0]

LINEAR_EDGE_ORDER = [(0, 1)]
LINEAR_EDGE_SPACING = 0


def get_track_segments(epoch_key, animals):
    '''

    Parameters
    ----------
    epoch_key : tuple
    animals : dict of namedtuples

    Returns
    -------
    track_segments : ndarray, shape (n_segments, n_nodes, n_space)
    center_well_position : ndarray, shape (n_space,)

    '''
    environment = np.asarray(
        make_epochs_dataframe(animals).loc[epoch_key].environment)[0]
    ENVIRONMENTS = {'lineartrack': 'linearTrack',
                    'wtrack': 'wTrack'}

    coordinate_path = os.path.join(
        animals[epoch_key[0]].directory,
        f'{ENVIRONMENTS[environment]}_coordinates.mat')
    linearcoord = loadmat(coordinate_path)['coords'][0]
    track_segments = [np.stack(((arm[:-1, :, 0], arm[1:, :, 0])), axis=1)
                      for arm in linearcoord]
    track_segments = np.concatenate(track_segments)
    _, unique_ind = np.unique(track_segments, return_index=True, axis=0)
    return track_segments[np.sort(unique_ind)]


def make_track_graph(epoch_key, animals, convert_to_pixels=False):
    '''

    Parameters
    ----------
    epoch_key : tuple, (animal, day, epoch)
    animals : dict of namedtuples

    Returns
    -------
    track_graph : networkx Graph

    '''
    track_segments = get_track_segments(epoch_key, animals)
    nodes = track_segments.copy().reshape((-1, 2))
    _, unique_ind = np.unique(nodes, return_index=True, axis=0)
    nodes = nodes[np.sort(unique_ind)]

    edges = np.zeros(track_segments.shape[:2], dtype=np.int)
    for node_id, node in enumerate(nodes):
        edge_ind = np.nonzero(np.isin(track_segments, node).sum(axis=2) > 1)
        edges[edge_ind] = node_id

    return _make_track_graph(nodes, edges)


def get_labels(times, time):
    ripple_labels = pd.DataFrame(np.zeros_like(time, dtype=np.int), index=time,
                                 columns=['replay_number'])
    for replay_number, start_time, end_time in times.itertuples():
        ripple_labels.loc[start_time:end_time] = replay_number

    return ripple_labels


def estimate_ripple_band_power(lfps, sampling_frequency):
    m = Multitaper(lfps.values, sampling_frequency=sampling_frequency,
                   time_halfbandwidth_product=1,
                   time_window_duration=0.020,
                   time_window_step=0.020,
                   start_time=lfps.index[0].total_seconds())
    c = Connectivity.from_multitaper(m)
    closest_200Hz_freq_ind = np.argmin(np.abs(c.frequencies - 200))
    power = c.power()[..., closest_200Hz_freq_ind, :].squeeze() + np.spacing(1)
    n_samples = int(0.020 * sampling_frequency)
    index = lfps.index[np.arange(1, power.shape[0] * n_samples + 1, n_samples)]
    power = pd.DataFrame(power, index=index)
    return power.reindex(lfps.index)


def get_ripple_consensus_trace(ripple_filtered_lfps, sampling_frequency):
    SMOOTHING_SIGMA = 0.004
    ripple_consensus_trace = np.full_like(ripple_filtered_lfps, np.nan)
    not_null = np.all(pd.notnull(ripple_filtered_lfps), axis=1)
    ripple_consensus_trace[not_null] = get_envelope(
        np.asarray(ripple_filtered_lfps)[not_null])
    ripple_consensus_trace = np.sum(ripple_consensus_trace ** 2, axis=1)
    ripple_consensus_trace[not_null] = gaussian_smooth(
        ripple_consensus_trace[not_null], SMOOTHING_SIGMA, sampling_frequency)
    return np.sqrt(ripple_consensus_trace)


def get_adhoc_ripple(epoch_key, tetrode_info, position_time,
                     position_to_linearize):
    LFP_SAMPLING_FREQUENCY = 1500

    # Get speed in terms of the LFP time
    time = get_trial_time(epoch_key, ANIMALS)
    position_df = get_position_info(
        epoch_key, skip_linearization=True)
    new_index = pd.Index(np.unique(np.concatenate(
        (position_df.index, time))), name='time')
    position_df = (position_df
                   .reindex(index=new_index)
                   .interpolate(method='linear')
                   .reindex(index=time)
                   )
    speed_feature = position_to_linearize[0].split('_')[0]
    speed = position_df[f'{speed_feature}_vel']

    # Load LFPs
    tetrode_keys = tetrode_info.loc[tetrode_info.area.isin(
        ['ca1R', 'ca1L'])].index
    ripple_lfps = get_LFPs(tetrode_keys, ANIMALS)

    # Get ripple filtered LFPs
    ripple_filtered_lfps = pd.DataFrame(
        filter_ripple_band(np.asarray(ripple_lfps)),
        index=ripple_lfps.index)

    # Get Ripple Times
    ripple_times = Kay_ripple_detector(
        time=ripple_filtered_lfps.index,
        filtered_lfps=ripple_filtered_lfps.values,
        speed=speed.values,
        sampling_frequency=LFP_SAMPLING_FREQUENCY,
        zscore_threshold=2.0,
        close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))

    ripple_times.index = ripple_times.index.rename('replay_number')
    ripple_labels = get_labels(ripple_times, position_time)
    is_ripple = ripple_labels > 0
    ripple_times = ripple_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    # Estmate ripple band power and change
    ripple_consensus_trace = pd.DataFrame(
        get_ripple_consensus_trace(
            ripple_filtered_lfps, LFP_SAMPLING_FREQUENCY),
        index=ripple_filtered_lfps.index,
        columns=['ripple_consensus_trace'])
    ripple_consensus_trace_zscore = pd.DataFrame(
        zscore(ripple_consensus_trace, nan_policy='omit'),
        index=ripple_filtered_lfps.index,
        columns=['ripple_consensus_trace_zscore'])

    instantaneous_ripple_power = np.full_like(ripple_filtered_lfps, np.nan)
    not_null = np.all(pd.notnull(ripple_filtered_lfps), axis=1)
    instantaneous_ripple_power[not_null] = get_envelope(
        np.asarray(ripple_filtered_lfps)[not_null])**2
    instantaneous_ripple_power_change = np.nanmedian(
        instantaneous_ripple_power /
        np.nanmean(instantaneous_ripple_power, axis=0),
        axis=1)
    instantaneous_ripple_power_change = pd.DataFrame(
        instantaneous_ripple_power_change,
        index=ripple_filtered_lfps.index,
        columns=['instantaneous_ripple_power_change'])

    return dict(
        ripple_times=ripple_times,
        ripple_labels=ripple_labels,
        ripple_filtered_lfps=ripple_filtered_lfps,
        ripple_consensus_trace=ripple_consensus_trace,
        ripple_lfps=ripple_lfps,
        ripple_consensus_trace_zscore=ripple_consensus_trace_zscore,
        instantaneous_ripple_power_change=instantaneous_ripple_power_change,
        is_ripple=is_ripple)


def get_adhoc_multiunit(position_info, tetrode_keys, time_function,
                        position_to_linearize):
    time = position_info.index
    multiunits = get_all_multiunit_indicators(
        tetrode_keys, ANIMALS, time_function)
    multiunit_spikes = (np.any(~np.isnan(multiunits.values), axis=1)
                        ).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY),
        index=position_info.index,
        columns=['firing_rate'])
    multiunit_rate_change = multiunit_firing_rate.transform(
        lambda df: df / df.mean())
    multiunit_rate_zscore = np.log(multiunit_firing_rate).transform(
        lambda df: (df - df.mean()) / df.std())

    speed_feature = position_to_linearize[0].split('_')[0]
    speed = np.asarray(position_info[f'{speed_feature}_vel'])

    multiunit_high_synchrony_times = multiunit_HSE_detector(
        time, multiunit_spikes, speed,
        SAMPLING_FREQUENCY,
        minimum_duration=np.timedelta64(15, 'ms'), zscore_threshold=2.0,
        close_event_threshold=np.timedelta64(0, 'ms'))
    multiunit_high_synchrony_times.index = (
        multiunit_high_synchrony_times.index.rename('replay_number'))
    multiunit_high_synchrony_labels = get_labels(
        multiunit_high_synchrony_times, time)
    is_multiunit_high_synchrony = multiunit_high_synchrony_labels > 0
    multiunit_high_synchrony_times = multiunit_high_synchrony_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    return dict(
        multiunits=multiunits,
        multiunit_spikes=multiunit_spikes,
        multiunit_firing_rate=multiunit_firing_rate,
        multiunit_high_synchrony_times=multiunit_high_synchrony_times,
        multiunit_high_synchrony_labels=multiunit_high_synchrony_labels,
        multiunit_rate_change=multiunit_rate_change,
        multiunit_rate_zscore=multiunit_rate_zscore,
        is_multiunit_high_synchrony=is_multiunit_high_synchrony)


def load_data(epoch_key,
              position_to_linearize=['nose_x', 'nose_y'],
              max_distance_from_well=30,
              min_distance_traveled=50,
              ):
    logger.info('Loading position info...')
    environment = np.asarray(
        make_epochs_dataframe(ANIMALS).loc[epoch_key].environment)[0]
    if environment == "lineartrack":
        edge_order, edge_spacing = LINEAR_EDGE_ORDER, LINEAR_EDGE_SPACING
    elif environment == "wtrack":
        edge_order, edge_spacing = WTRACK_EDGE_ORDER, WTRACK_EDGE_SPACING
    else:
        edge_order, edge_spacing = None, None
    position_info = get_interpolated_position_info(
        epoch_key,
        position_to_linearize=position_to_linearize,
        max_distance_from_well=max_distance_from_well,
        min_distance_traveled=min_distance_traveled,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).dropna(subset=["linear_position"])
    tetrode_info = make_tetrode_dataframe(
        ANIMALS, epoch_key=epoch_key)
    tetrode_keys = tetrode_info.loc[tetrode_info.area.isin(
        ['ca1R', 'ca1L'])].index

    logger.info('Loading multiunit...')

    def _time_function(*args, **kwargs):
        return position_info.index

    adhoc_multiunit = get_adhoc_multiunit(
        position_info, tetrode_keys, _time_function, position_to_linearize)

    logger.info('Loading spikes...')
    time = position_info.index
    try:
        neuron_info = make_neuron_dataframe(
            ANIMALS, exclude_animals=['Monty', 'Peanut']).xs(
                epoch_key, drop_level=False)
        neuron_info = neuron_info.loc[neuron_info.accepted.astype(bool)]
        spikes = get_all_spike_indicators(
            neuron_info.index, ANIMALS, _time_function).reindex(time)
    except (ValueError, KeyError):
        neuron_info = None
        spikes = None

    logger.info('Finding ripple times...')
    adhoc_ripple = get_adhoc_ripple(
        epoch_key, tetrode_info, time, position_to_linearize)

    track_graph = make_track_graph(epoch_key, ANIMALS)

    dio = get_DIO(epoch_key, ANIMALS)
    dio_indicator = get_DIO_indicator(
        epoch_key, ANIMALS, time_function=_time_function)

    return {
        'position_info': position_info,
        'tetrode_info': tetrode_info,
        'neuron_info': neuron_info,
        'spikes': spikes,
        'dio': dio,
        'dio_indicator': dio_indicator,
        'track_graph': track_graph,
        'edge_order': edge_order,
        'edge_spacing': edge_spacing,
        **adhoc_ripple,
        **adhoc_multiunit,
    }


def _get_pos_dataframe(epoch_key, animals):
    animal, day, epoch = epoch_key
    struct = get_data_structure(
        animals[animal], day, 'posdlc', 'posdlc')[epoch - 1]
    position_data = struct['data'][0, 0]
    field_names = struct['fields'][0, 0][0].split()
    time = pd.TimedeltaIndex(
        position_data[:, 0], unit='s', name='time')

    return pd.DataFrame(
        position_data[:, 1:], columns=field_names[1:], index=time)


def get_segments_df(epoch_key, animals, position_df, max_distance_from_well=30,
                    min_distance_traveled=50,
                    position_to_linearize=['nose_x', 'nose_y']):
    well_locations = get_well_locations(epoch_key, animals)
    position = position_df.loc[:, position_to_linearize].values
    segments_df, labeled_segments = segment_path(
        position_df.index, position, well_locations, epoch_key, animals,
        max_distance_from_well=max_distance_from_well)
    segments_df = score_inbound_outbound(
        segments_df, epoch_key, animals, min_distance_traveled)
    segments_df = segments_df.loc[
        :, ['from_well', 'to_well', 'task', 'is_correct', 'turn']]

    return segments_df, labeled_segments


def _get_linear_position_hmm(
    epoch_key, animals, position_df,
        max_distance_from_well=30,
        route_euclidean_distance_scaling=1,
        min_distance_traveled=50,
        sensor_std_dev=5,
        diagonal_bias=0.5,
        edge_order=WTRACK_EDGE_ORDER,
        edge_spacing=WTRACK_EDGE_SPACING,
        position_to_linearize=['nose_x', 'nose_y'],
        position_sampling_frequency=125):
    animal, day, epoch = epoch_key
    track_graph = make_track_graph(epoch_key, animals)
    position = np.asarray(position_df.loc[:, position_to_linearize])
    linearized_position_df = get_linearized_position(
        position=position,
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )
    position_df = pd.concat(
        (position_df,
         linearized_position_df.set_index(position_df.index)), axis=1)
    try:
        SEGMENT_ID_TO_ARM_NAME = {0.0: 'Center Arm',
                                  1.0: 'Left Arm',
                                  2.0: 'Right Arm',
                                  3.0: 'Left Arm',
                                  4.0: 'Right Arm'}
        position_df = position_df.assign(
            arm_name=lambda df: df.track_segment_id.map(SEGMENT_ID_TO_ARM_NAME)
        )
        segments_df, labeled_segments = get_segments_df(
            epoch_key, animals, position_df, max_distance_from_well,
            min_distance_traveled)

        segments_df = pd.merge(
            labeled_segments, segments_df, right_index=True,
            left_on='labeled_segments', how='outer')
        position_df = pd.concat((position_df, segments_df), axis=1)
        position_df['is_correct'] = position_df.is_correct.fillna(False)
    except ValueError:
        pass

    return position_df


def get_position_info(
    epoch_key, position_to_linearize=['nose_x', 'nose_y'],
        max_distance_from_well=30, min_distance_traveled=50,
        skip_linearization=False, route_euclidean_distance_scaling=1E-1,
        sensor_std_dev=5, diagonal_bias=0.5, position_sampling_frequency=125,
        edge_order=WTRACK_EDGE_ORDER, edge_spacing=WTRACK_EDGE_SPACING
):
    position_df = _get_pos_dataframe(epoch_key, ANIMALS)

    if not skip_linearization:
        position_df = _get_linear_position_hmm(
            epoch_key, ANIMALS, position_df,
            max_distance_from_well, route_euclidean_distance_scaling,
            min_distance_traveled, sensor_std_dev, diagonal_bias,
            edge_order=edge_order, edge_spacing=edge_spacing,
            position_to_linearize=position_to_linearize,
            position_sampling_frequency=position_sampling_frequency)

    return position_df

# max_distance_from_well=30 cms. This is perhaps ok for the tail but maybe the
# value needs to be lower for paws, nose etc.
# also eventually DIOs may help in the trajectory classification.


def get_interpolated_position_info(
    epoch_key, position_to_linearize=['nose_x', 'nose_y'],
        max_distance_from_well=30, min_distance_traveled=50,
        route_euclidean_distance_scaling=1E-1,
        sensor_std_dev=5, diagonal_bias=1E-1, edge_order=WTRACK_EDGE_ORDER,
        edge_spacing=WTRACK_EDGE_SPACING):
    position_info = get_position_info(
        epoch_key, skip_linearization=True)
    position_info = position_info.resample('2ms').mean().interpolate('linear')

    position_info = _get_linear_position_hmm(
        epoch_key, ANIMALS, position_info,
        max_distance_from_well, route_euclidean_distance_scaling,
        min_distance_traveled, sensor_std_dev, diagonal_bias,
        edge_order=edge_order, edge_spacing=edge_spacing,
        position_to_linearize=position_to_linearize,
        position_sampling_frequency=500)

    return position_info
