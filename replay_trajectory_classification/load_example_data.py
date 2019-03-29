from logging import getLogger
from os.path import abspath, dirname, join, pardir

import numpy as np
import pandas as pd
from loren_frank_data_processing import (Animal, get_all_multiunit_indicators,
                                         get_all_spike_indicators,
                                         get_interpolated_position_dataframe,
                                         get_LFPs, make_neuron_dataframe,
                                         make_tetrode_dataframe,
                                         get_trial_time)

from ripple_detection import (Kay_ripple_detector, filter_ripple_band,
                              get_multiunit_population_firing_rate)

logger = getLogger(__name__)

# LFP sampling frequency
SAMPLING_FREQUENCY = 1000

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'bon': Animal(directory=join(RAW_DATA_DIR, 'Bond'), short_name='bon'),
}

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max', 'channel_4_max']
_BRAIN_AREAS = ['CA1', 'CA3']


def load_data(epoch_key, brain_areas=None, speed_metric='linear_speed'):

    if brain_areas is None:
        brain_areas = _BRAIN_AREAS

    time = get_trial_time(epoch_key, ANIMALS)
    time = (pd.Series(np.ones_like(time, dtype=np.float), index=time)
            .resample('1ms').mean()
            .index)

    def _time_function(*args, **kwargs):
        return time

    position_info = (
        get_interpolated_position_dataframe(epoch_key, ANIMALS, _time_function)
        .dropna(subset=['linear_distance', 'linear_speed']))

    speed = position_info[speed_metric]
    time = position_info.index

    tetrode_info = make_tetrode_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    is_brain_areas = tetrode_info.area.isin(brain_areas)
    tetrode_keys = tetrode_info.loc[
        (tetrode_info.validripple == 1) & is_brain_areas].index
    lfps = get_LFPs(tetrode_keys, ANIMALS)
    lfps = lfps.resample('1ms').mean().fillna(method='pad').reindex(time)

    neuron_info = make_neuron_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    neuron_info = neuron_info.loc[
        (neuron_info.numspikes > 0) &
        neuron_info.area.isin(brain_areas)]
    spikes = get_all_spike_indicators(
        neuron_info.index, ANIMALS, _time_function).reindex(time)

    tetrode_info = tetrode_info.loc[is_brain_areas]
    multiunit = (get_all_multiunit_indicators(
        tetrode_info.index, ANIMALS, _time_function)
                 .sel(features=_MARKS)
                 .reindex({'time': time}))
    multiunit_spikes = (np.any(~np.isnan(multiunit), axis=1)
                        .values).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY), index=time,
        columns=['firing_rate'])

    logger.info('Finding ripple times...')
    ripple_times = Kay_ripple_detector(
        time, lfps.values, speed.values, SAMPLING_FREQUENCY,
        zscore_threshold=2.0, close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))

    ripple_band_lfps = pd.DataFrame(
        np.stack([filter_ripple_band(lfps.values[:, ind])
                  for ind in np.arange(lfps.shape[1])], axis=1),
        index=lfps.index)

    ripple_times = ripple_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    return {
        'position_info': position_info,
        'ripple_times': ripple_times,
        'spikes': spikes,
        'multiunit': multiunit,
        'lfps': lfps,
        'tetrode_info': tetrode_info,
        'ripple_band_lfps': ripple_band_lfps,
        'multiunit_firing_rate': multiunit_firing_rate,
        'sampling_frequency': SAMPLING_FREQUENCY,
    }
