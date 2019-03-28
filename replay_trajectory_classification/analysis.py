import numpy as np
import pandas as pd

from .classifier import SortedSpikesClassifier


def get_replay_info(results, ripple_spikes, ripple_position, ripple_times,
                    sampling_frequency, probablity_threshold):
    duration = (
        (SortedSpikesClassifier.predict_proba(results) > probablity_threshold)
        .sum('time') / sampling_frequency)
    duration = duration.acausal_posterior.to_dataframe().unstack(level=1)
    duration.columns = list(duration.columns.get_level_values('state'))
    duration = duration.rename(
        columns=lambda column_name: column_name + '_duration')
    is_category = (duration > 0.0).rename(columns=lambda c: c.split('_')[0])
    duration = pd.concat((duration, is_category), axis=1)
    duration['is_classified'] = np.any(duration > 0, axis=1)
    duration['n_unique_spiking'] = get_n_unique_spiking(ripple_spikes)
    duration['n_total_spikes'] = get_n_total_spikes(ripple_spikes)
    avg_ripple_position = ripple_position.groupby('ripple_number').mean()

    return pd.concat((ripple_times, duration, avg_ripple_position), axis=1)


def get_n_unique_spiking(ripple_spikes):
    return (ripple_spikes.groupby('ripple_number').sum() > 0).sum(axis=1)


def get_n_total_spikes(ripple_spikes):
    return ripple_spikes.groupby('ripple_number').sum().sum(axis=1)


def maximum_a_posteriori_estimate(posterior_density):
    '''

    Parameters
    ----------
    posterior_density : xarray.DataArray, shape (n_time, n_x_bins, n_y_bins)

    Returns
    -------
    map_estimate : ndarray, shape (n_time,)

    '''
    stacked_posterior = np.log(posterior_density.stack(
        z=['x_position', 'y_position']))
    map_estimate = stacked_posterior.z[stacked_posterior.argmax('z')]
    return np.asarray(map_estimate.values.tolist())


def get_place_field_max(classifier):
    max_ind = classifier.place_fields_.argmax('position')
    return np.asarray(
        classifier.place_fields_.position[max_ind].values.tolist())


def get_linear_position_order(position_info, place_field_max):
    position = position_info.loc[:, ['x_position', 'y_position']]
    linear_place_field_max = []

    for place_max in place_field_max:
        min_ind = np.sqrt(
            np.sum(np.abs(place_max - position) ** 2, axis=1)).argmin()
        linear_place_field_max.append(
            position_info.loc[min_ind, 'linear_position2'])

    linear_place_field_max = np.asarray(linear_place_field_max)
    return np.argsort(linear_place_field_max), linear_place_field_max
