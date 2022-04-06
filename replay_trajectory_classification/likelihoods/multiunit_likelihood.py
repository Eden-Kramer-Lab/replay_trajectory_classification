
import numpy as np
from replay_trajectory_classification.bins import atleast_2d


def fit_occupancy(position, place_bin_centers, model,
                  model_kwargs, is_track_interior):
    '''

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    model : model class
    model_kwargs : dict
    is_track_interior : ndarray, shape (n_bins,)

    Returns
    -------
    occupancy : ndarray, shape (n_bins,)
    occupancy_model : model class instance

    '''
    not_nan_position = np.all(~np.isnan(atleast_2d(position)), axis=1)
    occupancy_model = model(
        **model_kwargs).fit(atleast_2d(position[not_nan_position]))
    occupancy = np.zeros((place_bin_centers.shape[0],))
    occupancy[is_track_interior] = np.exp(occupancy_model.score_samples(
        atleast_2d(place_bin_centers[is_track_interior])))
    return occupancy, occupancy_model


def fit_marginal_model(multiunit, position, place_bin_centers,
                       model, model_kwargs, is_track_interior):
    '''

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_features)
    position : ndarray, shape (n_time, n_position_dims)
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    model : class
    model_kwargs : dict
    is_track_interior : ndarray, shape (n_bins,)

    Returns
    -------
    marginal_density : ndarray, shape (n_bins,)

    '''
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    marginal_density = np.zeros((place_bin_centers.shape[0],))
    if is_spike.sum() > 0:
        not_nan_position = np.all(~np.isnan(atleast_2d(position)), axis=1)
        marginal_model = (model(**model_kwargs)
                          .fit(atleast_2d(position)[is_spike &
                                                    not_nan_position]))

        marginal_density[is_track_interior] = np.exp(
            marginal_model.score_samples(
                atleast_2d(place_bin_centers[is_track_interior])))
    return marginal_density


def train_joint_model(multiunit, position, model, model_kwargs):
    '''Fits a density model to the joint pdf of position and mark.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_features)
    position : ndarray, shape (n_time, n_position_dims)
    model : model class
    model_kwargs : dict

    Returns
    -------
    fitted_joint_model : model class instance

    '''
    multiunit, position = atleast_2d(multiunit), atleast_2d(position)
    is_spike = (np.any(~np.isnan(multiunit), axis=1) &
                np.all(~np.isnan(position), axis=1))
    not_nan_marks = np.any(~np.isnan(multiunit), axis=0)

    return (model(**model_kwargs)
            .fit(np.concatenate((multiunit[is_spike][:, not_nan_marks],
                                 position[is_spike]), axis=1)))


def estimate_mean_rate(multiunit):
    '''

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_features)

    Returns
    -------
    mean_rate : float

    '''
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    return is_spike.mean()


def estimate_log_intensity(density, occupancy, mean_rate):
    '''

    Parameters
    ----------
    density : ndarray, shape (n_bins,)
    occupancy : ndarray, shape (n_bins,)
    mean_rate : float

    Returns
    -------
    intensity : ndarray, shape (n_bins,)

    '''
    return np.log(mean_rate) + np.log(density) - np.log(occupancy)


def estimate_log_intensity2(log_density, occupancy, mean_rate):
    '''

    Parameters
    ----------
    density : ndarray, shape (n_bins,)
    occupancy : ndarray, shape (n_bins,)
    mean_rate : float

    Returns
    -------
    intensity : ndarray, shape (n_bins,)

    '''
    return np.log(mean_rate) + log_density - np.log(occupancy)


def estimate_intensity(density, occupancy, mean_rate):
    '''

    Parameters
    ----------
    density : ndarray, shape (n_bins,)
    occupancy : ndarray, shape (n_bins,)
    mean_rate : float

    Returns
    -------
    intensity : ndarray, shape (n_bins,)

    '''
    return np.exp(estimate_log_intensity(density, occupancy, mean_rate))


def estimate_ground_process_intensity(multiunit, position, place_bin_centers,
                                      occupancy, mean_rate, model,
                                      model_kwargs, is_track_interior):
    '''

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_marks)
    position : ndarray, shape (n_time, n_position_dims)
    place_bin_centers : ndarray, (n_bins, n_position_dims)
    occupancy : ndarray, (n_bins, n_position_dims)
    mean_rate : float
    model : sklearn model
    model_kwargs : dict
    is_track_interior : ndarray, shape (n_bins,)

    Returns
    -------
    ground_process_intensity : ndarray, shape (1, n_bins)

    '''
    marginal_pdf = fit_marginal_model(
        multiunit, position, place_bin_centers, model, model_kwargs,
        is_track_interior)
    ground_process_intensity = np.zeros((1, place_bin_centers.shape[0],))
    ground_process_intensity[:, is_track_interior] = estimate_intensity(
        marginal_pdf[is_track_interior], occupancy[is_track_interior],
        mean_rate)
    return ground_process_intensity


def estimate_log_joint_mark_intensity(
        multiunit, place_bin_centers, occupancy, joint_model, mean_rate):
    '''

    Parameters
    ----------
    multiunit : ndarray, shape (n_decoding_spikes, n_marks)
    place_bin_centers : ndarray, (n_bins, n_position_dims)
    occupancy : ndarray, (n_bins,)
    joint_model : sklearn model
    mean_rate : float

    Returns
    -------
    joint_mark_intensity : ndarray, shape (n_decoding_spikes, n_bins)

    '''
    multiunit = np.atleast_2d(multiunit)
    n_bins = place_bin_centers.shape[0]
    n_decoding_spikes = multiunit.shape[0]

    log_pdf = joint_model.score_samples(get_marks_by_place_bin_centers(
        multiunit, place_bin_centers)
    ).reshape((n_decoding_spikes, n_bins), order='F')

    return estimate_log_intensity2(log_pdf, occupancy, mean_rate)


def fit_multiunit_likelihood(position, multiunits, place_bin_centers,
                             model, model_kwargs,
                             occupancy_model=None,
                             occupancy_kwargs=None,
                             is_track_interior=None,
                             is_track_boundary=None,
                             edges=None):
    '''

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    multiunits : ndarray, shape (n_time, n_marks, n_electrodes)
    place_bin_centers : ndarray, shape ( n_bins, n_position_dims)
    model : sklearn model
    model_kwargs : dict
    occupancy_model : sklearn model
    occupancy_kwargs : dict
    is_track_interior : None or ndarray, shape (n_bins,)

    Returns
    -------
    joint_pdf_models : list of sklearn models, shape (n_electrodes,)
    summed_ground_process_intensity : (1, n_bins)
    occupancy : ndarray, (n_bins, n_position_dims)
    mean_rates : ndarray, (n_electrodes,)

    '''

    if is_track_interior is None:
        is_track_interior = np.ones((place_bin_centers.shape[0],),
                                    dtype=np.bool)
    else:
        is_track_interior = is_track_interior.ravel(order='F')

    if occupancy_model is None:
        occupancy_model = model
    if occupancy_kwargs is None:
        occupancy_kwargs = model_kwargs
    occupancy, _ = fit_occupancy(position, place_bin_centers, occupancy_model,
                                 occupancy_kwargs, is_track_interior)
    mean_rates = []
    ground_process_intensities = []
    joint_pdf_models = []

    for multiunit in np.moveaxis(multiunits, -1, 0):
        mean_rates.append(estimate_mean_rate(multiunit))
        ground_process_intensities.append(
            estimate_ground_process_intensity(
                multiunit, position, place_bin_centers, occupancy,
                mean_rates[-1], model, model_kwargs, is_track_interior))
        joint_pdf_models.append(
            train_joint_model(multiunit, position, model, model_kwargs))

    summed_ground_process_intensity = np.sum(
        np.concatenate(ground_process_intensities, axis=0), axis=0,
        keepdims=True)

    return {
        'joint_pdf_models': joint_pdf_models,
        'summed_ground_process_intensity': summed_ground_process_intensity,
        'occupancy': occupancy,
        'mean_rates': mean_rates,
    }


def estimate_multiunit_likelihood(multiunits, place_bin_centers,
                                  joint_pdf_models,
                                  summed_ground_process_intensity, occupancy,
                                  mean_rates, is_track_interior=None,
                                  time_bin_size=1):
    '''

    Parameters
    ----------
    multiunits : ndarray, shape (n_time, n_marks, n_electrodes)
    place_bin_centers : ndarray, (n_bins, n_position_dims)
    joint_pdf_models : list of sklearn models, shape (n_electrodes,)
    summed_ground_process_intensity : (1, n_bins)
    occupancy : ndarray, (n_bins, n_position_dims)
    mean_rates : ndarray, (n_electrodes,)

    Returns
    -------
    log_likelihood : (n_time, n_bins)

    '''
    if is_track_interior is None:
        is_track_interior = np.ones((place_bin_centers.shape[0],),
                                    dtype=np.bool)
    else:
        is_track_interior = is_track_interior.ravel(order='F')

    n_time = multiunits.shape[0]
    log_likelihood = (-time_bin_size * summed_ground_process_intensity *
                      np.ones((n_time, 1)))

    for multiunit, joint_model, mean_rate in zip(
            np.moveaxis(multiunits, -1, 0), joint_pdf_models, mean_rates):
        is_spike = np.any(~np.isnan(multiunit), axis=1)
        log_joint_mark_intensity = estimate_log_joint_mark_intensity(
            multiunit[is_spike],
            place_bin_centers[is_track_interior],
            occupancy[is_track_interior],
            joint_model,
            mean_rate)
        log_likelihood[np.ix_(is_spike, is_track_interior)] += np.nan_to_num(
            log_joint_mark_intensity)

    log_likelihood[:, ~is_track_interior] = np.nan

    return log_likelihood


def get_marks_by_place_bin_centers(marks, place_bin_centers):
    """

    Parameters
    ----------
    marks : ndarray, shape (n_spikes, n_features)
    place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)

    Returns
    -------
    marks_by_place_bin_centers : ndarray, shape (n_spikes * n_position_bins,
                                                 n_features + n_position_dims)

    """
    n_spikes = marks.shape[0]
    n_place_bin_centers = place_bin_centers.shape[0]
    return np.concatenate(
        (np.tile(marks, reps=(n_place_bin_centers, 1)),
         np.repeat(place_bin_centers, n_spikes, axis=0)), axis=1)
