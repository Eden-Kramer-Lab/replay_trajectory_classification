from copy import deepcopy
from functools import partial
from logging import getLogger

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator

import joblib

from .core import (_acausal_classify, _causal_classify, atleast_2d,
                   get_centers, get_grid, get_track_interior)
from .initial_conditions import uniform, uniform_on_track
from .misc import WhitenedKDE
from .multiunit_likelihood import (estimate_multiunit_likelihood,
                                   fit_multiunit_likelihood)
from .spiking_likelihood import (estimate_place_fields,
                                 estimate_spiking_likelihood)
from .state_transition import (empirical_minus_identity, empirical_movement,
                               identity, identity_discrete,
                               inverse_random_walk, random_walk, random_walk2,
                               random_walk_minus_identity,
                               strong_diagonal_discrete, uniform_discrete,
                               uniform_minus_empirical,
                               uniform_minus_random_walk,
                               uniform_state_transition)

logger = getLogger(__name__)

_DEFAULT_MULTIUNIT_MODEL_KWARGS = dict(bandwidth=0.75, kernel='epanechnikov',
                                       rtol=1E-4)
_DEFAULT_CONTINUOUS_TRANSITIONS = (
    [['random_walk', 'uniform', 'identity'],
     ['uniform',   'uniform', 'uniform'],
     ['random_walk', 'uniform', 'identity']])
_DISCRETE_DIAG = 1 - 1E-3


class _ClassifierBase(BaseEstimator):
    def __init__(self, place_bin_size=2.0, replay_speed=40, movement_var=0.05,
                 position_range=None,
                 continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
                 discrete_transition_type='strong_diagonal',
                 initial_conditions_type='uniform_on_track',
                 discrete_transition_diag=_DISCRETE_DIAG,
                 infer_track_interior=True):
        self.place_bin_size = place_bin_size
        self.replay_speed = replay_speed
        self.movement_var = movement_var
        self.position_range = position_range
        self.continuous_transition_types = continuous_transition_types
        self.discrete_transition_type = discrete_transition_type
        self.initial_conditions_type = initial_conditions_type
        self.discrete_transition_diag = discrete_transition_diag
        self.infer_track_interior = infer_track_interior

    def fit_place_grid(self, position):
        (self.edges_, self.place_bin_edges_, self.place_bin_centers_,
         self.centers_shape_) = get_grid(
            position, self.place_bin_size, self.position_range,
            self.infer_track_interior)

    def fit_initial_conditions(self, position=None, is_track_interior=None):
        logger.info('Fitting initial conditions...')
        if is_track_interior is None and self.infer_track_interior:
            self.is_track_interior_ = get_track_interior(position, self.edges_)
        elif is_track_interior is None and not self.infer_track_interior:
            self.is_track_interior_ = np.ones(
                self.centers_shape_, dtype=np.bool)
        initial_conditions = {
            'uniform':  partial(
                uniform, self.place_bin_centers_),
            'uniform_on_track': partial(
                uniform_on_track, self.place_bin_centers_,
                self.is_track_interior_)
        }
        n_states = len(self.continuous_transition_types)
        initial_conditions = initial_conditions[self.initial_conditions_type]()
        self.initial_conditions_ = (
            np.stack([initial_conditions] * n_states, axis=0)[..., np.newaxis]
            / n_states)

    def fit_continuous_state_transition(
            self, position, is_training=None, replay_speed=None,
            is_track_interior=None,
            continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
            infer_track_interior=True):
        logger.info('Fitting state transition...')
        if is_training is None:
            is_training = np.ones((position.shape[0],), dtype=np.bool)
        is_training = np.asarray(is_training).squeeze()
        if replay_speed is not None:
            self.replay_speed = replay_speed
        self.continuous_transition_types = continuous_transition_types
        self.infer_track_interior = infer_track_interior
        if is_track_interior is None and self.infer_track_interior:
            self.is_track_interior_ = get_track_interior(position, self.edges_)
        elif is_track_interior is None and not self.infer_track_interior:
            self.is_track_interior_ = np.ones(
                self.centers_shape_, dtype=np.bool)

        transitions = {
            'empirical_movement': partial(
                empirical_movement, position, self.edges_, is_training,
                self.replay_speed),
            'random_walk': partial(
                random_walk,
                self.place_bin_centers_, self.movement_var,
                self.is_track_interior_, self.replay_speed),
            'random_walk2': partial(
                random_walk2,
                self.place_bin_centers_, self.movement_var,
                self.is_track_interior_, self.replay_speed),
            'uniform': partial(
                uniform_state_transition, self.place_bin_centers_,
                self.is_track_interior_),
            'identity': partial(
                identity, self.place_bin_centers_, self.is_track_interior_),
            'uniform_minus_empirical': partial(
                uniform_minus_empirical, self.place_bin_centers_,
                self.is_track_interior_, position, self.edges_, is_training,
                self.replay_speed, self.position_range
            ),
            'uniform_minus_random_walk': partial(
                uniform_minus_random_walk, self.place_bin_centers_,
                self.movement_var, self.is_track_interior_, self.replay_speed
            ),
            'empirical_minus_identity': partial(
                empirical_minus_identity, self.place_bin_centers_,
                self.is_track_interior_, position, self.edges_, is_training,
                self.replay_speed, self.position_range
            ),
            'random_walk_minus_identity': partial(
                random_walk_minus_identity, self.place_bin_centers_,
                self.movement_var, self.is_track_interior_, self.replay_speed
            ),
            'inverse_random_walk': partial(
                inverse_random_walk, self.place_bin_centers_,
                self.movement_var, self.is_track_interior_, self.replay_speed
            )
        }
        n_bins = self.place_bin_centers_.shape[0]
        n_states = len(self.continuous_transition_types)
        self.continuous_state_transition_ = np.zeros(
            (n_states, n_states, n_bins, n_bins))
        for row_ind, row in enumerate(self.continuous_transition_types):
            for column_ind, transition_type in enumerate(row):
                self.continuous_state_transition_[row_ind, column_ind] = (
                    transitions[transition_type]()
                )

    def fit_discrete_state_transition(self, discrete_transition_diag=None):
        if discrete_transition_diag is not None:
            self.discrete_transition_diag = discrete_transition_diag

        n_states = len(self.continuous_transition_types)
        transitions = {
            'strong_diagonal': partial(
                strong_diagonal_discrete, n_states,
                self.discrete_transition_diag),
            'identity': partial(
                identity_discrete, n_states),
            'uniform': partial(
                uniform_discrete, n_states),
        }

        self.discrete_state_transition_ = transitions[
            self.discrete_transition_type]()

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_model(self, filename='model.pkl'):
        joblib.dump(self, filename)

    @staticmethod
    def load_model(filename='model.pkl'):
        return joblib.load(filename)

    @staticmethod
    def predict_proba(results):
        return results.sum(['x_position', 'y_position'])

    def copy(self):
        return deepcopy(self)


class SortedSpikesClassifier(_ClassifierBase):
    '''

    Attributes
    ----------
    place_bin_size : float, optional
        Approximate size of the position bins.
    replay_speed : int, optional
        How many times faster the replay movement is than normal movement.
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    position_range : sequence, optional
        A sequence of `n_position_dims`, each an optional (lower, upper)
        tuple giving the outer bin edges for position.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of
        `n_position_dims` None values.
    continuous_transition_types : list of ('empirical_movement',
                                           'random_walk', 'uniform',
                                           'identity',
                                           'uniform_minus_empirical',
                                           'uniform_minus_random_walk',
                                           'empirical_minus_identity'
                                           'random_walk_minus_identity')
    discrete_transition_type : 'strong_diagonal' | 'identity' | 'uniform'
    initial_conditions_type : ('uniform' | 'uniform_on_track')
    discrete_transition_diag : float, optional
    infer_track_interior : bool, optional
    knot_spacing : float, optional
    spike_model_penalty : float, optional

    '''

    def __init__(self, place_bin_size=2.0, replay_speed=40, movement_var=0.05,
                 position_range=None,
                 continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
                 discrete_transition_type='strong_diagonal',
                 initial_conditions_type='uniform_on_track', knot_spacing=10,
                 spike_model_penalty=1E1,
                 discrete_transition_diag=_DISCRETE_DIAG,
                 infer_track_interior=True):
        super().__init__(place_bin_size, replay_speed, movement_var,
                         position_range, continuous_transition_types,
                         discrete_transition_type, initial_conditions_type,
                         discrete_transition_diag, infer_track_interior)
        self.knot_spacing = knot_spacing
        self.spike_model_penalty = spike_model_penalty

    def fit_place_fields(self, position, spikes, is_training=None):
        logger.info('Fitting place fields...')
        if is_training is None:
            is_training = np.ones((position.shape[0],), dtype=np.bool)
        is_training = np.asarray(is_training).squeeze()
        self.place_fields_ = estimate_place_fields(
            position[is_training], spikes[is_training],
            self.place_bin_centers_, penalty=self.spike_model_penalty,
            knot_spacing=self.knot_spacing)

    def plot_place_fields(self, spikes=None, position=None,
                          sampling_frequency=1):
        '''Plots the fitted 2D place fields for each neuron.

        Parameters
        ----------
        spikes : None or array_like, shape (n_time, n_neurons), optional
        position : None or array_like, shape (n_time, 2), optional
        sampling_frequency : float, optional

        Returns
        -------
        g : xr.plot.FacetGrid instance

        '''
        g = (self.place_fields_.unstack() * sampling_frequency).plot(
            x='x_position', y='y_position', col='neuron', col_wrap=5, vmin=0.0,
            vmax=10.0)
        if spikes is not None and position is not None:
            spikes, position = np.asarray(spikes), np.asarray(position)
            for ax, is_spike in zip(g.axes.flat, spikes.T):
                is_spike = is_spike > 0
                ax.plot(position[:, 0], position[:, 1], color='lightgrey',
                        alpha=0.3, zorder=1)
                ax.scatter(position[is_spike, 0], position[is_spike, 1],
                           color='red', s=1, alpha=0.5, zorder=1)
        elif position is not None:
            position = np.asarray(position)
            for ax in g.axes.flat:
                ax.plot(position[:, 0], position[:, 1], color='lightgrey',
                        alpha=0.3, zorder=1)

        return g

    def fit(self, position, spikes, is_training=None, is_track_interior=None):
        '''

        Parameters
        ----------
        position : ndarray, shape (n_time, n_position_dims)
        spikes : ndarray, shape (n_time, n_neurons)
        is_training : None or bool ndarray, shape (n_time), optional
            Time bins to be used for encoding.
        is_track_interior : None or bool ndaarray, shape (n_x_bins, n_y_bins)
        '''
        position = atleast_2d(np.asarray(position))
        spikes = np.asarray(spikes)
        self.fit_place_grid(position)
        self.fit_initial_conditions(position, is_track_interior)
        self.fit_continuous_state_transition(
            position, is_training, is_track_interior=is_track_interior,
            continuous_transition_types=self.continuous_transition_types,
            infer_track_interior=self.infer_track_interior)
        self.fit_discrete_state_transition()
        self.fit_place_fields(position, spikes, is_training)

        return self

    def predict(self, spikes, time=None, is_compute_acausal=True):
        '''

        Parameters
        ----------
        spikes : ndarray, shape (n_time, n_neurons)
        time : ndarray or None, shape (n_time,), optional
        is_compute_acausal : bool, optional

        Returns
        -------
        results : xarray.Dataset

        '''
        n_states = self.continuous_state_transition_.shape[0]
        spikes = np.asarray(spikes)

        results = {}
        results['likelihood'] = estimate_spiking_likelihood(
            spikes, np.asarray(self.place_fields_), self.is_track_interior_)
        results['likelihood'] = np.stack([results['likelihood']] *
                                         n_states, axis=1)[..., np.newaxis]

        results['causal_posterior'] = _causal_classify(
            self.initial_conditions_, self.continuous_state_transition_,
            self.discrete_state_transition_, results['likelihood'])

        if is_compute_acausal:
            results['acausal_posterior'] = _acausal_classify(
                results['causal_posterior'], self.continuous_state_transition_,
                self.discrete_state_transition_)

        n_time = spikes.shape[0]

        if time is None:
            time = np.arange(n_time)

        n_position_dims = self.place_bin_centers_.shape[1]
        if n_position_dims > 1:
            new_shape = (n_time, n_states, *self.centers_shape_)
            dims = ['time', 'state', 'x_position', 'y_position']
            coords = dict(
                time=time,
                x_position=get_centers(self.edges_[0]),
                y_position=get_centers(self.edges_[1]),
                state=np.diag(np.asarray(self.continuous_transition_types)),
            )
            results = xr.Dataset(
                {key: (dims, (value.squeeze(axis=-1)
                              .reshape(new_shape).swapaxes(-1, -2)))
                 for key, value in results.items()},
                coords=coords)
        else:
            dims = ['time', 'state', 'position']
            coords = dict(
                time=time,
                position=get_centers(self.edges_[0]),
                state=np.diag(np.asarray(self.continuous_transition_types)),
            )
            results = xr.Dataset(
                {key: (dims, (value.squeeze(axis=-1)))
                 for key, value in results.items()},
                coords=coords)

        return results


class ClusterlessClassifier(_ClassifierBase):
    '''

    Attributes
    ----------
    place_bin_size : float, optional
        Approximate size of the position bins.
    replay_speed : int, optional
        How many times faster the replay movement is than normal movement.
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    position_range : sequence, optional
        A sequence of `n_position_dims`, each an optional (lower, upper)
        tuple giving the outer bin edges for position.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of
        `n_position_dims` None values.
    continuous_transition_types : list of ('empirical_movement',
                                           'random_walk', 'uniform',
                                           'identity',
                                           'uniform_minus_empirical',
                                           'uniform_minus_random_walk',
                                           'empirical_minus_identity'
                                           'random_walk_minus_identity')
    discrete_transition_type : 'strong_diagonal' | 'identity' | 'uniform'
    initial_conditions_type : ('uniform' | 'uniform_on_track')
    discrete_transition_diag : float, optional
    infer_track_interior : bool, optional
    model : scikit-learn density estimator, optional
    model_kwargs : dict, optional
    occupancy_model : scikit-learn density estimator, optional
    occupancy_kwargs : dict, optional

    '''

    def __init__(self, place_bin_size=2.0, replay_speed=40, movement_var=0.05,
                 position_range=None,
                 continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
                 discrete_transition_type='strong_diagonal',
                 initial_conditions_type='uniform_on_track',
                 discrete_transition_diag=_DISCRETE_DIAG,
                 infer_track_interior=True,
                 model=WhitenedKDE,
                 model_kwargs=_DEFAULT_MULTIUNIT_MODEL_KWARGS,
                 occupancy_model=None, occupancy_kwargs=None):
        super().__init__(place_bin_size, replay_speed, movement_var,
                         position_range, continuous_transition_types,
                         discrete_transition_type, initial_conditions_type,
                         discrete_transition_diag, infer_track_interior)

        self.model = model
        self.model_kwargs = model_kwargs

        if occupancy_model is None:
            self.occupancy_model = model
            self.occupancy_kwargs = model_kwargs
        else:
            self.occupancy_model = occupancy_model
            self.occupancy_kwargs = occupancy_kwargs

    def fit_multiunits(self, position, multiunits, is_training=None,
                       is_track_interior=None):
        '''

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)
        is_track_interior : None or ndarray, shape (n_x_bins, n_y_bins)

        '''
        logger.info('Fitting multiunits...')
        if is_training is None:
            is_training = np.ones((position.shape[0],), dtype=np.bool)
        is_training = np.asarray(is_training).squeeze()
        if is_track_interior is None:
            self.is_track_interior_ = get_track_interior(position, self.edges_)

        (self.joint_pdf_models_, self.ground_process_intensities_,
         self.occupancy_, self.mean_rates_) = fit_multiunit_likelihood(
            position[is_training], multiunits[is_training],
            self.place_bin_centers_, self.model, self.model_kwargs,
            self.occupancy_model, self.occupancy_kwargs,
            self.is_track_interior_.ravel(order='F'))

    def fit(self, position, multiunits, is_training=None,
            is_track_interior=None):
        '''

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)
        is_track_interior : None or ndarray, shape (n_x_bins, n_y_bins)

        Returns
        -------
        self

        '''
        position = atleast_2d(np.asarray(position))
        multiunits = np.asarray(multiunits)

        self.fit_place_grid(position)
        self.fit_initial_conditions(position, is_track_interior)
        self.fit_continuous_state_transition(
            position, is_training, is_track_interior=is_track_interior,
            continuous_transition_types=self.continuous_transition_types,
            infer_track_interior=self.infer_track_interior)
        self.fit_discrete_state_transition()
        self.fit_multiunits(position, multiunits, is_training,
                            is_track_interior)

        return self

    def predict(self, multiunits, time=None, is_compute_acausal=True):
        '''

        Parameters
        ----------
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        time : None or ndarray, shape (n_time,)
        is_compute_acausal : bool, optional
            Use future information to compute the posterior.

        Returns
        -------
        results : xarray.Dataset

        '''
        n_states = self.continuous_state_transition_.shape[0]
        multiunits = np.asarray(multiunits)

        results = {}
        results['likelihood'] = estimate_multiunit_likelihood(
            multiunits, self.place_bin_centers_,
            self.joint_pdf_models_, self.ground_process_intensities_,
            self.occupancy_, self.mean_rates_,
            self.is_track_interior_.ravel(order='F'))

        results['likelihood'] = np.stack([results['likelihood']] *
                                         n_states, axis=1)[..., np.newaxis]

        results['causal_posterior'] = _causal_classify(
            self.initial_conditions_, self.continuous_state_transition_,
            self.discrete_state_transition_, results['likelihood'])

        if is_compute_acausal:
            results['acausal_posterior'] = _acausal_classify(
                results['causal_posterior'], self.continuous_state_transition_,
                self.discrete_state_transition_)

        n_time = multiunits.shape[0]

        if time is None:
            time = np.arange(n_time)

        n_position_dims = self.place_bin_centers_.shape[1]
        if n_position_dims > 1:
            new_shape = (n_time, n_states, *self.centers_shape_)
            dims = ['time', 'state', 'x_position', 'y_position']
            coords = dict(
                time=time,
                x_position=get_centers(self.edges_[0]),
                y_position=get_centers(self.edges_[1]),
                state=np.diag(np.asarray(self.continuous_transition_types)),
            )
            results = xr.Dataset(
                {key: (dims, (value.squeeze(axis=-1)
                              .reshape(new_shape).swapaxes(-1, -2)))
                 for key, value in results.items()},
                coords=coords)
        else:
            dims = ['time', 'state', 'position']
            coords = dict(
                time=time,
                position=get_centers(self.edges_[0]),
                state=np.diag(np.asarray(self.continuous_transition_types)),
            )
            results = xr.Dataset(
                {key: (dims, (value.squeeze(axis=-1)))
                 for key, value in results.items()},
                coords=coords)

        return results
