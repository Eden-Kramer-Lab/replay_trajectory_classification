from copy import deepcopy
from logging import getLogger

import joblib
import numpy as np
import sklearn
import xarray as xr
from replay_trajectory_classification.bins import atleast_2d, get_centers
from replay_trajectory_classification.continuous_state_transitions import (
    EmpiricalMovement, RandomWalk, Uniform)
from replay_trajectory_classification.core import (_acausal_classify,
                                                   _acausal_classify_gpu,
                                                   _causal_classify,
                                                   _causal_classify_gpu,
                                                   _ClUSTERLESS_ALGORITHMS,
                                                   mask, scaled_likelihood)
from replay_trajectory_classification.discrete_state_transitions import \
    DiagonalDiscrete
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.initial_conditions import \
    UniformInitialConditions
from replay_trajectory_classification.misc import NumbaKDE
from replay_trajectory_classification.spiking_likelihood import (
    estimate_place_fields, estimate_spiking_likelihood)
from sklearn.base import BaseEstimator

logger = getLogger(__name__)

sklearn.set_config(print_changed_only=False)

_DEFAULT_CLUSTERLESS_MODEL_KWARGS = {
    'model': NumbaKDE,
    'model_kwargs': {
        'bandwidth': np.array([24.0, 24.0, 24.0, 24.0, 6.0, 6.0])
    }
}

_DEFAULT_CONTINUOUS_TRANSITIONS = (
    [[RandomWalk(), Uniform()],
     [Uniform(),    Uniform()]])

_DEFAULT_ENVIRONMENT = Environment(environment_name='')


class _ClassifierBase(BaseEstimator):
    def __init__(self,
                 environments=_DEFAULT_ENVIRONMENT,
                 continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
                 discrete_transition_type=DiagonalDiscrete(0.968),
                 initial_conditions_type=UniformInitialConditions(),
                 infer_track_interior=True):
        if isinstance(environments, Environment):
            environments = (environments,)
        self.environments = environments
        self.continuous_transition_types = continuous_transition_types
        self.discrete_transition_type = discrete_transition_type
        self.initial_conditions_type = initial_conditions_type
        self.infer_track_interior = infer_track_interior

    def fit_environments(self, position, environment_labels=None):
        for environment in self.environments:
            if environment_labels is None:
                is_environment = np.ones((position.shape[0],), dtype=np.bool)
            else:
                is_environment = (environment_labels ==
                                  environment.environment_name)
            environment.fit_place_grid(
                position[is_environment],
                infer_track_interior=self.infer_track_interior)

        self.max_pos_bins_ = np.max([env.place_bin_centers_.shape[0]
                                     for env in self.environments])

    def fit_initial_conditions(self, environment_names_to_state=None):
        logger.info('Fitting initial conditions...')
        n_states = len(self.continuous_transition_types)
        if environment_names_to_state is None:
            environment_names_to_state = [
                self.environments[0].environment_name] * n_states
        initial_conditions = (
            self.initial_conditions_type.make_initial_conditions(
                self.environments, environment_names_to_state))

        self.initial_conditions_ = np.zeros((n_states, self.max_pos_bins_, 1),
                                            dtype=np.float64)
        for state_ind, ic in enumerate(initial_conditions):
            self.initial_conditions_[state_ind,
                                     :ic.shape[0]] = ic[..., np.newaxis]

    def fit_continuous_state_transition(
            self,
            continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
            position=None,
            is_training=None,
            encoding_group_labels=None,
            environment_labels=None,
    ):
        logger.info('Fitting state transition...')

        if is_training is None:
            n_time = position.shape[0]
            is_training = np.ones((n_time,), dtype=np.bool)

        if encoding_group_labels is None:
            n_time = position.shape[0]
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        is_training = np.asarray(is_training).squeeze()

        self.continuous_transition_types = continuous_transition_types
        continuous_state_transition = []

        for row in self.continuous_transition_types:
            continuous_state_transition.append([])
            for transition in row:
                if isinstance(transition, EmpiricalMovement):
                    continuous_state_transition[-1].append(
                        transition.make_state_transition(
                            self.environments, position, is_training,
                            encoding_group_labels, environment_labels))
                else:
                    continuous_state_transition[-1].append(
                        transition.make_state_transition(self.environments))

        n_states = len(self.continuous_transition_types)
        self.continuous_state_transition_ = np.zeros(
            (n_states, n_states, self.max_pos_bins_, self.max_pos_bins_))

        for row_ind, row in enumerate(continuous_state_transition):
            for column_ind, st in enumerate(row):
                self.continuous_state_transition_[
                    row_ind, column_ind, :st.shape[0], :st.shape[1]] = st

    def fit_discrete_state_transition(self):
        n_states = len(self.continuous_transition_types)
        self.discrete_state_transition_ = (
            self.discrete_transition_type.make_state_transition(n_states))

    def _return_results(self, likelihood, n_time, time, state_names, use_gpu,
                        is_compute_acausal):
        n_states = self.discrete_state_transition_.shape[0]
        states = tuple(zip(self.environment_names_to_state_,
                           self.encoding_group_to_state_))

        results = {}
        results['likelihood'] = np.full(
            (n_time, n_states, self.max_pos_bins_, 1), np.nan)
        for state_ind, state in enumerate(states):
            n_bins = likelihood[state].shape[1]
            results['likelihood'][:, state_ind,
                                  :n_bins] = likelihood[state][..., np.newaxis]
        results['likelihood'] = scaled_likelihood(
            results['likelihood'], axis=(1, 2))
        results['likelihood'][np.isnan(results['likelihood'])] = 0.0

        n_environments = len(self.environments)
        if n_environments == 1:
            logger.info('Estimating causal posterior...')
            is_track_interior = (self.environments[0]
                                 .is_track_interior_.ravel(order='F'))
            n_position_bins = len(is_track_interior)
            is_states = np.ones((n_states,), dtype=bool)
            st_interior_ind = np.ix_(
                is_states, is_states, is_track_interior, is_track_interior)
            if not use_gpu:
                results['causal_posterior'] = np.full(
                    (n_time, n_states, n_position_bins, 1), np.nan,
                    dtype=np.float64)
                results['causal_posterior'][:, :, is_track_interior] = (
                    _causal_classify(
                        self.initial_conditions_[:, is_track_interior],
                        self.continuous_state_transition_[st_interior_ind],
                        self.discrete_state_transition_,
                        results['likelihood'][:, :, is_track_interior]))
            else:
                results['causal_posterior'] = np.full(
                    (n_time, n_states, n_position_bins, 1), np.nan,
                    dtype=np.float32)
                results['causal_posterior'][:, :, is_track_interior] = (
                    _causal_classify_gpu(
                        self.initial_conditions_[:, is_track_interior],
                        self.continuous_state_transition_[st_interior_ind],
                        self.discrete_state_transition_,
                        results['likelihood'][:, :, is_track_interior]))

            if is_compute_acausal:
                logger.info('Estimating acausal posterior...')
                if not use_gpu:
                    results['acausal_posterior'] = np.full(
                        (n_time, n_states, n_position_bins, 1), np.nan,
                        dtype=np.float64)
                    results['acausal_posterior'][:, :, is_track_interior] = (
                        _acausal_classify(
                            results['causal_posterior'][:,
                                                        :, is_track_interior],
                            self.continuous_state_transition_[st_interior_ind],
                            self.discrete_state_transition_))
                else:
                    results['acausal_posterior'] = np.full(
                        (n_time, n_states, n_position_bins, 1), np.nan,
                        dtype=np.float32)
                    results['acausal_posterior'][:, :, is_track_interior] = (
                        _acausal_classify_gpu(
                            results['causal_posterior'][:,
                                                        :, is_track_interior],
                            self.continuous_state_transition_[st_interior_ind],
                            self.discrete_state_transition_))

            if time is None:
                time = np.arange(n_time)

            return self._convert_results_to_xarray(results, time, state_names)

        else:
            logger.info('Estimating causal posterior...')
            if not use_gpu:
                results['causal_posterior'] = _causal_classify(
                    self.initial_conditions_,
                    self.continuous_state_transition_,
                    self.discrete_state_transition_,
                    results['likelihood'])
            else:
                results['causal_posterior'] = _causal_classify_gpu(
                    self.initial_conditions_,
                    self.continuous_state_transition_,
                    self.discrete_state_transition_,
                    results['likelihood'])

            if is_compute_acausal:
                logger.info('Estimating acausal posterior...')
                if not use_gpu:
                    results['acausal_posterior'] = _acausal_classify(
                        results['causal_posterior'],
                        self.continuous_state_transition_,
                        self.discrete_state_transition_)
                else:
                    results['acausal_posterior'] = _acausal_classify_gpu(
                        results['causal_posterior'],
                        self.continuous_state_transition_,
                        self.discrete_state_transition_)

            if time is None:
                time = np.arange(n_time)

            return self._convert_results_to_xarray_mutienvironment(
                results, time, state_names)

    def _convert_results_to_xarray(self, results, time, state_names=None):
        n_position_dims = self.environments[0].place_bin_centers_.shape[1]
        diag_transition_names = np.diag(
            np.asarray(self.continuous_transition_types))
        if state_names is None:
            if len(np.unique(self.encoding_group_to_state_)) == 1:
                state_names = diag_transition_names
            else:
                state_names = [
                    f"{state}-{transition}" for state, transition
                    in zip(self.encoding_group_to_state_,
                           diag_transition_names)]
        n_time = time.shape[0]
        n_states = len(state_names)
        is_track_interior = (
            self.environments[0].is_track_interior_.ravel(order='F'))
        edges = self.environments[0].edges_

        if n_position_dims > 1:
            centers_shape = self.environments[0].centers_shape_
            new_shape = (n_time, n_states, *centers_shape)
            dims = ['time', 'state', 'x_position', 'y_position']
            coords = dict(
                time=time,
                x_position=get_centers(edges[0]),
                y_position=get_centers(edges[1]),
                state=state_names,
            )
            results = xr.Dataset(
                {key: (dims,
                       (mask(value, is_track_interior).squeeze(axis=-1)
                        .reshape(new_shape).swapaxes(-1, -2)))
                 for key, value in results.items()},
                coords=coords)
        else:
            dims = ['time', 'state', 'position']
            coords = dict(
                time=time,
                position=get_centers(edges[0]),
                state=state_names,
            )
            results = xr.Dataset(
                {key: (dims, (mask(value, is_track_interior).squeeze(axis=-1)))
                 for key, value in results.items()},
                coords=coords)

        return results

    def _convert_results_to_xarray_mutienvironment(self, results, time,
                                                   state_names=None):
        if state_names is None:
            states = tuple(zip(self.environment_names_to_state_,
                               self.encoding_group_to_state_))
            state_names = [f'{env}-{grp}' for env, grp in states]

        results = {key: val.squeeze(axis=-1) for key, val in results.items()}

        return results, time, state_names

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
        try:
            return results.sum(['x_position', 'y_position'])
        except ValueError:
            return results.sum(['position'])

    def copy(self):
        return deepcopy(self)


class SortedSpikesClassifier(_ClassifierBase):
    '''

    Attributes
    ----------
    place_bin_size : float, optional
        Approximate size of the position bins.
    replay_speed : int, optional
        How many times faster the replay movement is than normal movement. It​
        is only used with the empirical transition matrix---a transition matrix
        trained on the animal's actual movement. It can be used to make the
        empirical transition matrix "faster", means allowing for all the same
        transitions made by the animal but sped up by replay_speed​ times.
        So replay_speed​=20 means 20x faster than the animal's movement.
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
                                           'random_walk',
                                           'uniform',
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

    def __init__(self,
                 environments=_DEFAULT_ENVIRONMENT,
                 continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
                 discrete_transition_type=DiagonalDiscrete(0.98),
                 initial_conditions_type=UniformInitialConditions(),
                 infer_track_interior=True,
                 knot_spacing=10,
                 spike_model_penalty=1E1):
        super().__init__(environments,
                         continuous_transition_types,
                         discrete_transition_type,
                         initial_conditions_type,
                         infer_track_interior)
        self.knot_spacing = knot_spacing
        self.spike_model_penalty = spike_model_penalty

    def fit_place_fields(self,
                         position,
                         spikes,
                         is_training=None,
                         encoding_group_labels=None,
                         environment_labels=None,
                         encoding_group_to_state=None,
                         environment_names_to_state=None):
        logger.info('Fitting place fields...')
        n_states = len(self.continuous_transition_types)
        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=np.bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if encoding_group_to_state is None:
            self.encoding_group_to_state_ = np.zeros(
                (n_states,), dtype=np.int32)
        else:
            self.encoding_group_to_state_ = np.asarray(encoding_group_to_state)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time)

        if environment_names_to_state is None:
            self.environment_names_to_state_ = [
                self.environments[0].environment_name] * n_states
        else:
            self.environment_names_to_state_ = environment_names_to_state

        is_training = np.asarray(is_training).squeeze()

        states = tuple(zip(self.environment_names_to_state_,
                           self.encoding_group_to_state_))
        self.place_fields_ = {}
        for environment_name, encoding_group in set(states):
            environment = self.environments[
                self.environments.index(environment_name)]

            is_encoding = (encoding_group_labels == encoding_group)
            is_environment = (environment_labels == environment_name)
            likelihood_name = (environment_name, encoding_group)

            self.place_fields_[likelihood_name] = estimate_place_fields(
                position=position[is_training & is_encoding & is_environment],
                spikes=spikes[is_training & is_encoding & is_environment],
                place_bin_centers=environment.place_bin_centers_,
                place_bin_edges=environment.place_bin_edges_,
                penalty=self.spike_model_penalty,
                knot_spacing=self.knot_spacing)

    def plot_place_fields(self, sampling_frequency=1, col_wrap=5):
        '''Plots the fitted 2D place fields for each neuron.

        Parameters
        ----------
        sampling_frequency : float, optional
        col_wrap : int, optional

        Returns
        -------
        g : xr.plot.FacetGrid instance

        '''
        try:
            g = (self.place_fields_.unstack('position') * sampling_frequency
                 ).plot(x='x_position', y='y_position', col='neuron',
                        hue='encoding_group', col_wrap=col_wrap)
        except ValueError:
            g = (self.place_fields_ * sampling_frequency).plot(
                x='position', col='neuron', hue='encoding_group',
                col_wrap=col_wrap)

        return g

    def fit(self,
            position,
            spikes,
            is_training=None,
            encoding_group_labels=None,
            encoding_group_to_state=None,
            environment_labels=None,
            environment_names_to_state=None,
            ):
        '''

        Parameters
        ----------
        position : ndarray, shape (n_time, n_position_dims)
        spikes : ndarray, shape (n_time, n_neurons)
        is_training : None or bool ndarray, shape (n_time), optional
            Time bins to be used for encoding.
        is_track_interior : None or bool ndaarray, shape (n_x_bins, n_y_bins)
        encoding_group_labels : None or ndarray, shape (n_time,)
        encoding_group_to_state : None or ndarray, shape (n_states,)
        track_graph : networkx.Graph
        edge_order : array_like
        edge_spacing : None, float or array_like

        Returns
        -------
        self

        '''
        position = atleast_2d(np.asarray(position))
        spikes = np.asarray(spikes)
        self.fit_environments(position, environment_labels)
        self.fit_initial_conditions(environment_names_to_state)
        self.fit_continuous_state_transition(
            self.continuous_transition_types,
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        self.fit_discrete_state_transition()
        self.fit_place_fields(position,
                              spikes,
                              is_training,
                              encoding_group_labels,
                              environment_labels,
                              encoding_group_to_state,
                              environment_names_to_state)

        return self

    def predict(self, spikes, time=None, is_compute_acausal=True,
                use_gpu=False,
                state_names=None):
        '''

        Parameters
        ----------
        spikes : ndarray, shape (n_time, n_neurons)
        time : ndarray or None, shape (n_time,), optional
        is_compute_acausal : bool, optional
        use_gpu : bool, optional
            Use GPU for the state space part of the model, not the likelihood.
        state_names : None or array_like, shape (n_states,)

        Returns
        -------
        results : xarray.Dataset

        '''
        spikes = np.asarray(spikes)
        n_time = spikes.shape[0]

        # likelihood
        logger.info('Estimating likelihood...')
        likelihood = {}
        for (env_name, enc_group), place_fields in self.place_fields_.items():
            env_ind = self.environments.index(env_name)
            is_track_interior = self.environments[env_ind].is_track_interior_
            likelihood[(env_name, enc_group)] = estimate_spiking_likelihood(
                spikes,
                place_fields.values,
                is_track_interior)

        self._return_results(
            likelihood, n_time, time, state_names, use_gpu, is_compute_acausal)


class ClusterlessClassifier(_ClassifierBase):
    '''

    Attributes
    ----------
    place_bin_size : float, optional
        Approximate size of the position bins.
    replay_speed : int, optional
        How many times faster the replay movement is than normal movement. It​
        is only used with the empirical transition matrix---a transition matrix
        trained on the animal's actual movement. It can be used to make the
        empirical transition matrix "faster", means allowing for all the same
        transitions made by the animal but sped up by replay_speed​ times.
        So replay_speed​=20 means 20x faster than the animal's movement.
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
                                           'random_walk',
                                           'uniform',
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

    def __init__(self,
                 environments=_DEFAULT_ENVIRONMENT,
                 continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
                 discrete_transition_type=DiagonalDiscrete(0.98),
                 initial_conditions_type=UniformInitialConditions(),
                 infer_track_interior=True,
                 clusterless_algorithm='multiunit_likelihood',
                 clusterless_algorithm_params=_DEFAULT_CLUSTERLESS_MODEL_KWARGS
                 ):
        super().__init__(environments,
                         continuous_transition_types,
                         discrete_transition_type,
                         initial_conditions_type,
                         infer_track_interior)

        self.clusterless_algorithm = clusterless_algorithm
        self.clusterless_algorithm_params = clusterless_algorithm_params

    def fit_multiunits(self,
                       position,
                       multiunits,
                       is_training=None,
                       encoding_group_labels=None,
                       environment_labels=None,
                       encoding_group_to_state=None,
                       environment_names_to_state=None):
        '''

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)
        is_track_interior : None or ndarray, shape (n_x_bins, n_y_bins)

        '''
        logger.info('Fitting multiunits...')
        n_states = len(self.continuous_transition_types)
        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=np.bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if encoding_group_to_state is None:
            self.encoding_group_to_state_ = np.zeros(
                (n_states,), dtype=np.int32)
        else:
            self.encoding_group_to_state_ = np.asarray(encoding_group_to_state)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time)

        if environment_names_to_state is None:
            self.environment_names_to_state_ = [
                self.environments[0].environment_name] * n_states
        else:
            self.environment_names_to_state_ = environment_names_to_state

        is_training = np.asarray(is_training).squeeze()

        states = tuple(zip(self.environment_names_to_state_,
                           self.encoding_group_to_state_))

        kwargs = self.clusterless_algorithm_params
        if kwargs is None:
            kwargs = {}

        self.encoding_model_ = {}

        states = tuple(zip(self.environment_names_to_state_,
                           self.encoding_group_to_state_))
        for environment_name, encoding_group in set(states):
            environment = self.environments[
                self.environments.index(environment_name)]

            is_encoding = (encoding_group_labels == encoding_group)
            is_environment = (environment_labels == environment_name)
            is_group = is_training & is_encoding & is_environment

            likelihood_name = (environment_name, encoding_group)

            self.encoding_model_[likelihood_name] = _ClUSTERLESS_ALGORITHMS[
                self.clusterless_algorithm][0](
                    position=position[is_group],
                    multiunits=multiunits[is_group],
                    place_bin_centers=environment.place_bin_centers_,
                    is_track_interior=environment.is_track_interior_.ravel(
                        order='F'),
                    **kwargs
            )

    def fit(self,
            position,
            multiunits,
            is_training=None,
            encoding_group_labels=None,
            encoding_group_to_state=None,
            environment_labels=None,
            environment_names_to_state=None,
            ):
        '''

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)
        is_track_interior : None or ndarray, shape (n_x_bins, n_y_bins)
        encoding_group_labels : None or ndarray, shape (n_time,)
        encoding_group_to_state : None or ndarray, shape (n_states,)

        Returns
        -------
        self

        '''
        position = atleast_2d(np.asarray(position))
        multiunits = np.asarray(multiunits)

        self.fit_environments(position, environment_labels)
        self.fit_initial_conditions(environment_names_to_state)
        self.fit_continuous_state_transition(
            self.continuous_transition_types,
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        self.fit_discrete_state_transition()
        self.fit_multiunits(position,
                            multiunits,
                            is_training,
                            encoding_group_labels,
                            environment_labels,
                            encoding_group_to_state,
                            environment_names_to_state)

        return self

    def predict(self, multiunits, time=None, is_compute_acausal=True,
                use_gpu=False, state_names=None):
        '''

        Parameters
        ----------
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        time : None or ndarray, shape (n_time,)
        is_compute_acausal : bool, optional
            Use future information to compute the posterior.
        use_gpu : bool, optional
            Use GPU for the state space part of the model, not the likelihood.
        state_names : None or array_like, shape (n_states,)

        Returns
        -------
        results : xarray.Dataset

        '''
        multiunits = np.asarray(multiunits)
        n_time = multiunits.shape[0]

        logger.info('Estimating likelihood...')
        likelihood = {}
        for (env_name, enc_group), encoding_params in self.encoding_model_.items():
            env_ind = self.environments.index(env_name)
            is_track_interior = self.environments[env_ind].is_track_interior_
            place_bin_centers = self.environments[env_ind].place_bin_centers_
            likelihood[(env_name, enc_group)] = _ClUSTERLESS_ALGORITHMS[
                self.clusterless_algorithm][1](
                    multiunits=multiunits,
                    place_bin_centers=place_bin_centers,
                    is_track_interior=is_track_interior,
                    **encoding_params
            )

        self._return_results(
            likelihood, n_time, time, state_names, use_gpu, is_compute_acausal)
