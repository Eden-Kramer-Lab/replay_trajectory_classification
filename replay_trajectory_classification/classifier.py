from copy import deepcopy
from logging import getLogger

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import xarray as xr
from replay_trajectory_classification.bins import atleast_2d, get_centers
from replay_trajectory_classification.continuous_state_transitions import (
    EmpiricalMovement, RandomWalk, Uniform)
from replay_trajectory_classification.core import (_acausal_classify,
                                                   _acausal_classify_gpu,
                                                   _causal_classify,
                                                   _causal_classify_gpu, mask,
                                                   scaled_likelihood)
from replay_trajectory_classification.discrete_state_transitions import \
    DiagonalDiscrete
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.initial_conditions import \
    UniformInitialConditions
from replay_trajectory_classification.likelihoods import (
    _ClUSTERLESS_ALGORITHMS, estimate_place_fields,
    estimate_spiking_likelihood)
from replay_trajectory_classification.misc import NumbaKDE
from replay_trajectory_classification.observation_model import ObservationModel
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
                 observation_models=None,
                 continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
                 discrete_transition_type=DiagonalDiscrete(0.968),
                 initial_conditions_type=UniformInitialConditions(),
                 infer_track_interior=True):
        if isinstance(environments, Environment):
            environments = (environments,)
        if observation_models is None:
            n_states = len(continuous_transition_types)
            env_name = environments[0].environment_name
            observation_models = (ObservationModel(env_name),) * n_states
        self.environments = environments
        self.observation_models = observation_models
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

    def fit_initial_conditions(self):
        logger.info('Fitting initial conditions...')
        environment_names_to_state = [
            obs.environment_name for obs in self.observation_models]
        n_states = len(self.observation_models)
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

    def _get_results(self, likelihood, n_time, time, state_names, use_gpu,
                     is_compute_acausal):
        n_states = self.discrete_state_transition_.shape[0]
        results = {}
        results['likelihood'] = np.full(
            (n_time, n_states, self.max_pos_bins_, 1), np.nan)
        for state_ind, obs in enumerate(self.observation_models):
            likelihood_name = (obs.environment_name, obs.encoding_group)
            n_bins = likelihood[likelihood_name].shape[1]
            results['likelihood'][:, state_ind, :n_bins] = (
                likelihood[likelihood_name][..., np.newaxis])
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
            if len(np.unique(self.observation_models)) == 1:
                state_names = diag_transition_names
            else:
                state_names = [
                    f"{obs.encoding_group}-{transition}" for obs, transition
                    in zip(self.observation_models,
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
            state_names = [f'{obs.environment_name}-{obs.encoding_group}'
                           for obs in self.observation_models]

        n_position_dims = self.environments[0].place_bin_centers_.shape[1]

        if n_position_dims > 1:
            dims = ['time', 'state', 'position']
            coords = dict(
                time=time,
                state=state_names,
            )
            for env in self.environments:
                coords[env.environment_name +
                       '_x_position'] = get_centers(env.edges_[0])
                coords[env.environment_name +
                       '_y_position'] = get_centers(env.edges_[1])
            results = xr.Dataset(
                {key: (dims, value.squeeze(axis=-1))
                 for key, value in results.items()},
                coords=coords)
        else:
            dims = ['time', 'state', 'position']
            coords = dict(
                time=time,
                state=state_names,
            )
            for env in self.environments:
                coords[env.environment_name +
                       '_position'] = get_centers(env.edges_[0])

            results = xr.Dataset(
                {key: (dims, value.squeeze(axis=-1))
                 for key, value in results.items()},
                coords=coords)

        return results

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
    environment : Environment
        The spatial environment and topology to fit
    continuous_transition_types : tuple of tuples
        The continuous state transition class instances to fit
    discrete_transition_type : Discrete
        The type of transition between states
    initial_conditions_type : InitialConditions
        The initial conditions class instance to fit
    infer_track_interior : bool, optional
        Whether to infer the valid position bins
    knot_spacing : float, optional
        How far apart the spline knots are in position
    spike_model_penalty : float, optional
        L2 penalty (ridge) for the size of the regression coefficients

    '''

    def __init__(self,
                 environments=_DEFAULT_ENVIRONMENT,
                 observation_models=None,
                 continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
                 discrete_transition_type=DiagonalDiscrete(0.98),
                 initial_conditions_type=UniformInitialConditions(),
                 infer_track_interior=True,
                 knot_spacing=10,
                 spike_model_penalty=1E1):
        super().__init__(environments,
                         observation_models,
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
                         environment_labels=None):
        logger.info('Fitting place fields...')
        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=np.bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time)

        is_training = np.asarray(is_training).squeeze()

        self.place_fields_ = {}
        for obs in np.unique(self.observation_models):
            environment = self.environments[
                self.environments.index(obs.environment_name)]

            is_encoding = np.isin(encoding_group_labels, obs.encoding_group)
            is_environment = (environment_labels == obs.environment_name)
            likelihood_name = (obs.environment_name, obs.encoding_group)

            self.place_fields_[likelihood_name] = estimate_place_fields(
                position=position[is_training & is_encoding & is_environment],
                spikes=spikes[is_training & is_encoding & is_environment],
                place_bin_centers=environment.place_bin_centers_,
                place_bin_edges=environment.place_bin_edges_,
                penalty=self.spike_model_penalty,
                knot_spacing=self.knot_spacing)

    def plot_place_fields(self, sampling_frequency=1, figsize=(10, 7)):
        try:
            for (env, enc) in self.place_fields_:
                is_track_interior = self.environments[self.environments.index(
                    env)].is_track_interior_[np.newaxis]
                ((self.place_fields_[(env, enc)] * sampling_frequency)
                 .unstack('position')
                 .where(is_track_interior)
                 .plot(x='x_position', y='y_position',  col='neuron',
                       col_wrap=8, vmin=0.0, vmax=3.0))
        except ValueError:
            fig, axes = plt.subplots(
                len(self.place_fields_), 1, constrained_layout=True,
                figsize=figsize)
            for ax, ((env_name, enc_group), place_fields) in zip(
                    axes.flat, self.place_fields_.items()):
                is_track_interior = self.environments[self.environments.index(
                    env_name)].is_track_interior_[:, np.newaxis]
                ((place_fields * sampling_frequency)
                 .where(is_track_interior)
                 .plot(
                    x='position', hue='neuron', add_legend=False, ax=ax))
                ax.set_title(
                    f'Environment = {env_name}, Encoding Group = {enc_group}')
                ax.set_ylabel('Firing Rate\n[spikes/s]')

    def fit(self,
            position,
            spikes,
            is_training=None,
            encoding_group_labels=None,
            environment_labels=None,
            ):
        '''

        Parameters
        ----------
        position : ndarray, shape (n_time, n_position_dims)
        spikes : ndarray, shape (n_time, n_neurons)
        is_training : None or bool ndarray, shape (n_time), optional
            Time bins to be used for encoding.
        encoding_group_labels : None or ndarray, shape (n_time,)
            Label for the corresponding encoding group for each time point
        environment_labels : None or ndarray, shape (n_time,)
            Label for the corresponding environment for each time point

        Returns
        -------
        self

        '''
        position = atleast_2d(np.asarray(position))
        spikes = np.asarray(spikes)
        self.fit_environments(position, environment_labels)
        self.fit_initial_conditions()
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
                              environment_labels)

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

        return self._get_results(
            likelihood, n_time, time, state_names, use_gpu, is_compute_acausal)


class ClusterlessClassifier(_ClassifierBase):
    '''

    Attributes
    ----------
    environment : Environment
        The spatial environment and topology to fit
    continuous_transition_types : tuple of tuples
        The continuous state transition class instances to fit
    discrete_transition_type : Discrete
        The type of transition between states
    initial_conditions_type : InitialConditions
        The initial conditions class instance to fit
    infer_track_interior : bool, optional
        Whether to infer the valid position bins
    clusterless_algorithm : str
        The type of clusterless algorithm. See _ClUSTERLESS_ALGORITHMS for keys
    clusterless_algorithm_params : dict
        Parameters for the clusterless algorithms.

    '''

    def __init__(self,
                 environments=_DEFAULT_ENVIRONMENT,
                 observation_models=None,
                 continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
                 discrete_transition_type=DiagonalDiscrete(0.98),
                 initial_conditions_type=UniformInitialConditions(),
                 infer_track_interior=True,
                 clusterless_algorithm='multiunit_likelihood',
                 clusterless_algorithm_params=_DEFAULT_CLUSTERLESS_MODEL_KWARGS
                 ):
        super().__init__(environments,
                         observation_models,
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
                       environment_labels=None):
        '''

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)
        encoding_group_labels : None or ndarray, shape (n_time,)
            Label for the corresponding encoding group for each time point
        environment_labels : None or ndarray, shape (n_time,)
            Label for the corresponding environment for each time point

        '''
        logger.info('Fitting multiunits...')
        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=np.bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time)

        is_training = np.asarray(is_training).squeeze()

        kwargs = self.clusterless_algorithm_params
        if kwargs is None:
            kwargs = {}

        self.encoding_model_ = {}

        for obs in np.unique(self.observation_models):
            environment = self.environments[
                self.environments.index(obs.environment_name)]

            is_encoding = np.isin(encoding_group_labels, obs.encoding_group)
            is_environment = (environment_labels == obs.environment_name)
            is_group = is_training & is_encoding & is_environment

            likelihood_name = (obs.environment_name, obs.encoding_group)

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
            environment_labels=None,
            ):
        '''

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)
        encoding_group_labels : None or ndarray, shape (n_time,)
            Label for the corresponding encoding group for each time point
        environment_labels : None or ndarray, shape (n_time,)
            Label for the corresponding environment for each time point

        Returns
        -------
        self

        '''
        position = atleast_2d(np.asarray(position))
        multiunits = np.asarray(multiunits)

        self.fit_environments(position, environment_labels)
        self.fit_initial_conditions()
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
                            environment_labels)

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

        return self._get_results(
            likelihood, n_time, time, state_names, use_gpu, is_compute_acausal)
