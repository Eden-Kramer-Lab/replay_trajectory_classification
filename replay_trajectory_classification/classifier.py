from copy import deepcopy
from logging import getLogger

import joblib
import numpy as np
import pandas as pd
import sklearn
import xarray as xr
from replay_trajectory_classification.bins import (atleast_2d, get_centers,
                                                   get_grid, get_track_grid,
                                                   get_track_interior)
from replay_trajectory_classification.core import (_acausal_classify,
                                                   _acausal_classify_gpu,
                                                   _causal_classify,
                                                   _causal_classify_gpu,
                                                   _ClUSTERLESS_ALGORITHMS,
                                                   mask, scaled_likelihood)
from replay_trajectory_classification.initial_conditions import \
    uniform_on_track
from replay_trajectory_classification.misc import NumbaKDE
from replay_trajectory_classification.spiking_likelihood import (
    estimate_place_fields, estimate_spiking_likelihood)
from replay_trajectory_classification.state_transition import (
    CONTINUOUS_TRANSITIONS, DISCRETE_TRANSITIONS)
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
    [['random_walk', 'uniform', 'identity'],
     ['uniform',   'uniform', 'uniform'],
     ['random_walk', 'uniform', 'identity']])
_DISCRETE_DIAG = 1 - 1E-2


class _ClassifierBase(BaseEstimator):
    def __init__(self, place_bin_size=2.0, replay_speed=1, movement_var=6.0,
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

        if 2 * np.sqrt(replay_speed * movement_var) < place_bin_size:
            logger.warning('Place bin size is too small for a random walk '
                           'continuous state transition')

    def fit_place_grid(self, position, track_graph=None,
                       edge_order=None, edge_spacing=15,
                       infer_track_interior=True, is_track_interior=None):
        if track_graph is None:
            (self.edges_, self.place_bin_edges_, self.place_bin_centers_,
             self.centers_shape_) = get_grid(
                position, self.place_bin_size, self.position_range,
                self.infer_track_interior)

            self.infer_track_interior = infer_track_interior

            if is_track_interior is None and self.infer_track_interior:
                self.is_track_interior_ = get_track_interior(
                    position, self.edges_)
            elif is_track_interior is None and not self.infer_track_interior:
                self.is_track_interior_ = np.ones(
                    self.centers_shape_, dtype=np.bool)
        else:
            (
                self.place_bin_centers_,
                self.place_bin_edges_,
                self.is_track_interior_,
                self.distance_between_nodes_,
                self.centers_shape_,
                self.edges_,
                self.track_graph_with_bin_centers_edges_,
                self.original_nodes_df_,
                self.place_bin_edges_nodes_df_,
                self.place_bin_centers_nodes_df_,
                self.nodes_df_
            ) = get_track_grid(track_graph, edge_order,
                               edge_spacing, self.place_bin_size)

    def fit_initial_conditions(self, position=None):
        logger.info('Fitting initial conditions...')
        n_states = len(self.continuous_transition_types)
        initial_conditions = uniform_on_track(self.place_bin_centers_,
                                              self.is_track_interior_)
        self.initial_conditions_ = (
            np.stack([initial_conditions] * n_states, axis=0)[..., np.newaxis]
            / n_states)

    def fit_continuous_state_transition(
            self, position, is_training=None, replay_speed=None,
            continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS):
        logger.info('Fitting state transition...')
        if is_training is None:
            is_training = np.ones((position.shape[0],), dtype=np.bool)
        is_training = np.asarray(is_training).squeeze()
        if replay_speed is not None:
            self.replay_speed = replay_speed
        self.continuous_transition_types = continuous_transition_types

        n_bins = self.place_bin_centers_.shape[0]
        n_states = len(self.continuous_transition_types)
        self.continuous_state_transition_ = np.zeros(
            (n_states, n_states, n_bins, n_bins))
        for row_ind, row in enumerate(self.continuous_transition_types):
            for column_ind, transition_type in enumerate(row):
                try:
                    self.continuous_state_transition_[row_ind, column_ind] = (
                        CONTINUOUS_TRANSITIONS[transition_type](
                            self.place_bin_centers_, self.is_track_interior_,
                            position, self.edges_, is_training,
                            self.replay_speed, self.position_range,
                            self.movement_var,
                            np.asarray(
                                self.place_bin_centers_nodes_df_.node_id),
                            self.distance_between_nodes_)
                    )
                except AttributeError:
                    self.continuous_state_transition_[row_ind, column_ind] = (
                        CONTINUOUS_TRANSITIONS[transition_type](
                            self.place_bin_centers_, self.is_track_interior_,
                            position, self.edges_, is_training,
                            self.replay_speed, self.position_range,
                            self.movement_var, None, None)
                    )

    def fit_discrete_state_transition(self, discrete_transition_diag=None):
        if discrete_transition_diag is not None:
            self.discrete_transition_diag = discrete_transition_diag

        n_states = len(self.continuous_transition_types)
        self.discrete_state_transition_ = DISCRETE_TRANSITIONS[
            self.discrete_transition_type](
                n_states, self.discrete_transition_diag)

    def convert_results_to_xarray(self, results, time, state_names=None):
        n_position_dims = self.place_bin_centers_.shape[1]
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

        if n_position_dims > 1:
            new_shape = (n_time, n_states, *self.centers_shape_)
            dims = ['time', 'state', 'x_position', 'y_position']
            coords = dict(
                time=time,
                x_position=get_centers(self.edges_[0]),
                y_position=get_centers(self.edges_[1]),
                state=state_names,
            )
            results = xr.Dataset(
                {key: (dims,
                       (mask(value,
                             self.is_track_interior_.ravel(order='F')
                             ).squeeze(axis=-1)
                        .reshape(new_shape).swapaxes(-1, -2)))
                 for key, value in results.items()},
                coords=coords)
        else:
            dims = ['time', 'state', 'position']
            coords = dict(
                time=time,
                position=get_centers(self.edges_[0]),
                state=state_names,
            )
            results = xr.Dataset(
                {key: (dims,
                       (mask(value,
                             self.is_track_interior_.ravel(order='F')
                             ).squeeze(axis=-1)))
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
    place_bin_size : float, optional
        Approximate size of the position bins.
    replay_speed : int, optional
        How many times faster the replay movement is than normal movement. It​ is
        only used with the empirical transition matrix---a transition matrix
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

    def __init__(self, place_bin_size=2.0, replay_speed=1, movement_var=6.0,
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

    def fit_place_fields(self, position, spikes, is_training=None,
                         encoding_group_labels=None,
                         encoding_group_to_state=None):
        logger.info('Fitting place fields...')
        if is_training is None:
            n_time = position.shape[0]
            is_training = np.ones((n_time,), dtype=np.bool)

        if encoding_group_labels is None:
            n_time = position.shape[0]
            encoding_group_labels = np.zeros((n_time,), dtype=np.int)

        if encoding_group_to_state is None:
            n_states = len(self.continuous_transition_types)
            self.encoding_group_to_state_ = np.zeros((n_states,), dtype=np.int)
        else:
            self.encoding_group_to_state_ = np.asarray(encoding_group_to_state)

        is_training = np.asarray(is_training).squeeze()
        self.place_fields_ = []
        unique_labels = np.unique(encoding_group_labels[is_training])
        for encoding_group in unique_labels:
            self.place_fields_.append(estimate_place_fields(
                position=position[is_training & (
                    encoding_group_labels == encoding_group)],
                spikes=spikes[is_training & (
                    encoding_group_labels == encoding_group)],
                place_bin_centers=self.place_bin_centers_,
                place_bin_edges=self.place_bin_edges_,
                penalty=self.spike_model_penalty,
                knot_spacing=self.knot_spacing))
        self.place_fields_ = xr.concat(
            objs=self.place_fields_,
            dim=pd.Index(unique_labels, name='encoding_group'))

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
            is_track_interior=None,
            encoding_group_labels=None,
            encoding_group_to_state=None,
            track_graph=None,
            edge_order=None,
            edge_spacing=15):
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
        self.fit_place_grid(position, track_graph,
                            edge_order, edge_spacing,
                            self.infer_track_interior, is_track_interior)
        self.fit_initial_conditions(position)
        self.fit_continuous_state_transition(
            position, is_training,
            continuous_transition_types=self.continuous_transition_types)
        self.fit_discrete_state_transition()
        self.fit_place_fields(position, spikes, is_training,
                              encoding_group_labels,
                              encoding_group_to_state)

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
        is_track_interior = self.is_track_interior_.ravel(order='F')
        n_time = spikes.shape[0]
        n_position_bins = is_track_interior.shape[0]
        n_states = self.discrete_state_transition_.shape[0]
        is_states = np.ones((n_states,), dtype=bool)
        st_interior_ind = np.ix_(
            is_states, is_states, is_track_interior, is_track_interior)

        results = {}

        logger.info('Estimating likelihood...')
        likelihood = {}
        for encoding_group in np.asarray(self.place_fields_.encoding_group):
            likelihood[encoding_group] = estimate_spiking_likelihood(
                spikes,
                np.asarray(self.place_fields_.sel(
                    encoding_group=encoding_group)),
                is_track_interior)

        results['likelihood'] = np.stack(
            [likelihood[encoding_group]
             for encoding_group in self.encoding_group_to_state_],
            axis=1)
        results['likelihood'] = scaled_likelihood(
            results['likelihood'], axis=(1, 2))[..., np.newaxis]

        logger.info('Estimating causal posterior...')
        if not use_gpu:
            results['causal_posterior'] = np.full(
                (n_time, n_states, n_position_bins, 1), np.nan,
                dtype=np.float64)
            results['causal_posterior'][:, :, is_track_interior] = _causal_classify(
                self.initial_conditions_[:, is_track_interior],
                self.continuous_state_transition_[st_interior_ind],
                self.discrete_state_transition_,
                results['likelihood'][:, :, is_track_interior])
        else:
            results['causal_posterior'] = np.full(
                (n_time, n_states, n_position_bins, 1), np.nan,
                dtype=np.float32)
            results['causal_posterior'][:, :, is_track_interior] = _causal_classify_gpu(
                self.initial_conditions_[:, is_track_interior],
                self.continuous_state_transition_[st_interior_ind],
                self.discrete_state_transition_,
                results['likelihood'][:, :, is_track_interior])

        if is_compute_acausal:
            logger.info('Estimating acausal posterior...')
            if not use_gpu:
                results['acausal_posterior'] = np.full(
                    (n_time, n_states, n_position_bins, 1), np.nan,
                    dtype=np.float64)
                results['acausal_posterior'][:, :, is_track_interior] = _acausal_classify(
                    results['causal_posterior'][:, :, is_track_interior],
                    self.continuous_state_transition_[st_interior_ind],
                    self.discrete_state_transition_)
            else:
                results['acausal_posterior'] = np.full(
                    (n_time, n_states, n_position_bins, 1), np.nan,
                    dtype=np.float32)
                results['acausal_posterior'][:, :, is_track_interior] = _acausal_classify_gpu(
                    results['causal_posterior'][:, :, is_track_interior],
                    self.continuous_state_transition_[st_interior_ind],
                    self.discrete_state_transition_)

        n_time = spikes.shape[0]

        if time is None:
            time = np.arange(n_time)

        return self.convert_results_to_xarray(results, time, state_names)


class ClusterlessClassifier(_ClassifierBase):
    '''

    Attributes
    ----------
    place_bin_size : float, optional
        Approximate size of the position bins.
    replay_speed : int, optional
        How many times faster the replay movement is than normal movement. It​ is
        only used with the empirical transition matrix---a transition matrix
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

    def __init__(self, place_bin_size=2.0, replay_speed=1, movement_var=6.0,
                 position_range=None,
                 continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
                 discrete_transition_type='strong_diagonal',
                 initial_conditions_type='uniform_on_track',
                 discrete_transition_diag=_DISCRETE_DIAG,
                 infer_track_interior=True,
                 clusterless_algorithm='multiunit_likelihood',
                 clusterless_algorithm_params=_DEFAULT_CLUSTERLESS_MODEL_KWARGS
                 ):
        super().__init__(place_bin_size, replay_speed, movement_var,
                         position_range, continuous_transition_types,
                         discrete_transition_type, initial_conditions_type,
                         discrete_transition_diag, infer_track_interior)

        self.clusterless_algorithm = clusterless_algorithm
        self.clusterless_algorithm_params = clusterless_algorithm_params

    def fit_multiunits(self, position, multiunits, is_training=None,
                       encoding_group_labels=None,
                       encoding_group_to_state=None):
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
            n_time = position.shape[0]
            is_training = np.ones((n_time,), dtype=np.bool)

        if encoding_group_labels is None:
            n_time = position.shape[0]
            encoding_group_labels = np.zeros((n_time,), dtype=np.int)

        if encoding_group_to_state is None:
            n_states = len(self.continuous_transition_types)
            self.encoding_group_to_state_ = np.zeros((n_states,), dtype=np.int)
        else:
            self.encoding_group_to_state_ = np.asarray(encoding_group_to_state)

        kwargs = self.clusterless_algorithm_params
        if kwargs is None:
            kwargs = {}

        is_training = np.asarray(is_training).squeeze()

        self.encoding_model_ = {}

        for encoding_group in np.unique(encoding_group_labels[is_training]):
            is_group = is_training & (
                encoding_group == encoding_group_labels)
            self.encoding_model_[encoding_group] = _ClUSTERLESS_ALGORITHMS[
                self.clusterless_algorithm][0](
                    position=position[is_group],
                    multiunits=multiunits[is_group],
                    place_bin_centers=self.place_bin_centers_,
                    is_track_interior=self.is_track_interior_.ravel(order='F'),
                    **kwargs
            )

    def fit(self,
            position,
            multiunits,
            is_training=None,
            is_track_interior=None,
            encoding_group_labels=None,
            encoding_group_to_state=None,
            track_graph=None,
            edge_order=None,
            edge_spacing=15):
        '''

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)
        is_track_interior : None or ndarray, shape (n_x_bins, n_y_bins)
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
        multiunits = np.asarray(multiunits)

        self.fit_place_grid(position, track_graph,
                            edge_order, edge_spacing,
                            self.infer_track_interior, is_track_interior)
        self.fit_initial_conditions(position)
        self.fit_continuous_state_transition(
            position, is_training,
            continuous_transition_types=self.continuous_transition_types)
        self.fit_discrete_state_transition()
        self.fit_multiunits(position, multiunits, is_training,
                            encoding_group_labels, encoding_group_to_state)

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
        is_track_interior = self.is_track_interior_.ravel(order='F')
        n_time = multiunits.shape[0]
        n_position_bins = is_track_interior.shape[0]
        n_states = self.discrete_state_transition_.shape[0]
        is_states = np.ones((n_states,), dtype=bool)
        st_interior_ind = np.ix_(
            is_states, is_states, is_track_interior, is_track_interior)

        results = {}

        logger.info('Estimating likelihood...')
        likelihood = {}
        for encoding_group, encoding_params in self.encoding_model_.items():
            likelihood[encoding_group] = _ClUSTERLESS_ALGORITHMS[
                self.clusterless_algorithm][1](
                    multiunits=multiunits,
                    place_bin_centers=self.place_bin_centers_,
                    is_track_interior=is_track_interior,
                    **encoding_params
            )

        results['likelihood'] = np.stack(
            [likelihood[encoding_group]
             for encoding_group in self.encoding_group_to_state_],
            axis=1)
        results['likelihood'] = scaled_likelihood(
            results['likelihood'], axis=(1, 2))[..., np.newaxis]

        logger.info('Estimating causal posterior...')
        results['causal_posterior'] = np.full(
            (n_time, n_states, n_position_bins, 1), np.nan)
        if not use_gpu:
            results['causal_posterior'][:, :, is_track_interior] = _causal_classify(
                self.initial_conditions_[:, is_track_interior],
                self.continuous_state_transition_[st_interior_ind],
                self.discrete_state_transition_,
                results['likelihood'][:, :, is_track_interior])
        else:
            results['causal_posterior'][:, :, is_track_interior] = _causal_classify_gpu(
                self.initial_conditions_[:, is_track_interior],
                self.continuous_state_transition_[st_interior_ind],
                self.discrete_state_transition_,
                results['likelihood'][:, :, is_track_interior])

        if is_compute_acausal:
            logger.info('Estimating acausal posterior...')
            results['acausal_posterior'] = np.full(
                (n_time, n_states, n_position_bins, 1), np.nan)

            if not use_gpu:
                results['acausal_posterior'][:, :, is_track_interior] = _acausal_classify(
                    results['causal_posterior'][:, :, is_track_interior],
                    self.continuous_state_transition_[st_interior_ind],
                    self.discrete_state_transition_)
            else:
                results['acausal_posterior'][:, :, is_track_interior] = _acausal_classify_gpu(
                    results['causal_posterior'][:, :, is_track_interior],
                    self.continuous_state_transition_[st_interior_ind],
                    self.discrete_state_transition_)

        if time is None:
            time = np.arange(n_time)

        return self.convert_results_to_xarray(results, time, state_names)
