"""State space models that decode trajectories from population spiking"""

from copy import deepcopy
from logging import getLogger

import joblib
import numpy as np
import sklearn
import xarray as xr
from replay_trajectory_classification.continuous_state_transitions import (
    EmpiricalMovement,
    RandomWalk,
)
from replay_trajectory_classification.core import (
    _acausal_decode,
    _acausal_decode_gpu,
    _causal_decode,
    _causal_decode_gpu,
    atleast_2d,
    get_centers,
    mask,
    scaled_likelihood,
)
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.initial_conditions import UniformInitialConditions
from replay_trajectory_classification.likelihoods import (
    _SORTED_SPIKES_ALGORITHMS,
    _ClUSTERLESS_ALGORITHMS,
)
from sklearn.base import BaseEstimator

logger = getLogger(__name__)

sklearn.set_config(print_changed_only=False)

_DEFAULT_CLUSTERLESS_MODEL_KWARGS = {
    "mark_std": 24.0,
    "position_std": 6.0,
}

_DEFAULT_SORTED_SPIKES_MODEL_KWARGS = {
    "position_std": 6.0,
    "use_diffusion": False,
    "block_size": None,
}


class _DecoderBase(BaseEstimator):
    def __init__(
        self,
        environment=Environment(environment_name=""),
        transition_type=RandomWalk(),
        initial_conditions_type=UniformInitialConditions(),
        infer_track_interior=True,
    ):
        self.environment = environment
        self.transition_type = transition_type
        self.initial_conditions_type = initial_conditions_type
        self.infer_track_interior = infer_track_interior

    def fit_environment(self, position):
        self.environment.fit_place_grid(
            position, infer_track_interior=self.infer_track_interior
        )

    def fit_initial_conditions(self):
        logger.info("Fitting initial conditions...")
        self.initial_conditions_ = self.initial_conditions_type.make_initial_conditions(
            [self.environment], [self.environment.environment_name]
        )[0]

    def fit_state_transition(
        self, position, is_training=None, transition_type=RandomWalk()
    ):
        logger.info("Fitting state transition...")
        if isinstance(self.transition_type, EmpiricalMovement):
            if is_training is None:
                is_training = np.ones((position.shape[0],), dtype=np.bool)
            is_training = np.asarray(is_training).squeeze()
            n_time = position.shape[0]
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)
            self.state_transition_ = self.transition_type.make_state_transition(
                (self.environment,),
                position,
                is_training,
                encoding_group_labels,
                environment_labels=None,
            )
        else:
            self.state_transition_ = self.transition_type.make_state_transition(
                (self.environment,)
            )

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_model(self, filename="model.pkl"):
        joblib.dump(self, filename)

    @staticmethod
    def load_model(filename="model.pkl"):
        return joblib.load(filename)

    def copy(self):
        return deepcopy(self)

    def _get_results(
        self,
        results,
        n_time,
        time=None,
        is_compute_acausal=True,
        use_gpu=False,
    ):

        is_track_interior = self.environment.is_track_interior_.ravel(order="F")
        n_position_bins = is_track_interior.shape[0]
        st_interior_ind = np.ix_(is_track_interior, is_track_interior)

        logger.info("Estimating causal posterior...")
        if not use_gpu:
            results["causal_posterior"] = np.full(
                (n_time, n_position_bins), np.nan, dtype=np.float64
            )
            (
                results["causal_posterior"][:, is_track_interior],
                data_log_likelihood,
            ) = _causal_decode(
                self.initial_conditions_[is_track_interior].astype(np.float64),
                self.state_transition_[st_interior_ind].astype(np.float64),
                results["likelihood"][:, is_track_interior].astype(np.float64),
            )
        else:
            results["causal_posterior"] = np.full(
                (n_time, n_position_bins), np.nan, dtype=np.float32
            )
            (
                results["causal_posterior"][:, is_track_interior],
                data_log_likelihood,
            ) = _causal_decode_gpu(
                self.initial_conditions_[is_track_interior],
                self.state_transition_[st_interior_ind],
                results["likelihood"][:, is_track_interior],
            )

        if is_compute_acausal:
            logger.info("Estimating acausal posterior...")
            if not use_gpu:
                results["acausal_posterior"] = np.full(
                    (n_time, n_position_bins, 1), np.nan, dtype=np.float64
                )
                results["acausal_posterior"][:, is_track_interior] = _acausal_decode(
                    results["causal_posterior"][
                        :, is_track_interior, np.newaxis
                    ].astype(np.float64),
                    self.state_transition_[st_interior_ind].astype(np.float64),
                )
            else:
                results["acausal_posterior"] = np.full(
                    (n_time, n_position_bins, 1), np.nan, dtype=np.float32
                )
                results["acausal_posterior"][
                    :, is_track_interior
                ] = _acausal_decode_gpu(
                    results["causal_posterior"][:, is_track_interior, np.newaxis],
                    self.state_transition_[st_interior_ind],
                )

        if time is None:
            time = np.arange(n_time)

        return self.convert_results_to_xarray(results, time, data_log_likelihood)

    def convert_results_to_xarray(self, results, time, data_log_likelihood):
        n_position_dims = self.environment.place_bin_centers_.shape[1]
        n_time = time.shape[0]

        attrs = {"data_log_likelihood": data_log_likelihood}

        if n_position_dims > 1:
            dims = ["time", "x_position", "y_position"]
            coords = dict(
                time=time,
                x_position=get_centers(self.environment.edges_[0]),
                y_position=get_centers(self.environment.edges_[1]),
            )
        else:
            dims = ["time", "position"]
            coords = dict(
                time=time,
                position=get_centers(self.environment.edges_[0]),
            )
        new_shape = (n_time, *self.environment.centers_shape_)
        is_track_interior = self.environment.is_track_interior_.ravel(order="F")
        try:
            results = xr.Dataset(
                {
                    key: (
                        dims,
                        mask(value, is_track_interior)
                        .reshape(new_shape)
                        .swapaxes(-1, -2),
                    )
                    for key, value in results.items()
                },
                coords=coords,
                attrs=attrs,
            )
        except ValueError:
            results = xr.Dataset(
                {
                    key: (dims, mask(value, is_track_interior).reshape(new_shape))
                    for key, value in results.items()
                },
                coords=coords,
                attrs=attrs,
            )

        return results


class SortedSpikesDecoder(_DecoderBase):
    def __init__(
        self,
        environment=Environment(environment_name=""),
        transition_type=RandomWalk(),
        initial_conditions_type=UniformInitialConditions(),
        infer_track_interior=True,
        sorted_spikes_algorithm="spiking_likelihood_kde",
        sorted_spikes_algorithm_params=_DEFAULT_SORTED_SPIKES_MODEL_KWARGS,
    ):
        """

        Attributes
        ----------
        environment : Environment
            The spatial environment and topology to fit
        transition_type : (RandomWalk, Uniform, ...)
            The continuous state transition class instance to fit
        initial_conditions_type : InitialConditions
            The initial conditions class instance to fit
        infer_track_interior : bool, optional
            Whether to infer the valid position bins
        knot_spacing : float, optional
            How far apart the spline knots are in position
        spike_model_penalty : float, optional
            L2 penalty (ridge) for the size of the regression coefficients
        """
        super().__init__(
            environment, transition_type, initial_conditions_type, infer_track_interior
        )
        self.sorted_spikes_algorithm = sorted_spikes_algorithm
        self.sorted_spikes_algorithm_params = sorted_spikes_algorithm_params

    def fit_place_fields(self, position, spikes, is_training=None):
        logger.info("Fitting place fields...")
        if is_training is None:
            is_training = np.ones((position.shape[0],), dtype=np.bool)
        is_training = np.asarray(is_training).squeeze()
        kwargs = self.sorted_spikes_algorithm_params
        if kwargs is None:
            kwargs = {}
        self.place_fields_ = _SORTED_SPIKES_ALGORITHMS[self.sorted_spikes_algorithm][0](
            position[is_training],
            spikes[is_training],
            place_bin_centers=self.environment.place_bin_centers_,
            place_bin_edges=self.environment.place_bin_edges_,
            edges=self.environment.edges_,
            is_track_interior=self.environment.is_track_interior_,
            is_track_boundary=self.environment.is_track_boundary_,
            **kwargs
        )

    def plot_place_fields(self, sampling_frequency=1, col_wrap=5):
        """Plots the fitted 2D place fields for each neuron.

        Parameters
        ----------
        sampling_frequency : float, optional
        col_wrap : int, optional

        Returns
        -------
        g : xr.plot.FacetGrid instance

        """
        try:
            g = (
                self.place_fields_.unstack("position").where(
                    self.environment.is_track_interior_
                )
                * sampling_frequency
            ).plot(x="x_position", y="y_position", col="neuron", col_wrap=col_wrap)
        except ValueError:
            g = (self.place_fields_ * sampling_frequency).plot(
                x="position", col="neuron", col_wrap=col_wrap
            )

        return g

    def fit(self, position, spikes, is_training=None):
        """

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
        spikes : np.ndarray, shape (n_time, n_neurons)
        is_training : None or bool np.ndarray, shape (n_time), optional
            Time bins to be used for encoding.

        Returns
        -------
        self

        """
        position = atleast_2d(np.asarray(position))
        spikes = np.asarray(spikes)
        self.fit_environment(position)
        self.fit_initial_conditions()
        self.fit_state_transition(
            position, is_training, transition_type=self.transition_type
        )
        self.fit_place_fields(position, spikes, is_training)

        return self

    def predict(self, spikes, time=None, is_compute_acausal=True, use_gpu=False):
        """

        Parameters
        ----------
        spikes : np.ndarray, shape (n_time, n_neurons)
        time : ndarray or None, shape (n_time,), optional
        is_compute_acausal : bool, optional
        use_gpu : bool, optional
            Use GPU for the state space part of the model, not the likelihood.

        Returns
        -------
        results : xarray.Dataset

        """
        spikes = np.asarray(spikes)
        n_time = spikes.shape[0]

        logger.info("Estimating likelihood...")
        results = {}
        results["likelihood"] = scaled_likelihood(
            _SORTED_SPIKES_ALGORITHMS[self.sorted_spikes_algorithm][1](
                spikes, np.asarray(self.place_fields_)
            )
        )
        return self._get_results(results, n_time, time, is_compute_acausal, use_gpu)


class ClusterlessDecoder(_DecoderBase):
    """

    Attributes
    ----------
    environment : Environment
        The spatial environment and topology to fit
    transition_type : (RandomWalk, Uniform, ...)
        The continuous state transition class instance to fit
    initial_conditions_type : InitialConditions
        The initial conditions class instance to fit
    infer_track_interior : bool, optional
        Whether to infer the valid position bins
    clusterless_algorithm : str
        The type of clusterless algorithm. See _ClUSTERLESS_ALGORITHMS for keys
    clusterless_algorithm_params : dict
        Parameters for the clusterless algorithms.
    """

    def __init__(
        self,
        environment=Environment(environment_name=""),
        transition_type=RandomWalk(),
        initial_conditions_type=UniformInitialConditions(),
        infer_track_interior=True,
        clusterless_algorithm="multiunit_likelihood",
        clusterless_algorithm_params=_DEFAULT_CLUSTERLESS_MODEL_KWARGS,
    ):
        super().__init__(
            environment, transition_type, initial_conditions_type, infer_track_interior
        )
        self.clusterless_algorithm = clusterless_algorithm
        self.clusterless_algorithm_params = clusterless_algorithm_params

    def fit_multiunits(self, position, multiunits, is_training=None):
        """

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)

        """
        logger.info("Fitting multiunits...")
        if is_training is None:
            is_training = np.ones((position.shape[0],), dtype=np.bool)
        is_training = np.asarray(is_training).squeeze()

        kwargs = self.clusterless_algorithm_params
        if kwargs is None:
            kwargs = {}

        self.encoding_model_ = _ClUSTERLESS_ALGORITHMS[self.clusterless_algorithm][0](
            position=position[is_training],
            multiunits=multiunits[is_training],
            place_bin_centers=self.environment.place_bin_centers_,
            is_track_interior=self.environment.is_track_interior_.ravel(order="F"),
            **kwargs
        )

    def fit(self, position, multiunits, is_training=None):
        """

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)

        Returns
        -------
        self

        """
        position = atleast_2d(np.asarray(position))
        multiunits = np.asarray(multiunits)

        self.fit_environment(position)
        self.fit_initial_conditions()
        self.fit_state_transition(
            position, is_training, transition_type=self.transition_type
        )
        self.fit_multiunits(position, multiunits, is_training)

        return self

    def predict(self, multiunits, time=None, is_compute_acausal=True, use_gpu=False):
        """

        Parameters
        ----------
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        time : None or np.ndarray, shape (n_time,)
        is_compute_acausal : bool, optional
            Use future information to compute the posterior.
        use_gpu : bool, optional
            Use GPU for the state space part of the model, not the likelihood.

        Returns
        -------
        results : xarray.Dataset

        """
        multiunits = np.asarray(multiunits)
        is_track_interior = self.environment.is_track_interior_.ravel(order="F")
        n_time = multiunits.shape[0]

        logger.info("Estimating likelihood...")
        results = {}
        results["likelihood"] = scaled_likelihood(
            _ClUSTERLESS_ALGORITHMS[self.clusterless_algorithm][1](
                multiunits=multiunits,
                place_bin_centers=self.environment.place_bin_centers_,
                is_track_interior=is_track_interior,
                **self.encoding_model_
            )
        )

        return self._get_results(results, n_time, time, is_compute_acausal, use_gpu)
