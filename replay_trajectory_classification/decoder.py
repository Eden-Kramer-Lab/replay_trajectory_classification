"""Main classes for decoding trajectories from population spiking"""

from copy import deepcopy
from logging import getLogger
from typing import Optional, Union

import joblib
import numpy as np
import sklearn
import xarray as xr
from sklearn.base import BaseEstimator

from replay_trajectory_classification.continuous_state_transitions import (
    EmpiricalMovement,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
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
    """Base class for decoder objects.

    Parameters
    ----------
    environment : Environment, optional
        The spatial environment to fit
    transition_type : EmpiricalMovement | RandomWalk | RandomWalkDirection1 | RandomWalkDirection2 | Uniform
        The continuous state transition matrix
    initial_conditions_type : UniformInitialConditions, optional
        The initial conditions class instance
    infer_track_interior : bool, optional
        Whether to infer the spatial geometry of track from position

    """

    def __init__(
        self,
        environment: Environment = Environment(environment_name=""),
        transition_type: Union[
            EmpiricalMovement,
            RandomWalk,
            RandomWalkDirection1,
            RandomWalkDirection2,
            Uniform,
        ] = RandomWalk(),
        initial_conditions_type: UniformInitialConditions = UniformInitialConditions(),
        infer_track_interior: bool = True,
    ):
        self.environment = environment
        self.transition_type = transition_type
        self.initial_conditions_type = initial_conditions_type
        self.infer_track_interior = infer_track_interior

    def fit_environment(self, position: np.ndarray) -> None:
        """Discretize the spatial environment into bins. Determine valid track positions.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
            Position of the animal in the environment.
        """
        self.environment.fit_place_grid(
            position, infer_track_interior=self.infer_track_interior
        )

    def fit_initial_conditions(self):
        """Set the initial probability of position."""
        logger.info("Fitting initial conditions...")
        self.initial_conditions_ = self.initial_conditions_type.make_initial_conditions(
            [self.environment], [self.environment.environment_name]
        )[0]

    def fit_state_transition(
        self,
        position: np.ndarray,
        is_training: Optional[np.ndarray] = None,
        transition_type: Optional[
            Union[
                EmpiricalMovement,
                RandomWalk,
                RandomWalkDirection1,
                RandomWalkDirection2,
                Uniform,
            ]
        ] = None,
    ):
        logger.info("Fitting state transition...")

        if transition_type is not None:
            self.transition_type = transition_type

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
        """To be implemented by inheriting class"""
        raise NotImplementedError

    def predict(self):
        """To be implemented by inheriting class"""
        raise NotImplementedError

    def save_model(self, filename="model.pkl"):
        """Save the classifier to a pickled file.

        Parameters
        ----------
        filename : str, optional

        """
        joblib.dump(self, filename)

    @staticmethod
    def load_model(filename="model.pkl"):
        """Load the classifier from a file.

        Parameters
        ----------
        filename : str, optional

        Returns
        -------
        classifier instance

        """
        return joblib.load(filename)

    def copy(self):
        """Makes a copy of the classifier"""
        return deepcopy(self)

    def project_1D_position_to_2D(
        self, results: xr.Dataset, posterior_type="acausal_posterior"
    ) -> np.ndarray:
        """Project the 1D most probable position into the 2D track graph space.

        Only works for single environment.

        Parameters
        ----------
        results : xr.Dataset
        posterior_type : causal_posterior | acausal_posterior | likelihood

        Returns
        -------
        map_position2D : np.ndarray

        """
        map_position_ind = (
            results[posterior_type].sum("state").argmax("position").to_numpy().squeeze()
        )

        return self.environment.place_bin_centers_nodes_df_.iloc[map_position_ind][
            ["x_position", "y_position"]
        ].to_numpy()

    def _get_results(
        self,
        results: dict,
        n_time: int,
        time: Optional[np.ndarray] = None,
        is_compute_acausal: bool = True,
        use_gpu: bool = False,
    ):
        """Converts the results dict into a collection of labeled arrays.

        Parameters
        ----------
        results : dict
        n_time : int
        time : np.ndarray
        is_compute_acausal : bool
        use_gpu : bool

        Returns
        -------
        results : xr.Dataset

        """
        is_track_interior = self.environment.is_track_interior_.ravel(order="F")
        n_position_bins = is_track_interior.shape[0]
        st_interior_ind = np.ix_(is_track_interior, is_track_interior)

        logger.info("Estimating causal posterior...")
        if not use_gpu:
            results["causal_posterior"] = np.full(
                (n_time, n_position_bins), np.nan, dtype=float
            )
            (
                results["causal_posterior"][:, is_track_interior],
                data_log_likelihood,
            ) = _causal_decode(
                self.initial_conditions_[is_track_interior].astype(float),
                self.state_transition_[st_interior_ind].astype(float),
                results["likelihood"][:, is_track_interior].astype(float),
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
                    (n_time, n_position_bins, 1), np.nan, dtype=float
                )
                results["acausal_posterior"][:, is_track_interior] = _acausal_decode(
                    results["causal_posterior"][
                        :, is_track_interior, np.newaxis
                    ].astype(float),
                    self.state_transition_[st_interior_ind].astype(float),
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

    def convert_results_to_xarray(
        self, results: dict, time: np.ndarray, data_log_likelihood: float
    ) -> xr.Dataset:
        """Converts the results dict into a collection of labeled arrays.

        Parameters
        ----------
        results : dict
        time : np.ndarray
        data_log_likelihood : float

        Returns
        -------
        results : xr.Dataset

        """
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
        environment: Environment = Environment(environment_name=""),
        transition_type: Union[
            EmpiricalMovement,
            RandomWalk,
            RandomWalkDirection1,
            RandomWalkDirection2,
            Uniform,
        ] = RandomWalk(),
        initial_conditions_type: UniformInitialConditions = UniformInitialConditions(),
        infer_track_interior: bool = True,
        sorted_spikes_algorithm: str = "spiking_likelihood_kde",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_MODEL_KWARGS,
    ):
        """Decodes neural population representation of position from clustered cells.

        Parameters
        ----------
        environment : Environment instance, optional
            The spatial environment to fit
        transition_type : transition matrix instance, optional
            Movement model for the continuous state
        discrete_transition_type : discrete transition instance, optional
        initial_conditions_type : initial conditions instance, optional
            The initial conditions class instance
        infer_track_interior : bool, optional
            Whether to infer the spatial geometry of track from position
        sorted_spikes_algorithm : str, optional
            The type of algorithm. See _SORTED_SPIKES_ALGORITHMS for keys
        sorted_spikes_algorithm_params : dict, optional
            Parameters for the algorithm.

        """
        super().__init__(
            environment, transition_type, initial_conditions_type, infer_track_interior
        )
        self.sorted_spikes_algorithm = sorted_spikes_algorithm
        self.sorted_spikes_algorithm_params = sorted_spikes_algorithm_params

    def fit_place_fields(
        self,
        position: np.ndarray,
        spikes: np.ndarray,
        is_training: Optional[np.ndarray] = None,
    ):
        """Fits the place intensity function.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
            Position of the animal.
        spikes : np.ndarray, (n_time, n_neurons)
            Binary indicator of whether there was a spike in a given time bin for a given neuron.
        is_training : np.ndarray, shape (n_time,), optional
            Boolean array to indicate which data should be included in fitting of place fields, by default None

        """
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

    def plot_place_fields(
        self, sampling_frequency: int = 1, col_wrap: int = 5
    ) -> xr.plot.FacetGrid:
        """Plots the fitted place fields for each neuron.

        Parameters
        ----------
        sampling_frequency : int, optional
            Number of samples per second
        col_wrap : int, optional
            Number of columns in the subplot.

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

    def fit(
        self,
        position: np.ndarray,
        spikes: np.ndarray,
        is_training: Optional[np.ndarray] = None,
    ):
        """Fit the spatial grid, initial conditions, place field model, and
        transition matrix.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
            Position of the animal.
        spikes : np.ndarray, shape (n_time, n_neurons)
            Binary indicator of whether there was a spike in a given time bin for a given neuron.
        is_training : None or np.ndarray, shape (n_time), optional
            Boolean array to indicate which data should be included in fitting of place fields, by default None

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

    def predict(
        self,
        spikes: np.ndarray,
        time: Optional[np.ndarray] = None,
        is_compute_acausal: bool = True,
        use_gpu: bool = False,
    ) -> xr.Dataset:
        """Predict the probability of spatial position from the spikes.

        Parameters
        ----------
        spikes : np.ndarray, shape (n_time, n_neurons)
            Binary indicator of whether there was a spike in a given time bin for a given neuron.
        time : np.ndarray or None, shape (n_time,), optional
            Label the time axis with these values.
        is_compute_acausal : bool, optional
            If True, compute the acausal posterior.
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
    """Classifies neural population representation of position from multiunit spikes and waveforms.

    Parameters
    ----------
    environment : Environment, optional
        The spatial environment to fit
    transition_type : EmpiricalMovement | RandomWalk | RandomWalkDirection1 | RandomWalkDirection2 | Uniform
        The continuous state transition matrix
    initial_conditions_type : UniformInitialConditions, optional
        The initial conditions class instance
    infer_track_interior : bool, optional
        Whether to infer the spatial geometry of track from position
    clusterless_algorithm : str
        The type of clusterless algorithm. See _ClUSTERLESS_ALGORITHMS for keys
    clusterless_algorithm_params : dict
        Parameters for the clusterless algorithms.

    """

    def __init__(
        self,
        environment: Environment = Environment(environment_name=""),
        transition_type: Union[
            EmpiricalMovement,
            RandomWalk,
            RandomWalkDirection1,
            RandomWalkDirection2,
            Uniform,
        ] = RandomWalk(),
        initial_conditions_type: UniformInitialConditions = UniformInitialConditions(),
        infer_track_interior: bool = True,
        clusterless_algorithm: str = "multiunit_likelihood",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_MODEL_KWARGS,
    ):
        super().__init__(
            environment, transition_type, initial_conditions_type, infer_track_interior
        )
        self.clusterless_algorithm = clusterless_algorithm
        self.clusterless_algorithm_params = clusterless_algorithm_params

    def fit_multiunits(
        self,
        position: np.ndarray,
        multiunits: np.ndarray,
        is_training: Optional[np.ndarray] = None,
    ):
        """

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
        multiunits : np.ndarray, shape (n_time, n_marks, n_electrodes)
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

    def fit(
        self,
        position: np.ndarray,
        multiunits: np.ndarray,
        is_training: Optional[np.ndarray] = None,
    ):
        """Fit the spatial grid, initial conditions, place field model, and
        transition matrices.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
            Position of the animal.
        multiunits : np.ndarray, shape (n_time, n_marks, n_electrodes)
            Array where spikes are indicated by non-Nan values that correspond to the waveform features
            for each electrode.
        is_training : None or np.ndarray, shape (n_time), optional
            Boolean array to indicate which data should be included in fitting of place fields, by default None

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

    def predict(
        self,
        multiunits: np.ndarray,
        time: Optional[np.ndarray] = None,
        is_compute_acausal: bool = True,
        use_gpu: bool = False,
    ) -> xr.Dataset:
        """Predict the probability of spatial position and category from the multiunit spikes and waveforms.

        Parameters
        ----------
        multiunits : np.ndarray, shape (n_time, n_marks, n_electrodes)
            Array where spikes are indicated by non-Nan values that correspond to the waveform features
            for each electrode.
        time : np.ndarray or None, shape (n_time,), optional
            Label the time axis with these values.
        is_compute_acausal : bool, optional
            If True, compute the acausal posterior.
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
