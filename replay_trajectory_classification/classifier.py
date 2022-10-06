"""State space models that classify trajectories as well as decode the
trajectory from population spiking
"""
from copy import deepcopy
from logging import getLogger

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import xarray as xr
from replay_trajectory_classification.continuous_state_transitions import (
    EmpiricalMovement,
    RandomWalk,
    Uniform,
)
from replay_trajectory_classification.core import (
    _acausal_classify,
    _acausal_classify_gpu,
    _causal_classify,
    _causal_classify_gpu,
    atleast_2d,
    check_converged,
    get_centers,
    mask,
    scaled_likelihood,
)
from replay_trajectory_classification.discrete_state_transitions import (
    DiagonalDiscrete,
    estimate_discrete_state_transition,
)
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.initial_conditions import UniformInitialConditions
from replay_trajectory_classification.likelihoods import (
    _SORTED_SPIKES_ALGORITHMS,
    _ClUSTERLESS_ALGORITHMS,
)
from replay_trajectory_classification.observation_model import ObservationModel
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

_DEFAULT_CONTINUOUS_TRANSITIONS = [[RandomWalk(), Uniform()], [Uniform(), Uniform()]]

_DEFAULT_ENVIRONMENT = Environment(environment_name="")


class _ClassifierBase(BaseEstimator):
    def __init__(
        self,
        environments=_DEFAULT_ENVIRONMENT,
        observation_models=None,
        continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
        discrete_transition_type=DiagonalDiscrete(0.968),
        initial_conditions_type=UniformInitialConditions(),
        infer_track_interior=True,
    ):
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
                is_environment = np.ones((position.shape[0],), dtype=bool)
            else:
                is_environment = environment_labels == environment.environment_name
            environment.fit_place_grid(
                position[is_environment], infer_track_interior=self.infer_track_interior
            )

        self.max_pos_bins_ = np.max(
            [env.place_bin_centers_.shape[0] for env in self.environments]
        )

    def fit_initial_conditions(self):
        logger.info("Fitting initial conditions...")
        environment_names_to_state = [
            obs.environment_name for obs in self.observation_models
        ]
        n_states = len(self.observation_models)
        initial_conditions = self.initial_conditions_type.make_initial_conditions(
            self.environments, environment_names_to_state
        )

        self.initial_conditions_ = np.zeros(
            (n_states, self.max_pos_bins_, 1), dtype=np.float64
        )
        for state_ind, ic in enumerate(initial_conditions):
            self.initial_conditions_[state_ind, : ic.shape[0]] = ic[..., np.newaxis]

    def fit_continuous_state_transition(
        self,
        continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
        position=None,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
    ):
        logger.info("Fitting continuous state transition...")

        if is_training is None:
            n_time = position.shape[0]
            is_training = np.ones((n_time,), dtype=bool)

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
                            self.environments,
                            position,
                            is_training,
                            encoding_group_labels,
                            environment_labels,
                        )
                    )
                else:
                    continuous_state_transition[-1].append(
                        transition.make_state_transition(self.environments)
                    )

        n_states = len(self.continuous_transition_types)
        self.continuous_state_transition_ = np.zeros(
            (n_states, n_states, self.max_pos_bins_, self.max_pos_bins_)
        )

        for row_ind, row in enumerate(continuous_state_transition):
            for column_ind, st in enumerate(row):
                self.continuous_state_transition_[
                    row_ind, column_ind, : st.shape[0], : st.shape[1]
                ] = st

    def fit_discrete_state_transition(self):
        logger.info("Fitting discrete state transition")
        n_states = len(self.continuous_transition_types)
        self.discrete_state_transition_ = (
            self.discrete_transition_type.make_state_transition(n_states)
        )

    def plot_discrete_state_transition(
        self,
        state_names=None,
        cmap="Oranges",
        ax=None,
        convert_to_seconds=False,
        sampling_frequency=1,
    ):

        if ax is None:
            ax = plt.gca()

        if state_names is None:
            state_names = [
                f"state {ind + 1}"
                for ind in range(self.discrete_state_transition_.shape[0])
            ]

        if convert_to_seconds:
            discrete_state_transition = (
                1 / (1 - self.discrete_state_transition_)
            ) / sampling_frequency
            vmin, vmax, fmt = 0.0, None, "0.03f"
            label = "Seconds"
        else:
            discrete_state_transition = self.discrete_state_transition_
            vmin, vmax, fmt = 0.0, 1.0, "0.03f"
            label = "Probability"

        sns.heatmap(
            data=discrete_state_transition,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=state_names,
            yticklabels=state_names,
            ax=ax,
            cbar_kws={"label": label},
        )
        ax.set_ylabel("Previous State", fontsize=12)
        ax.set_xlabel("Current State", fontsize=12)
        ax.set_title("Discrete State Transition", fontsize=16)

    def estimate_parameters(
        self,
        fit_args,
        predict_args,
        tolerance=1e-4,
        max_iter=10,
        verbose=True,
        store_likelihood=True,
        estimate_initial_conditions=True,
        estimate_discrete_transition=True,
    ):

        if "store_likelihood" in predict_args:
            store_likelihood = predict_args["store_likelihood"]
        else:
            predict_args["store_likelihood"] = store_likelihood

        self.fit(**fit_args)
        results = self.predict(**predict_args)

        data_log_likelihoods = [results.data_log_likelihood]
        log_likelihood_change = np.inf
        converged = False
        increasing = True
        n_iter = 0
        n_time = len(results.time)

        logger.info(f"iteration {n_iter}, likelihood: {data_log_likelihoods[-1]}")
        get_results_args = {
            key: value
            for key, value in predict_args.items()
            if key in ["time", "state_names", "use_gpu", "is_compute_acausal"]
        }

        while not converged and (n_iter < max_iter):
            if estimate_initial_conditions:
                self.initial_conditions_ = results.isel(
                    time=0
                ).acausal_posterior.values[..., np.newaxis]
            if estimate_discrete_transition:
                self.discrete_state_transition_ = estimate_discrete_state_transition(
                    self, results
                )

            if store_likelihood:
                results = self._get_results(
                    self.likelihood_, n_time, **get_results_args
                )
            else:
                results = self.predict(**predict_args)

            data_log_likelihoods.append(results.data_log_likelihood)
            log_likelihood_change = data_log_likelihoods[-1] - data_log_likelihoods[-2]
            n_iter += 1

            converged, increasing = check_converged(
                data_log_likelihoods[-1], data_log_likelihoods[-2], tolerance
            )

            if verbose:
                logger.info(
                    f"iteration {n_iter}, "
                    f"likelihood: {data_log_likelihoods[-1]}, "
                    f"change: {log_likelihood_change}"
                )

        if not converged and (n_iter == max_iter):
            logger.warning("Max iterations reached...")

        return results, data_log_likelihoods

    @staticmethod
    def convert_2D_to_1D_results(
        results2D: xr.Dataset, environment2D: Environment, environment1D: Environment
    ) -> xr.Dataset:
        """Projects a 2D position decoding result to a 1D decoding result.

        Parameters
        ----------
        results : xarray.core.dataset.Dataset
        environment2D : replay_trajectory_classification.environments.Environment
        environment1D : replay_trajectory_classification.environments.Environment

        Returns
        -------
        results1D : xarray.core.dataset.Dataset

        Examples
        --------
        results = classifier.predict(spikes)
        environment1D = (
            Environment(track_graph=track_graph,
                        place_bin_size=2.0,
                        edge_order=edge_order,
                        edge_spacing=edge_spacing)
            .fit_place_grid())
        results1D = convert_2D_to_1D_results(
            results, classifier.environments[0], environment1D)
        """
        projected_1D_position = np.asarray(
            environment1D.place_bin_centers_nodes_df_[["x_position", "y_position"]]
        )
        bin_centers_2D = environment2D.place_bin_centers_
        closest_1D_bin_ind = np.asarray(
            [
                np.argmin(np.linalg.norm(projected_1D_position - bin_center, axis=1))
                for bin_center in bin_centers_2D
            ]
        )

        non_position_dims = [
            n_elements
            for dim, n_elements in results2D.dims.items()
            if dim not in ["x_position", "y_position"]
        ]
        results1D_shape = (
            *non_position_dims,
            environment1D.place_bin_centers_.shape[0],
        )
        results1D = {
            variable: np.zeros(results1D_shape) for variable in results2D.data_vars
        }

        is_track_interior = environment2D.is_track_interior_.ravel(order="F")
        interior_bin_ind = np.unravel_index(
            np.nonzero(is_track_interior)[0],
            (len(results2D.x_position), len(results2D.y_position)),
            order="F",
        )

        for linear_bin_ind, x_ind, y_ind in zip(
            closest_1D_bin_ind[is_track_interior], *interior_bin_ind
        ):
            for variable in results2D.data_vars:
                results1D[variable][:, linear_bin_ind] += results2D.isel(
                    x_position=x_ind, y_position=y_ind
                )[variable].values

        dims = [
            dim for dim in results2D.dims if dim not in ["x_position", "y_position"]
        ]
        coords = {dim: results2D.coords[dim].values for dim in dims}

        dims.append("position")
        coords["position"] = environment1D.place_bin_centers_.squeeze()

        return xr.Dataset(
            {key: (dims, value) for key, value in results1D.items()},
            coords=coords,
            attrs=results2D.attrs,
        )

    def _get_results(
        self,
        likelihood,
        n_time,
        time=None,
        is_compute_acausal=True,
        use_gpu=False,
        state_names=None,
    ):
        n_states = self.discrete_state_transition_.shape[0]
        results = {}
        dtype = np.float32 if use_gpu else np.float64

        results["likelihood"] = np.full(
            (n_time, n_states, self.max_pos_bins_, 1), np.nan, dtype=dtype
        )
        compute_causal = _causal_classify_gpu if use_gpu else _causal_classify
        compute_acausal = _acausal_classify_gpu if use_gpu else _acausal_classify

        for state_ind, obs in enumerate(self.observation_models):
            likelihood_name = (obs.environment_name, obs.encoding_group)
            n_bins = likelihood[likelihood_name].shape[1]
            results["likelihood"][:, state_ind, :n_bins] = likelihood[likelihood_name][
                ..., np.newaxis
            ]

        results["likelihood"] = scaled_likelihood(results["likelihood"], axis=(1, 2))
        results["likelihood"][np.isnan(results["likelihood"])] = 0.0

        n_environments = len(self.environments)

        if time is None:
            time = np.arange(n_time)

        if n_environments == 1:
            logger.info("Estimating causal posterior...")
            is_track_interior = self.environments[0].is_track_interior_.ravel(order="F")
            n_position_bins = len(is_track_interior)
            is_states = np.ones((n_states,), dtype=bool)
            st_interior_ind = np.ix_(
                is_states, is_states, is_track_interior, is_track_interior
            )

            results["causal_posterior"] = np.full(
                (n_time, n_states, n_position_bins, 1), np.nan, dtype=dtype
            )
            (
                results["causal_posterior"][:, :, is_track_interior],
                data_log_likelihood,
            ) = compute_causal(
                self.initial_conditions_[:, is_track_interior].astype(dtype),
                self.continuous_state_transition_[st_interior_ind].astype(dtype),
                self.discrete_state_transition_.astype(dtype),
                results["likelihood"][:, :, is_track_interior].astype(dtype),
            )

            if is_compute_acausal:
                logger.info("Estimating acausal posterior...")

                results["acausal_posterior"] = np.full(
                    (n_time, n_states, n_position_bins, 1), np.nan, dtype=dtype
                )
                results["acausal_posterior"][:, :, is_track_interior] = compute_acausal(
                    results["causal_posterior"][:, :, is_track_interior].astype(dtype),
                    self.continuous_state_transition_[st_interior_ind].astype(dtype),
                    self.discrete_state_transition_.astype(dtype),
                )

            return self._convert_results_to_xarray(
                results, time, state_names, data_log_likelihood
            )

        else:
            logger.info("Estimating causal posterior...")
            (results["causal_posterior"], data_log_likelihood) = compute_causal(
                self.initial_conditions_.astype(dtype),
                self.continuous_state_transition_.astype(dtype),
                self.discrete_state_transition_.astype(dtype),
                results["likelihood"].astype(dtype),
            )

            if is_compute_acausal:
                logger.info("Estimating acausal posterior...")
                results["acausal_posterior"] = compute_acausal(
                    results["causal_posterior"].astype(dtype),
                    self.continuous_state_transition_.astype(dtype),
                    self.discrete_state_transition_.astype(dtype),
                )

            return self._convert_results_to_xarray_mutienvironment(
                results, time, state_names, data_log_likelihood
            )

    def _convert_results_to_xarray(
        self, results, time, state_names, data_log_likelihood
    ):
        attrs = {"data_log_likelihood": data_log_likelihood}
        n_position_dims = self.environments[0].place_bin_centers_.shape[1]
        diag_transition_names = np.diag(np.asarray(self.continuous_transition_types))
        if state_names is None:
            if len(np.unique(self.observation_models)) == 1:
                state_names = diag_transition_names
            else:
                state_names = [
                    f"{obs.encoding_group}-{transition}"
                    for obs, transition in zip(
                        self.observation_models, diag_transition_names
                    )
                ]
        n_time = time.shape[0]
        n_states = len(state_names)
        is_track_interior = self.environments[0].is_track_interior_.ravel(order="F")
        edges = self.environments[0].edges_

        if n_position_dims > 1:
            centers_shape = self.environments[0].centers_shape_
            new_shape = (n_time, n_states, *centers_shape)
            dims = ["time", "state", "x_position", "y_position"]
            coords = dict(
                time=time,
                x_position=get_centers(edges[0]),
                y_position=get_centers(edges[1]),
                state=state_names,
            )
            results = xr.Dataset(
                {
                    key: (
                        dims,
                        (
                            mask(value, is_track_interior)
                            .squeeze(axis=-1)
                            .reshape(new_shape, order="F")
                        ),
                    )
                    for key, value in results.items()
                },
                coords=coords,
                attrs=attrs,
            )
        else:
            dims = ["time", "state", "position"]
            coords = dict(
                time=time,
                position=get_centers(edges[0]),
                state=state_names,
            )
            results = xr.Dataset(
                {
                    key: (dims, (mask(value, is_track_interior).squeeze(axis=-1)))
                    for key, value in results.items()
                },
                coords=coords,
                attrs=attrs,
            )

        return results

    def _convert_results_to_xarray_mutienvironment(
        self, results, time, state_names, data_log_likelihood
    ):
        if state_names is None:
            state_names = [
                f"{obs.environment_name}-{obs.encoding_group}"
                for obs in self.observation_models
            ]

        attrs = {"data_log_likelihood": data_log_likelihood}
        n_position_dims = self.environments[0].place_bin_centers_.shape[1]

        if n_position_dims > 1:
            dims = ["time", "state", "position"]
            coords = dict(
                time=time,
                state=state_names,
            )
            for env in self.environments:
                coords[env.environment_name + "_x_position"] = get_centers(
                    env.edges_[0]
                )
                coords[env.environment_name + "_y_position"] = get_centers(
                    env.edges_[1]
                )
            results = xr.Dataset(
                {key: (dims, value.squeeze(axis=-1)) for key, value in results.items()},
                coords=coords,
                attrs=attrs,
            )
        else:
            dims = ["time", "state", "position"]
            coords = dict(
                time=time,
                state=state_names,
            )
            for env in self.environments:
                coords[env.environment_name + "_position"] = get_centers(env.edges_[0])

            results = xr.Dataset(
                {key: (dims, value.squeeze(axis=-1)) for key, value in results.items()},
                coords=coords,
                attrs=attrs,
            )

        return results

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_model(self, filename="model.pkl"):
        joblib.dump(self, filename)

    @staticmethod
    def load_model(filename="model.pkl"):
        return joblib.load(filename)

    @staticmethod
    def predict_proba(results):
        try:
            return results.sum(["x_position", "y_position"])
        except ValueError:
            return results.sum(["position"])

    def copy(self):
        return deepcopy(self)


class SortedSpikesClassifier(_ClassifierBase):
    """

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
    sorted_spikes_algorithm : str
        The type of algorithm. See _SORTED_SPIKES_ALGORITHMS for keys
    sorted_spikes_algorithm_params : dict
        Parameters for the algorithm.

    """

    def __init__(
        self,
        environments=_DEFAULT_ENVIRONMENT,
        observation_models=None,
        continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
        discrete_transition_type=DiagonalDiscrete(0.98),
        initial_conditions_type=UniformInitialConditions(),
        infer_track_interior=True,
        sorted_spikes_algorithm="spiking_likelihood_kde",
        sorted_spikes_algorithm_params=_DEFAULT_SORTED_SPIKES_MODEL_KWARGS,
    ):
        super().__init__(
            environments,
            observation_models,
            continuous_transition_types,
            discrete_transition_type,
            initial_conditions_type,
            infer_track_interior,
        )
        self.sorted_spikes_algorithm = sorted_spikes_algorithm
        self.sorted_spikes_algorithm_params = sorted_spikes_algorithm_params

    def fit_place_fields(
        self,
        position,
        spikes,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
    ):
        logger.info("Fitting place fields...")
        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time
            )

        is_training = np.asarray(is_training).squeeze()

        kwargs = self.sorted_spikes_algorithm_params
        if kwargs is None:
            kwargs = {}

        self.place_fields_ = {}
        for obs in np.unique(self.observation_models):
            environment = self.environments[
                self.environments.index(obs.environment_name)
            ]

            is_encoding = np.isin(encoding_group_labels, obs.encoding_group)
            is_environment = environment_labels == obs.environment_name
            likelihood_name = (obs.environment_name, obs.encoding_group)

            self.place_fields_[likelihood_name] = _SORTED_SPIKES_ALGORITHMS[
                self.sorted_spikes_algorithm
            ][0](
                position=position[is_training & is_encoding & is_environment],
                spikes=spikes[is_training & is_encoding & is_environment],
                place_bin_centers=environment.place_bin_centers_,
                place_bin_edges=environment.place_bin_edges_,
                edges=environment.edges_,
                is_track_interior=environment.is_track_interior_,
                is_track_boundary=environment.is_track_boundary_,
                **kwargs,
            )

    def plot_place_fields(self, sampling_frequency=1, figsize=(10, 7)):
        """Plots all place fields.

        Parameters
        ----------
        sampling_frequency : int, optional
            samples per second, by default 1
        figsize : tuple, optional
            figure dimensions, by default (10, 7)
        """
        try:
            for (env, enc) in self.place_fields_:
                is_track_interior = self.environments[
                    self.environments.index(env)
                ].is_track_interior_[np.newaxis]
                (
                    (self.place_fields_[(env, enc)] * sampling_frequency)
                    .unstack("position")
                    .where(is_track_interior)
                    .plot(
                        x="x_position",
                        y="y_position",
                        col="neuron",
                        col_wrap=8,
                        vmin=0.0,
                        vmax=3.0,
                    )
                )
        except ValueError:
            n_enc_env = len(self.place_fields_)
            fig, axes = plt.subplots(
                n_enc_env, 1, constrained_layout=True, figsize=figsize
            )
            if n_enc_env == 1:
                axes = np.asarray([axes])
            for ax, ((env_name, enc_group), place_fields) in zip(
                axes.flat, self.place_fields_.items()
            ):
                is_track_interior = self.environments[
                    self.environments.index(env_name)
                ].is_track_interior_[:, np.newaxis]
                (
                    (place_fields * sampling_frequency)
                    .where(is_track_interior)
                    .plot(x="position", hue="neuron", add_legend=False, ax=ax)
                )
                ax.set_title(f"Environment = {env_name}, Encoding Group = {enc_group}")
                ax.set_ylabel("Firing Rate\n[spikes/s]")

    def fit(
        self,
        position,
        spikes,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
    ):
        """Fit the spatial grid, initial conditions, place field model, and
        transition matrices.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, n_position_dims)
        spikes : np.ndarray, shape (n_time, n_neurons)
        is_training : None or bool np.ndarray, shape (n_time), optional
            Time bins to be used for encoding.
        encoding_group_labels : None or np.ndarray, shape (n_time,)
            Label for the corresponding encoding group for each time point
        environment_labels : None or np.ndarray, shape (n_time,)
            Label for the corresponding environment for each time point

        Returns
        -------
        self

        """
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
        self.fit_place_fields(
            position, spikes, is_training, encoding_group_labels, environment_labels
        )

        return self

    def predict(
        self,
        spikes,
        time=None,
        is_compute_acausal=True,
        use_gpu=False,
        state_names=None,
        store_likelihood=False,
    ):
        """Run the state space model on spikes.

        Parameters
        ----------
        spikes : np.ndarray, shape (n_time, n_neurons)
        time : ndarray or None, shape (n_time,), optional
        is_compute_acausal : bool, optional
        use_gpu : bool, optional
            Use GPU for the state space part of the model, not the likelihood.
        state_names : None or array_like, shape (n_states,)

        Returns
        -------
        results : xarray.Dataset

        """
        spikes = np.asarray(spikes)
        n_time = spikes.shape[0]

        # likelihood
        logger.info("Estimating likelihood...")
        likelihood = {}
        for (env_name, enc_group), place_fields in self.place_fields_.items():
            env_ind = self.environments.index(env_name)
            is_track_interior = self.environments[env_ind].is_track_interior_.ravel(
                order="F"
            )
            likelihood[(env_name, enc_group)] = _SORTED_SPIKES_ALGORITHMS[
                self.sorted_spikes_algorithm
            ][1](spikes, place_fields.values, is_track_interior)
        if store_likelihood:
            self.likelihood_ = likelihood

        return self._get_results(
            likelihood, n_time, time, is_compute_acausal, use_gpu, state_names
        )


class ClusterlessClassifier(_ClassifierBase):
    """

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

    """

    def __init__(
        self,
        environments=_DEFAULT_ENVIRONMENT,
        observation_models=None,
        continuous_transition_types=_DEFAULT_CONTINUOUS_TRANSITIONS,
        discrete_transition_type=DiagonalDiscrete(0.98),
        initial_conditions_type=UniformInitialConditions(),
        infer_track_interior=True,
        clusterless_algorithm="multiunit_likelihood",
        clusterless_algorithm_params=_DEFAULT_CLUSTERLESS_MODEL_KWARGS,
    ):
        super().__init__(
            environments,
            observation_models,
            continuous_transition_types,
            discrete_transition_type,
            initial_conditions_type,
            infer_track_interior,
        )

        self.clusterless_algorithm = clusterless_algorithm
        self.clusterless_algorithm_params = clusterless_algorithm_params

    def fit_multiunits(
        self,
        position,
        multiunits,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
    ):
        """Fit the clusterless place field model.

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)
        encoding_group_labels : None or np.ndarray, shape (n_time,)
            Label for the corresponding encoding group for each time point
        environment_labels : None or np.ndarray, shape (n_time,)
            Label for the corresponding environment for each time point

        """
        logger.info("Fitting multiunits...")
        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time
            )

        is_training = np.asarray(is_training).squeeze()

        kwargs = self.clusterless_algorithm_params
        if kwargs is None:
            kwargs = {}

        self.encoding_model_ = {}

        for obs in np.unique(self.observation_models):
            environment = self.environments[
                self.environments.index(obs.environment_name)
            ]

            is_encoding = np.isin(encoding_group_labels, obs.encoding_group)
            is_environment = environment_labels == obs.environment_name
            is_group = is_training & is_encoding & is_environment

            likelihood_name = (obs.environment_name, obs.encoding_group)

            self.encoding_model_[likelihood_name] = _ClUSTERLESS_ALGORITHMS[
                self.clusterless_algorithm
            ][0](
                position=position[is_group],
                multiunits=multiunits[is_group],
                place_bin_centers=environment.place_bin_centers_,
                is_track_interior=environment.is_track_interior_,
                is_track_boundary=environment.is_track_boundary_,
                edges=environment.edges_,
                **kwargs,
            )

    def fit(
        self,
        position,
        multiunits,
        is_training=None,
        encoding_group_labels=None,
        environment_labels=None,
    ):
        """Fit the spatial grid, initial conditions, place field model, and
        transition matrices.

        Parameters
        ----------
        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)
        encoding_group_labels : None or np.ndarray, shape (n_time,)
            Label for the corresponding encoding group for each time point
        environment_labels : None or np.ndarray, shape (n_time,)
            Label for the corresponding environment for each time point

        Returns
        -------
        self

        """
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
        self.fit_multiunits(
            position, multiunits, is_training, encoding_group_labels, environment_labels
        )

        return self

    def predict(
        self,
        multiunits,
        time=None,
        is_compute_acausal=True,
        use_gpu=False,
        state_names=None,
        store_likelihood=False,
    ):
        """Run the state space model.

        Parameters
        ----------
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        time : None or np.ndarray, shape (n_time,)
        is_compute_acausal : bool, optional
            Use future information to compute the posterior.
        use_gpu : bool, optional
            Use GPU for the state space part of the model, not the likelihood.
        state_names : None or array_like, shape (n_states,)
        store_likelihood : bool, optional
            Save the likelihood for EM computations

        Returns
        -------
        results : xarray.Dataset

        """
        multiunits = np.asarray(multiunits)
        n_time = multiunits.shape[0]

        logger.info("Estimating likelihood...")
        likelihood = {}
        for (env_name, enc_group), encoding_params in self.encoding_model_.items():
            env_ind = self.environments.index(env_name)
            is_track_interior = self.environments[env_ind].is_track_interior_.ravel(
                order="F"
            )
            place_bin_centers = self.environments[env_ind].place_bin_centers_
            likelihood[(env_name, enc_group)] = _ClUSTERLESS_ALGORITHMS[
                self.clusterless_algorithm
            ][1](
                multiunits=multiunits,
                place_bin_centers=place_bin_centers,
                is_track_interior=is_track_interior,
                **encoding_params,
            )
        if store_likelihood:
            self.likelihood_ = likelihood

        return self._get_results(
            likelihood, n_time, time, is_compute_acausal, use_gpu, state_names
        )
