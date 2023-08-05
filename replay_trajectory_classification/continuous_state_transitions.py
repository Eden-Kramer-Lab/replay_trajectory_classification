"""Classes for constructing different types of movement models."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal

from replay_trajectory_classification.core import atleast_2d
from replay_trajectory_classification.environments import Environment, diffuse_each_bin


def _normalize_row_probability(x: np.ndarray) -> np.ndarray:
    """Ensure the state transition matrix rows sum to 1.

    Parameters
    ----------
    x : np.ndarray, shape (n_rows, n_cols)

    Returns
    -------
    normalized_x : np.ndarray, shape (n_rows, n_cols)

    """
    x /= x.sum(axis=1, keepdims=True)
    x[np.isnan(x)] = 0
    return x


def estimate_movement_var(
    position: np.ndarray, sampling_frequency: int = 1
) -> np.ndarray:
    """Estimates the movement variance based on position.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dim)
        Position of the animal
    sampling_frequency : int, optional
        Number of samples per second.

    Returns
    -------
    movement_var : np.ndarray, shape (n_position_dim,)
        Variance of the movement.

    """
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    movement_var = np.cov(np.diff(position, axis=0), rowvar=False)
    return movement_var * sampling_frequency


def _random_walk_on_track_graph(
    place_bin_centers: np.ndarray,
    movement_mean: float,
    movement_var: float,
    place_bin_center_ind_to_node: np.ndarray,
    distance_between_nodes: dict[dict],
) -> np.ndarray:
    """Estimates the random walk probabilities based on the spatial
    topology given by the track graph.

    Parameters
    ----------
    place_bin_centers : np.ndarray, shape (n_position_bins,)
    movement_mean : float
    movement_var : float
    place_bin_center_ind_to_node : np.ndarray
        Mapping of place bin center to track graph node
    distance_between_nodes : dict[dict]
        Distance between each pair of track graph nodes with an edge.

    Returns
    -------
    random_walk : np.ndarray, shape (n_position_bins, n_position_bins)

    """
    state_transition = np.zeros((place_bin_centers.size, place_bin_centers.size))
    gaussian = multivariate_normal(mean=movement_mean, cov=movement_var)

    for bin_ind1, node1 in enumerate(place_bin_center_ind_to_node):
        for bin_ind2, node2 in enumerate(place_bin_center_ind_to_node):
            try:
                state_transition[bin_ind1, bin_ind2] = gaussian.pdf(
                    distance_between_nodes[node1][node2]
                )
            except KeyError:
                # bins not on track interior will be -1 and not in distance
                # between nodes
                continue

    return state_transition


@dataclass
class RandomWalk:
    """A transition where the movement stays locally close in space

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    movement_mean : float, optional
    use_diffusion : bool, optional
        Use diffusion to respect the geometry of the environment
    """

    environment_name: str = ""
    movement_var: float = 6.0
    movement_mean: float = 0.0
    use_diffusion: bool = False

    def make_state_transition(self, environments: tuple[Environment]) -> np.ndarray:
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]

        if self.environment.track_graph is None:
            n_position_dims = self.environment.place_bin_centers_.shape[1]

            if (n_position_dims == 1) or not self.use_diffusion:
                transition_matrix = np.stack(
                    [
                        multivariate_normal(
                            mean=center + self.movement_mean, cov=self.movement_var
                        ).pdf(self.environment.place_bin_centers_)
                        for center in self.environment.place_bin_centers_
                    ],
                    axis=1,
                )
            else:
                dx = self.environment.edges_[0][1] - self.environment.edges_[0][0]
                dy = self.environment.edges_[1][1] - self.environment.edges_[1][0]
                n_total_bins = np.prod(self.environment.is_track_interior_.shape)
                transition_matrix = diffuse_each_bin(
                    self.environment.is_track_interior_,
                    self.environment.is_track_boundary_,
                    dx,
                    dy,
                    std=np.sqrt(self.movement_var),
                ).reshape((n_total_bins, -1), order="F")
        else:
            place_bin_center_ind_to_node = np.asarray(
                self.environment.place_bin_centers_nodes_df_.node_id
            )
            transition_matrix = _random_walk_on_track_graph(
                self.environment.place_bin_centers_,
                self.movement_mean,
                self.movement_var,
                place_bin_center_ind_to_node,
                self.environment.distance_between_nodes_,
            )

        is_track_interior = self.environment.is_track_interior_.ravel(order="F")
        transition_matrix[~is_track_interior] = 0.0
        transition_matrix[:, ~is_track_interior] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class Uniform:
    """
    Attributes
    ----------
    environment_name : str, optional
        Name of first environment to fit
    environment_name2 : str, optional
        Name of second environment to fit if going from one environment to
        another
    """

    environment_name: str = ""
    environment2_name: str = None

    def make_state_transition(self, environments: tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment1 = environments[environments.index(self.environment_name)]
        n_bins1 = self.environment1.place_bin_centers_.shape[0]
        is_track_interior1 = self.environment1.is_track_interior_.ravel(order="F")

        if self.environment2_name is None:
            n_bins2 = n_bins1
            is_track_interior2 = is_track_interior1.copy()
        else:
            self.environment2 = environments[environments.index(self.environment2_name)]
            n_bins2 = self.environment2.place_bin_centers_.shape[0]
            is_track_interior2 = self.environment2.is_track_interior_.ravel(order="F")

        transition_matrix = np.ones((n_bins1, n_bins2))

        transition_matrix[~is_track_interior1] = 0.0
        transition_matrix[:, ~is_track_interior2] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class Identity:
    """A transition where the movement stays within a place bin

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    """

    environment_name: str = ""

    def make_state_transition(self, environments: tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]
        n_bins = self.environment.place_bin_centers_.shape[0]

        transition_matrix = np.identity(n_bins)

        is_track_interior = self.environment.is_track_interior_.ravel(order="F")
        transition_matrix[~is_track_interior] = 0.0
        transition_matrix[:, ~is_track_interior] = 0.0

        return _normalize_row_probability(transition_matrix)


@dataclass
class EmpiricalMovement:
    """A transition matrix trained on the animal's actual movement

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    encoding_group : str, optional
        Name of encoding group to fit
    speedup : int, optional
        Used to make the empirical transition matrix "faster", means allowing for
        all the same transitions made by the animal but sped up by
        `speedup` times. So `speedup​=20` means 20x faster than the
        animal's movement.
    """

    environment_name: str = ""
    encoding_group: str = None
    speedup: int = 1

    def make_state_transition(
        self,
        environments: tuple[Environment],
        position: np.ndarray,
        is_training: np.ndarray = None,
        encoding_group_labels: np.ndarray = None,
        environment_labels: np.ndarray = None,
    ):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model
        position : np.ndarray
            Position of the animal
        is_training : np.ndarray, optional
            Boolean array that determines what data to train the place fields on, by default None
        encoding_group_labels : np.ndarray, shape (n_time,), optional
            If place fields should correspond to each state, label each time point with the group name
            For example, Some points could correspond to inbound trajectories and some outbound, by default None
        environment_labels : np.ndarray, shape (n_time,), optional
            If there are multiple environments, label each time point with the environment name, by default None

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]

        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=bool)

        if encoding_group_labels is None:
            is_encoding = np.ones((n_time,), dtype=bool)
        else:
            is_encoding = encoding_group_labels == self.encoding_group

        if environment_labels is None:
            is_environment = np.ones((n_time,), dtype=bool)
        else:
            is_environment = environment_labels == self.environment_name

        position = atleast_2d(position)[is_training & is_encoding & is_environment]
        state_transition, _ = np.histogramdd(
            np.concatenate((position[1:], position[:-1]), axis=1),
            bins=self.environment.edges_ * 2,
            range=self.environment.position_range,
        )
        original_shape = state_transition.shape
        n_position_dims = position.shape[1]
        shape_2d = np.product(original_shape[:n_position_dims])
        state_transition = _normalize_row_probability(
            state_transition.reshape((shape_2d, shape_2d), order="F")
        )

        return np.linalg.matrix_power(state_transition, self.speedup)


@dataclass
class RandomWalkDirection1:
    """A Gaussian random walk in that can only go one direction

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    """

    environment_name: str = ""
    movement_var: float = 6.0

    def make_state_transition(self, environments: tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]
        random = RandomWalk(
            self.environment_name, self.movement_var
        ).make_state_transition(environments)

        return _normalize_row_probability(np.triu(random))


@dataclass
class RandomWalkDirection2:
    """A Gaussian random walk in that can only go one direction

    Attributes
    ----------
    environment_name : str, optional
        Name of environment to fit
    movement_var : float, optional
        How far the animal is can move in one time bin during normal
        movement.
    """

    environment_name: str = ""
    movement_var: float = 6.0

    def make_state_transition(self, environments: tuple[Environment]):
        """Creates a transition matrix for a given environment.

        Parameters
        ----------
        environments : tuple[Environment]
            The existing environments in the model

        Returns
        -------
        state_transition_matrix : np.ndarray, shape (n_position_bins, n_position_bins)

        """
        self.environment = environments[environments.index(self.environment_name)]
        random = RandomWalk(
            self.environment_name, self.movement_var
        ).make_state_transition(environments)

        return _normalize_row_probability(np.tril(random))
