from dataclasses import dataclass

import numpy as np
from replay_trajectory_classification.state_transition import (
    _normalize_row_probability, atleast_2d)
from scipy.stats import multivariate_normal


def _random_walk_on_track_graph(
    place_bin_centers, movement_mean, movement_var,
    place_bin_center_ind_to_node,
    distance_between_nodes
):
    state_transition = np.zeros(
        (place_bin_centers.size, place_bin_centers.size))
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
    """
    environment_name: str = ''
    movement_var: float = 6.0
    movement_mean: float = 0.0

    def make_state_transition(self, environments: tuple):
        self.environment = environments[
            environments.index(self.environment_name)]

        if self.environment.track_graph is None:
            transition_matrix = np.stack(
                [multivariate_normal(
                    mean=center + self.movement_mean,
                    cov=self.movement_var
                ).pdf(self.environment.place_bin_centers_)
                    for center in self.environment.place_bin_centers_], axis=1)
        else:
            place_bin_center_ind_to_node = np.asarray(
                self.environment.place_bin_centers_nodes_df_.node_id)
            transition_matrix = _random_walk_on_track_graph(
                self.environment.place_bin_centers_,
                self.movement_mean,
                self.movement_var,
                place_bin_center_ind_to_node,
                self.environment.distance_between_nodes_
            )

        is_track_interior = self.environment.is_track_interior_.ravel(
            order='F')
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
    environment_name: str = ''
    environment2_name: str = None

    def make_state_transition(self, environments: tuple):
        self.environment1 = environments[
            environments.index(self.environment_name)]
        n_bins1 = self.environment1.place_bin_centers_.shape[0]
        is_track_interior1 = self.environment1.is_track_interior_.ravel(
            order='F')

        if self.environment2_name is None:
            n_bins2 = n_bins1
            is_track_interior2 = is_track_interior1.copy()
        else:
            self.environment2 = environments[
                environments.index(self.environment2_name)]
            n_bins2 = self.environment2.place_bin_centers_.shape[0]
            is_track_interior2 = self.environment2.is_track_interior_.ravel(
                order='F')

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
    environment_name: str = ''

    def make_state_transition(self, environments: tuple):
        self.environment = environments[
            environments.index(self.environment_name)]
        n_bins = self.environment.place_bin_centers_.shape[0]

        transition_matrix = np.identity(n_bins)

        is_track_interior = self.environment.is_track_interior_.ravel(
            order='F')
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
        Used to
        make the empirical transition matrix "faster", means allowing for
        all the same transitions made by the animal but sped up by
        `speedup` times. So `speedup​=20` means 20x faster than the
        animal's movement.
    """
    environment_name: str = ''
    encoding_group: str = None
    speedup: int = 1

    def make_state_transition(self,
                              environments: tuple,
                              position: np.ndarray,
                              is_training: np.ndarray = None,
                              encoding_group_labels: np.ndarray = None,
                              environment_labels: np.ndarray = None):

        self.environment = environments[
            environments.index(self.environment_name)]

        n_time = position.shape[0]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=np.bool)

        if encoding_group_labels is None:
            is_encoding = np.ones((n_time,), dtype=np.bool)
        else:
            is_encoding = (encoding_group_labels == self.encoding_group)

        if environment_labels is None:
            is_environment = np.ones((n_time,), dtype=np.bool)
        else:
            is_environment = (environment_labels == self.environment_name)

        position = atleast_2d(position)[
            is_training & is_encoding & is_environment]
        state_transition, _ = np.histogramdd(
            np.concatenate((position[1:], position[:-1]), axis=1),
            bins=self.environment.edges_ * 2,
            range=self.environment.position_range)
        original_shape = state_transition.shape
        n_position_dims = position.shape[1]
        shape_2d = np.product(original_shape[:n_position_dims])
        state_transition = _normalize_row_probability(
            state_transition.reshape((shape_2d, shape_2d), order='F'))

        return np.linalg.matrix_power(
            state_transition, self.speedup)


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
    environment_name: str = ''
    movement_var: float = 6.0

    def make_state_transition(self, environments: tuple):
        self.environment = environments[
            environments.index(self.environment_name)]
        random = (RandomWalk(self.environment_name, self.movement_var)
                  .make_state_transition(environments))

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
    environment_name: str = ''
    movement_var: float = 6.0

    def make_state_transition(self, environments: tuple):
        self.environment = environments[
            environments.index(self.environment_name)]
        random = (RandomWalk(self.environment_name, self.movement_var)
                  .make_state_transition(environments))

        return _normalize_row_probability(np.tril(random))
