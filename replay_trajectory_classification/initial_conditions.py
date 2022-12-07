"""Classes for setting the initial conditions for the state space models."""

from dataclasses import dataclass

import numpy as np


@dataclass
class UniformInitialConditions:
    """Initial conditions where all discrete states and position bins are
    equally likely."""

    def make_initial_conditions(
        self, environments: tuple, environment_names_to_state: tuple
    ):
        n_total_place_bins = 0
        initial_conditions = []

        for environment_name in environment_names_to_state:
            is_track_interior = environments[
                environments.index(environment_name)
            ].is_track_interior_.ravel(order="F")
            n_total_place_bins += is_track_interior.sum()
            initial_conditions.append(is_track_interior)

        return [ic / n_total_place_bins for ic in initial_conditions]


@dataclass
class UniformOneEnvironmentInitialConditions:
    """Initial conditions where all position bins are
    equally likely for one environment and zero for other environments."""

    environment_name: str = ""

    def make_initial_conditions(
        self, environments: tuple, environment_names_to_state: tuple
    ):
        n_total_place_bins = 0
        initial_conditions = []

        for environment_name in environment_names_to_state:
            is_track_interior = environments[
                environments.index(environment_name)
            ].is_track_interior_.ravel(order="F")

            if self.environment_name == environment_name:
                initial_conditions.append(is_track_interior)
                n_total_place_bins += is_track_interior.sum()
            else:
                initial_conditions.append(np.zeros_like(is_track_interior))

        return [ic / n_total_place_bins for ic in initial_conditions]
