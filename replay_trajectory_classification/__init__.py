# flake8: noqa
from track_linearization import (
    get_linearized_position,
    make_actual_vs_linearized_position_movie,
    make_track_graph,
    plot_graph_as_1D,
    plot_track_graph,
)
from trajectory_analysis_tools.distance1D import (
    get_ahead_behind_distance,
    get_map_speed,
    get_trajectory_data,
)
from trajectory_analysis_tools.distance2D import (
    get_2D_distance,
    get_ahead_behind_distance2D,
    get_map_estimate_direction_from_track_graph,
    get_speed,
    get_velocity,
    head_direction_simliarity,
    make_2D_track_graph_from_environment,
)
from trajectory_analysis_tools.highest_posterior_density import (
    get_highest_posterior_threshold,
    get_HPD_spatial_coverage,
)
from trajectory_analysis_tools.posterior import (
    maximum_a_posteriori_estimate,
    sample_posterior,
)

from replay_trajectory_classification.classifier import (
    ClusterlessClassifier,
    SortedSpikesClassifier,
)
from replay_trajectory_classification.continuous_state_transitions import (
    EmpiricalMovement,
    Identity,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
    estimate_movement_var,
)
from replay_trajectory_classification.decoder import (
    ClusterlessDecoder,
    SortedSpikesDecoder,
)
from replay_trajectory_classification.discrete_state_transitions import (
    DiagonalDiscrete,
    RandomDiscrete,
    UniformDiscrete,
)
from replay_trajectory_classification.environments import Environment
from replay_trajectory_classification.initial_conditions import UniformInitialConditions
from replay_trajectory_classification.observation_model import ObservationModel

__version__ = "1.3.8"
