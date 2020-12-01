import numpy as np
from replay_trajectory_classification.bins import atleast_2d
from scipy.stats import multivariate_normal


def _normalize_row_probability(x):
    '''Ensure the state transition matrix rows sum to 1
    '''
    x /= x.sum(axis=1, keepdims=True)
    x[np.isnan(x)] = 0
    return x


def empirical_movement(
    place_bin_centers, is_track_interior, position, edges,
        is_training, replay_speed, position_extent, movement_var,
        place_bin_center_ind_to_node,
        distance_between_nodes):
    '''Estimate the probablity of the next position based on the movement
     data, given the movment is sped up by the
     `replay_speed`

    Place cell firing during a hippocampal replay event is a "sped-up"
    version of place cell firing when the animal is actually moving.
    Here we use the animal's actual movements to constrain which place
    cell is likely to fire next.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    edges : sequence
        A sequence of arrays describing the bin edges along each dimension.
    is_training : None or bool ndarray, shape (n_time,), optional
    replay_speed : int, optional
        How much the movement is sped-up during a replay event
    position_extent : sequence, optional
        A sequence of `n_position_dims`, each an optional (lower, upper)
        tuple giving the outer bin edges for position.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of
        `n_position_dims` None values.

    Returns
    -------
    transition_matrix : ndarray, shape (n_position_bins, n_position_bins)

    '''
    if is_training is None:
        is_training = np.ones((position.shape[0]), dtype=np.bool)
    position = atleast_2d(position)[is_training]
    movement_bins, _ = np.histogramdd(
        np.concatenate((position[1:], position[:-1]), axis=1),
        bins=edges * 2, range=position_extent)
    original_shape = movement_bins.shape
    n_position_dims = position.shape[1]
    shape_2d = np.product(original_shape[:n_position_dims])
    movement_bins = _normalize_row_probability(
        movement_bins.reshape((shape_2d, shape_2d), order='F'))
    movement_bins = np.linalg.matrix_power(movement_bins, replay_speed)

    return movement_bins


def random_walk(place_bin_centers, is_track_interior, position, edges,
                is_training, replay_speed, position_extent, movement_var,
                place_bin_center_ind_to_node,
                distance_between_nodes):
    '''Zero mean random walk.

    This version makes the sped up gaussian without constraints and then
    imposes the constraints.

    Parameters
    ----------
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    movement_var : float,
    is_track_interior : bool ndarray, shape (n_x_bins, n_y_bins)
    replay_speed : int

    Returns
    -------
    transition_matrix : ndarray, shape (n_bins, n_bins)

    '''
    if place_bin_center_ind_to_node is None:
        transition_matrix = np.stack(
            [multivariate_normal(
                mean=bin, cov=movement_var * replay_speed).pdf(
                place_bin_centers)
             for bin in place_bin_centers], axis=1)
    else:
        transition_matrix = _random_walk_on_track_graph(
            place_bin_centers, is_track_interior, position, edges,
            is_training, replay_speed, position_extent, movement_var,
            place_bin_center_ind_to_node,
            distance_between_nodes,
        )
    is_track_interior = is_track_interior.ravel(order='F')
    transition_matrix[~is_track_interior] = 0.0
    transition_matrix[:, ~is_track_interior] = 0.0

    return _normalize_row_probability(transition_matrix)


def _random_walk_on_track_graph(
    place_bin_centers, is_track_interior, position, edges,
    is_training, replay_speed, position_extent, movement_var,
    place_bin_center_ind_to_node,
    distance_between_nodes
):
    """

    Parameters
    ----------
    node_linear_position : np.ndarray, shape (n_nodes,)
    place_bin_centers : np.ndarray, shape (n_bins)
    is_track_interior : np.ndarray, shape (n_bins)
    distance_between_nodes : dict of dicts, shape (n_nodes, n_nodes)
    bin_to_node : tuple, shape (n_nodes,)
    movement_var : float, optional
    replay_speed : float, optional

    Returns
    -------
    state_transition : np.ndarray, shape (n_bins, n_bins)
    bin_to_node : np.array, shape (n_bins)

    """
    state_transition = np.zeros(
        (place_bin_centers.size, place_bin_centers.size))
    gaussian = multivariate_normal(mean=0, cov=movement_var * replay_speed)

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


def random_walk_direction1(*args):
    '''Zero mean random walk.

    This version makes the sped up gaussian without constraints and then
    imposes the constraints.

    Parameters
    ----------
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    movement_var : float,
    is_track_interior : bool ndarray, shape (n_x_bins, n_y_bins)
    replay_speed : int

    Returns
    -------
    transition_matrix : ndarray, shape (n_bins, n_bins)

    '''
    random = random_walk(*args)
    return _normalize_row_probability(np.triu(random))


def random_walk_direction2(*args):
    '''Zero mean random walk.

    This version makes the sped up gaussian without constraints and then
    imposes the constraints.

    Parameters
    ----------
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    movement_var : float,
    is_track_interior : bool ndarray, shape (n_x_bins, n_y_bins)
    replay_speed : int

    Returns
    -------
    transition_matrix : ndarray, shape (n_bins, n_bins)

    '''
    random = random_walk(*args)
    return _normalize_row_probability(np.tril(random))


def uniform_state_transition(
    place_bin_centers, is_track_interior, position, edges,
        is_training, replay_speed, position_extent, movement_var,
        place_bin_center_ind_to_node,
        distance_between_nodes):
    '''Equally likely to go somewhere on the track.

    Parameters
    ----------
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    is_track_interior : bool ndarray, shape (n_x_bins, n_y_bins)

    Returns
    -------
    transition_matrix : ndarray, shape (n_bins, n_bins)

    '''
    n_bins = place_bin_centers.shape[0]
    transition_matrix = np.ones((n_bins, n_bins))

    is_track_interior = is_track_interior.ravel(order='F')
    transition_matrix[~is_track_interior] = 0.0
    transition_matrix[:, ~is_track_interior] = 0.0

    return _normalize_row_probability(transition_matrix)


def identity(place_bin_centers, is_track_interior, position, edges,
             is_training, replay_speed, position_extent, movement_var,
             place_bin_center_ind_to_node,
             distance_between_nodes):
    '''Stay in one place on the track.

    Parameters
    ----------
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    is_track_interior : bool ndarray, shape (n_x_bins, n_y_bins)

    Returns
    -------
    transition_matrix : ndarray, shape (n_bins, n_bins)

    '''
    n_bins = place_bin_centers.shape[0]
    transition_matrix = np.identity(n_bins)

    is_track_interior = is_track_interior.ravel(order='F')
    transition_matrix[~is_track_interior] = 0.0
    transition_matrix[:, ~is_track_interior] = 0.0

    return _normalize_row_probability(transition_matrix)


def uniform_minus_empirical(*args):
    uniform = uniform_state_transition(*args)
    empirical = empirical_movement(*args)
    difference = uniform - empirical
    difference[difference < 0] = 0.0
    return _normalize_row_probability(difference)


def uniform_minus_random_walk(*args):
    uniform = uniform_state_transition(*args)
    random = random_walk(*args)
    difference = uniform - random
    difference[difference < 0] = 0.0
    return _normalize_row_probability(difference)


def empirical_minus_identity(*args):
    empirical = empirical_movement(*args)
    ident = identity(*args)
    difference = empirical - ident
    difference[difference < 0] = 0.0
    return _normalize_row_probability(difference)


def random_walk_minus_identity(*args):
    random = random_walk(*args)
    ident = identity(*args)
    difference = random - ident
    difference[difference < 0] = 0.0
    return _normalize_row_probability(difference)


def inverse_random_walk(
    place_bin_centers, is_track_interior, position, edges,
        is_training, replay_speed, position_extent, movement_var,
        place_bin_center_ind_to_node,
        distance_between_nodes):
    rw = random_walk(
        place_bin_centers, is_track_interior, position, edges,
        is_training, replay_speed, position_extent, movement_var,
        place_bin_center_ind_to_node,
        distance_between_nodes)
    transition_matrix = rw.max(axis=1, keepdims=True) - rw

    is_track_interior = is_track_interior.ravel(order='F')
    transition_matrix[~is_track_interior] = 0.0
    transition_matrix[:, ~is_track_interior] = 0.0
    return _normalize_row_probability(transition_matrix)


def random_walk_plus_uniform(
    place_bin_centers, is_track_interior, position, edges,
        is_training, replay_speed, position_extent, movement_var,
        place_bin_center_ind_to_node,
        distance_between_nodes):
    random = random_walk(
        place_bin_centers, is_track_interior, position, edges,
        is_training, replay_speed, position_extent, movement_var,
        place_bin_center_ind_to_node,
        distance_between_nodes)
    uniform = uniform_state_transition(
        place_bin_centers, is_track_interior, position, edges,
        is_training, replay_speed, position_extent, movement_var,
        place_bin_center_ind_to_node,
        distance_between_nodes)
    transition_matrix = uniform + random
    is_track_interior = is_track_interior.ravel(order='F')
    transition_matrix[~is_track_interior] = 0.0
    transition_matrix[:, ~is_track_interior] = 0.0

    return _normalize_row_probability(transition_matrix)


def identity_discrete(n_states, diag):
    '''

    Parameters
    ----------
    n_states : int

    Returns
    -------
    transition_matrix : ndarray, shape (n_states, n_states)

    '''
    return np.identity(n_states)


def strong_diagonal_discrete(n_states, diag):
    '''

    Parameters
    ----------
    n_states : int
    diag : float

    Returns
    -------
    transition_matrix : ndarray, shape (n_states, n_states)

    '''
    strong_diagonal = np.identity(n_states) * diag
    is_off_diag = ~np.identity(n_states, dtype=bool)
    strong_diagonal[is_off_diag] = (
        (1 - diag) / (n_states - 1))
    return strong_diagonal


def uniform_discrete(n_states, diag):
    '''

    Parameters
    ----------
    n_states : int

    Returns
    -------
    transition_matrix : ndarray, shape (n_states, n_states)

    '''
    return np.ones((n_states, n_states)) / n_states


def estimate_movement_var(position, sampling_frequency):
    '''Estimates the movement variance based on position.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dim)

    Returns
    -------
    movement_std : ndarray, shape (n_position_dim,)

    '''
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    movement_var = np.cov(np.diff(position, axis=0), rowvar=False)
    return movement_var * sampling_frequency


CONTINUOUS_TRANSITIONS = {
    'empirical_movement': empirical_movement,
    'random_walk': random_walk,
    'uniform': uniform_state_transition,
    'identity': identity,
    'uniform_minus_empirical': uniform_minus_empirical,
    'uniform_minus_random_walk': uniform_minus_random_walk,
    'empirical_minus_identity': empirical_minus_identity,
    'random_walk_minus_identity': random_walk_minus_identity,
    'random_walk_plus_uniform': random_walk_plus_uniform,
    'inverse_random_walk': inverse_random_walk,
    'random_walk_direction1': random_walk_direction1,
    'random_walk_direction2': random_walk_direction2,
}

DISCRETE_TRANSITIONS = {
    'strong_diagonal': strong_diagonal_discrete,
    'identity': identity_discrete,
    'uniform': uniform_discrete,
}
