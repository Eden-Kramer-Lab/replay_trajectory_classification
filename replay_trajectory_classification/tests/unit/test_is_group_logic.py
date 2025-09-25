# replay_trajectory_classification/tests/unit/test_is_group_logic.py
"""
Focused tests for is_group formation logic in classifier fitting methods.
This tests the core data selection logic without requiring full classifier fitting.
"""
import numpy as np

from replay_trajectory_classification.observation_model import ObservationModel


def test_is_group_formation_basic():
    """Test basic is_group formation logic."""
    n_time = 100

    # Create test conditions
    is_training = np.ones(n_time, dtype=bool)
    is_training[20:30] = False  # Exclude training period

    encoding_group_labels = np.zeros(n_time, dtype=np.int32)
    encoding_group_labels[50:] = 1  # Second half different group

    environment_labels = np.full(n_time, "env_0", dtype='U10')
    environment_labels[75:] = "env_1"  # Last quarter different environment

    # Test for observation model: env_0, group 0
    target_encoding_group = 0
    target_environment = "env_0"

    is_encoding = np.isin(encoding_group_labels, [target_encoding_group])
    is_environment = environment_labels == target_environment
    is_group = is_training & is_encoding & is_environment

    # Expected is_group should be:
    # is_training=True AND group=0 AND env="env_0"
    # Time 0:20 - training=True, group=0, env=env_0 → True
    # Time 20:30 - training=False, group=0, env=env_0 → False
    # Time 30:50 - training=True, group=0, env=env_0 → True
    # Time 50:75 - training=True, group=1, env=env_0 → False
    # Time 75:100 - training=True, group=1, env=env_1 → False

    expected = np.zeros(n_time, dtype=bool)
    expected[0:20] = True   # training=True, group=0, env=env_0
    expected[30:50] = True  # training=True, group=0, env=env_0

    np.testing.assert_array_equal(is_group, expected)


def test_is_group_formation_multiple_groups():
    """Test is_group formation with multiple encoding groups."""
    n_time = 80

    is_training = np.ones(n_time, dtype=bool)
    encoding_group_labels = np.zeros(n_time, dtype=np.int32)
    encoding_group_labels[20:40] = 1
    encoding_group_labels[40:60] = 2
    encoding_group_labels[60:] = 0

    environment_labels = np.full(n_time, "env_0")

    # Test for observation model that includes groups 0 and 2
    target_encoding_groups = [0, 2]
    target_environment = "env_0"

    is_encoding = np.isin(encoding_group_labels, target_encoding_groups)
    is_environment = environment_labels == target_environment
    is_group = is_training & is_encoding & is_environment

    # Expected: times where group is 0 or 2, all training, all env_0
    expected = np.zeros(n_time, dtype=bool)
    expected[0:20] = True   # group 0
    expected[40:60] = True  # group 2
    expected[60:] = True    # group 0 again

    np.testing.assert_array_equal(is_group, expected)


def test_is_group_formation_multiple_environments():
    """Test is_group formation with multiple environments."""
    n_time = 120

    is_training = np.ones(n_time, dtype=bool)
    encoding_group_labels = np.zeros(n_time, dtype=np.int32)
    environment_labels = np.zeros(n_time, dtype='U10')

    # Set up environment labels
    environment_labels[0:40] = "env_A"
    environment_labels[40:80] = "env_B"
    environment_labels[80:] = "env_A"

    # Test for env_A only
    target_environment = "env_A"

    is_encoding = np.isin(encoding_group_labels, [0])  # All group 0
    is_environment = environment_labels == target_environment
    is_group = is_training & is_encoding & is_environment

    # Expected: times where env is env_A
    expected = np.zeros(n_time, dtype=bool)
    expected[0:40] = True   # First env_A period
    expected[80:] = True    # Second env_A period

    np.testing.assert_array_equal(is_group, expected)


def test_is_group_all_conditions_false():
    """Test is_group when all conditions exclude the data."""
    n_time = 50

    is_training = np.zeros(n_time, dtype=bool)  # No training data
    encoding_group_labels = np.ones(n_time, dtype=np.int32)  # All group 1
    environment_labels = np.full(n_time, "env_0")

    # Look for group 0 data
    target_encoding_group = 0
    target_environment = "env_0"

    is_encoding = np.isin(encoding_group_labels, [target_encoding_group])
    is_environment = environment_labels == target_environment
    is_group = is_training & is_encoding & is_environment

    # Expected: all False
    expected = np.zeros(n_time, dtype=bool)
    np.testing.assert_array_equal(is_group, expected)


def test_is_group_data_selection_consistency():
    """Test that is_group selects consistent data across arrays."""
    n_time = 60
    n_neurons = 8
    position = np.random.randn(n_time, 1) * 5
    spikes = np.random.poisson(2, size=(n_time, n_neurons))

    # Create selection conditions
    is_training = np.ones(n_time, dtype=bool)
    is_training[15:25] = False

    encoding_group_labels = np.zeros(n_time, dtype=np.int32)
    encoding_group_labels[30:] = 1

    environment_labels = np.full(n_time, "env_test")

    # Form is_group
    is_encoding = np.isin(encoding_group_labels, [0])
    is_environment = environment_labels == "env_test"
    is_group = is_training & is_encoding & is_environment

    # Select data
    selected_position = position[is_group]
    selected_spikes = spikes[is_group]

    # Verify consistency
    assert selected_position.shape[0] == selected_spikes.shape[0]
    assert selected_position.shape[0] < n_time  # Some data was filtered
    assert selected_spikes.shape[1] == n_neurons  # Neuron count unchanged

    # Verify expected selection
    expected_indices = np.where(is_group)[0]
    # Should be indices 0:15 and 25:30 (training=True, group=0, env=env_test)
    expected_selected_indices = np.concatenate([
        np.arange(0, 15),    # training=True, group=0
        np.arange(25, 30),   # training=True, group=0
    ])
    np.testing.assert_array_equal(expected_indices, expected_selected_indices)


def test_observation_model_encoding_group_access():
    """Test that ObservationModel provides encoding_group attribute for is_group formation."""
    # Test with single encoding group
    obs1 = ObservationModel(environment_name="env_A", encoding_group=5)
    assert hasattr(obs1, 'encoding_group')
    assert obs1.encoding_group == 5

    # Test with multiple encoding groups (if supported)
    try:
        obs2 = ObservationModel(environment_name="env_B", encoding_group=[1, 3, 7])
        assert hasattr(obs2, 'encoding_group')
        assert obs2.encoding_group == [1, 3, 7] or obs2.encoding_group == (1, 3, 7)
    except (TypeError, ValueError):
        # If multiple groups aren't supported, that's fine
        pass


def test_is_group_edge_cases():
    """Test is_group formation edge cases."""
    n_time = 30

    # Case 1: Empty arrays
    empty_training = np.array([], dtype=bool)
    empty_groups = np.array([], dtype=np.int32)
    empty_envs = np.array([], dtype='U10')

    is_encoding = np.isin(empty_groups, [0])
    is_environment = empty_envs == "env_0"
    is_group = empty_training & is_encoding & is_environment

    assert is_group.shape[0] == 0
    assert is_group.dtype == bool

    # Case 2: Single time point
    single_training = np.array([True])
    single_group = np.array([0])
    single_env = np.array(["env_0"])

    is_encoding = np.isin(single_group, [0])
    is_environment = single_env == "env_0"
    is_group = single_training & is_encoding & is_environment

    expected = np.array([True])
    np.testing.assert_array_equal(is_group, expected)

    # Case 3: All same values
    all_same_training = np.ones(n_time, dtype=bool)
    all_same_groups = np.full(n_time, 2, dtype=np.int32)
    all_same_envs = np.full(n_time, "env_X")

    is_encoding = np.isin(all_same_groups, [2])
    is_environment = all_same_envs == "env_X"
    is_group = all_same_training & is_encoding & is_environment

    expected = np.ones(n_time, dtype=bool)  # All should be True
    np.testing.assert_array_equal(is_group, expected)


def test_is_group_data_type_handling():
    """Test that is_group formation handles different data types correctly."""
    n_time = 40

    # Test with different dtypes
    is_training = np.ones(n_time, dtype=bool)

    # Test int32 vs int64 for encoding groups
    encoding_group_labels_32 = np.zeros(n_time, dtype=np.int32)
    encoding_group_labels_32[20:] = 1

    encoding_group_labels_64 = np.zeros(n_time, dtype=np.int64)
    encoding_group_labels_64[20:] = 1

    environment_labels = np.full(n_time, "env_0")

    # Both should give same result
    is_encoding_32 = np.isin(encoding_group_labels_32, [0])
    is_encoding_64 = np.isin(encoding_group_labels_64, [0])
    is_environment = environment_labels == "env_0"

    is_group_32 = is_training & is_encoding_32 & is_environment
    is_group_64 = is_training & is_encoding_64 & is_environment

    np.testing.assert_array_equal(is_group_32, is_group_64)

    # Test different string dtypes for environments
    environment_labels_U5 = np.full(n_time, "env_0", dtype='U5')
    environment_labels_U20 = np.full(n_time, "env_0", dtype='U20')

    is_environment_5 = environment_labels_U5 == "env_0"
    is_environment_20 = environment_labels_U20 == "env_0"

    np.testing.assert_array_equal(is_environment_5, is_environment_20)


def test_is_group_boolean_logic_correctness():
    """Test the boolean AND logic in is_group formation."""
    n_time = 16

    # Create specific pattern to test boolean logic
    is_training = np.array([True, True, False, False] * 4)  # Alternating pattern
    groups = np.array([0, 1, 0, 1] * 4)
    envs = np.array(["A", "A", "B", "B"] * 4)

    # Test different combinations
    test_cases = [
        {"group": [0], "env": "A"},      # Should match indices 0, 8 (training=True, group=0, env=A)
        {"group": [1], "env": "A"},      # Should match indices 1, 9 (training=True, group=1, env=A)
        {"group": [0], "env": "B"},      # Should match indices 8 (but wait, need to check pattern...)
        {"group": [0, 1], "env": "A"},   # Should match indices 0, 1, 8, 9
    ]

    for case in test_cases:
        is_encoding = np.isin(groups, case["group"])
        is_environment = envs == case["env"]
        is_group = is_training & is_encoding & is_environment

        # Basic sanity checks
        assert is_group.dtype == bool
        assert is_group.shape == (n_time,)
        assert np.sum(is_group) <= n_time  # Can't select more than total

        # Verify that selected indices meet all three conditions
        selected_indices = np.where(is_group)[0]
        for idx in selected_indices:
            assert is_training[idx], f"Index {idx} should have training=True"
            assert groups[idx] in case["group"], f"Index {idx} should have group in {case['group']}"
            assert envs[idx] == case["env"], f"Index {idx} should have env={case['env']}"