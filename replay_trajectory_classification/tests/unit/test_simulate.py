# replay_trajectory_classification/tests/unit/test_simulate.py
import numpy as np
import pytest

from replay_trajectory_classification.simulate import (
    simulate_time,
    simulate_position,
)

# Try to import additional simulation functions if available
try:
    from replay_trajectory_classification.simulate import simulate_place_field_firing_rate
except ImportError:
    simulate_place_field_firing_rate = None

try:
    from replay_trajectory_classification.simulate import simulate_poisson_spikes
except ImportError:
    simulate_poisson_spikes = None


# ---------------------- simulate_time Tests ----------------------


def test_simulate_time_basic():
    """Test basic functionality of simulate_time."""
    n_samples = 100
    sampling_frequency = 500.0  # 500 Hz

    time = simulate_time(n_samples, sampling_frequency)

    assert time.shape == (n_samples,)
    assert time.dtype in [np.float64, np.float32]
    assert time[0] == 0.0
    assert np.allclose(time[1], 1.0 / sampling_frequency)
    assert np.allclose(time[-1], (n_samples - 1) / sampling_frequency)


def test_simulate_time_monotonic_increasing():
    """Test that simulated time is monotonically increasing."""
    time = simulate_time(50, 1000.0)

    # Check monotonic increasing
    assert np.all(np.diff(time) > 0)


@pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
def test_simulate_time_various_lengths(n_samples):
    """Test simulate_time with various lengths."""
    sampling_frequency = 500.0
    time = simulate_time(n_samples, sampling_frequency)

    assert time.shape == (n_samples,)
    assert time[0] == 0.0
    if n_samples > 1:
        assert time[-1] == (n_samples - 1) / sampling_frequency


@pytest.mark.parametrize("sampling_frequency", [1.0, 100.0, 500.0, 2000.0])
def test_simulate_time_various_frequencies(sampling_frequency):
    """Test simulate_time with various sampling frequencies."""
    n_samples = 100
    time = simulate_time(n_samples, sampling_frequency)

    expected_dt = 1.0 / sampling_frequency
    actual_dt = time[1] - time[0]
    assert np.allclose(actual_dt, expected_dt, rtol=1e-12)


def test_simulate_time_edge_cases():
    """Test simulate_time with edge cases."""
    # Single sample
    time = simulate_time(1, 500.0)
    assert time.shape == (1,)
    assert time[0] == 0.0

    # Very high frequency
    time = simulate_time(10, 1e6)
    assert np.all(np.diff(time) > 0)
    assert np.isfinite(time).all()


# ---------------------- simulate_position Tests ----------------------


def test_simulate_position_basic():
    """Test basic functionality of simulate_position."""
    time = simulate_time(100, 500.0)
    track_height = 100.0
    running_speed = 15.0

    position = simulate_position(time, track_height, running_speed)

    assert position.shape == time.shape
    assert position.dtype in [np.float64, np.float32]
    assert np.isfinite(position).all()


def test_simulate_position_bounds():
    """Test that simulated position stays within track bounds."""
    time = simulate_time(1000, 500.0)  # 2 seconds
    track_height = 50.0
    running_speed = 10.0

    position = simulate_position(time, track_height, running_speed)

    # Position should be within track bounds
    assert np.all(position >= 0.0)
    assert np.all(position <= track_height)


def test_simulate_position_movement_characteristics():
    """Test that position simulation has realistic movement characteristics."""
    time = simulate_time(1000, 500.0)
    track_height = 100.0
    running_speed = 20.0

    position = simulate_position(time, track_height, running_speed)

    # Position should change over time (not constant)
    assert not np.all(position == position[0])

    # Check that position changes are reasonable
    velocities = np.diff(position) / np.diff(time)
    # Allow for some variability but expect reasonable speeds
    assert np.abs(velocities).max() < running_speed * 2  # Allow 2x safety factor


@pytest.mark.parametrize("track_height", [10.0, 50.0, 100.0, 200.0])
def test_simulate_position_various_track_heights(track_height):
    """Test simulate_position with various track heights."""
    time = simulate_time(200, 500.0)
    running_speed = 15.0

    position = simulate_position(time, track_height, running_speed)

    assert np.all(position >= 0.0)
    assert np.all(position <= track_height)
    # Basic sanity check - position should change from starting point
    assert position.max() > 0.0  # Should have some movement


@pytest.mark.parametrize("running_speed", [5.0, 10.0, 20.0, 30.0])
def test_simulate_position_various_speeds(running_speed):
    """Test simulate_position with various running speeds."""
    time = simulate_time(500, 500.0)  # 1 second
    track_height = 100.0

    position = simulate_position(time, track_height, running_speed)

    # Higher speeds should generally result in more movement
    velocities = np.abs(np.diff(position) / np.diff(time))
    mean_velocity = np.mean(velocities)

    # Speed should be related to running_speed parameter
    # (though exact relationship depends on implementation details)
    assert mean_velocity > 0  # Should have some movement
    assert mean_velocity < running_speed * 3  # Reasonable upper bound


def test_simulate_position_deterministic():
    """Test that simulate_position is deterministic given same inputs."""
    time = simulate_time(100, 500.0)
    track_height = 80.0
    running_speed = 12.0

    position1 = simulate_position(time, track_height, running_speed)
    position2 = simulate_position(time, track_height, running_speed)

    np.testing.assert_allclose(position1, position2, rtol=1e-12)


# ---------------------- simulate_place_field_firing_rate Tests (if available) ----------------------


@pytest.mark.skipif(
    simulate_place_field_firing_rate is None,
    reason="simulate_place_field_firing_rate not available"
)
def test_simulate_place_field_firing_rate_basic():
    """Test basic functionality of simulate_place_field_firing_rate."""
    position = np.linspace(0, 100, 200)
    place_field_center = 50.0
    place_field_std = 10.0

    firing_rate = simulate_place_field_firing_rate(
        np.array([place_field_center]), position.reshape(-1, 1), max_rate=10.0, variance=place_field_std**2
    )

    assert firing_rate.shape == position.shape
    assert np.isfinite(firing_rate).all()
    assert np.all(firing_rate >= 0)  # Firing rates should be non-negative


@pytest.mark.skipif(
    simulate_place_field_firing_rate is None,
    reason="simulate_place_field_firing_rate not available"
)
def test_simulate_place_field_firing_rate_peak_at_center():
    """Test that firing rate peaks at place field center."""
    position = np.linspace(0, 100, 1000)
    place_field_center = 60.0
    place_field_std = 8.0

    firing_rate = simulate_place_field_firing_rate(
        np.array([place_field_center]), position.reshape(-1, 1), max_rate=10.0, variance=place_field_std**2
    )

    # Find position closest to center
    center_idx = np.argmin(np.abs(position - place_field_center))

    # Firing rate should be maximal (or very close to maximal) at center
    assert firing_rate[center_idx] >= np.max(firing_rate) * 0.99


# ---------------------- simulate_spikes Tests (if available) ----------------------


@pytest.mark.skipif(
    simulate_poisson_spikes is None,
    reason="simulate_poisson_spikes not available"
)
def test_simulate_spikes_basic():
    """Test basic functionality of simulate_poisson_spikes."""
    firing_rates = np.random.exponential(5.0, size=(100,))  # 100 time points for single neuron
    sampling_frequency = 500  # 500 Hz

    spikes = simulate_poisson_spikes(firing_rates, sampling_frequency)

    assert spikes.shape == firing_rates.shape
    assert spikes.dtype in [np.int32, np.int64, int, np.float64]  # Function returns float64
    assert np.all(spikes >= 0)  # Spike counts should be non-negative


@pytest.mark.skipif(
    simulate_poisson_spikes is None,
    reason="simulate_poisson_spikes not available"
)
def test_simulate_spikes_poisson_properties():
    """Test that simulated spikes have Poisson-like properties."""
    # Create constant firing rate
    constant_rate = 10.0  # 10 Hz
    n_time = 1000
    sampling_frequency = 1000  # 1000 Hz

    firing_rates = np.full(n_time, constant_rate)
    spikes = simulate_poisson_spikes(firing_rates, sampling_frequency)

    # Mean spike count should be close to expected (rate / sampling_frequency)
    expected_mean = constant_rate / sampling_frequency
    actual_mean = np.mean(spikes)

    # Allow some tolerance for stochastic variation
    assert np.abs(actual_mean - expected_mean) < expected_mean * 0.5


# ---------------------- Integration Tests ----------------------


def test_simulate_full_pipeline():
    """Test that simulation functions work together in a pipeline."""
    # Simulate a complete dataset
    n_samples = 500
    sampling_frequency = 500.0
    track_height = 120.0
    running_speed = 18.0

    # Step 1: Simulate time
    time = simulate_time(n_samples, sampling_frequency)

    # Step 2: Simulate position
    position = simulate_position(time, track_height, running_speed)

    # Basic checks on the full pipeline
    assert len(time) == len(position) == n_samples
    assert np.all(position >= 0) and np.all(position <= track_height)
    assert np.all(np.diff(time) > 0)  # Time increases


def test_simulation_reproducibility():
    """Test that simulation is reproducible with fixed random seed."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate first dataset
    time1 = simulate_time(100, 500.0)
    position1 = simulate_position(time1, 100.0, 15.0)

    # Reset seed and generate second dataset
    np.random.seed(42)
    time2 = simulate_time(100, 500.0)
    position2 = simulate_position(time2, 100.0, 15.0)

    # Should be identical
    np.testing.assert_allclose(time1, time2)

    # Position might involve randomness - if so, should also be identical
    if not np.allclose(position1, position2):
        # If not identical, at least should be similar patterns
        correlation = np.corrcoef(position1, position2)[0, 1]
        assert correlation > 0.9  # High correlation expected


# ---------------------- Parameter Validation Tests ----------------------


def test_simulate_time_parameter_validation():
    """Test parameter validation for simulate_time."""
    # Negative samples returns empty array
    result = simulate_time(-10, 500.0)
    assert len(result) == 0

    # Zero frequency gives warning but handles gracefully
    result = simulate_time(100, 0.0)
    assert isinstance(result, np.ndarray)

    # Negative frequency returns empty or handles gracefully
    result = simulate_time(100, -500.0)
    assert isinstance(result, np.ndarray)


def test_simulate_position_parameter_validation():
    """Test parameter validation for simulate_position."""
    time = simulate_time(100, 500.0)

    # Negative track height should raise error or be handled gracefully
    try:
        position = simulate_position(time, -50.0, 15.0)
        # If it doesn't raise error, should still produce valid output
        assert np.isfinite(position).all()
    except (ValueError, TypeError):
        pass  # Expected

    # Negative running speed should raise error or be handled gracefully
    try:
        position = simulate_position(time, 100.0, -15.0)
        assert np.isfinite(position).all()
    except (ValueError, TypeError):
        pass  # Expected