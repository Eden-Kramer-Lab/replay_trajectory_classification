# replay_trajectory_classification/tests/unit/test_continuous_transitions.py
import numpy as np
import pytest

from replay_trajectory_classification.continuous_state_transitions import (
    Identity,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
)
from replay_trajectory_classification.environments import Environment

# estimate_movement_var may or may not exist; import if present.
try:
    from replay_trajectory_classification.continuous_state_transitions import (
        estimate_movement_var,
    )
except Exception:  # pragma: no cover
    estimate_movement_var = None


# ---------------------- helpers ----------------------


def make_1d_env(name="E", n=11, bin_size=1.0):
    """
    Build a simple 1D environment by fitting a grid on synthetic positions.
    In practice, the resulting number of *interior* bins may differ from `n`.
    """
    x = np.linspace(0, (n - 1) * bin_size, n).reshape(-1, 1)
    env = Environment()
    env.environment_name = name
    env.fit_place_grid(position=x, infer_track_interior=True)
    return env


def flat_mask(env: Environment):
    """Return a flat boolean interior mask; if None, assume all interior."""
    mask = getattr(env, "is_track_interior_", None)
    if mask is None:
        # Infer interior length from centers; assume all interior
        n = getattr(env, "place_bin_centers_", None)
        n_states = len(n) if n is not None else None
        if n_states is None:
            raise RuntimeError("Cannot infer number of states for mask.")
        return np.ones(n_states, dtype=bool)
    return np.asarray(mask).ravel(order="F")


def assert_interior_rows_row_stochastic(
    A: np.ndarray, interior: np.ndarray, atol=1e-12
):
    """Check that interior rows sum to 1, masked rows/cols are all-zero."""
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Matrix must be square"
    assert np.isfinite(A).all(), "NaN/inf in transition matrix"
    assert (A >= -1e-14).all(), "Negative probabilities"

    # Masked rows & cols are zero
    masked_idx = np.where(~interior)[0]
    if masked_idx.size:
        assert np.all(A[masked_idx] == 0.0)
        assert np.all(A[:, masked_idx] == 0.0)

    # Interior rows are row-stochastic
    np.testing.assert_allclose(A[interior].sum(axis=1), 1.0, atol=atol)


# ---------------------- estimate_movement_var ----------------------


@pytest.mark.skipif(
    estimate_movement_var is None,
    reason="estimate_movement_var not available"
)
def test_estimate_movement_var_returns_cov_and_is_psd():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(1000, 2))
    cov = estimate_movement_var(v, sampling_frequency=50.0)
    assert cov.shape in {(2, 2), (2,)}  # accept either contract
    if cov.ndim == 2:
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        eigs = np.linalg.eigvalsh(cov)
        assert np.all(eigs >= -1e-9)


@pytest.mark.skipif(
    estimate_movement_var is None,
    reason="estimate_movement_var not available"
)
def test_estimate_movement_var_ignores_nans_and_scales():
    x = np.arange(100, dtype=float).reshape(-1, 1)
    x[10:15] = np.nan
    cov = estimate_movement_var(x, sampling_frequency=100.0)
    assert np.isfinite(cov).all()


# ---------------------- RandomWalk (non-diffusion) ----------------------


@pytest.mark.parametrize("movement_var", [0.5, 2.0, 6.0])
def test_random_walk_row_stochastic_and_peaked_near_self(movement_var):
    env = make_1d_env(n=9, bin_size=1.0)
    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=movement_var,
        movement_mean=0.0,
        use_diffusion=False,
    )
    A = rw.make_state_transition((env,))
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)
    # Argmax per interior row should be on the diagonal (peak near self)
    peak_cols = np.argmax(A[interior], axis=1)
    np.testing.assert_allclose(peak_cols, np.where(interior)[0], atol=0)


def test_random_walk_masks_off_track_bins_if_any():
    env = make_1d_env(n=7, bin_size=1.0)
    # If environment has at least 3 interior bins, mask the middle one
    interior = flat_mask(env).copy()
    interior_idx = np.where(interior)[0]
    if interior_idx.size < 3:
        pytest.skip("Not enough interior bins to test masking.")
    mid = interior_idx[len(interior_idx) // 2]
    interior[mid] = False
    # Apply the modified mask back in env with original shape
    env.is_track_interior_ = interior.reshape(env.is_track_interior_.shape, order="F")

    rw = RandomWalk(
        environment_name=env.environment_name, movement_var=1.0, use_diffusion=False
    )
    A = rw.make_state_transition((env,))
    assert_interior_rows_row_stochastic(A, interior)


# ---------------------- Uniform ----------------------


def test_uniform_is_uniform_over_interior_states():
    env = make_1d_env(n=6)
    uni = Uniform(environment_name=env.environment_name)
    A = uni.make_state_transition((env,))
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)

    # For uniform, each interior row should be identical and uniform over interior columns
    if interior.any():
        row0 = A[interior][0]
        for r in A[interior]:
            np.testing.assert_allclose(r, row0, atol=1e-12)
        n_int = interior.sum()
        expected = np.zeros_like(row0)
        expected[interior] = 1.0 / n_int
        np.testing.assert_allclose(row0, expected, atol=1e-12)


def test_uniform_respects_masked_bins():
    env = make_1d_env(n=5)
    interior = flat_mask(env).copy()
    idx = np.where(interior)[0]
    if idx.size < 2:
        pytest.skip("Not enough interior bins to mask one and keep others.")
    # mask one interior bin
    interior[idx[0]] = False
    env.is_track_interior_ = interior.reshape(env.is_track_interior_.shape, order="F")
    A = Uniform(environment_name=env.environment_name).make_state_transition((env,))
    assert_interior_rows_row_stochastic(A, interior)


# ---------------------- Identity ----------------------


def test_identity_transition_masks_and_normalizes():
    env = make_1d_env(n=4)
    interior = flat_mask(env).copy()
    idx = np.where(interior)[0]
    if idx.size < 2:
        pytest.skip("Not enough interior bins to mask one and test identity.")
    interior[idx[0]] = False
    env.is_track_interior_ = interior.reshape(env.is_track_interior_.shape, order="F")
    ident = Identity(environment_name=env.environment_name)
    A = ident.make_state_transition((env,))
    assert_interior_rows_row_stochastic(A, interior)

    # On the interior subspace, Identity should be identity
    sub = np.ix_(interior, interior)
    np.testing.assert_allclose(A[sub], np.eye(interior.sum()), atol=1e-12)


# ---------------------- Directional random walks ----------------------
# These models bias probability mass to one side; they are not strictly triangular-zeroed.


def _mass_left_right(row: np.ndarray, i: int):
    left = row[: i + 1].sum()
    right = row[i:].sum()  # includes self on both; we'll compare ratios
    return float(left), float(right)


def test_random_walk_direction1_biases_to_higher_indices():
    env = make_1d_env(n=8)
    rwd1 = RandomWalkDirection1(environment_name=env.environment_name, movement_var=2.0)
    A = rwd1.make_state_transition((env,))
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)

    # For each interior row i, mass on >= i should be >= mass on <= i (bias to the right)
    rows = np.where(interior)[0]
    for i in rows:
        left, right = _mass_left_right(A[i], i)
        assert right >= left - 1e-12


def test_random_walk_direction2_biases_to_lower_indices():
    env = make_1d_env(n=8)
    rwd2 = RandomWalkDirection2(environment_name=env.environment_name, movement_var=2.0)
    A = rwd2.make_state_transition((env,))
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)

    # For each interior row i, mass on <= i should be >= mass on >= i (bias to the left)
    rows = np.where(interior)[0]
    for i in rows:
        left, right = _mass_left_right(A[i], i)
        assert left >= right - 1e-12


# ---------------------- Diffusion path sanity (2D) ----------------------


def test_random_walk_with_diffusion_row_stochastic_2d():
    # Build a simple 2D environment by fitting grid on a small 2D point cloud
    xx, yy = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4), indexing="xy")
    pts = np.c_[xx.ravel(), yy.ravel()]  # 16 points on a grid
    env = Environment()
    env.environment_name = "E2"
    env.fit_place_grid(position=pts, infer_track_interior=True)

    rw = RandomWalk(
        environment_name=env.environment_name,
        movement_var=1.0,
        movement_mean=0.0,
        use_diffusion=True,
    )
    A = rw.make_state_transition((env,))
    interior = flat_mask(env)
    assert_interior_rows_row_stochastic(A, interior)
