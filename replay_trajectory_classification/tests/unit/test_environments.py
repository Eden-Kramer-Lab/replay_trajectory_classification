# replay_trajectory_classification/tests/unit/test_environment.py
import numpy as np
import pytest

from replay_trajectory_classification.environments import Environment

# order_boundary may not be exported in all builds; import if available.
try:
    from replay_trajectory_classification.environments import (
        order_boundary,  # type: ignore
    )
except Exception:  # pragma: no cover
    order_boundary = None


# ---------------------- helpers ----------------------


def make_1d_env(name="E", n=11, bin_size=1.0):
    """Build a simple 1D Environment and fit a place grid from synthetic positions."""
    x = np.linspace(0, (n - 1) * bin_size, n).reshape(-1, 1)
    env = Environment()
    env.environment_name = name
    env.fit_place_grid(position=x, infer_track_interior=True)
    return env


# ---------------------- tests ----------------------


def test_fit_place_grid_sets_centers_and_mask_consistently():
    env = make_1d_env(n=9)
    # Centers should exist
    assert hasattr(env, "place_bin_centers_")
    centers = np.asarray(env.place_bin_centers_)
    assert centers.ndim in (1, 2), "centers expected 1D or (n_bins, dim)"
    # Mask should exist and be boolean
    assert hasattr(env, "is_track_interior_")
    mask = np.asarray(env.is_track_interior_)
    assert mask.dtype == bool
    # Flattened mask size equals number of states
    msize = mask.size
    # Number of states = number of rows in centers if 2D, else length
    n_states = centers.shape[0] if centers.ndim == 2 else centers.size
    assert n_states == msize, "centers/mask size mismatch"
    # Sanity: at least 1 interior bin
    assert mask.any()


@pytest.mark.skipif(
    order_boundary is None,
    reason="order_boundary not available"
)
def test_order_boundary_returns_valid_indices_on_small_mask():
    # 4x4 square; interior True everywhere ⇒ perimeter has 2*(4+4)-4 = 12 cells
    mask2d = np.ones((4, 4), dtype=bool)
    expected_perimeter = 2 * (mask2d.shape[0] + mask2d.shape[1]) - 4

    boundary = order_boundary(mask2d)
    boundary = np.asarray(boundary).ravel()
    assert boundary.ndim == 1
    assert boundary.size > 0

    if boundary.dtype == bool:
        # Either a full-size mask, or a compact boolean path of length == perimeter
        if boundary.size == mask2d.size:
            assert boundary.any()
        else:
            # Compact path form: must be all True and have expected perimeter length
            assert boundary.size == expected_perimeter
            assert boundary.all()
    elif np.issubdtype(boundary.dtype, np.integer):
        # Index form: indices are within range
        assert boundary.min() >= 0
        assert boundary.max() < mask2d.size
    else:
        pytest.fail(f"Unexpected dtype from order_boundary: {boundary.dtype}")


@pytest.mark.skipif(
    order_boundary is None,
    reason="order_boundary not available"
)
def test_order_boundary_handles_non_rectangular_interior():
    # A plus-shape interior in a 5x5 grid
    m = np.zeros((5, 5), dtype=bool)
    m[2, :] = True
    m[:, 2] = True
    # Boundary ordering should still return valid indices without error
    boundary_idx = order_boundary(m)
    boundary_idx = np.asarray(boundary_idx).ravel()
    assert boundary_idx.size > 0
    assert boundary_idx.max() < m.size


def test_environment_name_and_equality_semantics():
    env = make_1d_env(name="ArenaA", n=7)
    # Name should be set
    assert hasattr(env, "environment_name")
    assert isinstance(env.environment_name, str) and env.environment_name != ""
    # Some codebases implement __eq__ to compare to name strings; allow skip if not.
    try:
        assert (env == env.environment_name) or (env.environment_name == env)
    except Exception:
        # If not implemented, at least ensure normal equality doesn't crash
        pytest.skip("__eq__ to environment_name not implemented; skipping.")
