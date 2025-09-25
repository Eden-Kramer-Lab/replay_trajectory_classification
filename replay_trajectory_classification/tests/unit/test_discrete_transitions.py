# tests/unit/test_discrete_transitions.py
import numpy as np
import pytest

# Required imports
from replay_trajectory_classification.discrete_state_transitions import DiagonalDiscrete

# Optional imports (skip tests gracefully if unavailable)
try:
    from replay_trajectory_classification.discrete_state_transitions import (
        RandomDiscrete,  # type: ignore
    )
except Exception:  # ImportError or any runtime import issue
    RandomDiscrete = None


# ---------- Helpers ----------


def assert_row_stochastic(A: np.ndarray, atol=1e-12):
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Matrix must be square"
    # No NaNs or infs
    assert np.isfinite(A).all(), "Transition matrix contains NaN/inf"
    # Non-negative
    assert (A >= -1e-14).all(), "Transition matrix has negative entries"
    # Rows sum to 1
    np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=atol)


# ---------- DiagonalDiscrete ----------


@pytest.mark.parametrize("n_states", [1, 2, 5, 17])
@pytest.mark.parametrize("diag", [0.0, 0.2, 0.7, 1.0])
def test_diagonal_discrete_shape_and_row_stochastic(n_states, diag):
    dd = DiagonalDiscrete(diagonal_value=diag)
    A = dd.make_state_transition(n_states)
    assert A.shape == (n_states, n_states)
    assert_row_stochastic(A)


@pytest.mark.parametrize("n_states", [2, 5, 11])
@pytest.mark.parametrize("diag", [0.0, 0.2, 0.7, 0.95])
def test_diagonal_discrete_values(n_states, diag):
    dd = DiagonalDiscrete(diagonal_value=diag)
    A = dd.make_state_transition(n_states)

    # Diagonal should be exactly the provided value (within FP tolerance)
    np.testing.assert_allclose(np.diag(A), diag, atol=1e-12)

    if n_states > 1:
        off = A - np.diag(np.diag(A))
        # All off-diagonals equal and sum to (1 - diag)
        expected_off = (1.0 - diag) / (n_states - 1)
        # Where off-diagonal exists, values should be equal to expected_off
        if diag < 1.0:
            np.testing.assert_allclose(off[off > 0], expected_off, atol=1e-12)
        else:
            # diag==1 => identity, so off-diagonals should be 0
            assert np.allclose(off, 0.0)


def test_diagonal_discrete_n_states_eq_1_is_identity():
    dd = DiagonalDiscrete(diagonal_value=1.0)
    A = dd.make_state_transition(1)
    np.testing.assert_allclose(A, np.array([[1.0]]), atol=0.0)


# ---------- Composition property (generic sanity check) ----------


@pytest.mark.parametrize("diag", [0.0, 0.3, 0.8, 1.0])
def test_composition_preserves_row_stochastic(diag):
    n = 7
    dd = DiagonalDiscrete(diagonal_value=diag)
    A = dd.make_state_transition(n)
    AA = A @ A
    assert_row_stochastic(AA, atol=1e-10)


# ---------- RandomDiscrete (optional) ----------


@pytest.mark.skipif(
    RandomDiscrete is None,
    reason="RandomDiscrete not available"
)
def test_random_discrete_is_row_stochastic_and_square():
    n = 9
    rd = RandomDiscrete()
    A = rd.make_state_transition(n)
    assert A.shape == (n, n)
    assert_row_stochastic(A)


@pytest.mark.skipif(
    RandomDiscrete is None,
    reason="RandomDiscrete not available"
)
def test_random_discrete_determinism_with_seed():
    # Try to use a seed if the implementation supports it;
    # fall back to global NumPy seed for reproducibility.
    n = 6
    try:
        rd1 = RandomDiscrete(random_state=123)
        rd2 = RandomDiscrete(random_state=123)
        A1 = rd1.make_state_transition(n)
        A2 = rd2.make_state_transition(n)
    except TypeError:
        np.random.seed(123)  # legacy support if code uses global seed
        rd1 = RandomDiscrete()
        A1 = rd1.make_state_transition(n)
        np.random.seed(123)
        rd2 = RandomDiscrete()
        A2 = rd2.make_state_transition(n)

    np.testing.assert_allclose(A1, A2, atol=1e-12)
    assert_row_stochastic(A1)
    assert_row_stochastic(A2)


# ---------- Boundary / sparsity scenarios (robustness) ----------


def test_diagonal_extremes_no_negative_or_nan():
    # diag=0 (no self-transition) and diag=1 (identity) are extremes
    for diag in (0.0, 1.0):
        A = DiagonalDiscrete(diagonal_value=diag).make_state_transition(5)
        assert np.isfinite(A).all()
        assert (A >= 0.0).all()
        assert_row_stochastic(A)
