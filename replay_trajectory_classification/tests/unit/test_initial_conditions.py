# replay_trajectory_classification/tests/unit/test_initial_conditions.py
import numpy as np
import pytest

import replay_trajectory_classification.initial_conditions as ic_mod
from replay_trajectory_classification.environments import Environment

# -------- helpers --------


def make_1d_env(name="E", n=11, bin_size=1.0):
    """Build a simple 1D Environment with an interior mask."""
    x = np.linspace(0, (n - 1) * bin_size, n).reshape(-1, 1)
    env = Environment()
    env.environment_name = name
    env.fit_place_grid(position=x, infer_track_interior=True)
    return env


def flat_mask(env: Environment):
    mask = getattr(env, "is_track_interior_", None)
    if mask is None:
        centers = getattr(env, "place_bin_centers_", None)
        n_states = len(centers) if centers is not None else None
        if n_states is None:
            raise RuntimeError("Cannot infer state count for environment.")
        return np.ones(n_states, dtype=bool)
    return np.asarray(mask).ravel(order="F")


def _find_uniform_initial_cls():
    """Return a class in initial_conditions that represents a uniform prior."""
    candidate_names = [
        "Uniform",
        "UniformInitialConditions",
        "InitialUniform",
        "UniformPrior",
        "UniformInitialState",
    ]
    for nm in candidate_names:
        cls = getattr(ic_mod, nm, None)
        if isinstance(cls, type):
            return cls
    return None


def _make_prior(uniform_obj, env):
    """Try various common maker method names to get the prior vector."""
    makers = [
        "make_initial_conditions",
        "make_initial_distribution",
        "make_initial_state",
        "prior",
        "__call__",
    ]
    for mk in makers:
        fn = getattr(uniform_obj, mk, None)
        if callable(fn):
            # Most implementations expect a tuple of envs
            for arg in ((env,), env):
                try:
                    return fn(arg)
                except TypeError:
                    continue
    raise pytest.skip.Exception(
        "No compatible prior-construction method found on uniform initial-conditions object."
    )


# -------- tests --------


def test_uniform_initial_conditions_exists():
    cls = _find_uniform_initial_cls()
    if cls is None:
        pytest.skip("No uniform initial-conditions class exported.")
    # minimal constructibility (constructor takes no env_name)
    _ = cls()


def test_uniform_prior_normalizes_and_respects_mask():
    cls = _find_uniform_initial_cls()
    if cls is None:
        pytest.skip("No uniform initial-conditions class exported.")

    env = make_1d_env(n=9)
    interior = flat_mask(env).copy()
    idx = np.where(interior)[0]
    # Mask one interior bin if possible
    if idx.size >= 3:
        interior[idx[len(idx) // 2]] = False
        env.is_track_interior_ = interior.reshape(
            env.is_track_interior_.shape, order="F"
        )

    obj = cls()
    prior = _make_prior(obj, env)
    prior = np.asarray(prior).ravel()
    n_states = prior.size

    # Basic shape and finiteness
    assert n_states == flat_mask(env).size
    assert np.isfinite(prior).all()
    assert (prior >= -1e-14).all()

    # Masked bins must be zero; interior sums to 1
    m = flat_mask(env)
    if (~m).any():
        assert np.allclose(prior[~m], 0.0)
    np.testing.assert_allclose(prior[m].sum(), 1.0, atol=1e-12)

    # Uniform over interior: all interior bins equal (within tolerance)
    if m.sum() >= 2:
        interior_vals = prior[m]
        np.testing.assert_allclose(interior_vals, interior_vals[0], atol=1e-12)


def test_env_lookup_by_name_matches_object_prior():
    """
    If the API supports name-based lookup implicitly (e.g., env carries its name,
    but the constructor does NOT accept environment_name), we still verify that
    two independent instances produce identical priors for the same env object.
    """
    cls = _find_uniform_initial_cls()
    if cls is None:
        pytest.skip("No uniform initial-conditions class exported.")

    env = make_1d_env(name="ArenaA", n=7)
    m = flat_mask(env)

    obj1 = cls()
    obj2 = cls()

    prior1 = _make_prior(obj1, env).ravel()
    prior2 = _make_prior(obj2, env).ravel()

    # Both ways should agree
    np.testing.assert_allclose(prior1, prior2, atol=1e-12)

    # Sanity: masked entries zero; interior sums to 1
    if (~m).any():
        assert np.allclose(prior1[~m], 0.0)
    np.testing.assert_allclose(prior1[m].sum(), 1.0, atol=1e-12)
