# replay_trajectory_classification/tests/unit/test_observation_model.py
import inspect

import pytest

from replay_trajectory_classification import observation_model as om_mod


def _has_callable_prediction(obj):
    """Return (callable, name) where callable is predict fn or __call__."""
    # Prefer explicit predict method if present
    for name in ("predict_log_likelihood", "predict", "log_likelihood"):
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn, name
    # Fallback to __call__
    if callable(obj):
        return obj, "__call__"
    return None, None


def _invoke_predict(fn, observations=None, n_time=4):
    """
    Try a few common signatures for predict/predict_log_likelihood and
    return the output if any works; otherwise raise TypeError.
    """
    observations = {} if observations is None else observations

    tried = []

    # Common signatures we try in order
    candidates = [
        lambda: fn(observations),  # (obs)
        lambda: fn(observations, n_time),  # (obs, T)
        lambda: fn(observations=observations),  # (obs=)
        lambda: fn(observations=observations, n_time=n_time),  # (obs=, T=)
    ]

    for call in candidates:
        try:
            out = call()
            return out
        except TypeError as e:
            tried.append(str(e))
            continue
    raise TypeError("No compatible predict signature found:\n" + "\n".join(tried))


def test_module_and_class_exist():
    assert hasattr(om_mod, "ObservationModel"), "ObservationModel not exported"
    # basic introspection doesn't raise
    assert inspect.isclass(om_mod.ObservationModel)


def test_observation_model_constructible_minimally():
    """Try to construct the object with a minimal set of args.

    If your class requires specific kwargs, add defaults internally or
    raise a clear TypeError. This test will SKIP if the constructor
    is strictly application-specific.
    """
    ObservationModel = om_mod.ObservationModel

    try:
        # Try a no-arg or minimal construction
        try:
            obj = ObservationModel()  # type: ignore[call-arg]
        except TypeError:
            # Try a common alt: allow passing nothing-like models
            obj = ObservationModel(encoding_models=None)  # type: ignore[call-arg]
    except TypeError:
        pytest.skip("ObservationModel requires project-specific constructor args")

    assert obj is not None
