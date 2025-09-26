"""Likelihood functions for different neural data types.

This subpackage provides likelihood estimation functions for various types
of neural data including sorted spikes, clusterless spikes, and calcium
imaging data.

The module supports:

- Sorted spikes: GLM and KDE-based likelihood estimation
- Clusterless data: Multiunit likelihood functions with mark information
- Calcium imaging: Gamma likelihood for calcium activity traces
- GPU acceleration: CUDA implementations for performance
"""

from __future__ import annotations

from typing import Callable, Any

# flake8: noqa
from replay_trajectory_classification.likelihoods.calcium_likelihood import (
    estimate_calcium_likelihood,
    estimate_calcium_place_fields,
)
from replay_trajectory_classification.likelihoods.multiunit_likelihood import (
    estimate_multiunit_likelihood,
    fit_multiunit_likelihood,
)
from replay_trajectory_classification.likelihoods.multiunit_likelihood_gpu import (
    estimate_multiunit_likelihood_gpu,
    fit_multiunit_likelihood_gpu,
)
from replay_trajectory_classification.likelihoods.multiunit_likelihood_integer import (
    estimate_multiunit_likelihood_integer,
    fit_multiunit_likelihood_integer,
)
from replay_trajectory_classification.likelihoods.multiunit_likelihood_integer_gpu import (
    estimate_multiunit_likelihood_integer_gpu,
    fit_multiunit_likelihood_integer_gpu,
)
from replay_trajectory_classification.likelihoods.spiking_likelihood_glm import (
    estimate_place_fields,
    estimate_spiking_likelihood,
)
from replay_trajectory_classification.likelihoods.spiking_likelihood_kde import (
    estimate_place_fields_kde,
    estimate_spiking_likelihood_kde,
)
from replay_trajectory_classification.likelihoods.spiking_likelihood_kde_gpu import (
    estimate_place_fields_kde_gpu,
    estimate_spiking_likelihood_kde_gpu,
)

from .multiunit_likelihood_integer_gpu_log import (
    estimate_multiunit_likelihood_integer_gpu_log,
    fit_multiunit_likelihood_integer_gpu_log,
)

_ClUSTERLESS_ALGORITHMS: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
    "multiunit_likelihood": (fit_multiunit_likelihood, estimate_multiunit_likelihood),
    "multiunit_likelihood_gpu": (
        fit_multiunit_likelihood_gpu,
        estimate_multiunit_likelihood_gpu,
    ),
    "multiunit_likelihood_integer": (
        fit_multiunit_likelihood_integer,
        estimate_multiunit_likelihood_integer,
    ),
    "multiunit_likelihood_integer_gpu": (
        fit_multiunit_likelihood_integer_gpu,
        estimate_multiunit_likelihood_integer_gpu,
    ),
    "multiunit_likelihood_integer_gpu_log": (
        fit_multiunit_likelihood_integer_gpu_log,
        estimate_multiunit_likelihood_integer_gpu_log,
    ),
}

_SORTED_SPIKES_ALGORITHMS: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
    "spiking_likelihood_glm": (estimate_place_fields, estimate_spiking_likelihood),
    "spiking_likelihood_kde": (
        estimate_place_fields_kde,
        estimate_spiking_likelihood_kde,
    ),
    "spiking_likelihood_kde_gpu": (
        estimate_place_fields_kde_gpu,
        estimate_spiking_likelihood_kde_gpu,
    ),
}

_CALCIUM_ALGORITHMS: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
    "calcium_likelihood": (estimate_calcium_place_fields, estimate_calcium_likelihood)
}
