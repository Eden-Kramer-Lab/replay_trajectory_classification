# flake8: noqa
from .calcium_likelihood import (estimate_calcium_likelihood,
                                 estimate_calcium_place_fields)
from .multiunit_likelihood import (estimate_multiunit_likelihood,
                                   fit_multiunit_likelihood)
from .multiunit_likelihood2 import (estimate_multiunit_likelihood2,
                                    fit_multiunit_likelihood2)
from .multiunit_likelihood_gpu import (estimate_multiunit_likelihood_gpu,
                                       fit_multiunit_likelihood_gpu)
from .multiunit_likelihood_integer import (
    estimate_multiunit_likelihood_integer, fit_multiunit_likelihood_integer)
from .multiunit_likelihood_integer_gpu import (
    estimate_multiunit_likelihood_integer_gpu,
    fit_multiunit_likelihood_integer_gpu)
from .multiunit_likelihood_integer_gpu_log import (
    estimate_multiunit_likelihood_integer_gpu2,
    fit_multiunit_likelihood_integer_gpu2)
from .spiking_likelihood_glm import (estimate_place_fields,
                                     estimate_spiking_likelihood)
from .spiking_likelihood_kde import (estimate_place_fields_kde,
                                     estimate_spiking_likelihood_kde)

_ClUSTERLESS_ALGORITHMS = {
    'multiunit_likelihood': (
        fit_multiunit_likelihood,
        estimate_multiunit_likelihood),
    'multiunit_likelihood2': (
        fit_multiunit_likelihood2,
        estimate_multiunit_likelihood2),
    'multiunit_likelihood_integer': (
        fit_multiunit_likelihood_integer,
        estimate_multiunit_likelihood_integer),
    'multiunit_likelihood_integer_gpu': (
        fit_multiunit_likelihood_integer_gpu,
        estimate_multiunit_likelihood_integer_gpu),
    'multiunit_likelihood_integer_gpu_log': (
        fit_multiunit_likelihood_integer_gpu2,
        estimate_multiunit_likelihood_integer_gpu2),
    'multiunit_likelihood_gpu': (
        fit_multiunit_likelihood_gpu,
        estimate_multiunit_likelihood_gpu),
}

_SORTED_SPIKES_ALGORITHMS = {
    'spiking_likelihood_glm': (estimate_place_fields,
                               estimate_spiking_likelihood),
    'spiking_likelihood_kde': (estimate_place_fields_kde,
                               estimate_spiking_likelihood_kde)

}
