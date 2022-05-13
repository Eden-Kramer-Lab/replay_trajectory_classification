#!/usr/bin/env python
# coding: utf-8

import logging

import numba
import numpy as np
from replay_trajectory_classification import ClusterlessDecoder
from replay_trajectory_classification.clusterless_simulation import \
    make_simulated_run_data
from replay_trajectory_classification.continuous_state_transitions import \
    estimate_movement_var

# Enable logging
logging.basicConfig(level=logging.INFO)


def main():
    (time, position, sampling_frequency,
     multiunits, multiunits_spikes) = make_simulated_run_data(n_tetrodes=10)

    movement_var = estimate_movement_var(position, sampling_frequency)

    clusterless_algorithm = 'multiunit_likelihood_gpu'
    clusterless_algorithm_params = {
        'mark_std': 1.0,
        'position_std': 12.5,
    }

    decoder = ClusterlessDecoder(
        movement_var=movement_var,
        replay_speed=1,
        place_bin_size=np.sqrt(movement_var),
        clusterless_algorithm=clusterless_algorithm,
        clusterless_algorithm_params=clusterless_algorithm_params)
    numba.cuda.profile_start()
    decoder.fit(position, multiunits)
    results = decoder.predict(multiunits, time=time, use_gpu=True)
    numba.cuda.profile_stop()
    logging.info('Done!')


if __name__ == "__main__":
    main()
