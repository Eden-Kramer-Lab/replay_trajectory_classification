import logging
import os
import sys
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run

import matplotlib.pyplot as plt
import xarray as xr
from loren_frank_data_processing import reshape_to_segments, save_xarray
from replay_trajectory_classification import SortedSpikesClassifier
from replay_trajectory_classification.analysis import (
    get_linear_position_order, get_place_field_max, get_replay_info)
from replay_trajectory_classification.load_example_data import (
    FIGURE_DIR, PROCESSED_DATA_DIR, SAMPLING_FREQUENCY, load_data)
from replay_trajectory_classification.visualization import (
    plot_neuron_place_field_2D_1D_position, plot_ripple_decode)

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')
plt.switch_backend('agg')


TRANSITION_TO_CATEGORY = {
    'identity': 'hover',
    'uniform': 'fragmented',
    'random_walk_with_absorbing_boundaries': 'continuous',
}

PROBABILITY_THRESHOLD = 0.8


def run_analysis(epoch_key, make_movies=False):
    animal, day, epoch = epoch_key

    logging.info('Loading data...')
    data = load_data(epoch_key)
    position = data['position_info'].loc[:, ['x_position', 'y_position']]
    is_training = data['position_info'].speed > 4
    ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]

    classifier_filename = os.path.join(
        PROCESSED_DATA_DIR, (f'{animal}_{day:02d}_{epoch:02d}_'
                             'sorted_spikes_classifier_replay_model.pkl'))
    if not os.path.isfile(classifier_filename):
        logging.info('Fitting classifier...')
        classifier = SortedSpikesClassifier().fit(
            position, data['spikes'], is_training=is_training)
        logging.info('Saving fitted classifier...')
        classifier.save_model(classifier_filename)
    else:
        logging.info('Loading encoding model...')
        classifier = SortedSpikesClassifier.load_model(classifier_filename)

    logging.info('Plotting place fields...')
    g = classifier.plot_place_fields(
        data['spikes'], position, SAMPLING_FREQUENCY)
    plt.suptitle(epoch_key, y=1.04, fontsize=16)

    fig_name = (f'{animal}_{day:02d}_{epoch:02d}_place_fields.png')
    fig_name = os.path.join(FIGURE_DIR, 'neuron_place_fields', fig_name)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(g.fig)

    logging.info('Decoding ripples...')
    ripple_spikes = reshape_to_segments(
        data['spikes'], ripple_times, sampling_frequency=SAMPLING_FREQUENCY)

    results = xr.concat(
        [classifier.predict(ripple_spikes.loc[ripple_number],
                            time=(ripple_spikes.loc[ripple_number].index -
                                  ripple_spikes.loc[ripple_number].index[0]))
         for ripple_number in data['ripple_times'].index],
        dim=data['ripple_times'].index).assign_coords(
            state=lambda ds: ds.state.to_index().map(TRANSITION_TO_CATEGORY)
    )

    logging.info('Saving results...')
    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                results.drop(['likelihood', 'causal_posterior']),
                group='/sorted_spikes/classifier/ripples/')

    logging.info('Saving replay_info...')
    ripple_position = reshape_to_segments(
        position, ripple_times, sampling_frequency=SAMPLING_FREQUENCY)
    replay_info = get_replay_info(
        results, ripple_spikes, ripple_position, data['ripple_times'],
        SAMPLING_FREQUENCY, PROBABILITY_THRESHOLD)
    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{animal}_{day:02d}_{epoch:02d}_replay_info.csv')
    replay_info.to_csv(replay_info_filename)

    logging.info('Plotting ripple figures...')
    place_field_max = get_place_field_max(classifier)
    linear_position_order, linear_place_field_max = get_linear_position_order(
        data['position_info'], place_field_max)
    plot_neuron_place_field_2D_1D_position(
            data['position_info'], place_field_max, linear_place_field_max,
            linear_position_order)
    fig_name = (f'{animal}_{day:02d}_{epoch:02d}_place_field_max.png')
    fig_name = os.path.join(FIGURE_DIR, 'neuron_place_fields', fig_name)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(plt.gcf())

    for ripple_number in ripple_times.index:
        plot_ripple_decode(ripple_number, results, ripple_position,
                           ripple_spikes, position, linear_position_order)
        plt.suptitle(
            f'ripple number = {animal}_{day:02d}_{epoch:02d}_'
            f'{ripple_number:04d}')
        fig_name = (f'{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_'
                    'sorted_spikes_acasual_classification.png')
        fig_name = os.path.join(FIGURE_DIR, 'sorted_spikes_ripples', fig_name)
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(plt.gcf())

    logging.info('Done...')


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument(
        '-d', '--debug',
        help='More verbose output for debugging',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT, level=args.log_level)

    def _signal_handler(signal_code, frame):
        logging.error('***Process killed with signal {signal}***'.format(
            signal=signal_code))
        exit()

    for code in [SIGUSR1, SIGUSR2]:
        signal(code, _signal_handler)

    epoch_key = (args.Animal, args.Day, args.Epoch)
    logging.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *epoch_key))
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logging.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    # Analysis Code
    run_analysis(epoch_key)


if __name__ == '__main__':
    sys.exit(main())
