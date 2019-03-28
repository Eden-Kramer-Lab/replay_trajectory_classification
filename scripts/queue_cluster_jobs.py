'''Script for executing run_by_epoch on the cluster
'''
from argparse import ArgumentParser
from os import environ, getcwd, makedirs
from os.path import join
from subprocess import run
from sys import exit

from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from replay_trajectory_classification.load_data import ANIMALS


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('--Animal', type=str, help='Short name of animal')
    parser.add_argument('--Day', type=int, help='Day of recording session')
    parser.add_argument('--Epoch', type=int,
                        help='Epoch number of recording session')
    return parser.parse_args()


def queue_job(python_cmd, directives=None, log_file='log.log',
              job_name='job'):
    queue_cmd = f'qsub {directives} -j y -o {log_file} -N {job_name}'
    cmd_line_script = f'echo python {python_cmd}  | {queue_cmd}'
    run(cmd_line_script, shell=True)


def main():
    # Set the maximum number of threads for openBLAS to use.
    NUM_THREADS = 16
    environ['OPENBLAS_NUM_THREADS'] = str(NUM_THREADS)
    environ['NUMBA_NUM_THREADS'] = str(NUM_THREADS)
    environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
    LOG_DIRECTORY = join(getcwd(), 'logs')
    makedirs(LOG_DIRECTORY,  exist_ok=True)

    python_function = 'run_by_epoch.py'
    directives = ' '.join(
        ['-l h_rt=7:00:00', f'-pe omp {NUM_THREADS}',
         '-P braincom', '-notify', '-l mem_total=125G',
         '-v OPENBLAS_NUM_THREADS', '-v NUMBA_NUM_THREADS',
         '-v OMP_NUM_THREADS'])

    args = get_command_line_arguments()
    if args.Animal is None and args.Day is None and args.Epoch is None:
        epoch_info = make_epochs_dataframe(ANIMALS)
        neuron_info = make_neuron_dataframe(ANIMALS)
        n_neurons = (neuron_info
                     .groupby(['animal', 'day', 'epoch'])
                     .neuron_id
                     .agg(len)
                     .rename('n_neurons')
                     .to_frame())

        epoch_info = epoch_info.join(n_neurons)
        is_w_track = (epoch_info.environment
                      .isin(['TrackA', 'TrackB', 'WTrackA', 'WTrackB']))
        epoch_keys = epoch_info[is_w_track & (epoch_info.n_neurons > 20)].index
    else:
        epoch_keys = [(args.Animal, args.Day, args.Epoch)]

    for animal, day, epoch in epoch_keys:
        print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')

        log_file = f'{animal}_{day:02d}_{epoch:02d}.log'
        function_name = python_function.replace('.py', '')
        job_name = f'{function_name}_{animal}_{day:02d}_{epoch:02d}'
        python_cmd = f'{python_function} {animal} {day} {epoch}'
        queue_job(python_cmd,
                  directives=directives,
                  log_file=join(LOG_DIRECTORY, log_file),
                  job_name=job_name)


if __name__ == '__main__':
    exit(main())
