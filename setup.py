#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy >= 1.11', 'statsmodels', 'numba', 'matplotlib',
                    'xarray', 'scipy', 'scikit-learn', 'regularized_glm',
                    'dask', 'patsy', 'loren_frank_data_processing']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='replay_trajectory_classification',
    version='0.1.0.dev0',
    license='MIT',
    description=('Classify replay trajectories.'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/Eden-Kramer-Lab/replay_trajectory_classification',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
