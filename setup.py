#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy', 'statsmodels', 'numba', 'matplotlib',
                    'pandas', 'xarray', 'scipy', 'scikit-learn', 'scikit-image',
                    'regularized_glm', 'dask', 'patsy', 'networkx', 'joblib',
                    'track_linearization', 'tqdm', 'seaborn']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='replay_trajectory_classification',
    version='1.2.9',
    license='MIT',
    description=('Classify replay trajectories.'),
    author='Eric Denovellis',
    author_email='eric.denovellis@ucsf.edu',
    url='https://github.com/Eden-Kramer-Lab/replay_trajectory_classification',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
