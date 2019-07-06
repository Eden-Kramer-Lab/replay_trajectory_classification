#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy', 'statsmodels', 'numba', 'matplotlib',
                    'pandas', 'xarray', 'scipy <= 1.2', 'scikit-learn',
                    'regularized_glm', 'dask', 'patsy', 'networkx']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='replay_trajectory_classification',
    version='0.5.8.dev0',
    license='MIT',
    description=('Classify replay trajectories.'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/Eden-Kramer-Lab/replay_trajectory_classification',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
