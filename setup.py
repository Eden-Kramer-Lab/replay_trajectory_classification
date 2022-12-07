#!/usr/bin/env python3

import codecs
import os.path

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


INSTALL_REQUIRES = [
    "numpy",
    "statsmodels",
    "numba",
    "matplotlib",
    "pandas",
    "xarray",
    "scipy",
    "scikit-learn",
    "scikit-image",
    "regularized_glm",
    "dask",
    "patsy",
    "networkx",
    "joblib",
    "track_linearization",
    "tqdm",
    "seaborn",
]
TESTS_REQUIRE = ["pytest >= 2.7.1"]

setup(
    name="replay_trajectory_classification",
    version=get_version("replay_trajectory_classification/__init__.py"),
    license="MIT",
    description=("Classify replay trajectories."),
    author="Eric Denovellis",
    author_email="eric.denovellis@ucsf.edu",
    url="https://github.com/Eden-Kramer-Lab/replay_trajectory_classification",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
