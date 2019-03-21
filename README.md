# replay_trajectory_classification
[![DOI](https://zenodo.org/badge/177004334.svg)](https://zenodo.org/badge/latestdoi/177004334)


### Installation

`replay_trajectory_classification` can be installed through pypi or conda. Conda is the best way to ensure that everything is installed properly.

```bash
pip install replay_trajectory_classification
python setup.py install
```
Or

```bash
conda install -c edeno replay_trajectory_classification
python setup.py install
```

### Developer Installation ###
1. Install miniconda (or anaconda) if it isn't already installed. Type into bash (or install from the anaconda website):
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

2. Go to the local repository on your computer (cd `.../replay_trajectory_classification`) and install the anaconda environment for the repository. Type into bash:
```bash
conda update -n base conda # make sure conda is up to date
conda env create -f environment.yml # create a conda environment
conda activate replay_trajectory_classification # activate conda environment
python setup.py develop
```
