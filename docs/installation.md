### Installation

`replay_trajectory_classification` can be installed through pypi or conda. Conda is the best way to ensure that all the dependencies are installed properly.

```bash
pip install replay_trajectory_classification
```

Or

```bash
conda install -c edeno replay_trajectory_classification
```

### Developer Installation

For people who want to expand upon the code for their own use:

1. Install miniconda (or anaconda) if it isn't already installed. Type into bash (or install from the anaconda website):

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

2. Go to the local repository on your computer (`cd replay_trajectory_classification`) and install the anaconda environment for the repository. Type into bash:

```bash
conda update -n base conda # make sure conda is up to date
conda env create -f environment.yml # create a conda environment
conda activate replay_trajectory_classification # activate conda environment
python setup.py develop
```
