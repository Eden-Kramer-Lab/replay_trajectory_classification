# replay_trajectory_classification
[![DOI](https://zenodo.org/badge/177004334.svg)](https://zenodo.org/badge/latestdoi/177004334)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Eden-Kramer-Lab/replay_trajectory_classification/master)

### Installation ###

`replay_trajectory_classification` can be installed through pypi or conda. Conda is the best way to ensure that everything is installed properly.

```bash
pip install replay_trajectory_classification
```
Or

```bash
conda install -c edeno replay_trajectory_classification
```

### Tutorials ###
There are three jupyter notebooks introducing the package:

1. [01-Introduction_and_Data_Format](notebooks/tutorial/01-Introduction_and_Data_Format.ipynb): How to get your data in the correct format to use with the decoder.
2. [02-Decoding_with_Sorted_Spikes](notebooks/tutorial/02-Decoding_with_Sorted_Spikes.ipynb): How to decode using a single movement model using sorted spikes.
3. [03-Decoding_with_Clusterless_Spikes](notebooks/tutorial/03-Decoding_with_Clusterless_Spikes.ipynb): How to decode using a single movement model using the "clusterless" approach --- which does not require spike sorting.

A fourth notebook explaining the classifier, which uses multiple movement models, is forthcoming.

### Developer Installation ###
For people who want to expand upon the code for their own use:

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
