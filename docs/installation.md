### Installation

`replay_trajectory_classification` can be installed through PyPI or conda. **Conda is strongly recommended** to ensure that all complex scientific dependencies are installed properly.

#### Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **Scientific Computing Stack**: NumPy, SciPy, pandas, matplotlib, scikit-learn, etc.

#### Recommended Installation (Conda)

```bash
conda install -c edeno replay_trajectory_classification
```

#### Alternative Installation (PyPI)

```bash
pip install replay_trajectory_classification
```

### Developer Installation

For contributors and researchers who want to modify the code:

#### Step 1: Prerequisites

Install miniconda or anaconda:

```bash
# Linux/macOS
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r

# macOS (alternative)
brew install --cask miniconda

# Windows: Download installer from https://docs.conda.io/en/latest/miniconda.html
```

#### Step 2: Clone Repository

```bash
git clone https://github.com/Eden-Kramer-Lab/replay_trajectory_classification.git
cd replay_trajectory_classification
```

#### Step 3: Environment Setup

```bash
# Update conda to latest version
conda update -n base conda

# Create isolated environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate replay_trajectory_classification
```

#### Step 4: Development Installation

Choose your installation type:

```bash
# Basic development installation
pip install -e .

# With development tools (recommended for contributors)
pip install -e '.[dev]'      # Includes ruff linter, jupyter, testing tools

# Specific dependency groups
pip install -e '.[test]'     # Testing dependencies only
pip install -e '.[docs]'     # Documentation building tools
```

#### Step 5: Verification

Test your installation:

```bash
# Verify package imports correctly
python -c "import replay_trajectory_classification; print('✓ Package installed successfully')"

# Run code quality checks
ruff check replay_trajectory_classification/

# Test a tutorial notebook
jupyter nbconvert --execute notebooks/tutorial/01-Introduction_and_Data_Format.ipynb
```

### Building Distribution Packages

For maintainers building releases:

```bash
# Install build tools
pip install build twine

# Build wheel and source distribution
python -m build

# Upload to PyPI (maintainers only)
twine upload dist/*
```

### Troubleshooting

#### Common Issues

**Import errors**: Ensure you've activated the conda environment:
```bash
conda activate replay_trajectory_classification
```

**Missing dependencies**: Recreate the environment:
```bash
conda env remove -n replay_trajectory_classification
conda env create -f environment.yml
```

**GPU support**: Install CuPy for GPU acceleration:
```bash
conda install cupy
# or
pip install cupy-cuda11x  # for CUDA 11.x
```
