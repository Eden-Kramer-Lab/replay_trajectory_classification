# Copilot Instructions for replay_trajectory_classification

## Repository Summary

`replay_trajectory_classification` is a Python package for decoding spatial position from neural activity and categorizing trajectory types, specifically designed for analyzing hippocampal replay events in neuroscience research. The package provides state-space models that can decode position from both spike-sorted cells and clusterless spikes, with support for GPU acceleration and complex 1D/2D environments.

## High-Level Repository Information

- **Size**: ~63MB with 28 Python files
- **Type**: Scientific Python package for computational neuroscience
- **Primary Language**: Python 3.10+ (configured for Python 3.13 in current environment)
- **Key Dependencies**: NumPy, SciPy, scikit-learn, numba, xarray, dask, matplotlib, pandas
- **Documentation**: Sphinx-based documentation with ReadTheDocs hosting
- **License**: MIT

## Environment Setup and Build Instructions

### Prerequisites
Always use conda for environment management due to complex scientific dependencies:

```bash
# Update conda first (required)
conda update -n base conda

# Create environment from environment.yml (required)
conda env create -f environment.yml

# Activate environment
conda activate replay_trajectory_classification
```

### Installation Commands
**ALWAYS install in development mode for code changes:**

```bash
# Modern development installation (recommended)
pip install -e .

# With optional development tools
pip install -e '.[dev]'      # Includes ruff, jupyter, testing tools
pip install -e '.[test]'     # Testing dependencies only
pip install -e '.[docs]'     # Documentation building tools
```

**Note**: The repository has been fully modernized to use `pyproject.toml`. The old `setup.py develop` command is no longer available.

### Validation Commands

#### Package Import Test
```bash
python -c "import replay_trajectory_classification; print('Package imported successfully')"
```
**Expected output**: "Cupy is not installed or GPU is not detected. Ignore this message if not using GPU" followed by "Package imported successfully"

#### Linting
```bash
# Modern linting (preferred)
ruff check replay_trajectory_classification/

# Legacy flake8 still works
flake8 replay_trajectory_classification/ --max-line-length=88 --select=E9,F63,F7,F82 --show-source --statistics
```
**Expected**: Minimal output (style issues are non-breaking)

#### Notebook Testing (CI Validation)
The main test suite runs Jupyter notebooks. Test individual notebooks:
```bash
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute notebooks/tutorial/01-Introduction_and_Data_Format.ipynb --output-dir=/tmp
```
**Time required**: ~2-3 minutes per notebook
**Expected**: Notebook executes without errors

#### Documentation Build
```bash
# First install docs dependencies
pip install -r docs/requirements-docs.txt

# Note: Documentation build has dependency issues with jupytext in Makefile
# The docs can be built but require manual intervention
```

## Continuous Integration

The repository uses GitHub Actions (`.github/workflows/PR-test.yml`):
- **Trigger**: All pushes
- **OS**: Ubuntu latest only
- **Python**: 3.11 (but environment.yml uses current conda defaults)
- **Test Process**: Executes all 5 tutorial notebooks sequentially
- **Environment**: Uses conda with channels: conda-forge, franklab, edeno
- **Installation**: `pip install -e .` after conda environment setup

## Project Architecture and Layout

### Core Package Structure (`replay_trajectory_classification/`)
- **`__init__.py`**: Main API exports (ClassifierBase, Decoders, Environment, etc.)
- **`classifier.py`**: Base classes for trajectory classification with both sorted/clusterless approaches
- **`decoder.py`**: Core decoding functionality
- **`environments.py`**: Spatial environment representation with discrete grids
- **`core.py`**: Low-level computational functions
- **`likelihoods/`**: Subpackage with various likelihood models (KDE, GLM, multiunit, GPU variants)

### Key Configuration Files
- **`environment.yml`**: Conda environment specification with scientific computing stack
- **`setup.py`**: Package configuration and dependencies
- **`.readthedocs.yaml`**: Documentation build configuration
- **`docs/conf.py`**: Sphinx documentation configuration
- **`docs/requirements-docs.txt`**: Documentation build dependencies

### Documentation (`docs/`)
- **Sphinx-based** with ReadTheDocs hosting
- **API docs**: Auto-generated from docstrings
- **Installation guide**: `installation.md`
- **Build system**: Makefile (but has jupytext dependency issues)

### Tutorials (`notebooks/tutorial/`)
Five comprehensive Jupyter notebooks demonstrate package usage:
1. **01-Introduction_and_Data_Format.ipynb**: Data format requirements
2. **02-Decoding_with_Sorted_Spikes.ipynb**: Single movement model with sorted spikes
3. **03-Decoding_with_Clusterless_Spikes.ipynb**: Single movement model with clusterless approach
4. **04-Classifying_with_Sorted_Spikes.ipynb**: Multiple movement models with sorted spikes
5. **05-Classifying_with_Clusterless_Spikes.ipynb**: Multiple movement models with clusterless spikes

### Dependencies Not Obvious from Structure
- **track_linearization**: External package for spatial track handling (imported in `__init__.py`)
- **regularized_glm**: Custom GLM implementation
- **GPU dependencies**: CuPy for GPU acceleration (optional)
- **franklab & edeno conda channels**: Required for specialized neuroscience packages

## Important Development Notes

### Environment Requirements
- **ALWAYS** use the conda environment - pip-only installations will fail due to complex scientific dependencies
- **GPU support** requires CuPy installation (optional, warnings are normal without GPU)
- **Documentation builds** may require manual intervention due to jupytext path issues

### Testing Approach
- **Integration testing**: All 5 tutorial notebooks must execute successfully
- **CI dependency**: Notebooks test real scientific workflows, not isolated functions

### Common Issues and Workarounds
- **Documentation build**: Makefile expects jupytext in PATH but may not find conda environment version
- **Setup.py warnings**: Deprecation warnings are expected but installation succeeds
- **GPU warnings**: "Cupy not installed" messages are normal for CPU-only environments
- **Long notebook execution**: Tutorial notebooks can take 2-3 minutes each to execute

### File Exclusions (from .gitignore)
Key files to exclude from commits:
- Jupyter checkpoint files (`.ipynb_checkpoints`)
- Build artifacts (`_build`, `_autosummary`, `dist/`)
- Data files (`*.mat`, `*.csv`, `*.nc`)
- Cache files (`__pycache__`, `*.prof`)

## Validation Checklist for Changes

1. **Environment setup**: Conda environment creates successfully
2. **Installation**: `python setup.py develop` or `pip install -e .` succeeds
3. **Import test**: Package imports without errors (GPU warnings OK)
4. **Lint check**: flake8 passes with specified parameters
5. **Notebook execution**: All tutorial notebooks run successfully
6. **CI compatibility**: Changes don't break the GitHub Actions workflow

## Final Note

This package serves active neuroscience research. Changes should maintain scientific accuracy and computational efficiency. The codebase prioritizes correctness over traditional software engineering practices (hence notebook-based testing). Trust these instructions and only search for additional information if specific technical details are missing or incorrect.