# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`replay_trajectory_classification` is a Python package for decoding spatial position from neural activity and categorizing trajectory types in hippocampal replay events. This is a computational neuroscience package that prioritizes scientific accuracy and computational efficiency.

## Essential Commands

### Environment Setup (REQUIRED)

Always use conda - pip-only installations will fail due to complex scientific dependencies:

```bash
# Update conda first (required)
conda update -n base conda

# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate replay_trajectory_classification

# Development installation (required for code changes)
pip install -e .

# Install with optional dependencies
pip install -e '.[dev]'     # Development tools (ruff, jupyter, etc.)
pip install -e '.[test]'    # Testing tools
pip install -e '.[docs]'    # Documentation tools
```

### Validation Commands

Test package installation:

```bash
python -c "import replay_trajectory_classification; print('Package imported successfully')"
```

*Note: "Cupy not installed" warnings are normal without GPU*

Lint check (modern):

```bash
ruff check replay_trajectory_classification/
```

Legacy flake8 also works:
```bash
flake8 replay_trajectory_classification/ --max-line-length=88 --select=E9,F63,F7,F82 --show-source --statistics
```

### Testing

The main test suite runs tutorial notebooks (takes ~10-15 minutes total):

```bash
# Test individual notebook
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python3 --execute notebooks/tutorial/01-Introduction_and_Data_Format.ipynb --output-dir=/tmp

# Test all notebooks (CI equivalent)
for nb in notebooks/tutorial/*.ipynb; do
  jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 --execute "$nb"
done
```

## Architecture Overview

### Core Package Structure (`replay_trajectory_classification/`)

**Main API Classes** (exported in `__init__.py`):

- `SortedSpikesClassifier/Decoder`: For spike-sorted neural data
- `ClusterlessClassifier/Decoder`: For clusterless (unsorted) neural data
- `Environment`: Spatial environment representation with discrete grids
- Movement models: `RandomWalk`, `EmpiricalMovement`, `Identity`, etc.

**Key Modules**:

- `classifier.py`: Base classes for trajectory classification (49k lines)
- `decoder.py`: Core decoding functionality
- `environments.py`: Spatial environment handling (33k lines)
- `core.py`: Low-level computational functions
- `likelihoods/`: Subpackage with various likelihood models

**Likelihood Algorithms** (`likelihoods/`):

- **Sorted spikes**: GLM-based (`spiking_likelihood_glm.py`), KDE-based (`spiking_likelihood_kde.py`)
- **Clusterless**: Multiunit likelihood variants (`multiunit_likelihood*.py`)
- **GPU support**: `*_gpu.py` variants require CuPy installation
- **Calcium imaging**: `calcium_likelihood.py`

### Dependencies Architecture

**Essential External Packages**:

- `track_linearization`: Spatial track handling (imported in `__init__.py`)
- `regularized_glm`: Custom GLM implementation
- Standard scientific stack: NumPy, SciPy, scikit-learn, pandas, xarray

**Conda Channels Required**:

- `conda-forge`: Standard scientific packages
- `franklab`: Lab-specific neuroscience tools
- `edeno`: Author's specialized packages

## Development Guidelines

### Code Patterns

- **State-space models**: Core pattern throughout codebase
- **Fit/Estimate pattern**: Likelihood functions use `fit_*` (training) and `estimate_*` (inference)
- **GPU/CPU variants**: Many functions have `*_gpu` equivalents requiring CuPy
- **Clusterless vs Sorted**: Dual pathways throughout for different data types

### Critical Requirements

1. **Always use conda environment** - complex scientific dependencies
2. **Development mode installation** - `python setup.py develop` for code changes
3. **Notebook-based testing** - integration tests via tutorial notebooks
4. **Scientific accuracy priority** - computational correctness over traditional software practices

### Common Issues

- **GPU warnings**: "Cupy not installed" messages are normal for CPU-only setups
- **Long notebook execution**: Tutorial notebooks take 2-3 minutes each
- **Setup.py deprecation warnings**: Expected but installation succeeds
- **Documentation builds**: May require manual intervention due to jupytext dependency issues

### Files to Never Modify

- Tutorial notebooks in `notebooks/tutorial/` - these are integration tests
- `environment.yml` - carefully balanced scientific dependencies
- Version string in `replay_trajectory_classification/__init__.py` - managed by maintainers

## Testing Strategy

**Primary Testing**: Execute all 5 tutorial notebooks sequentially

1. `01-Introduction_and_Data_Format.ipynb`: Data format requirements
2. `02-Decoding_with_Sorted_Spikes.ipynb`: Basic sorted spike decoding
3. `03-Decoding_with_Clusterless_Spikes.ipynb`: Basic clusterless decoding
4. `04-Classifying_with_Sorted_Spikes.ipynb`: Multi-model classification
5. `05-Classifying_with_Clusterless_Spikes.ipynb`: Multi-model clusterless

**CI Pipeline**: GitHub Actions runs all notebooks on Ubuntu with Python 3.11

## Modernization Status

**✅ COMPLETE - All 3 Phases Implemented**:

**Phase 1 & 2 (Foundation + Metadata)**:
- ✅ Added `pyproject.toml` with modern PEP 621 project metadata
- ✅ Configured `ruff` as modern linter/formatter
- ✅ Migrated all dependencies and optional dependency groups
- ✅ Updated Python version support (3.10+, was incorrectly 3.11+)

**Phase 3 (Build System Modernization)**:
- ✅ **Removed `setup.py` completely** - now pure `pyproject.toml`
- ✅ Explicit package configuration for clean builds
- ✅ Updated CI/CD to include modern `build` tool
- ✅ Wheel building works perfectly without legacy setup.py

**Current State**:
- **Pure modern installation**: `pip install -e .` (no more setup.py)
- **Optional dependencies**: `pip install -e '.[dev]'`, `'.[test]'`, `'.[docs]'`
- **Modern linting**: `ruff check` (67 style issues detected but non-breaking)
- **Modern building**: `python -m build --wheel` creates proper wheels
- **Full compatibility**: All notebooks and functionality preserved

## Key Architectural Decisions

1. **Notebook-based testing**: Integration tests using real scientific workflows rather than isolated unit tests
2. **GPU-optional design**: All functionality works on CPU with GPU acceleration available
3. **Dual data pathways**: Separate but parallel implementations for sorted vs clusterless data
4. **Conda-first distribution**: Complex dependency tree requires conda package management
5. **Scientific reproducibility**: Version pinning and environment specification prioritized over flexibility
6. **Incremental modernization**: Modern tooling added alongside legacy support for seamless transition
