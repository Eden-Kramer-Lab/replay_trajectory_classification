# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import shutil
import sys

import replay_trajectory_classification

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "replay_trajectory_classification"
copyright = "2022, Eric Denovellis"
author = "Eric Denovellis"

# The full version, including alpha/beta/rc tags

# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
version = re.sub(
    r"(\d+\.\d+)\.\d+(.*)", r"\1\2", replay_trajectory_classification.__version__
)
version = re.sub(r"(\.dev\d+).*?$", r"\1", version)
# The full version, including alpha/beta/rc tags.
release = replay_trajectory_classification.__version__
print("%s %s" % (version, release))

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "nbsphinx",  # Integrate Jupyter Notebooks and Sphinx
    "numpydoc",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",  # syntax highlighting
]
autosummary_generate = True
add_module_names = False
numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = {".rst": "restructuredtext", ".myst": "myst-nb", ".ipynb": "myst-nb"}

language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "setup.py",
    "README.md",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "search_bar_text": "Search this site...",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Eden-Kramer-Lab/replay_trajectory_classification/",
            "icon": "fa-brands fa-github",
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
}

# -- MyST and MyST-NB ---------------------------------------------------

# MyST
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]

# MyST-NB
nb_execution_mode = "cache"
nb_execution_mode = "off"


def copy_tree(src, tar):
    """Copies over notebooks into the documentation folder, so get around an issue where nbsphinx
    requires notebooks to be in the same folder as the documentation folder"""
    if os.path.exists(tar):
        shutil.rmtree(tar)
    shutil.copytree(src, tar)


# -- Get Jupyter Notebooks ---------------------------------------------------

copy_tree("../notebooks/tutorial", "./_copied_over/notebooks")

# Report warnings for all validation checks except GL01, GL02, and GL05
numpydoc_validation_checks = {"all", "GL01", "GL02", "GL05"}
