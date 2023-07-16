# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Teras'
copyright = '2023, Khawaja Abaid Ullah'
author = 'Khawaja Abaid Ullah'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

autosummary_generate = True

extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode", "sphinx.ext.autosummary", "sphinx.ext.napoleon"]

templates_path = ['_templates']
exclude_patterns = ['config', 'teras/config']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


def skip_member(app, what, name, obj, skip, options):
    exclusions = ['call', 'compile', 'build', 'train_step', 'predict_step', 'get_config']
    # Add the names of the methods you want to exclude from the documentation

    if name in exclusions:
        return True

    return None


def setup(app):
    app.connect('autodoc-skip-member', skip_member)
