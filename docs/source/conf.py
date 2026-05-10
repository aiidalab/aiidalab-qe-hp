"""Configuration file for the Sphinx documentation builder."""
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
import sys
import time

sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'AiiDAlab QE HP Plugin'
copyright_owners = 'The AiiDAlab Team'
copyright_first_year = '2024'
current_year = str(time.localtime().tm_year)
copyright_year_string = (
    current_year if current_year == copyright_first_year else f'{copyright_first_year}-{current_year}'
)
copyright = f'{copyright_year_string}, {copyright_owners}. All rights reserved'


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_design',
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaste
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Using this instead of `html_logo` to handle light/dark modes
# See https://pydata-sphinx-theme.readthedocs.io
html_theme_options = {
    'source_repository': 'https://github.com/aiidaplugins/aiida-qe-xspec/',
    'source_branch': 'main',
    'source_directory': 'docs/source',
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/aiidaplugins/aiida-qe-xspec',
            'icon': 'fa-brands fa-square-github',
            'type': 'fontawesome',
        },
    ],
}
