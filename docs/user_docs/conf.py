# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Segmenter ML Plugin for napari"
copyright = "2024, Allen Institute for Cell Science"
author = "Segmenter ML plugin team"
release = "1.0.0rc8"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinxext.opengraph",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_togglebutton",
    "sphinx_design",
]

myst_enable_extensions = [
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_image",
    "linkify",
    "substitution",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


def setup(app):
    app.add_css_file("theme_1_TD.css")


html_theme = "furo"
html_title = "Segmenter ML - User Guide"
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "images/thumb_segmenterML_2_onLight.png",
    "dark_logo": "images/thumb_segmenterML_2_onDark.png",
}


# html_css_files = ["custom.css"]
