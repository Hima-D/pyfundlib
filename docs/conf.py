# docs/conf.py
import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add src/ directory so Sphinx can import pyfund
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "PyFundLib"
author = "Himanshu Dixit"
copyright = f"{datetime.now().year}, {author}"

# For release versions, you can use hatch or manually set
release = "0.1.6"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",           # Extract docstrings
    "sphinx.ext.napoleon",          # Google / NumPy style
    "sphinx.ext.viewcode",          # Link to source code
    "sphinx.ext.autosummary",       # Generate summary tables
    "sphinx_autodoc_typehints",     # Show type hints in docs
    "myst_parser",                  # Markdown support
    "sphinx_copybutton",            # Copy code button
    "sphinxext.opengraph",          # OpenGraph social previews
    "autoapi.extension",            # Automatic API documentation
]

# Templates and static
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autosummary settings
autosummary_generate = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__weakref__, __dict__, __module__",
}

# Type hints
set_type_checking_flag = True
typehints_fully_qualified = False
always_document_param_types = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "PyFundLib ⚡"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#2E86AB",
        "color-brand-content": "#F18F01",
    },
    "dark_css_variables": {
        "color-brand-primary": "#F18F01",
        "color-brand-content": "#2E86AB",
    },
    "announcement": "<b>🚀 PyFundLib 1.0 is here!</b> The future of open-source quant trading.",
}

# OpenGraph
ogp_site_url = "https://pyfundlib.com"
ogp_image = "_static/og-image.png"
ogp_description_length = 200
ogp_type = "website"

# MyST Parser (Markdown support)
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# AutoAPI configuration: scan all subpackages
autoapi_type = "python"
autoapi_dirs = ["../src/pyfund"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_keep_files = True
autoapi_add_toctree_entry = True

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "xgboost": ("https://xgboost.readthedocs.io/en/latest/", None),
}

# -- End of configuration ----------------------------------------------------
