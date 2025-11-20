.. pyfundlib documentation master file, created by
   sphinx-quickstart on Thu Nov 21 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyFundLib's documentation!
=====================================

.. image:: https://img.shields.io/pypi/v/pyfundlib?style=flat-square
   :target: https://pypi.org/project/pyfundlib/
   :alt: PyPI

.. image:: https://img.shields.io/github/stars/Hima-D/pyfundlib?style=social
   :target: https://github.com/Hima-D/pyfundlib
   :alt: GitHub stars

**PyFundLib** — The Ultimate Python Framework for Algorithmic Trading & ML Alpha

A complete, production-grade quant stack: backtesting, ML predictions, live execution, automation, and beautiful reporting — all in pure Python.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Features
--------

- Institutional-grade backtester
- Full ML pipeline (LSTM, XGBoost, RF) with MLflow
- Per-ticker model registry & versioning
- Live trading (Alpaca, Zerodha, IBKR)
- 24/7 automation with scheduler
- Beautiful CLI, plots, and reports
- One-command deployment

Quickstart
----------

.. code-block:: bash

   pip install pyfundlib
   pyfundlib backtest SPY --strategy ml_random_forest
   pyfundlib automate --mode paper

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`