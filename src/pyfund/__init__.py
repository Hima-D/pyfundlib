"""pyfund - End-to-End Python Toolkit for Algo Trading."""

__version__ = "0.1.0"
__author__ = "Himanshu Dixit"

from .automation.runner import AutomationRunner
from .backtester import Backtester
from .data import features, fetcher, processor
from .execution.live import LiveExecutor
from .indicators import macd, rsi, sma
from .ml.predictor import MLPredictor
from .portfolio.allocator import PortfolioAllocator
from .strategies import MLRandomForestStrategy, RSIMeanReversionStrategy

__all__ = [
    "AutomationRunner",
    "AutomationRunner",
    "Backtester",
    "Backtester",
    "LiveExecutor",
    "LiveExecutor",
    "MLPredictor",
    "MLPredictor",
    "MLRandomForestStrategy",
    "PortfolioAllocator",
    "PortfolioAllocator",
    "RSIMeanReversionStrategy",
    "auto_report",
    "deflated_sharpe_ratio",
    "describe_financial",
    "features",
    "fetcher",
    "macd",
    "probabilistic_sharpe_ratio",
    "processor",
    "rsi",
    "sma",
]
from .backtester.auto_report import run as auto_report
from .econometrics.core import (
    deflated_sharpe_ratio,
    describe_financial,
    probabilistic_sharpe_ratio,
)
