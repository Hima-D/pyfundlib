"""
PyFundLib: Institutional-Grade Algorithmic Trading Framework
Version: 2025.1.0

End-to-end Python library for:
- Backtesting and paper trading
- ML-powered alpha generation (LSTM, XGBoost, Random Forest)
- Multi-broker live execution (Alpaca, Zerodha, IBKR, Binance)
- Statistical validation (Deflated Sharpe Ratio, Parameter Stability)
- Real-time monitoring and automated rebalancing
"""

__version__ = "2025.1.0"
__author__ = "Himanshu Dixit"
__email__ = "hima@pyfund.tech"

# Core imports
from pyfundlib.data import DataFetcher, DataCached, UniverseManager
from pyfundlib.backtester import Backtester, PerformanceReport, TradeLog
from pyfundlib.strategies import (
    StrategyBase,
    SMACrossover,
    RSIMeanReversion,
    PairsTradingStrategy,
    DonchianBreakout,
)
from pyfundlib.ml import MLPredictor, XGBoostModel, LSTMModel, RandomForestModel
from pyfundlib.brokers import BrokerFactory, PaperBroker, AlpacaBroker, ZerodhaBroker
from pyfundlib.utils import (
    SystemMonitor,
    Scheduler,
    StatisticalValidator,
    PortfolioAnalyzer,
    Logger,
)
from pyfundlib.config import Config

__all__ = [
    "DataFetcher",
    "DataCached",
    "UniverseManager",
    "Backtester",
    "PerformanceReport",
    "TradeLog",
    "StrategyBase",
    "SMACrossover",
    "RSIMeanReversion",
    "PairsTradingStrategy",
    "DonchianBreakout",
    "MLPredictor",
    "XGBoostModel",
    "LSTMModel",
    "RandomForestModel",
    "BrokerFactory",
    "PaperBroker",
    "AlpacaBroker",
    "ZerodhaBroker",
    "SystemMonitor",
    "Scheduler",
    "StatisticalValidator",
    "PortfolioAnalyzer",
    "Logger",
    "Config",
]
