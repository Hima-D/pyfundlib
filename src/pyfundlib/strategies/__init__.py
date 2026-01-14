# src/pyfund/strategies/__init__.py
"""
pyfund.strategies - Complete collection of professional trading strategies

Includes:
- Classic technical (RSI, SMA, Donchian)
- Statistical arbitrage (Pairs Trading)
- Machine Learning
- High-Frequency (Micro Reversion, Market Making)
- Event-driven (Earnings Drift)
- Volatility arbitrage (Delta-Hedged Straddle)
- Risk Parity allocation
- And more...
"""

# === Portfolio Construction (sometimes used as strategy overlay) ===
from .risk_parity import RiskParityAllocator
from .base import BaseStrategy, SignalResult

# === Options & Volatility Arbitrage ===
from .delta_hedged_straddle import DeltaHedgedStraddleStrategy
from .donchian_breakout import DonchianBreakoutStrategy

# === Event-Driven ===
from .earnings_drift import EarningDriftStrategy

# === High-Frequency & Market Making ===
from .hft_micro_reversion import HFTMicroReversionStrategy
from .market_making import MarketMakingStrategy

# === Machine Learning ===
from .ml_random_forest import MLRandomForestStrategy
from .time_series_momentum import TimeSeriesMomentumStrategy

# === Statistical & Quantitative ===
from .pairs_trading import PairsTradingStrategy

# === Classic Technical Analysis ===
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .sma_crossover import SMACrossoverStrategy
from .triangular_arb import TriangularArbitrageStrategy

# === Execution Algorithms ===
from .vwap_execution import VWAPExecutor

# Export everything for easy importing
__all__ = [
    # Core
    "BaseStrategy",
    "SignalResult",
    # Technical
    "RSIMeanReversionStrategy",
    "SMACrossoverStrategy",
    "DonchianBreakoutStrategy",
    # Stat Arb
    "PairsTradingStrategy",
    "TriangularArbitrageStrategy",
    # ML
    "MLRandomForestStrategy",
    "TimeSeriesMomentumStrategy",
    # HFT
    "HFTMicroReversionStrategy",
    "MarketMakingStrategy",
    # Event
    "EarningDriftStrategy",
    # Options/Vol
    "DeltaHedgedStraddleStrategy",
    # Execution
    "VWAPExecutor",
    # Portfolio
    "RiskParityAllocator",
]

# Optional: convenience dictionary for strategy registry
STRATEGY_REGISTRY = {
    "rsi": RSIMeanReversionStrategy,
    "sma_crossover": SMACrossoverStrategy,
    "donchian": DonchianBreakoutStrategy,
    "pairs": PairsTradingStrategy,
    "triangular_arb": TriangularArbitrageStrategy,
    "ml_rf": MLRandomForestStrategy,
    "ts_momentum": TimeSeriesMomentumStrategy,
    "hft_micro": HFTMicroReversionStrategy,
    "market_making": MarketMakingStrategy,
    "earning_drift": EarningDriftStrategy,
    "hedged_straddle": DeltaHedgedStraddleStrategy,
    "vwap": VWAPExecutor,
    "risk_parity": RiskParityAllocator,
}


def get_strategy(name: str, **params):
    """Factory function to instantiate strategy by name"""
    cls = STRATEGY_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return cls(params=params)


# Make it pretty when imported
__version__ = "1.0.0"
__author__ = "Himanshu Dixit"

print(f"pyfund.strategies loaded - {len(__all__)} professional strategies ready")
from .arima_garch_reversion import ARIMAGARCHStrategy
