from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pyfundlib.agents.orchestrator import QuantCrew
from pyfundlib.backtester.engine import Backtester
from pyfundlib.data.fetcher import DataFetcher
from pyfundlib.strategies import STRATEGY_REGISTRY, get_strategy
from pyfundlib.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class VoiceCommandResult:
    intent: str
    raw_text: str
    response: str


class VoiceCommandRouter:
    def __init__(self, crew: Optional[QuantCrew] = None):
        self.crew = crew or QuantCrew()

    def _parse_ticker(self, text: str) -> Optional[str]:
        tokens = [t.strip().upper() for t in text.replace(",", " ").split()]
        for t in tokens:
            if len(t) >= 3 and t.isalnum():
                return t
        return None

    def _parse_strategy_name(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for name in STRATEGY_REGISTRY.keys():
            if name in lowered:
                return name
        if "momentum" in lowered:
            return "ts_momentum"
        if "rsi" in lowered:
            return "rsi"
        if "sma" in lowered or "crossover" in lowered:
            return "sma_crossover"
        return None

    def handle(self, text: str) -> VoiceCommandResult:
        cleaned = text.strip()
        lowered = cleaned.lower()

        if any(k in lowered for k in ["backtest", "run test", "simulate"]):
            return self._handle_backtest(cleaned)
        if any(k in lowered for k in ["analyze", "analysis", "view", "report"]):
            return self._handle_quant_analysis(cleaned)
        return self._fallback(cleaned)

    def _handle_backtest(self, text: str) -> VoiceCommandResult:
        ticker = self._parse_ticker(text) or "AAPL"
        strat_name = self._parse_strategy_name(text) or "ts_momentum"
        try:
            df = DataFetcher.get_price(ticker, period="3y")
            strategy = get_strategy(strat_name)
            bt = Backtester(strategy=strategy, data=df, name=f"voice_{strat_name}_{ticker}")
            result = bt.run()
            metrics = result.metrics
            msg = (
                f"Backtest for {strat_name} on {ticker}: "
                f"CAGR {metrics.get('cagr', 0.0):.1%}, "
                f"Sharpe {metrics.get('sharpe', 0.0):.2f}, "
                f"Max drawdown {metrics.get('max_drawdown', 0.0):.1%}."
            )
            return VoiceCommandResult(intent="backtest", raw_text=text, response=msg)
        except Exception as e:
            logger.error("voice_backtest_failed", error=str(e))
            return VoiceCommandResult(
                intent="backtest",
                raw_text=text,
                response="Backtest failed due to an internal error.",
            )

    def _handle_quant_analysis(self, text: str) -> VoiceCommandResult:
        ticker = self._parse_ticker(text) or "AAPL"
        try:
            report = self.crew.analyze_ticker(ticker)
            return VoiceCommandResult(
                intent="quant_analysis",
                raw_text=text,
                response=f"Quant analysis for {ticker}:\n{report}",
            )
        except Exception as e:
            logger.error("voice_quant_analysis_failed", error=str(e))
            return VoiceCommandResult(
                intent="quant_analysis",
                raw_text=text,
                response="Quant analysis failed due to an internal error.",
            )

    def _fallback(self, text: str) -> VoiceCommandResult:
        msg = "I can run backtests or analyze a ticker. For example, say: run a momentum backtest on AAPL."
        return VoiceCommandResult(intent="fallback", raw_text=text, response=msg)

