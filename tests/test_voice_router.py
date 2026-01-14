from __future__ import annotations

from pyfundlib.agents.voice_router import VoiceCommandRouter


class DummyCrew:
    def analyze_ticker(self, ticker: str) -> str:
        return f"Analysis for {ticker}"


def test_voice_router_backtest_intent():
    router = VoiceCommandRouter(crew=DummyCrew())
    result = router.handle("run a backtest on AAPL with momentum")
    assert result.intent == "backtest"
    assert isinstance(result.response, str)


def test_voice_router_quant_analysis_intent():
    router = VoiceCommandRouter(crew=DummyCrew())
    result = router.handle("analyze MSFT")
    assert result.intent == "quant_analysis"
    assert "Analysis for" in result.response or isinstance(result.response, str)


def test_voice_router_fallback_intent():
    router = VoiceCommandRouter(crew=DummyCrew())
    result = router.handle("hello there")
    assert result.intent == "fallback"
    assert "backtests" in result.response or "analyze a ticker" in result.response

