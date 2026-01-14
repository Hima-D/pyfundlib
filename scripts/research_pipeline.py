from __future__ import annotations

import argparse

from pyfundlib.backtester.engine import Backtester
from pyfundlib.data.fetcher import DataFetcher
from pyfundlib.reporting.validation import validate_equity_curve, summarize_validation
from pyfundlib.strategies import get_strategy


def run_research(
    ticker: str,
    strategy_name: str,
    period: str,
) -> None:
    df = DataFetcher.get_price(ticker, period=period)
    if df.empty:
        print(f"No data for {ticker} over period {period}")
        return

    strategy = get_strategy(strategy_name)
    bt = Backtester(strategy=strategy, data=df, name=f"research_{strategy_name}_{ticker}")
    result = bt.run()

    print(f"Backtest completed for {strategy_name} on {ticker}")
    print(f"Final equity: {result.equity_curve.iloc[-1]:,.2f}")
    print(f"Core metrics: {result.metrics}")

    validation = validate_equity_curve(result.equity_curve)
    summary = summarize_validation(validation)

    print("\nValidation summary:")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full research pipeline for a ticker and strategy.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ts_momentum",
        help="Strategy name from STRATEGY_REGISTRY (e.g., rsi, sma_crossover, ts_momentum)",
    )
    parser.add_argument("--period", type=str, default="5y", help="Data period (e.g., 1y, 3y, 5y, max)")

    args = parser.parse_args()
    run_research(args.ticker, args.strategy, args.period)


if __name__ == "__main__":
    main()

