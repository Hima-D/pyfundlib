# src/pyfundlib/automation/jobs/rebalance_portfolio.py
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from pyfund.data.fetcher import DataFetcher
from pyfund.utils.alerts import send_alert

from ...data.storage import DataStorage
from ...execution.live import LiveExecutor, OrderRequest
from ...ml.predictor import MLPredictor
from ...portfolio.allocator import PortfolioAllocator  # You'll love this one next
from ...risk.constraints import RiskConstraints
from ...utils.logger import get_logger

logger = get_logger(__name__)

# Config
REBALANCE_HOUR = 15  # 3 PM ET — end of day rebalance
MAX_TRADE_SIZE_PCT = 0.20  # No single trade >20% of portfolio
MIN_TRADE_THRESHOLD = 0.02  # Only trade if allocation diff >2%


def rebalance_job() -> None:
    """
    Daily portfolio rebalancing job.
    Runs at market close — turns ML signals into real trades.
    """
    logger.info("=== Starting Portfolio Rebalance Job ===")
    start_time = datetime.now()

    executor = LiveExecutor(dry_run=False)  # Set dry_run=True for paper
    predictor = MLPredictor()
    allocator = PortfolioAllocator()
    storage = DataStorage()
    constraints = RiskConstraints(
        max_position=0.25,
        max_sector=0.50,
        leverage_limit=2.0,
    )

    try:
        # 1. Get current account state
        account = executor.get_account()
        positions = executor.get_positions()  # {ticker: qty}
        # # cash = account["cash"]  # unused
        portfolio_value = account["portfolio_value"]

        current_weights = {}
        for ticker, qty in positions.items():
            if ticker == "CASH":
                continue
            price = DataFetcher.get_price(ticker).iloc[-1]["Close"]
            value = qty * price
            current_weights[ticker] = value / portfolio_value

        logger.info(
            f"Current portfolio value: ${portfolio_value:,.0f} | Positions: {len(current_weights)}"
        )

        # 2. Generate latest signals
        signals = {}
        for ticker in allocator.watchlist:
            try:
                pred = predictor.predict(ticker, period="60d")[-1]
                signals[ticker] = float(pred)
            except:
                signals[ticker] = 0.0

        # 3. Compute target weights
        target_weights = allocator.allocate(signals, method="signal_strength")  # or "risk_parity"

        # 4. Check risk constraints
        compliance = constraints.check_compliance(pd.Series(target_weights), allocator.sector_map)
        if not compliance["compliant"]:
            logger.warning(f"Risk violation: {compliance['violations']}")
            # Optional: adjust weights or abort
            return

        # 5. Calculate trades
        trades = []
        for ticker, target_w in target_weights.items():
            current_w = current_weights.get(ticker, 0.0)
            diff = target_w - current_w

            if abs(diff) < MIN_TRADE_THRESHOLD:
                continue

            dollar_amount = diff * portfolio_value
            if abs(dollar_amount / portfolio_value) > MAX_TRADE_SIZE_PCT:
                dollar_amount = np.sign(dollar_amount) * MAX_TRADE_SIZE_PCT * portfolio_value

            price = DataFetcher.get_price(ticker).iloc[-1]["Close"]
            qty = int(dollar_amount / price)

            if qty == 0:
                continue

            side = "buy" if qty > 0 else "sell"
            qty = abs(qty)

            trades.append(
                {
                    "ticker": ticker,
                    "side": side,
                    "qty": qty,
                    "dollar": dollar_amount,
                    "weight_diff": diff,
                }
            )

        if not trades:
            logger.info("No rebalance needed — portfolio already optimal")
            return

        # 6. Execute trades
        logger.info(f"Executing {len(trades)} trades...")
        for trade in trades:
            # order = OrderRequest(
                ticker=trade["ticker"],
                qty=trade["qty"],
                side=trade["side"],
                type="market",
                time_in_force="day",
            )
            # response = executor.place_order(order)  # unused
            logger.info(
                f"{trade['side'].upper()} {trade['qty']} {trade['ticker']} | ~${trade['dollar']:,.0f}"
            )

        # 7. Save rebalance record
        record = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": portfolio_value,
            "num_trades": len(trades),
            "trades": trades,
            "target_weights": target_weights,
        }
        storage.save(pd.DataFrame([record]), name=f"rebalance/log_{datetime.now().date()}")

        elapsed = (datetime.now() - start_time).seconds
        logger.info(
            f"Rebalance completed successfully in {elapsed}s | {len(trades)} trades executed"
        )

        # Alert
        send_rebalance_alert(len(trades), portfolio_value)

    except Exception as e:
        logger.error(f"Rebalance failed: {e}", exc_info=True)
        send_alert("REBALANCE FAILED", str(e))


def send_rebalance_alert(num_trades: int, value: float):
    message = (
        f"Portfolio Rebalanced\n"
        f"{num_trades} trades executed\n"
        f"Value: ${value:,.0f}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    # Send via Telegram/email/Slack
    logger.info(f"Alert: {message}")


# Schedule it
# scheduler.add_job(rebalance_job, "cron", hour=15, minute=55, timezone="US/Eastern")
