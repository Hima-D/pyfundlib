# src/pyfund/automation/jobs/rebalance_portfolio.py
"""
Event-Driven Portfolio Rebalancer
────────────────────────────────
Fully framework-native. No hardcoding. Zero magic numbers.

Designed for:
- Daily EOD rebalance
- Signal-triggered rebalance
- Risk breach rebalance
- Manual trigger via API

Used by:
    pyfund run rebalance
    scheduler.add_job(..., trigger="signal")
    FastAPI endpoint
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import numpy as np

from ...core.events import Event, EventBus, EventType
from ...data.storage import DataStorage
from ...execution.live import LiveExecutor
from ...ml.predictor import MLPredictor
from ...portfolio.allocator import PortfolioAllocator
from ...risk.constraints import RiskConstraints
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RebalanceConfig:
    """User-configurable rebalance rules"""
    min_trade_threshold: float = 0.02      # 2% weight diff to trigger trade
    max_trade_size_pct: float = 0.25       # Max 25% of portfolio in one trade
    risk_limits: Dict[str, float] = None   # Passed to RiskConstraints
    allocation_method: str = "signal_strength"
    dry_run: bool = True
    alert_on_rebalance: bool = True


class PortfolioRebalancer:
    """
    Event-driven rebalancer — the heart of live execution.
    Subscribes to signals, runs on timer, or triggered manually.
    """

    def __init__(
        self,
        config: RebalanceConfig | None = None,
        executor: Optional[LiveExecutor] = None,
        predictor: Optional[MLPredictor] = None,
        allocator: Optional[PortfolioAllocator] = None,
        constraints: Optional[RiskConstraints] = None,
        storage: Optional[DataStorage] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.config = config or RebalanceConfig()
        self.executor = executor or LiveExecutor(dry_run=self.config.dry_run)
        self.predictor = predictor or MLPredictor()
        self.allocator = allocator or PortfolioAllocator()
        self.constraints = constraints or RiskConstraints(**(self.config.risk_limits or {}))
        self.storage = storage or DataStorage()
        self.event_bus = event_bus or EventBus.get_default()

        # Subscribe to events
        self.event_bus.subscribe(EventType.SIGNALS_GENERATED, self.on_signals)
        self.event_bus.subscribe(EventType.MANUAL_TRIGGER, self.on_manual_rebalance)

        logger.info(f"PortfolioRebalancer initialized | dry_run={self.config.dry_run}")

    def rebalance(self, trigger: str = "scheduled") -> None:
        """Main rebalance logic — can be called anytime"""
        logger.info(f"REBALANCE TRIGGERED: {trigger.upper()}")

        start_time = datetime.now()
        try:
            # 1. Get current state
            account = self.executor.get_account()
            positions = self.executor.get_positions()
            portfolio_value = float(account["portfolio_value"])

            current_weights = self._calculate_current_weights(positions, portfolio_value)
            logger.info(f"Portfolio value: ${portfolio_value:,.0f} | {len(current_weights)} positions")

            # 2. Generate fresh signals
            signals = self._generate_signals()
            if not signals:
                logger.info("No valid signals → skipping rebalance")
                return

            # 3. Compute target allocation
            target_weights = self.allocator.allocate(
                signals=signals,
                method=self.config.allocation_method,
                current_weights=current_weights,
            )

            # 4. Risk check
            compliance = self.constraints.check(target_weights, self.allocator.get_sector_map())
            if not compliance["compliant"]:
                logger.warning(f"Risk violation → aborting: {compliance['violations']}")
                self.event_bus.publish(EventType.RISK_BREACH, compliance)
                return

            # 5. Compute trades
            trades = self._compute_trades(current_weights, target_weights, portfolio_value)
            if not trades:
                logger.info("No trades needed — portfolio in tolerance")
                return

            # 6. Execute
            self._execute_trades(trades)

            # 7. Record & alert
            self._save_rebalance_record(trades, target_weights, portfolio_value, trigger)
            if self.config.alert_on_rebalance:
                self.event_bus.publish(EventType.REBALANCE_COMPLETED, {
                    "num_trades": len(trades),
                    "value": portfolio_value,
                    "trigger": trigger,
                })

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Rebalance completed | {len(trades)} trades | {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Rebalance failed: {e}", exc_info=True)
            self.event_bus.publish(EventType.REBALANCE_FAILED, {"error": str(e)})

    # Event handlers
    def on_signals(self, event: Event) -> None:
        """Auto-rebalance when new signals arrive"""
        self.rebalance(trigger="signals")

    def on_manual_rebalance(self, event: Event) -> None:
        """API or CLI trigger"""
        self.rebalance(trigger="manual")

    # Helpers
    def _calculate_current_weights(self, positions: Dict[str, float], value: float) -> Dict[str, float]:
        weights = {}
        for ticker, qty in positions.items():
            if ticker in ("CASH", "") or qty == 0:
                continue
            try:
                price = self.executor.get_price(ticker)
                weights[ticker] = (qty * price) / value
            except:
                pass
        return weights

    def _generate_signals(self) -> Dict[str, float]:
        signals = {}
        for ticker in self.allocator.get_watchlist():
            try:
                pred = self.predictor.predict(ticker, period="90d")
                if pred is not None and len(pred) > 0:
                    signals[ticker] = float(pred[-1])
            except Exception as e:
                logger.debug(f"Signal failed for {ticker}: {e}")
        return signals

    def _compute_trades(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        value: float,
    ) -> List[Dict[str, Any]]:
        trades = []
        for ticker, target_w in target.items():
            current_w = current.get(ticker, 0.0)
            diff = target_w - current_w

            if abs(diff) < self.config.min_trade_threshold:
                continue

            dollar = diff * value
            if abs(dollar / value) > self.config.max_trade_size_pct:
                dollar = np.sign(diff) * self.config.max_trade_size_pct * value

            try:
                price = self.executor.get_price(ticker)
                qty = int(dollar / price)
                if abs(qty) == 0:
                    continue

                trades.append({
                    "ticker": ticker,
                    "side": "buy" if qty > 0 else "sell",
                    "qty": abs(qty),
                    "dollar": dollar,
                    "target_weight": target_w,
                    "current_weight": current_w,
                })
            except:
                pass
        return trades

    def _execute_trades(self, trades: List[Dict[str, Any]]) -> None:
        for t in trades:
            logger.info(f"{t['side'].upper()} {t['qty']} {t['ticker']} | ~${t['dollar']:,.0f}")
            if not self.config.dry_run:
                self.executor.place_order(ticker=t["ticker"], qty=t["qty"], side=t["side"])

    def _save_rebalance_record(self, trades, target_weights, value, trigger):
        record = {
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger,
            "portfolio_value": value,
            "num_trades": len(trades),
            "trades": trades,
            "target_weights": target_weights,
        }
        self.storage.save(pd.DataFrame([record]), name=f"rebalance/{datetime.now().date()}_{trigger}")


# Framework entry points
def rebalance_job(config: RebalanceConfig | None = None) -> None:
    """CLI/Scheduler entry point"""
    rebalancer = PortfolioRebalancer(config)
    rebalancer.rebalance(trigger="scheduled")


def trigger_rebalance() -> None:
    """Manual trigger (API, button, etc.)"""
    EventBus.get_default().publish(EventType.MANUAL_TRIGGER, {})