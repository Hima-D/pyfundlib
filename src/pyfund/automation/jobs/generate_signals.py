# src/pyfund/automation/jobs/generate_signals.py
"""
Multi-Strategy Daily Signal Generator
────────────────────────────────────
Runs every trading day (e.g., 4:30 AM ET) and generates buy/sell/hold signals
for ALL user-defined strategies defined in config/strategies/*.yaml

Features:
- Fully YAML-configurable strategies
- Per-strategy watchlists, thresholds, data periods
- Real-time alerts on strong signals
- Daily summary reports
- Automatic saving of signals with versioning
- Graceful error handling (one ticker failure ≠ job crash)
- Designed to run via scheduler (cron, Airflow, Prefect, etc.)
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from ...data.fetcher import DataFetcher
from ...data.storage import DataStorage
from ...ml.predictor import MLPredictor
from ...notifications import Notifier
from ...utils.logger import get_logger

logger = get_logger(__name__)

# Auto-discover strategy configs
CONFIG_DIR = Path(__file__).parents[3] / "config" / "strategies"  # project_root/config/strategies


class TradingStrategy:
    """One user-defined trading strategy loaded from YAML"""

    def __init__(self, config_path: Path):
        self.path = config_path
        self.config = self._load_config(config_path)
        self.name = self.config.get("name", config_path.stem.replace("_", " ").title())
        self.description = self.config.get("description", "No description")

        # Core services
        self.predictor = MLPredictor()
        self.storage = DataStorage()
        self.notifier = Notifier()

    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load and validate strategy YAML"""
        if not path.exists():
            raise FileNotFoundError(f"Strategy config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # Basic validation
        required = ["watchlist"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Strategy {path.name} missing required fields: {missing}")

        return config

    def run(self) -> None:
        """Execute the full signal generation pipeline for this strategy"""
        if not self.config.get("enabled", True):
            logger.info(f"Strategy '{self.name}' is disabled → skipping")
            return

        logger.info(f"{'='*60}")
        logger.info(f"RUNNING STRATEGY: {self.name}")
        logger.info(f"Description: {self.description}")
        logger.info(f"Config: {self.path.name}")
        logger.info(f"{'='*60}")

        start_time = datetime.now()
        signals_today: List[Dict[str, Any]] = []

        # Configurable parameters with smart defaults
        watchlist = self.config["watchlist"]
        data_period = self.config.get("data_period", "3y")
        min_points = self.config.get("min_data_points", 200)
        window = self.config.get("prediction_window", 252)

        strong_thresh = self.config["signal_threshold"]["strong"]
        medium_thresh = self.config["signal_threshold"]["medium"]

        alerting = self.config.get("alerting", {})
        strong_only_alerts = alerting.get("strong_only", True)

        for ticker in watchlist:
            ticker = ticker.strip().upper()
            try:
                logger.debug(f"Fetching data for {ticker}...")

                df = DataFetcher.get_price(ticker, period=data_period)

                if df is None or len(df) < min_points:
                    logger.warning(f"{ticker}: Only {len(df) if df is not None else 0} rows → skipped")
                    signals_today.append({
                        "strategy": self.name,
                        "ticker": ticker,
                        "position": 0,
                        "strength": "INSUFFICIENT_DATA",
                        "error": "Not enough data"
                    })
                    continue

                # ML Prediction
                raw_pred = self.predictor.predict(ticker, df.tail(window))
                signal = float(raw_pred[-1]) if raw_pred is not None and len(raw_pred) > 0 else 0.0
                confidence = abs(signal)

                # Position logic
                if confidence > strong_thresh:
                    position = 1 if signal > 0 else -1
                    strength = "STRONG BUY" if position > 0 else "STRONG SELL"
                elif confidence > medium_thresh:
                    position = 1 if signal > 0 else -1
                    strength = "BUY" if position > 0 else "SELL"
                else:
                    position = 0
                    strength = "HOLD"

                # Build record
                latest_model = self.predictor.load_latest(ticker)
                model_version = latest_model.version if latest_model is not None else "unknown"
                record = {
                    "strategy": self.name,
                    "ticker": ticker,
                    "date": datetime.now().date().isoformat(),
                    "timestamp": datetime.now().isoformat(),
                    "raw_signal": round(signal, 6),
                    "position": int(position),
                    "strength": strength,
                    "confidence": round(confidence, 4),
                    "model_version": model_version,
                }
                signals_today.append(record)

                # Real-time alert
                if position != 0 and (not strong_only_alerts or "STRONG" in strength):
                    self.notifier.send_alert(
                        ticker=ticker,
                        direction=strength,
                        confidence=confidence,
                        strategy=self.name,
                        raw_signal=signal
                    )

                logger.info(f"{ticker:<8} {strength:<12} (signal={signal:+.4f}, conf={confidence:.1%})")

            except Exception as e:
                logger.error(f"Failed {ticker}: {e}", exc_info=True)
                signals_today.append({
                    "strategy": self.name,
                    "ticker": ticker,
                    "position": 0,
                    "strength": "ERROR",
                    "error": str(e)
                })

        # === Save Results ===
        if signals_today:
            df_signals = pd.DataFrame(signals_today)
            today_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"signals/{self.name.lower().replace(' ', '_')}_{today_str}"
            self.storage.save(df_signals, name=filename)
            logger.info(f"Saved {len(df_signals)} signals → {filename}")

            # Daily summary
            self.notifier.send_daily_summary(df_signals, strategy_name=self.name)

        # Final stats
        active = len([s for s in signals_today if s["position"] != 0])
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"COMPLETED {self.name} | {active}/{len(watchlist)} active | {elapsed:.1f}s")


def generate_signals_job() -> None:
    """
    Main entry point — called daily by scheduler
    Discovers and runs ALL enabled strategies in config/strategies/
    """
    logger.info("PYFUND SIGNAL ENGINE STARTED")
    logger.info(f"Looking for strategies in: {CONFIG_DIR}")

    if not CONFIG_DIR.exists():
        logger.error(f"Config directory not found: {CONFIG_DIR}")
        logger.error("Create it and add *.yaml strategy files")
        return

    strategy_files = sorted(CONFIG_DIR.glob("*.yaml")) + sorted(CONFIG_DIR.glob("*.yml"))

    if not strategy_files:
        logger.warning("No strategy files found in config/strategies/")
        logger.warning("Create one using: pyfund new-strategy momentum_alpha")
        return

    logger.info(f"Found {len(strategy_files)} strategy file(s)")

    for config_file in strategy_files:
        try:
            strategy = TradingStrategy(config_file)
            strategy.run()
        except Exception as e:
            logger.error(f"Strategy failed: {config_file.name} → {e}", exc_info=True)

    logger.info("ALL STRATEGIES COMPLETED")