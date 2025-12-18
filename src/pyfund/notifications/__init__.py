# src/pyfund/notifications/__init__.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Notifier:
    """
    Multi-channel notifier with graceful fallback.
    Works in dry-run mode if dependencies/tokens are missing.
    """

    def __init__(self):
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat = os.getenv("TELEGRAM_CHAT_ID", "@pyfund_alerts")
        self.dry_run = True

        # Only enable real sending if token + library are available
        if self.telegram_token:
            try:
                import telegram  # type: ignore
                self._bot = telegram.Bot(token=self.telegram_token)
                self.dry_run = False
                logger.info("Telegram notifier enabled (live mode)")
            except ImportError:
                logger.warning("python-telegram-bot not installed → falling back to dry-run")
                self.dry_run = True
            except Exception as e:
                logger.warning(f"Telegram bot init failed ({e}) → dry-run mode")
                self.dry_run = True
        else:
            logger.info("No TELEGRAM_BOT_TOKEN → running in dry-run mode")

    def send_alert(
        self,
        ticker: str,
        direction: str,
        confidence: float,
        strategy: str,
        raw_signal: float | None = None,
    ) -> None:
        lines = [
            f"{direction} | {ticker}",
            f"Strategy: {strategy}",
            f"Confidence: {confidence:.1%}",
        ]
        if raw_signal is not None:
            lines.append(f"Signal: {raw_signal:+.4f}")
        lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        message = "\n".join(lines)
        self._send(message)

    def send_daily_summary(self, df: pd.DataFrame, strategy_name: str) -> None:
        if df.empty:
            return

        buys = len(df[df.position == 1])
        sells = len(df[df.position == -1])
        holds = len(df) - buys - sells

        top = df[df.position != 0].nlargest(5, "confidence")

        lines = [
            f"Daily Summary — {strategy_name}",
            f"Date: {datetime.now().strftime('%Y-%m-%d')}\n",
            f"BUY: {buys} | SELL: {sells} | HOLD: {holds}\n",
            "Top Signals:",
        ]
        for _, row in top.iterrows():
            lines.append(f"• {row['strength']} {row['ticker']} ({row['confidence']:.1%})")

        message = "\n".join(lines)
        self._send(message)

    def _send(self, text: str) -> None:
        if self.dry_run:
            logger.info(f"[DRY RUN ALERT]\n{text}\n{'─' * 50}")
            return

        try:
            # This will only be reached if telegram was successfully imported
            import telegram  # type: ignore
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # schedule the coroutine when an event loop is already running
                loop.create_task(
                    self._bot.send_message(
                        chat_id=self.telegram_chat,
                        text=f"<pre>{text}</pre>",
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                )
            else:
                # run the coroutine to completion in a newly running loop
                loop.run_until_complete(
                    self._bot.send_message(
                        chat_id=self.telegram_chat,
                        text=f"<pre>{text}</pre>",
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                )
            logger.info(f"Live alert sent → {text.splitlines()[0]}")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            logger.info(f"Falling back to log:\n{text}")