from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from .engine import Backtester, BacktestResult
from pyfundlib.strategies.base import BaseStrategy
from pyfundlib.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class WalkForwardSliceResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: Dict[str, float]


@dataclass
class WalkForwardResult:
    slices: List[WalkForwardSliceResult]

    @property
    def metrics_frame(self) -> pd.DataFrame:
        rows = []
        for s in self.slices:
            row = {
                "train_start": s.train_start,
                "train_end": s.train_end,
                "test_start": s.test_start,
                "test_end": s.test_end,
            }
            row.update(s.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def mean_sharpe(self) -> float:
        df = self.metrics_frame
        return float(df["sharpe"].mean()) if "sharpe" in df.columns and not df.empty else 0.0

    @property
    def mean_cagr(self) -> float:
        df = self.metrics_frame
        return float(df["cagr"].mean()) if "cagr" in df.columns and not df.empty else 0.0


class WalkForwardBacktester:
    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        params: Optional[Dict[str, Any]] = None,
        window_train: int = 252 * 3,
        window_test: int = 252,
        min_length: int = 252,
    ):
        self.strategy_class = strategy_class
        self.params = params or {}
        self.window_train = int(window_train)
        self.window_test = int(window_test)
        self.min_length = int(min_length)

    def run(self, data: pd.DataFrame) -> WalkForwardResult:
        if len(data) < self.min_length:
            raise ValueError(f"Not enough data for walk-forward: need {self.min_length} rows")

        idx = data.index
        slices: List[WalkForwardSliceResult] = []

        start = 0
        n = len(data)

        while True:
            train_end = start + self.window_train
            test_end = train_end + self.window_test
            if test_end > n:
                break

            train_df = data.iloc[start:train_end]
            test_df = data.iloc[train_end:test_end]

            strategy = self.strategy_class(self.params)
            bt = Backtester(strategy=strategy, data=test_df, name="walk_forward_slice")
            result: BacktestResult = bt.run()

            wf_slice = WalkForwardSliceResult(
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                metrics=result.metrics,
            )
            slices.append(wf_slice)

            logger.info(
                "walk_forward_slice_completed",
                extra={
                    "train_start": str(wf_slice.train_start),
                    "train_end": str(wf_slice.train_end),
                    "test_start": str(wf_slice.test_start),
                    "test_end": str(wf_slice.test_end),
                    "cagr": wf_slice.metrics.get("cagr", 0.0),
                    "sharpe": wf_slice.metrics.get("sharpe", 0.0),
                },
            )

            start += self.window_test

        return WalkForwardResult(slices=slices)

