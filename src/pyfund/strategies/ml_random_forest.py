from __future__ import annotations
from typing import Any, Optional

import pandas as pd

from ..data.features import FeatureEngineer
from ..ml.predictor import MLPredictor
from ..utils.logger import logger
from base import BaseStrategy


class MLRandomForestStrategy(BaseStrategy):
    """
    A machine-learning trading strategy using Random Forest (or any ML model).

    Key features:
    - Safe model loading (with fallbacks)
    - Automatic feature engineering
    - Probability-based signals with confidence thresholds
    - Neutral signal on all failures
    - Optional fallback to RSI strategy
    """

    default_params = {
        "model_name": "random_forest",
        "confidence_threshold": 0.6,
        "use_probability": True,
        "min_feature_count": 10,
        "fallback_to_rsi": True,
    }

    def __init__(self, ticker: str, params: Optional[dict[str, Any]] = None):
        super().__init__({**self.default_params, **(params or {})})

        self.ticker = ticker.upper()
        self.predictor = MLPredictor()

        self.model: Optional[Any] = None
        self.feature_names: Optional[list[str]] = None
        self.last_prediction_date = None

    # ----------------------------------------------------------------------
    # Model Loading
    # ----------------------------------------------------------------------
    def _load_model_safely(self) -> bool:
        """Load ML model with error handling. Returns True if successful."""
        try:
            model = self.predictor.load_latest(
                self.ticker,
                self.params["model_name"]
            )
            if model is None:
                logger.warning(f"No ML model found for {self.ticker}")
                return False

            self.model = model
            logger.info(f"Loaded ML model for {self.ticker}")
            return True

        except Exception as e:
            logger.error(f"Error loading ML model for {self.ticker}: {e}")
            self.model = None
            return False

    # ----------------------------------------------------------------------
    # Feature Preparation
    # ----------------------------------------------------------------------
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate input features for prediction."""
        df = data.copy()

        # If only OHLCV is present → add indicators
        if set(df.columns).issubset({"Open", "High", "Low", "Close", "Volume"}):
            df = FeatureEngineer.add_technical_features(df)

        # Drop non-feature columns
        feature_cols = [
            c for c in df.columns
            if c not in {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        ]

        if len(feature_cols) < self.params["min_feature_count"]:
            logger.warning(
                f"Feature count too small for {self.ticker}: {len(feature_cols)} found"
            )
            return pd.DataFrame()

        X = df[feature_cols].dropna()
        self.feature_names = feature_cols

        return X

    # ----------------------------------------------------------------------
    # Signal Generation
    # ----------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals:

            +1 → strong long
             0 → neutral / not confident
            -1 → strong short
        """

        # Initialize all signals to 0
        signals = pd.Series(0, index=data.index, dtype=int)

        # 1. Load model if missing
        if self.model is None:
            if not self._load_model_safely():

                # Optional: fallback to RSI
                if self.params["fallback_to_rsi"]:
                    logger.info(f"Falling back to RSI strategy for {self.ticker}")
                    from .rsi_mean_reversion import RSIMeanReversionStrategy

                    return RSIMeanReversionStrategy().generate_signals(data)

                return signals  # stay neutral

        # If model still None → return flat
        if self.model is None:
            return signals

        # 2. Prepare features
        X = self._prepare_features(data)
        if X.empty or len(X) < 20:
            logger.warning(f"Not enough feature rows for {self.ticker}")
            return signals

        pred_index = X.index

        try:
            # ------------------------------------------------------------------
            # Probability-based predictions
            # ------------------------------------------------------------------
            if (
                self.params["use_probability"]
                and hasattr(self.model, "predict_proba")
            ):
                probabilities = self.model.predict_proba(X)

                if probabilities.shape[1] == 2:
                    prob_short = probabilities[:, 0]
                    prob_long = probabilities[:, 1]
                else:
                    # multi-class fallback: treat the last column as "long"
                    prob_long = probabilities[:, -1]
                    prob_short = 1 - prob_long

                long_mask = prob_long >= self.params["confidence_threshold"]
                short_mask = prob_short >= self.params["confidence_threshold"]

                signals.loc[pred_index[long_mask]] = 1
                signals.loc[pred_index[short_mask]] = -1

            # ------------------------------------------------------------------
            # Hard predictions
            # ------------------------------------------------------------------
            elif hasattr(self.model, "predict"):
                hard_pred = self.model.predict(X)

                # Ensure valid output
                hard_pred = pd.Series(hard_pred, index=pred_index).clip(-1, 1)
                signals.loc[pred_index] = hard_pred.values

            else:
                logger.error(f"Model for {self.ticker} has no predict() method.")
                return signals

        except Exception as e:
            logger.error(f"Prediction error for {self.ticker}: {e}")
            return signals

        logger.info(
            f"Signals generated for {self.ticker}: "
            f"Long={int((signals==1).sum())}, "
            f"Short={int((signals==-1).sum())}, "
            f"Flat={int((signals==0).sum())}"
        )

        return signals

    # ----------------------------------------------------------------------
    # String repr
    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        status = "loaded" if self.model is not None else "not_loaded"
        return f"MLRandomForestStrategy({self.ticker}, model={status})"


# For manual testing
if __name__ == "__main__":
    from ..data.fetcher import DataFetcher

    df = DataFetcher.get_price("AAPL", period="2y")
    strategy = MLRandomForestStrategy("AAPL", {"confidence_threshold": 0.55})
    signals = strategy.generate_signals(df)

    print("ML Strategy signals for AAPL:")
    print(f"Non-zero signals: {len(signals[signals != 0])}")
    print(f"Latest signal: {signals.iloc[-1]}")
    print(signals.tail(10))
