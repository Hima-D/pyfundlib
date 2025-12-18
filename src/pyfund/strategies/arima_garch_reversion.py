import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


class ARIMAGARCHStrategy:
    """Universal Statistical Arbitrage — Works on ANY asset, ANY timeframe"""

    def __init__(
        self,
        p: int = 1,
        d: int = 1,
        q: int = 1,
        garch_p: int = 1,
        garch_q: int = 1,
        lookback: int = 252,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        use_log_prices: bool = True,
        min_periods: int = 100,
    ):
        self.p, self.d, self.q = p, d, q
        self.garch_p, self.garch_q = garch_p, garch_q
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.use_log_prices = use_log_prices
        self.min_periods = min_periods

    def _prepare_series(self, prices: pd.Series) -> pd.Series:
        if self.use_log_prices:
            return pd.Series(np.log(prices.to_numpy()), index=prices.index)
        return prices.copy()

    def fit_forecast(self, prices: pd.Series) -> tuple[float, float, float]:
        if len(prices) < self.min_periods:
            return float(prices.iloc[-1]), float(prices.std()), float(prices.std())

        series = self._prepare_series(prices)[-self.lookback :]

        try:
            model = ARIMA(series, order=(self.p, self.d, self.q))
            fitted = model.fit(method="statespace")  # Remove disp=0
            forecast_series = fitted.forecast(steps=1)
            forecast = float(forecast_series.iloc[0])
            residuals = fitted.resid[-self.lookback // 2 :]

            if len(residuals) > 50 and residuals.std() > 1e-8:
                garch = arch_model(
                    residuals, vol="GARCH", p=self.garch_p, q=self.garch_q, dist="normal"  # Uppercase GARCH, lowercase normal
                )
                garch_fit = garch.fit(disp="off")
                cond_vol = float(np.sqrt(np.asarray(garch_fit.conditional_volatility)[-1]))
            else:
                cond_vol = float(residuals.std())

            residual_std = float(residuals.std()) if len(residuals) > 0 else cond_vol

        except Exception:
            forecast = float(series.iloc[-1])
            pct_change_std = series.pct_change().std()
            cond_vol = float(pct_change_std) if len(series) > 10 else float(series.std())
            residual_std = cond_vol

        return forecast, residual_std, cond_vol

    def generate_signal(self, prices: pd.Series) -> tuple[int, float, dict]:
        if len(prices) < self.min_periods:
            return 0, 0.0, {"error": "insufficient_data"}

        current_price = prices.iloc[-1]
        forecast, _, cond_vol = self.fit_forecast(prices)
        error = current_price - forecast
        z_score = error / cond_vol if cond_vol > 1e-8 else 0

        confidence = min(abs(z_score) / 4.0, 1.0)
        signal = 0
        if z_score > self.z_entry:
            signal = -1
        elif z_score < -self.z_entry:
            signal = 1
        elif abs(z_score) < self.z_exit:
            signal = 0

        return (
            signal,
            confidence,
            {
                "z_score": round(z_score, 3),
                "forecast": round(forecast, 4),
                "cond_vol": round(cond_vol, 6),
                "confidence": round(confidence, 3),
            },
        )