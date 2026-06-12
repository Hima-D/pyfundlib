"""Tests for PyFundLib ML module."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyfundlib.ml import MLPredictor


def create_sample_ohlcv(days: int = 100):
    """Create sample OHLCV data."""
    dates = pd.date_range(start='2023-01-01', periods=days)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(days) * 0.5)
    
    return pd.DataFrame({
        'Open': prices,
        'High': prices + np.abs(np.random.randn(days) * 0.5),
        'Low': prices - np.abs(np.random.randn(days) * 0.5),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, days)
    }, index=pd.date_range(start='2023-01-01', periods=days))


def test_ml_predictor_init():
    \"\"\"Test MLPredictor initialization.\"\"\"\n    predictor = MLPredictor(model_type=\"xgboost\", lookback=20)\n    assert predictor.model_type == \"xgboost\"\n    assert predictor.lookback == 20\n    assert predictor.model is None\n    print(\"✓ MLPredictor initialization test passed\")\n\n\ndef test_ml_prepare_features():\n    \"\"\"Test feature preparation.\"\"\"\n    predictor = MLPredictor(lookback=20)\n    ohlcv = create_sample_ohlcv()\n    X, y = predictor.prepare_features(ohlcv)\n    \n    assert len(X) > 0\n    assert len(y) > 0\n    assert X.shape[1] == 5  # 5 features\n    print(\"✓ MLPredictor feature preparation test passed\")\n\n\ndef test_ml_model_training():\n    \"\"\"Test model training.\"\"\"\n    predictor = MLPredictor(model_type=\"xgboost\")\n    ohlcv = create_sample_ohlcv(200)\n    X, y = predictor.prepare_features(ohlcv)\n    \n    predictor.train(X, y, test_size=0.2)\n    assert predictor.model is not None\n    print(\"✓ MLPredictor training test passed\")\n\n\ndef test_ml_prediction():\n    \"\"\"Test model prediction.\"\"\"\n    predictor = MLPredictor(model_type=\"xgboost\")\n    ohlcv = create_sample_ohlcv(200)\n    X, y = predictor.prepare_features(ohlcv)\n    \n    predictor.train(X, y, test_size=0.2)\n    predictions = predictor.predict(X[:10])\n    \n    assert len(predictions) == 10\n    assert set(predictions).issubset({0, 1})\n    print(\"✓ MLPredictor prediction test passed\")\n\n\nif __name__ == \"__main__\":\n    test_ml_predictor_init()\n    test_ml_prepare_features()\n    test_ml_model_training()\n    test_ml_prediction()\n    print(\"\\n✅ All ML tests passed!\")\n