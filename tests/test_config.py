"""Tests for PyFundLib configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyfundlib.config import Config, BrokerConfig, DataConfig, BacktestConfig


def test_config_defaults():
    """Test default configuration."""
    cfg = Config()
    assert cfg.broker.name == "paper"
    assert cfg.backtest.initial_capital == 100000
    assert cfg.live_trading is False
    print("✓ Default config test passed")


def test_broker_config():
    """Test broker configuration."""
    broker_cfg = BrokerConfig(name="alpaca", api_key="test_key", sandbox=True)
    assert broker_cfg.name == "alpaca"
    assert broker_cfg.api_key == "test_key"
    assert broker_cfg.sandbox is True
    print("✓ Broker config test passed")


def test_data_config():
    """Test data configuration."""
    data_cfg = DataConfig(cache_dir="./test_cache", compression="zstd")
    assert data_cfg.cache_dir == "./test_cache"
    assert data_cfg.compression == "zstd"
    print("✓ Data config test passed")


def test_backtest_config():
    """Test backtest configuration."""
    backtest_cfg = BacktestConfig(initial_capital=50000, commission=0.001)
    assert backtest_cfg.initial_capital == 50000
    assert backtest_cfg.commission == 0.001
    print("✓ Backtest config test passed")


if __name__ == "__main__":
    test_config_defaults()
    test_broker_config()
    test_data_config()
    test_backtest_config()
    print("\n✅ All config tests passed!")
