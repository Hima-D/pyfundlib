"""Institutional configuration management for PyFundLib."""

from typing import Any, Dict, Optional
import os
import yaml
from pydantic import BaseModel, Field, ConfigDict


class BrokerConfig(BaseModel):
    """Broker configuration."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="paper", description="Broker name")
    api_key: Optional[str] = Field(default=None, description="API key")
    api_secret: Optional[str] = Field(default=None, description="API secret")
    sandbox: bool = Field(default=True, description="Use sandbox/paper trading")
    base_url: Optional[str] = Field(default=None, description="Custom broker URL")


class DataConfig(BaseModel):
    """Data fetching and caching configuration."""

    model_config = ConfigDict(extra="allow")

    source: str = Field(default="yfinance", description="Default data source")
    cache_dir: str = Field(default="./cache", description="Cache directory")
    cache_format: str = Field(default="parquet", description="Cache format")
    compression: str = Field(default="zstd", description="Compression type")
    refresh_interval_hours: int = Field(default=24, description="Cache refresh interval")


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    model_config = ConfigDict(extra="allow")

    initial_capital: float = Field(default=100000, description="Starting capital")
    commission: float = Field(default=0.001, description="Commission per trade")
    slippage: float = Field(default=0.0005, description="Slippage factor")
    max_position_size: float = Field(default=0.1, description="Max position size")
    risk_per_trade: float = Field(default=0.02, description="Risk per trade")


class MLConfig(BaseModel):
    """Machine Learning configuration."""

    model_config = ConfigDict(extra="allow")

    model_type: str = Field(default="xgboost", description="Default model type")
    lookback_days: int = Field(default=252, description="Lookback period")
    train_test_split: float = Field(default=0.8, description="Train/test split")
    validation_split: float = Field(default=0.2, description="Validation split")
    random_seed: int = Field(default=42, description="Random seed")
    mlflow_enabled: bool = Field(default=True, description="Enable MLflow")
    mlflow_uri: Optional[str] = Field(default=None, description="MLflow server URI")


class MonitoringConfig(BaseModel):
    """System monitoring configuration."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(default=True, description="Enable monitoring")
    update_interval_seconds: int = Field(default=60, description="Update interval")
    alert_on_drawdown_pct: float = Field(default=0.15, description="Alert threshold")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")


class Config(BaseModel):
    """Main configuration class."""

    model_config = ConfigDict(extra="allow")

    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    live_trading: bool = Field(default=False, description="Enable live trading")
    paper_trading: bool = Field(default=True, description="Use paper trading")
    universe: list[str] = Field(default_factory=list, description="Asset universe")
    default_strategy: str = Field(default="sma_crossover", description="Default strategy")

    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        """Load configuration from YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config_path = os.getenv("PYFUNDLIB_CONFIG", "config.yaml")
        if os.path.exists(config_path):
            return cls.from_yaml(config_path)
        return cls()

    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
