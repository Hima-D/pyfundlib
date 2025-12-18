# src/pyfund/core/broker_registry.py
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .broker import Broker

# Thread-safe global registry
_BROKER_REGISTRY: dict[str, type[Broker]] = {}
_registry_lock = threading.RLock()


def register_broker(
    name: str | None = None, *, override: bool = False
) -> Callable[[type[Broker]], type[Broker]]:
    """
    Decorator to register a broker class.

    Usage:
        @register_broker("zerodha")
        class ZerodhaBroker(Broker): ...

        # Or auto-register using class name
        @register_broker()
        class AlpacaBroker(Broker): ...
    """

    def decorator(cls: type[Broker]) -> type[Broker]:
        if not issubclass(cls, Broker):
            raise TypeError(f"Class {cls.__name__} must inherit from Broker")

        broker_name = name or cls.__name__.replace("Broker", "").lower()

        with _registry_lock:
            if broker_name in _BROKER_REGISTRY:
                if not override:
                    raise ValueError(
                        f"Broker '{broker_name}' already registered. Use override=True to replace."
                    )
                else:
                    print(f"Warning: Overriding existing broker '{broker_name}'")

            _BROKER_REGISTRY[broker_name] = cls
            # store registered name on the class for debugging; use setattr to satisfy static type checkers
            setattr(cls, "_registered_name", broker_name)  # For debugging

        return cls

    return decorator


def get_broker(name: str, **credentials: Any) -> Broker:
    """
    Factory function to instantiate a broker by name.

    Example:
        broker = get_broker("zerodha", api_key="xxx", access_token="yyy")
    """
    broker_name = name.lower().strip()

    with _registry_lock:
        broker_class = _BROKER_REGISTRY.get(broker_name)

    if not broker_class:
        available = sorted(_BROKER_REGISTRY.keys())
        raise ValueError(
            f"Broker '{broker_name}' not found!\n"
            f"Available brokers: {', '.join(available) or 'None'}\n"
            f"Did you forget to import the broker module?"
        )

    try:
        instance = broker_class(**credentials)
        return instance
    except Exception as e:
        raise RuntimeError(f"Failed to initialize broker '{broker_name}': {e}") from e


def list_available_brokers() -> list[str]:
    """Return sorted list of registered broker names"""
    with _registry_lock:
        return sorted(_BROKER_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all registered brokers (useful for testing)"""
    with _registry_lock:
        _BROKER_REGISTRY.clear()


# Auto-import all brokers in pyfund/brokers/ when this module is imported
def _auto_discover_brokers() -> None:
    import importlib
    from pathlib import Path

    brokers_path = Path(__file__).parent.parent / "brokers"
    if not brokers_path.exists():
        return

    for file in brokers_path.glob("*.py"):
        if file.name.startswith("_") or file.name == "base.py":
            continue

        module_name = f"pyfund.brokers.{file.stem}"
        try:
            importlib.import_module(module_name)
        except Exception as e:
            print(f"Failed to auto-import broker module {module_name}: {e}")


# Auto-discover on import
_auto_discover_brokers()


# Make it pretty
__all__ = [
    "Broker",
    "clear_registry",
    "get_broker",
    "list_available_brokers",
    "register_broker",
]
