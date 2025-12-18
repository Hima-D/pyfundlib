# src/pyfund/core/events.py
from __future__ import annotations
from enum import Enum, auto
from typing import Any, Callable, Dict, List
import threading
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """All event types in the system."""
    SIGNALS_GENERATED = auto()
    MANUAL_TRIGGER = auto()
    REBALANCE_COMPLETED = auto()
    REBALANCE_FAILED = auto()
    RISK_BREACH = auto()
    # Add more events as needed


class Event:
    """A simple Event object."""
    def __init__(self, type: EventType, payload: Any = None):
        self.type = type
        self.payload = payload
        self.timestamp = None  # Can be added if needed


class EventBus:
    """Thread-safe publish-subscribe event bus."""
    _default_bus: EventBus | None = None
    _lock = threading.RLock()

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._lock = threading.RLock()

    # -----------------------
    # Singleton / default bus
    # -----------------------
    @classmethod
    def get_default(cls) -> EventBus:
        with cls._lock:
            if cls._default_bus is None:
                cls._default_bus = EventBus()
            return cls._default_bus

    # -----------------------
    # Subscribe
    # -----------------------
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe a callback function to an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed {callback.__name__} to {event_type.name}")

    # -----------------------
    # Unsubscribe
    # -----------------------
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Remove a callback from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed {callback.__name__} from {event_type.name}")

    # -----------------------
    # Publish
    # -----------------------
    def publish(self, event_type: EventType, payload: Any = None) -> None:
        """Publish an event to all subscribers."""
        event = Event(event_type, payload)
        subscribers: List[Callable[[Event], None]] = []

        with self._lock:
            subscribers = self._subscribers.get(event_type, []).copy()

        if not subscribers:
            logger.debug(f"No subscribers for event {event_type.name}")
            return

        logger.info(f"Publishing event {event_type.name} to {len(subscribers)} subscribers")
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in subscriber {callback.__name__} for event {event_type.name}: {e}")
