import threading
import logging

logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe pub/sub event bus.

    Each subscriber's handler is invoked in its own daemon thread so that
    slow handlers (e.g. TTS, network) never block the publisher.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}
        self._lock = threading.Lock()

    def subscribe(self, event: str, handler) -> None:
        with self._lock:
            self._handlers.setdefault(event, []).append(handler)

    def publish(self, event: str, **data) -> None:
        with self._lock:
            handlers = list(self._handlers.get(event, []))
        for h in handlers:
            threading.Thread(
                target=self._invoke,
                args=(h, event, data),
                daemon=True,
            ).start()

    def _invoke(self, handler, event: str, data: dict) -> None:
        try:
            handler(**data)
        except Exception:
            logger.exception("EventBus handler error for event '%s'", event)


# Module-level singleton — import this everywhere
bus = EventBus()
