"""In-process event bus. Synchronous; no broker."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["Event", "EventBus", "EventType"]


class EventType(Enum):
    TARGET_SELECTED = "target_selected"
    STATE_CHANGED = "state_changed"
    TRACK_REMOVED = "track_removed"
    COMMAND_REJECTED = "command_rejected"


@dataclass(frozen=True, slots=True)
class Event:
    type: EventType
    frame_id: int
    ns: int
    track_id: int | None = None
    payload: dict = field(default_factory=dict)


class EventBus:
    def __init__(self):
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._subscribers.append(fn)

    def publish(self, event: Event) -> None:
        for fn in self._subscribers:
            fn(event)
