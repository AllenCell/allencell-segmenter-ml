from abc import ABC
from typing import Dict, Callable

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event


class Publisher(ABC):
    """
    ABC publisher class
    """

    def __init__(self):
        # Map of Event -> Subscriber -> Handler function
        self._events_to_subscriber_handlers: Dict[
            Dict[Subscriber, Callable]
        ] = {event: dict() for event in [e.value for e in Event]}

    def dispatch(self, event: Event) -> None:
        """
        Dispatches an event to all subscribers
        """
        for _, handler in self._events_to_subscriber_handlers[
            event.value
        ].items():
            handler(event)

    def subscribe(self, event: Event, subscriber: Subscriber, handler) -> None:
        """
        subscribes a subscriber to this publisher
        """
        self._events_to_subscriber_handlers[event.value][subscriber] = handler

    def unsubscribe(self, event: Event, subscriber: Subscriber) -> None:
        """
        unsubscribes a subscriber from this publisher
        """
        if subscriber in self._events_to_subscriber_handlers[event.value]:
            del self._events_to_subscriber_handlers[event.value][subscriber]
