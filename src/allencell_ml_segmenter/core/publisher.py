from abc import ABC
from typing import List

from allencell_ml_segmenter.core import Subscriber, Event


class Publisher(ABC):
    """
    ABC publisher class
    """

    def __init__(self):
        # list of subscribers subscribed to this publisher
        self._subscribers: List[Subscriber] = list()

    def dispatch(self, event: Event):
        """
        Dispatches an event to all subscribers
        """
        for i in self._subscribers:
            i.handle_event(event)

    def subscribe(self, subscriber):
        """
        subscribes a subscriber to this publisher
        """
        self._subscribers.append(subscriber)
