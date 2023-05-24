from abc import ABC
from typing import List
from allencell_ml_segmenter.model.subscriber import Subscriber
from allencell_ml_segmenter.model.event import Event


class Publisher(ABC):
    def __init__(self):
        self._subscribers: List[Subscriber] = list()

    def dispatch(self, event: Event):
        for i in self._subscribers:
            i.handle_event(event)

    def subscribe(self, subscriber):
        self._subscribers.append(subscriber)
