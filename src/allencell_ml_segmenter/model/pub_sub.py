from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class Event(Enum):
    TRAINING = "training"

class Publisher(ABC):
    def __init__(self):
        self._subscribers: List[Subscriber] = list()

    def dispatch(self, event: Event):
        for i in self._subscribers:
            i.handle_event(event)

    def subscribe(self, subscriber):
        self._subscribers.append(subscriber)


class Subscriber(ABC):

    @abstractmethod
    def handle_event(self, event: Event):
        pass






