from abc import ABC, abstractmethod
from allencell_ml_segmenter.model.event import Event


class Subscriber(ABC):
    @abstractmethod
    def handle_event(self, event: Event):
        pass
