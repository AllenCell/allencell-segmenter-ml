from abc import ABC, abstractmethod

from allencell_ml_segmenter.core import Event


class Subscriber(ABC):
    """
    ABC subscriber class
    """

    @abstractmethod
    def handle_event(self, event: Event):
        pass
