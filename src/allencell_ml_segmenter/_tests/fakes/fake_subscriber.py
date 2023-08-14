from typing import Dict

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event


class FakeSubscriber(Subscriber):
    """
    Testing publishers with this fake subscriber class
    using the fake event handler.
    """

    def __init__(self):
        self.handled: Dict[Event, bool] = {}

    def handle(self, event: Event) -> None:
        """
        Fake event handler that indicates that the given event was handled.
        """
        self.handled[event] = True

    def was_handled(self, event: Event) -> bool:
        """
        Returns True if the given event was handled.
        """
        return event in self.handled
