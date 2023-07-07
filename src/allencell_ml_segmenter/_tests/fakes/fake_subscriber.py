from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event


class FakeSubscriber(Subscriber):
    """
    Testing publishers with this fake subscriber class
    using the fake event handler
    """

    def __init__(self):
        self.handled = {}

    def handle(self, event: Event):
        self.handled[event] = True

    def was_handled(self, event: Event):
        return event in self.handled
