from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event


class FakeSubscriber(Subscriber):
    """
    Testing publishers with this fake subscriber class
    that implements handle_event
    """

    def __init__(self):
        self.handled_event = None

    def handle_event(self, event: Event):
        self.handled_event = event
