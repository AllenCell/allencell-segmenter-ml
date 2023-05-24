from allencell_ml_segmenter.model.subscriber import Subscriber
from allencell_ml_segmenter.model.event import Event
# Testing publishers with this mock subscriber class
# that implements handle_event
class FakeSubscriber(Subscriber):
    """
    Testing publishers with this mock subscriber class
    that implements handle_event
    """
    def __init__(self):
        self.handled_event = None

    def handle_event(self, event: Event):
        self.handled_event = event
