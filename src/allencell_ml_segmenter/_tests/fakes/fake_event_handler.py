from allencell_ml_segmenter.core.event import Event


class FakeEventHandler():
    """
    """

    def __init__(self):
        self.handled = False

    def handle(self, event: Event):
        self.handled = True

    def is_handled(self):
        return self.handled
