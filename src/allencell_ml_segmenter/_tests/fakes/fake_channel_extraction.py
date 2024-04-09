from pathlib import Path
from typing import Callable, Optional

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.training.training_model import TrainingModel


class FakeChannelsReady:
    def __init__(self):
        self.connected: Callable = None

    def connect(self, connect: Callable):
        self.connected = connect


class FakeChannelExtractionThread:
    """
    Fake of Channel Extraction Thread
    """

    def __init__(self):
        self.channels_ready = FakeChannelsReady()
        self.started: bool = False

    def start(self):
        self.started = True
