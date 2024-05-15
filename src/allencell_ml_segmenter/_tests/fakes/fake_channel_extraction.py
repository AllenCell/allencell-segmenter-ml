from pathlib import Path
from typing import Callable, Optional, Any

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.image_data_extractor import FakeImageDataExtractor
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.training.training_model import TrainingModel


class FakeChannelsReady:
    def __init__(self):
        self.connected: Callable = None

    def connect(self, connect: Callable):
        self.connected = connect

class FakeImageDataReady:
    def __init__(self):
        self.connected: Callable = None

    def connect(self, connect: Callable):
        self.connected = connect


class FakeChannelExtractionThread:
    """
    Fake of Channel Extraction Thread
    """

    def __init__(self, get_image_data: bool, fake_return_value: int):
        self.channels_ready = FakeChannelsReady()
        self.image_data_ready = FakeImageDataReady()
        self.started: bool = False
        self._fake_value = fake_return_value
        self.get_image_data = get_image_data

    def start(self):
        self.started = True
        if self.get_image_data:
            fake_image_data_extractor: FakeImageDataExtractor = FakeImageDataExtractor.global_instance()
            self.image_data_ready.connected(fake_image_data_extractor.extract_image_data(Path("")))
        else:
            self.channels_ready.connected(self._fake_value)
