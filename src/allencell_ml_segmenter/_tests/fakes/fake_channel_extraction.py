from pathlib import Path
from typing import Callable, Optional, Any

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.image_data_extractor import (
    FakeImageDataExtractor,
)
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.training.training_model import TrainingModel


class FakeChannelsReady:
    def __init__(self):
        self.function_called_when_emitted: Callable = None

    def connect(self, connect: Callable):
        self.function_called_when_emitted = connect


class FakeImageDataReady:
    def __init__(self):
        self.function_called_when_emitted: Callable = None

    def connect(self, connect: Callable):
        self.function_called_when_emitted = connect


class FakeChannelExtractionThread:
    """
    Fake of Channel Extraction Thread
    """

    def __init__(self, emit_image_data: bool, fake_return_value: int):
        self.channels_ready = FakeChannelsReady()
        self.image_data_ready = FakeImageDataReady()
        self.started: bool = False
        self._fake_value = fake_return_value
        self.emit_image_data = emit_image_data

    def start(self):
        self.started = True
        if self.emit_image_data:
            fake_image_data_extractor: FakeImageDataExtractor = (
                FakeImageDataExtractor.global_instance()
            )
            self.image_data_ready.function_called_when_emitted(
                fake_image_data_extractor.extract_image_data(Path(""))
            )
        else:
            self.channels_ready.function_called_when_emitted(self._fake_value)
