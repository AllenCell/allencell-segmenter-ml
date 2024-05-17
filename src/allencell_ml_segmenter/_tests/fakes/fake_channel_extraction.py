from pathlib import Path
from typing import Callable
from qtpy.QtCore import Signal

from allencell_ml_segmenter.core.image_data_extractor import (
    FakeImageDataExtractor,
    ImageData,
)
from allencell_ml_segmenter.core.channel_extraction import (
    ChannelExtractionThreadSignals,
)


class FakeChannelExtractionThread:
    """
    Fake of Channel Extraction Thread
    """

    def __init__(self, emit_image_data: bool, fake_return_value: int):
        self.signals: ChannelExtractionThreadSignals = (
            ChannelExtractionThreadSignals()
        )
        self.started: bool = False
        self._fake_value = fake_return_value
        self.emit_image_data = emit_image_data

    def start(self):
        self.started = True
        if self.emit_image_data:
            fake_image_data_extractor: FakeImageDataExtractor = (
                FakeImageDataExtractor.global_instance()
            )
            self.signals.image_data_ready.emit(
                fake_image_data_extractor.extract_image_data(Path(""))
            )
        else:
            self.signals.channels_ready.emit(self._fake_value)
