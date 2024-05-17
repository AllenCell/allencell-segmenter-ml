from pathlib import Path

from allencell_ml_segmenter.core.i_channel_extraction import IChannelExtractionThread
from allencell_ml_segmenter.core.image_data_extractor import (
    FakeImageDataExtractor,
)
from allencell_ml_segmenter.core.task_executor import SynchroTaskExecutor


class FakeChannelExtractionThread(IChannelExtractionThread):
    """
    Fake of Channel Extraction Thread
    """

    def __init__(self, emit_image_data: bool, fake_return_value: int):
        super().__init__(
            FakeImageDataExtractor.global_instance(),
            SynchroTaskExecutor.global_instance(),
            None,
            emit_image_data
        )
        self.started: bool = False
        self._fake_value = fake_return_value

    def start(self):
        self.started = True
        if self._emit_image_data:
            self.signals.image_data_ready.emit(
                self._image_extractor.extract_image_data(Path(""))
            )
        else:
            self.signals.channels_ready.emit(self._fake_value)
