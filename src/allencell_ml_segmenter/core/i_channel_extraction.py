from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from qtpy.QtCore import QObject, Signal

from allencell_ml_segmenter.core.image_data_extractor import (
    ImageData,
    IImageDataExtractor,
)
from allencell_ml_segmenter.core.task_executor import ITaskExecutor


class ChannelExtractionThreadSignals(QObject):
    # emitted when channels ready, with number of channels in image
    channels_ready: Signal = Signal(int)  # num_channels
    # emitted when image data ready, with ImageData object
    image_data_ready: Signal = Signal(ImageData)
    # emitted when error occurred during image data extraction
    task_failed: Signal = Signal(Exception)


class IChannelExtractionThread(ABC):
    def __init__(
        self,
        img_data_extractor: IImageDataExtractor,
        task_executor: ITaskExecutor,
        img_path: Optional[Path],
        emit_image_data: bool = False,
        parent: QObject = None,
    ):
        super().__init__()
        self.signals: ChannelExtractionThreadSignals = (
            ChannelExtractionThreadSignals()
        )
        self._img_path: Optional[Path] = img_path
        self._emit_image_data = emit_image_data
        self._image_extractor = img_data_extractor
        self.task_executor = task_executor

    @abstractmethod
    def start(self):
        pass
