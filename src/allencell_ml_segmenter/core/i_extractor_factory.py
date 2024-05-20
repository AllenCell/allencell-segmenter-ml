from abc import ABC, abstractmethod
from pathlib import Path

from allencell_ml_segmenter.core.i_channel_extraction import (
    IChannelExtractionThread,
)


class IExtractorFactory(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(
        self, img_path: Path, emit_image_data: bool
    ) -> IChannelExtractionThread:
        pass
