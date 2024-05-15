from pathlib import Path

from allencell_ml_segmenter._tests.fakes.fake_channel_extraction import (
    FakeChannelExtractionThread,
)
from allencell_ml_segmenter.core.channel_extraction import (
    ChannelExtractionThread,
)
from allencell_ml_segmenter.core.i_extractor_factory import IExtractorFactory


class ExtractorFactory(IExtractorFactory):

    def create(
        self, img_path: Path, get_image_data: bool = False
    ) -> ChannelExtractionThread:
        return ChannelExtractionThread(img_path, get_image_data=get_image_data)


class FakeExtractorFactory(IExtractorFactory):
    def __init__(self, fake_value: int):
        super().__init__()
        self._fake_value = fake_value

    def create(
        self, img_path: Path, get_image_data: bool = False
    ) -> FakeChannelExtractionThread:
        return FakeChannelExtractionThread(self._fake_value)
