from pathlib import Path
from typing import List

from aicsimageio import AICSImage

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class CurationModel(Publisher):
    """
    Stores state relevant to prediction processes.
    """

    def __init__(self):
        super().__init__()
        self.current_view = None

        self._raw_directory: Path = None
        self._seg1_directory: Path = None
        self._seg2_directory: Path = (
            None  # optional, if None was never selected
        )
        # These are what the user has selected in the input view
        self._raw_image_channel: int = None
        self._seg1_image_channel: int = None
        self._seg2_image_channel: int = None
        # these are the total number of channels for the images in the folder
        self._raw_image_channel_count: int = None
        self._seg1_image_channel_count: int = None
        self._seg2_image_channel_count: int = None

    def set_raw_directory(self, dir: Path):
        self._raw_directory = dir
        self._raw_image_channel_count = self.get_total_num_channels(
            self._raw_directory
        )
        self.dispatch(Event.ACTION_CURATION_RAW_SELECTED)

    def get_raw_directory(self) -> Path:
        return self._raw_directory

    def set_seg1_directory(self, dir: Path):
        self._seg1_directory = dir
        self._seg1_image_channel_count = self.get_total_num_channels(
            self._seg1_directory
        )
        self.dispatch(Event.ACTION_CURATION_SEG1_SELECTED)

    def get_seg1_directory(self) -> Path:
        return self._seg1_directory

    def set_seg2_directory(self, dir: Path):
        self._seg2_directory = dir
        self._seg2_image_channel_count = self.get_total_num_channels(
            self._seg2_directory
        )
        self.dispatch(Event.ACTION_CURATION_SEG2_SELECTED)

    def get_seg2_directory(self) -> Path:
        return self._seg2_directory

    def set_raw_channel(self, channel: int):
        self._raw_image_channel = channel

    def get_raw_channel(self) -> int:
        return self._raw_image_channel

    def set_seg1_channel(self, channel: int):
        self._seg1_image_channel = channel

    def get_seg1_channel(self) -> int:
        return self._seg1_image_channel

    def set_seg2_channel(self, channel: int):
        self._seg2_image_channel = channel

    def get_seg2_channel(self) -> int:
        return self._seg2_image_channel

    def set_view(self):
        self.dispatch(Event.PROCESS_CURATION_INPUT_STARTED)

    def get_view(self):
        return self.current_view

    def get_total_num_channels_raw(self):
        return self._raw_image_channel_count

    def get_total_num_channels_seg1(self):
        return self._seg1_image_channel_count

    def get_total_num_channels_seg2(self):
        return self._seg2_image_channel_count

    def get_total_num_channels(self, path) -> int:
        # we expect user to have the same number of channels for all images in their folders
        # and that only images are stored in those folders
        first_image = path.iterdir().__next__()
        img = AICSImage(str(first_image.resolve()))
        # return num channel
        return img.dims.C
