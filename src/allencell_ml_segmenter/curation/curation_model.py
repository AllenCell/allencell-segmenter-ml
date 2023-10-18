from pathlib import Path

from aicsimageio import AICSImage

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class CurationModel(Publisher):
    """
    Stores state relevant to prediction processes.
    """

    def __init__(
        self,
        raw_path: Path = None,
        seg1_path: Path = None,
        seg2_path: Path = None,
    ) -> None:
        super().__init__()
        self.current_view = None

        self._raw_directory: Path = raw_path
        self._seg1_directory: Path = seg1_path
        self._seg2_directory: Path = (
            seg2_path  # optional, if None was never selected
        )
        # These are what the user has selected in the input view
        self._raw_image_channel: int = None
        self._seg1_image_channel: int = None
        self._seg2_image_channel: int = None
        # these are the total number of channels for the images in the folder
        self._raw_image_channel_count: int = None
        self._seg1_image_channel_count: int = None
        self._seg2_image_channel_count: int = None

    def set_raw_directory(self, dir: Path) -> None:
        """
        Set the raw image directory path
        """
        self._raw_directory = dir

    def get_raw_directory(self) -> Path:
        """
        Get the raw image directory path
        """
        return self._raw_directory

    def set_seg1_directory(self, dir: Path) -> None:
        """
        Set the seg1 image directory path
        """
        self._seg1_directory = dir

    def get_seg1_directory(self) -> Path:
        """
        Get the seg1 image directory path
        """
        return self._seg1_directory

    def set_seg2_directory(self, dir: Path) -> None:
        """
        Set the seg2 image directory path
        """
        self._seg2_directory = dir

    def get_seg2_directory(self) -> Path:
        """
        Get the seg2 image directory path
        """
        return self._seg2_directory

    def set_raw_channel(self, channel: int) -> None:
        """
        Set the raw image channel
        """
        self._raw_image_channel = channel

    def get_raw_channel(self) -> int:
        """
        Get the raw image channel
        """
        return self._raw_image_channel

    def set_seg1_channel(self, channel: int) -> None:
        """
        Set the seg1 image channel
        """
        self._seg1_image_channel = channel

    def get_seg1_channel(self) -> int:
        """
        get the seg1 image channel
        """
        return self._seg1_image_channel

    def set_seg2_channel(self, channel: int) -> None:
        """
        Set the seg2 image channel
        """
        self._seg2_image_channel = channel

    def get_seg2_channel(self) -> int:
        """
        Get the seg2 image channel
        """
        return self._seg2_image_channel

    def set_view(self) -> None:
        """
        Set current curation view
        """
        self.dispatch(Event.PROCESS_CURATION_INPUT_STARTED)

    def get_view(self) -> str:
        """
        Get current curation view
        """
        return self.current_view

    def get_total_num_channels_raw(self) -> int:
        """
        Get total number of raw channels
        """
        return self._raw_image_channel_count

    def get_total_num_channels_seg1(self) -> int:
        """
        Get total number of seg1 channels
        """
        return self._seg1_image_channel_count

    def get_total_num_channels_seg2(self) -> int:
        """
        Get total number of seg2 channels
        """
        return self._seg2_image_channel_count

    def set_total_num_channels_raw(self, channels: int) -> int:
        self._raw_image_channel_count = channels

    def set_total_num_channels_seg1(self, channels: int) -> int:
        self._seg1_image_channel_count = channels

    def set_total_num_channels_seg2(self, channels: int) -> int:
        self._seg2_image_channel_count = channels

    def get_total_num_channels(self, path: Path) -> int:
        """
        Determine total number of channels for image in a set folder
        """
        # we expect user to have the same number of channels for all images in their folders
        # and that only images are stored in those folders
        first_image: Path = path.iterdir().__next__()
        img: AICSImage = AICSImage(str(first_image.resolve()))
        # return num channel
        return img.dims.C
