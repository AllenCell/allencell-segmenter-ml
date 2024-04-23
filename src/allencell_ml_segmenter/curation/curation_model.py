from pathlib import Path
from typing import Tuple, List, Optional

from napari.layers import Shapes

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.curation.curation_image_loader import (
    ICurationImageLoader,
)


class CurationModel(Publisher):
    """
    Stores state relevant to prediction processes.
    """

    def __init__(
        self,
        raw_path: Path = None,
        seg1_path: Path = None,
        seg2_path: Path = None,
        experiments_model: ExperimentsModel = None,
    ) -> None:
        super().__init__()
        self.current_view = None

        self._raw_directory: Path = raw_path
        self._seg1_directory: Path = seg1_path
        self._seg2_directory: Path = (
            seg2_path  # optional, if None was never selected
        )
        self.experiments_model = experiments_model
        # These are what the user has selected in the input view
        self._raw_image_channel: int = None
        self._seg1_image_channel: int = None
        self._seg2_image_channel: int = None
        # these are the total number of channels for the images in the folder
        self._raw_image_channel_count: int = None
        self._seg1_image_channel_count: int = None
        self._seg2_image_channel_count: int = None
        self._curation_record: List[CurationRecord] = []
        self._curation_image_dims: Tuple[int, int, int] = None
        self._merging_mask_base_layer: str = None

        self._image_loader: Optional[ICurationImageLoader] = None

    def set_image_loader(self, image_loader: ICurationImageLoader) -> None:
        self._image_loader = image_loader

    def get_image_loader(self) -> Optional[ICurationImageLoader]:
        return self._image_loader

    def get_merging_mask_base_layer(self) -> str:
        return self._merging_mask_base_layer

    def set_merging_mask_base_layer(self, layer_name: str) -> None:
        self._merging_mask_base_layer = layer_name

    def get_curation_image_dims(self) -> Tuple[int, int, int]:
        return self._curation_image_dims

    def set_curation_image_dims(
        self, image_dims: Tuple[int, int, int]
    ) -> None:
        self._curation_image_dims = image_dims

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

    def set_raw_image_channel_count(self, channels: int) -> int:
        self._raw_image_channel_count = channels
        self.dispatch(Event.ACTION_CURATION_RAW_CHANNELS_SET)

    def set_seg1_image_channel_count(self, channels: int) -> int:
        self._seg1_image_channel_count = channels
        self.dispatch(Event.ACTION_CURATION_SEG1_CHANNELS_SET)

    def set_seg2_image_channel_count(self, channels: int) -> int:
        self._seg2_image_channel_count = channels
        self.dispatch(Event.ACTION_CURATION_SEG2_CHANNELS_SET)

    def get_save_masks_path(self) -> Path:
        return (
            self.experiments_model.get_user_experiments_path()
            / self.experiments_model.get_experiment_name()
        )

    def get_curation_record(self) -> List[CurationRecord]:
        return self._curation_record

    def is_user_experiment_selected(self) -> bool:
        if self.experiments_model.get_experiment_name() is None:
            return False
        else:
            return True

    def append_curation_record(self, record: CurationRecord) -> None:
        self._curation_record.append(record)
