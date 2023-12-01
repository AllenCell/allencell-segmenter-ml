from pathlib import Path
from typing import Dict, Tuple, List, Optional


from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel


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
        self.excluding_mask_shape_layers = []
        self.merging_mask_shape_layers = []
        self.curation_record: List[CurationRecord] = []
        self.curation_image_dims: Tuple[int, int, int] = None

        self._current_excluding_mask_path: Path = None
        self._current_merging_mask_path: Path = None
        self._current_loaded_images: Tuple[Path, Path] = (None, None)
        self.merging_mask_base_layer: str = None

        self.curation_index: int = 0
        self.raw_images: List[Path] = list()
        self.seg1_images: List[Path] = list()
        self.seg2_images: List[Path] = list()

    def get_raw_images(self) -> List[Path]:
        return self.raw_images

    def get_current_raw_image(self) -> Path:
        return self.get_raw_images()[self.curation_index]

    def set_raw_images(self, images: List[Path]) -> None:
        self.raw_images = images

    def get_seg1_images(self) -> List[Path]:
        return self.seg1_images

    def set_seg1_images(self, images: List[Path]) -> None:
        self.seg1_images = images

    def get_current_seg1_image(self) -> Path:
        return self.get_seg1_images()[self.curation_index]

    def get_seg2_images(self) -> List[Path]:
        return self.seg2_images

    def set_seg2_images(self, images: List[Path]) -> None:
        self.seg2_images = images

    def get_current_seg2_image(self) -> Path:
        return self.get_seg2_images()[self.curation_index]

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

    def get_save_masks_path(self) -> Path:
        return (
            self.experiments_model.get_user_experiments_path()
            / self.experiments_model.get_experiment_name()
        )

    def set_current_loaded_images(self, images: Tuple[Path, Path, Optional[Path]]):
        self._current_loaded_images = images

    def get_current_loaded_images(self):
        return self._current_loaded_images

    def get_curation_record(self) -> List[CurationRecord]:
        return self.curation_record

    def set_current_excluding_mask_path(self, path: Path):
        self._current_excluding_mask_path = path

    def get_current_excluding_mask_path(self) -> Path:
        current_mask_path: Path = self._current_excluding_mask_path
        self._current_excluding_mask_path = None
        return current_mask_path

    def set_current_merging_mask_path(self, path: Path):
        self._current_merging_mask_path = path

    def get_current_merging_mask_path(self):
        return self._current_merging_mask_path

    def get_excluding_mask_shape_layers(self) -> List:
        return self.excluding_mask_shape_layers

    def get_user_experiment_selected(self) -> bool:
        if self.experiments_model.get_experiment_name() is None:
            return False
        else:
            return True

    def image_available(self) -> bool:
        return self.curation_index < len(self.raw_images)

    def get_curation_index(self) -> int:
        return self.curation_index

    def set_curation_index(self, i: int) -> None:
        self.curation_index = i
