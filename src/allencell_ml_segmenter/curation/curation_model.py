import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from enum import Enum

from qtpy.QtCore import Signal, QObject

from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.curation.curation_image_loader import (
    ICurationImageLoader,
    ICurationImageLoaderFactory,
)
from allencell_ml_segmenter.core.image_data_extractor import ImageData


class CurationView(Enum):
    INPUT_VIEW = "input_view"
    MAIN_VIEW = "main_view"

class 

class CurationModel(QObject):
    """
    Stores state relevant to prediction processes.
    """

    current_view_changed: Signal = Signal()

    raw_directory_set: Signal = Signal()
    seg1_directory_set: Signal = Signal()
    seg2_directory_set: Signal = Signal()

    raw_image_channel_count_set: Signal = Signal()
    seg1_image_channel_count_set: Signal = Signal()
    seg2_image_channel_count_set: Signal = Signal()

    cursor_moved: Signal = Signal()
    image_loading_finished: Signal = Signal()

    save_to_disk_requested: Signal = Signal()
    saved_to_disk: Signal = Signal()

    def __init__(
        self,
        experiments_model: ExperimentsModel,
    ) -> None:
        super().__init__()
        # should always start at input view

        self._experiments_model: ExperimentsModel = experiments_model
        self._current_view: CurationView = CurationView.INPUT_VIEW

        self._raw_directory: Optional[Path] = None
        self._seg1_directory: Optional[Path] = None
        self._seg2_directory: Optional[Path] = None

        self._raw_directory_paths: Optional[List[Path]] = None
        self._seg1_directory_paths: Optional[List[Path]] = None
        self._seg2_directory_paths: Optional[List[Path]] = None

        # These are what the user has selected in the input view
        self._raw_image_channel: Optional[int] = None
        self._seg1_image_channel: Optional[int] = None
        self._seg2_image_channel: Optional[int] = None
        # these are the total number of channels for the images in the folder
        self._raw_image_channel_count: Optional[int] = None
        self._seg1_image_channel_count: Optional[int] = None
        self._seg2_image_channel_count: Optional[int] = None

        self._curation_record: Optional[List[CurationRecord]] = None
        # None until start_image_loading is called
        self._cursor: Optional[int] = None
        self._curation_record_saved_to_disk: bool = False

        # private invariant: _next_img_data will only have < self._get_num_data_dict_keys() keys if
        # a thread is currently updating _next_img_data. Same goes for prev and curr
        self._curr_img_data: Dict[str, Optional[ImageData]] = (
            self._get_placeholder_dict()
        )
        self._next_img_data: Dict[str, Optional[ImageData]] = (
            self._get_placeholder_dict()
        )
        self._prev_img_data: Dict[str, Optional[ImageData]] = (
            self._get_placeholder_dict()
        )

    def get_merging_mask(self) -> Optional[np.ndarray]:
        return self._curation_record[self._cursor].merging_mask

    def set_merging_mask(self, mask: np.ndarray) -> None:
        self._curation_record[self._cursor].merging_mask = mask

    def get_excluding_mask(self) -> Optional[np.ndarray]:
        return self._curation_record[self._cursor].excluding_mask

    def set_excluding_mask(self, mask: np.ndarray) -> None:
        self._curation_record[self._cursor].excluding_mask = mask

    def get_base_image(self) -> Optional[str]:
        return self._curation_record[self._cursor].base_image

    def set_base_image(self, base: str) -> None:
        self._curation_record[self._cursor].base_image = base

    def get_use_image(self) -> bool:
        return self._curation_record[self._cursor].to_use

    def set_use_image(self, use: bool) -> None:
        self._curation_record[self._cursor].to_use = use

    def set_raw_directory(self, dir: Path) -> None:
        """
        Set the raw image directory path
        """
        self._raw_directory = dir
        self.raw_directory_set.emit()

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
        self.seg1_directory_set.emit()

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
        self.seg2_directory_set.emit()

    def get_seg2_directory(self) -> Path:
        """
        Get the seg2 image directory path
        """
        return self._seg2_directory

    def set_raw_directory_paths(self, paths: List[Path]) -> None:
        self._raw_directory_paths = paths

    def set_seg1_directory_paths(self, paths: List[Path]) -> None:
        self._seg1_directory_paths = paths

    def set_seg2_directory_paths(self, paths: List[Path]) -> None:
        self._seg2_directory_paths = paths

    def set_raw_channel(self, channel: int) -> None:
        """
        Set the raw image channel
        """
        self._raw_image_channel = channel

    def get_raw_channel(self) -> Optional[int]:
        """
        Get the raw image channel
        """
        return self._raw_image_channel

    def set_seg1_channel(self, channel: int) -> None:
        """
        Set the seg1 image channel
        """
        self._seg1_image_channel = channel

    def get_seg1_channel(self) -> Optional[int]:
        """
        get the seg1 image channel
        """
        return self._seg1_image_channel

    def set_seg2_channel(self, channel: int) -> None:
        """
        Set the seg2 image channel
        """
        self._seg2_image_channel = channel

    def get_seg2_channel(self) -> Optional[int]:
        """
        Get the seg2 image channel
        """
        return self._seg2_image_channel

    def set_current_view(self, view: CurationView) -> None:
        """
        Set current curation view
        """
        # TODO: reset all state? only relevant if we expect a nonlinear path through curation, or multiple curation
        # rounds in a single session
        if view != self._current_view:
            if view == CurationView.MAIN_VIEW:
                self._curation_record = self._generate_new_curation_record()
                self._curation_record_saved_to_disk = False
            else:
                self._curation_record = None
            self._current_view = view
            self.current_view_changed.emit()

    def get_current_view(self) -> CurationView:
        """
        Get current curation view
        """
        return self._current_view

    def get_raw_image_channel_count(self) -> int:
        """
        Get total number of raw channels
        """
        return self._raw_image_channel_count

    def get_seg1_image_channel_count(self) -> int:
        """
        Get total number of seg1 channels
        """
        return self._seg1_image_channel_count

    def get_seg2_image_channel_count(self) -> int:
        """
        Get total number of seg2 channels
        """
        return self._seg2_image_channel_count

    def set_raw_image_channel_count(self, channels: int) -> None:
        self._raw_image_channel_count = channels
        self.raw_image_channel_count_set.emit()

    def set_seg1_image_channel_count(self, channels: int) -> None:
        self._seg1_image_channel_count = channels
        self.seg1_image_channel_count_set.emit()

    def set_seg2_image_channel_count(self, channels: int) -> None:
        self._seg2_image_channel_count = channels
        self.seg2_image_channel_count_set.emit()

    def get_save_masks_path(self) -> Path:
        return (
            self._experiments_model.get_user_experiments_path()
            / self._experiments_model.get_experiment_name()
        )

    def get_curation_record(self) -> List[CurationRecord]:
        return self._curation_record

    def is_user_experiment_selected(self) -> bool:
        if self._experiments_model.get_experiment_name() is None:
            return False
        else:
            return True

    def get_raw_image_data(self) -> ImageData:
        return self._curr_img_data["raw"]

    def get_seg1_image_data(self) -> ImageData:
        return self._image_loader.get_seg1_image_data()

    def get_seg2_image_data(self) -> Optional[ImageData]:
        return self._image_loader.get_seg2_image_data()

    def get_num_images(self) -> int:
        return len(self._raw_directory_paths)

    def get_curr_image_index(self) -> int:
        return self._cursor

    def has_next_image(self) -> bool:
        return self._cursor + 1 < self.get_num_images()

    def is_loading_images(self) -> bool:
        return (
            len(self._curr_img_data) != self._get_num_data_dict_keys()
            or len(self._prev_img_data) != self._get_num_data_dict_keys()
            or len(self._next_img_data) != self._get_num_data_dict_keys()
        )

    def start_loading_images(self) -> None:
        """
        Must be called before attempting to get image data.
        Signals emitted:
        cursor_moved
        """
        self._cursor = 0
        # need to set use image to true since we want this to be the default
        self.set_use_image(True)
        self._curr_img_data.clear()
        if self.has_next_image():
            self._next_img_data.clear()
        self.cursor_moved.emit()

    def next_image(self) -> None:
        """
        Move to the next image for curation.
        Signals emitted:
        cursor_moved
        """
        if self.is_loading_images():
            raise RuntimeError(
                "Image loader is busy. Please see image_loading_finished signal."
            )
        self._prev_img_data = self._curr_img_data
        self._curr_img_data = self._next_img_data
        self._next_img_data = {}
        self._cursor += 1
        # need to set use image to true since we want this to be the default
        self.set_use_image(True)
        self.cursor_moved.emit()

    def set_curation_record_saved_to_disk(self, saved: bool) -> None:
        self._curation_record_saved_to_disk = saved
        if saved:
            self.saved_to_disk.emit()

    def get_curation_record_saved_to_disk(self) -> bool:
        return self._curation_record_saved_to_disk

    def save_curr_curation_record_to_disk(self) -> None:
        if not self._curation_record_saved_to_disk:
            self.save_to_disk_requested.emit()

    def get_csv_path(self) -> Path:
        return (
            self._experiments_model.get_user_experiments_path()
            / self._experiments_model.get_experiment_name()
            / "data"
            / "train.csv"
        )

    def _generate_new_curation_record(self) -> List[CurationRecord]:
        if len(self._raw_directory_paths) != len(
            self._seg1_directory_paths
        ) or (
            self._seg2_directory_paths is not None
            and len(self._seg1_directory_paths)
            != len(self._seg2_directory_paths)
        ):
            raise ValueError("provided image dirs must be of same length")
        elif len(self._raw_directory_paths) < 1:
            raise ValueError("cannot load images from empty image dir")

        return [
            CurationRecord(
                self._raw_directory_paths[i],
                self._seg1_directory_paths[i],
                (
                    self._seg2_directory_paths[i]
                    if self._seg2_directory_paths is not None
                    else None
                ),
                None,
                None,
                "seg1",
                False,
            )
            for i in range(len(self._raw_directory_paths))
        ]

    def _get_num_data_dict_keys(self) -> int:
        """
        Returns expected number of keys in an img data dict that is not being written to
        asynchronously.
        """
        return 3 if self._seg2_directory_paths else 2

    def _get_placeholder_dict(self) -> Dict[str, Optional[ImageData]]:
        """
        Returns placeholder image data dict with keys mapped to None. Necessary to use
        placeholders instead of empty dicts so that calls to is_loading_images() can return False
        when at the beginning or end of curation.
        """
        return (
            {"raw": None, "seg1": None}
            if self._get_num_data_dict_keys() == 2
            else {"raw": None, "seg1": None, "seg2": None}
        )
