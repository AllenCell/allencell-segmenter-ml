import numpy as np
from pathlib import Path
from typing import List, Optional
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

    image_loading_finished: Signal = Signal()

    save_to_disk_requested: Signal = Signal()
    saved_to_disk: Signal = Signal()

    def __init__(
        self,
        experiments_model: ExperimentsModel,
        img_loader_factory: ICurationImageLoaderFactory,
    ) -> None:
        super().__init__()
        # should always start at input view

        self._experiments_model: ExperimentsModel = experiments_model
        self._img_loader_factory: ICurationImageLoaderFactory = (
            img_loader_factory
        )
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

        self._curation_record: List[CurationRecord] = []
        self._curation_record_saved_to_disk: bool = False

        self._merging_mask: Optional[np.ndarray] = None
        self._excluding_mask: Optional[np.ndarray] = None

        self._base_image: Optional[str] = None
        self._use_image: bool = True

        self._image_loader: Optional[ICurationImageLoader] = None

    def get_merging_mask(self) -> Optional[np.ndarray]:
        return self._merging_mask

    def set_merging_mask(self, mask: np.ndarray) -> None:
        self._merging_mask = mask

    def get_excluding_mask(self) -> Optional[np.ndarray]:
        return self._excluding_mask

    def set_excluding_mask(self, mask: np.ndarray) -> None:
        self._excluding_mask = mask

    def get_base_image(self) -> Optional[str]:
        return self._base_image

    def set_base_image(self, base: str) -> None:
        self._base_image = base

    def get_use_image(self) -> bool:
        return self._use_image

    def set_use_image(self, use: bool) -> None:
        self._use_image = use

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
                self._curation_record = []
                self._curation_record_saved_to_disk = False
                self._merging_mask = None
                self._excluding_mask = None
                self._base_image = "seg1"
                self._use_image = True

                self._image_loader = self._img_loader_factory.create(
                    self._raw_directory_paths,
                    self._seg1_directory_paths,
                    self._seg2_directory_paths,
                )
                self._image_loader.signals.is_idle.connect(
                    lambda: self.image_loading_finished.emit()
                )
            else:
                self._image_loader = None
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
        return self._image_loader.get_raw_image_data()

    def get_seg1_image_data(self) -> ImageData:
        return self._image_loader.get_seg1_image_data()

    def get_seg2_image_data(self) -> Optional[ImageData]:
        return self._image_loader.get_seg2_image_data()

    def get_num_images(self) -> int:
        return self._image_loader.get_num_images()

    def get_curr_image_index(self) -> int:
        return self._image_loader.get_current_index()

    def has_next_image(self) -> bool:
        return self._image_loader.has_next()

    def save_curr_curation_record(self):
        idx: int = self.get_curr_image_index()
        record: CurationRecord = CurationRecord(
            self.get_raw_image_data().path,
            self.get_seg1_image_data().path,
            (
                self.get_seg2_image_data().path
                if self.get_seg2_image_data() is not None
                else None
            ),
            self.get_excluding_mask(),
            (
                self.get_merging_mask()
                if self.get_seg2_image_data() is not None
                else None
            ),
            (
                self.get_base_image()
                if self.get_base_image() is not None
                and self.get_seg2_image_data() is not None
                else "seg1"
            ),
            self.get_use_image(),
        )
        if idx == len(self._curation_record):
            self._curation_record.append(record)
        else:
            self._curation_record[idx] = record

        # here, I don't actually check that the record changed, seems unnecessary
        self._curation_record_saved_to_disk = False

    def start_loading_images(self) -> None:
        """
        Must be called before attempting to get image data.
        Signals emitted:
        image_loading_finished
        """
        if self._image_loader is None:
            raise RuntimeError(
                "Image loader not initialized. Current view must be main view to load images."
            )
        self._image_loader.start()

    def next_image(self) -> None:
        """
        Move to the next image for curation.
        Signals emitted:
        image_loading_finished
        """
        if self._image_loader.is_busy():
            raise RuntimeError(
                "Image loader is busy. Please see image_data_ready signal."
            )
        self._image_loader.next()
        self._merging_mask = None
        self._excluding_mask = None
        self._base_image = "seg1"
        self._use_image = True

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
        )
