import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

from qtpy.QtCore import Signal, QObject

from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel, ImageType
from allencell_ml_segmenter.core.image_data_extractor import ImageData


class CurationView(Enum):
    INPUT_VIEW = "input_view"
    MAIN_VIEW = "main_view"


class CurationModel(QObject):
    """
    Stores state relevant to prediction processes.
    """

    current_view_changed: Signal = Signal()
    image_directory_set: Signal = Signal(ImageType)
    channel_count_set: Signal = Signal(ImageType)

    cursor_moved: Signal = Signal()
    image_loading_finished: Signal = Signal()

    save_to_disk_requested: Signal = Signal()
    saved_to_disk: Signal = Signal(bool)

    def __init__(
        self,
        experiments_model: ExperimentsModel,
        main_model: MainModel,
    ) -> None:
        super().__init__()
        # should always start at input view
        self._experiments_model: ExperimentsModel = experiments_model
        self._main_model: MainModel = main_model
        self._current_view: CurationView = CurationView.INPUT_VIEW

        self._img_dirs: Dict[ImageType, Optional[Path]] = (
            self._get_placeholder_dict()
        )
        self._img_dir_paths: Dict[ImageType, Optional[List[Path]]] = (
            self._get_placeholder_dict()
        )

        # These are what the user has selected in the input view
        self._selected_channels: Dict[ImageType, Optional[int]] = (
            self._get_placeholder_dict()
        )
        # these are the total number of channels for the images in the folder
        self._channel_counts: Dict[ImageType, Optional[int]] = (
            self._get_placeholder_dict()
        )

        self._curation_record: Optional[List[CurationRecord]] = None
        # None until start_image_loading is called
        self._cursor: Optional[int] = None
        # True when images have been dropped from memory
        self._image_loading_stopped: bool = False
        # private invariant: _next_img_data will only have < self._get_num_data_dict_keys() keys if
        # a thread is currently updating _next_img_data. Same goes for curr
        self._curr_img_data: Optional[Dict[str, Optional[ImageData]]] = None
        self._next_img_data: Optional[Dict[str, Optional[ImageData]]] = None

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

    def set_image_directory(self, img_type: ImageType, dir: Path) -> None:
        self._img_dirs[img_type] = dir
        self.image_directory_set.emit(img_type)

    def get_image_directory(self, img_type: ImageType) -> Optional[Path]:
        return self._img_dirs[img_type]

    def set_image_directory_paths(
        self, img_type: ImageType, paths: List[Path]
    ) -> None:
        self._img_dir_paths[img_type] = paths

    def get_image_directory_paths(
        self, img_type: ImageType
    ) -> Optional[List[Path]]:
        return self._img_dir_paths[img_type]

    def set_selected_channel(self, img_type: ImageType, channel: int) -> None:
        self._selected_channels[img_type] = channel

    def get_selected_channel(self, img_type: ImageType) -> Optional[int]:
        return self._selected_channels[img_type]

    def set_current_view(self, view: CurationView) -> None:
        """
        Set current curation view
        """
        # TODO: reset all state? only relevant if we expect a nonlinear path through curation, or multiple curation
        # rounds in a single session
        if view != self._current_view:
            if view == CurationView.MAIN_VIEW:
                self._curation_record = self._generate_new_curation_record()
                seg2_exists: bool = (
                    self._img_dir_paths[ImageType.SEG2] is not None
                )
                self._curr_img_data = self._get_placeholder_dict(
                    incl_seg2=seg2_exists
                )
                self._next_img_data = self._get_placeholder_dict(
                    incl_seg2=seg2_exists
                )
                self._curation_record_saved_to_disk = False
                # set the central selected channels for the app once the user clicks 'start curation'
                self._main_model.set_selected_channels(self._selected_channels)
            else:
                self._curation_record = None
                self._curr_img_data = None
                self._next_img_data = None
            self._current_view = view
            self.current_view_changed.emit()

    def get_current_view(self) -> CurationView:
        """
        Get current curation view
        """
        return self._current_view

    def set_channel_count(self, img_type: ImageType, count: int) -> None:
        self._channel_counts[img_type] = count
        self.channel_count_set.emit(img_type)

    def get_channel_count(self, img_type: ImageType) -> Optional[int]:
        return self._channel_counts[img_type]

    def get_save_masks_path(self) -> Path:
        return (
            self._experiments_model.get_user_experiments_path()
            / self._experiments_model.get_experiment_name()
        )

    def get_curation_record(self) -> List[CurationRecord]:
        return self._curation_record

    # WARNING: methods that access data dicts must only be called from the main thread
    def set_curr_image_data(
        self, img_type: ImageType, img_data: ImageData
    ) -> None:
        self._curr_img_data[img_type] = img_data
        if not self.is_waiting_for_images():
            self.image_loading_finished.emit()

    def get_curr_image_data(self, img_type: ImageType) -> Optional[ImageData]:
        return self._curr_img_data[img_type]

    def set_next_image_data(
        self, img_type: ImageType, img_data: ImageData
    ) -> None:
        self._next_img_data[img_type] = img_data
        if not self.is_waiting_for_images():
            self.image_loading_finished.emit()

    # note: I don't see a reason why we would need to get the next image data instead of
    # calling next_image, so leaving that out
    def has_seg2_data(self) -> bool:
        return self._img_dir_paths[ImageType.SEG2] is not None

    def get_num_images(self) -> int:
        return len(self._img_dir_paths[ImageType.RAW])

    def get_curr_image_index(self) -> int:
        return self._cursor

    def has_next_image(self) -> bool:
        return self._cursor + 1 < self.get_num_images()

    def is_waiting_for_curr_images(self) -> bool:
        return len(self._curr_img_data) != self._get_num_data_dict_keys()

    def is_waiting_for_next_images(self) -> bool:
        return len(self._next_img_data) != self._get_num_data_dict_keys()

    def is_waiting_for_images(self) -> bool:
        return (
            self.is_waiting_for_curr_images()
            or self.is_waiting_for_next_images()
        )

    def get_image_loading_stopped(self) -> bool:
        return self._image_loading_stopped

    def start_loading_images(self) -> None:
        """
        Must be called before attempting to get image data.
        Signals emitted:
        immediate: cursor_moved
        at some point: image_loading_finished
        """
        self._cursor = 0
        # need to set use image to true since we want this to be the default
        self.set_use_image(True)
        self._curr_img_data.clear()
        if self.has_next_image():
            self._next_img_data.clear()
        self.cursor_moved.emit()

    def stop_loading_images(self) -> None:
        """
        Drops pre-loaded images from memory and prevents further
        image loading from occurring in curation.
        """
        self._image_loading_stopped = True
        self._curr_img_data.clear()
        self._next_img_data.clear()

    def next_image(self) -> None:
        """
        Move to the next image for curation.
        Signals emitted:
        immediate: cursor_moved
        at some point: image_loading_finished
        """
        if self.get_image_loading_stopped():
            raise RuntimeError("Image loader is stopped.")
        if self.is_waiting_for_images():
            raise RuntimeError(
                "Image loader is busy. Please see image_loading_finished signal."
            )
        if not self.has_next_image():
            raise RuntimeError("No next image available")

        self._curr_img_data = self._next_img_data
        self._cursor += 1
        self._next_img_data = (
            {}
            if self.has_next_image()
            else self._get_placeholder_dict(incl_seg2=self.has_seg2_data())
        )
        # need to set use image to true since we want this to be the default
        self.set_use_image(True)
        self.cursor_moved.emit()
        # client expects that calling next will eventually result in a image_loading_finished signal
        if not self.has_next_image():
            self.image_loading_finished.emit()

    def set_curation_record_saved_to_disk(self, saved: bool) -> None:
        self.saved_to_disk.emit(saved)

    def save_curr_curation_record_to_disk(self) -> None:
        self.save_to_disk_requested.emit()

    def get_csv_path(self) -> Path:
        return (
            self._experiments_model.get_user_experiments_path()
            / self._experiments_model.get_experiment_name()
            / "data"
        )

    def get_max_num_images_to_use(self) -> int:
        """
        Returns the maximum size of the resulting dataset at this point in curation. For example,
        if the user has 12 images and has marked 3 as 'do not use', this method will return 9.
        Note that the current 'to_use' selection is not included, as it is not yet finalized.
        """
        possible: int = self.get_num_images()
        for i in range(self._cursor):
            if not self._curation_record[i].to_use:
                possible -= 1
        return possible

    def get_num_images_selected_to_use(self) -> int:
        """
        Returns the number of images that the user has marked as 'use this image'. Current
        'to_use' selection is included.
        """
        return sum([1 if rec.to_use else 0 for rec in self._curation_record])

    def _generate_new_curation_record(self) -> List[CurationRecord]:
        """
        Returns a list of curation records populated with the file paths in self._img_dir_paths. See
        CurationRecord constructor below for the default values.
        """
        raw_paths: List[Path] = self._img_dir_paths[ImageType.RAW]
        seg1_paths: List[Path] = self._img_dir_paths[ImageType.SEG1]
        seg2_paths: Optional[List[Path]] = self._img_dir_paths[ImageType.SEG2]
        if len(raw_paths) != len(seg1_paths) or (
            seg2_paths is not None and len(seg1_paths) != len(seg2_paths)
        ):
            raise ValueError("provided image dirs must be of same length")
        elif len(raw_paths) < 1:
            raise ValueError("cannot load images from empty image dir")

        return [
            CurationRecord(
                raw_paths[i],
                seg1_paths[i],
                (seg2_paths[i] if seg2_paths is not None else None),
                None,
                None,
                "seg1",
                False,
            )
            for i in range(len(raw_paths))
        ]

    def _get_num_data_dict_keys(self) -> int:
        """
        Returns expected number of keys in an img data dict that is not being written to
        asynchronously.
        """
        return 3 if self.has_seg2_data() else 2

    def _get_placeholder_dict(
        self, incl_seg2: bool = True
    ) -> Dict[ImageType, Optional[Any]]:
        """
        Returns placeholder image data dict with keys mapped to None. Only includes a SEG2 key if
        :param incl_seg2: is True.
        """
        output: Dict[ImageType, Optional[Any]] = {
            ImageType.RAW: None,
            ImageType.SEG1: None,
        }
        if incl_seg2:
            output[ImageType.SEG2] = None
        return output
