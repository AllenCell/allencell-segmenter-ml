import aicsimageio.exceptions

from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.channel_extraction import (
    ChannelExtractionThread,
)
from allencell_ml_segmenter.core.image_data_extractor import ImageData
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.curation.curation_image_loader import (
    ICurationImageLoader,
    ICurationImageLoaderFactory,
)
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.main.viewer import Viewer
from allencell_ml_segmenter.utils.file_utils import FileUtils

from pathlib import Path
from typing import List, Union, Optional, Callable
import csv
import numpy as np
from aicsimageio import AICSImage
from enum import Enum
from napari.utils.notifications import show_info
from napari.layers.shapes.shapes import Shapes


MERGING_MASK_LAYER_NAME: str = "Merging Mask"
EXCLUDING_MASK_LAYER_NAME: str = "Excluding Mask"


class CurationService(Subscriber):
    """ """

    def __init__(
        self,
        curation_model: CurationModel,
        viewer: IViewer,
        img_loader_factory: ICurationImageLoaderFactory,
    ) -> None:
        super().__init__()
        self._curation_model: CurationModel = curation_model
        self._viewer: Viewer = viewer
        self._img_loader_factory = img_loader_factory
        self._raw_thread: Optional[ChannelExtractionThread] = None
        self._seg1_thread: Optional[ChannelExtractionThread] = None
        self._seg2_thread: Optional[ChannelExtractionThread] = None

    def build_raw_images_list(self) -> List[Path]:
        """
        Return all raw images in the raw images path as a list of Paths
        """
        raw_path: Path = self._curation_model.get_raw_directory()
        if raw_path is None:
            raise ValueError(
                "Raw directory not set. Please set raw directory."
            )
        return FileUtils.get_all_files_in_dir_ignore_hidden(raw_path)

    def build_seg1_images_list(self) -> List[Path]:
        """
        Return all seg1 images in the seg1 images path as a list of Paths
        """
        seg1_path: Path = self._curation_model.get_seg1_directory()
        if seg1_path is None:
            raise ValueError(
                "Seg1 directory not set. Please set seg1 directory."
            )
        return FileUtils.get_all_files_in_dir_ignore_hidden(seg1_path)

    def build_seg2_images_list(self) -> List[Path]:
        """
        Return all seg2 images in the seg2 images path as a list of Paths
        """
        seg2_path: Path = self._curation_model.get_seg2_directory()
        if seg2_path is None:
            raise ValueError(
                "Seg2 directory not set. Please set seg2 directory."
            )
        return FileUtils.get_all_files_in_dir_ignore_hidden(seg2_path)

    def open_image_from_path(self, path: Path) -> AICSImage:
        """
        Open an image from a path
        """
        return AICSImage(str(path))

    def write_curation_record(
        self, curation_record: List[CurationRecord], path: Path
    ) -> None:
        """
        Save the curation record as a csv at the specified path. will create parent directories to path as needed.

        curation_record (List[CurationRecord]): record to save to csv
        path (Path): path to save csv
        """
        parent_path: Path = path.parents[0]
        if not parent_path.is_dir():
            parent_path.mkdir(parents=True)

        with open(path, "w") as f:
            # need file header
            writer: csv.writer = csv.writer(f, delimiter=",")
            writer.writerow(
                [
                    "",
                    "raw",
                    "seg1",
                    "seg2",
                    "excluding_mask",
                    "merging_mask",
                    "merging_col",
                ]
            )
            for idx, record in enumerate(curation_record):
                if record.to_use:
                    writer.writerow(
                        [
                            str(idx),
                            str(record.raw_file),
                            str(record.seg1),
                            str(record.seg2) if record.seg2 else "",
                            str(record.excluding_mask),
                            str(record.merging_mask),
                            str(record.base_image_index),
                        ]
                    )
                f.flush()

        # TODO: WRITE ACTUAL VALIDATION AND TEST SETS

    def remove_all_images_from_viewer_layers(self) -> None:
        """
        Remove all images from the napari viewer
        """
        self._viewer.clear_layers()

    def add_image_to_viewer(self, img_data: ImageData, title: str) -> None:
        """
        Add an image to the napari viewer as its own layer

        :param img_data: data for image to display in napari
        """
        self._curation_model.set_curation_image_dims(
            (
                img_data.dim_x,
                img_data.dim_y,
                img_data.dim_z,
            )
        )
        self._viewer.add_image(img_data.np_data, name=title)

    # TODO: move to viewer?
    def _get_layer_by_name(self, name: str):
        matching_layer = None
        for layer in self._viewer.get_layers():
            if layer.name == name:
                matching_layer = layer
                break
        return matching_layer
    
    def _create_mask_layer(self, name: str, color: Optional[str]=None):
        matching_layer = self._get_layer_by_name(name)

        if matching_layer is not None:
            discard_layer_prompt = DialogBox(
                f"There is already a '{name}' layer in the viewer. Would you like to discard this layer and lose any unsaved progress?"
            )
            discard_layer_prompt.exec()
            if not discard_layer_prompt.selection:
                return
            else:
                # TODO: rename to 'clear layers?'
                self._viewer.clear_mask_layers([matching_layer])

        # TODO: encapsulate mode / color in add_shapes?
        points_layer: Shapes = self._viewer.add_shapes(
            name=name
        )
        points_layer.mode = "add_polygon"
        if color is not None:
            points_layer.face_color = color

    def create_excluding_mask_layer(self) -> None:
        self._create_mask_layer(EXCLUDING_MASK_LAYER_NAME, color="coral")
    
    def create_merging_mask_layer(self) -> None:
        self._create_mask_layer(MERGING_MASK_LAYER_NAME, color="royalblue")

    def _stop_channel_extraction_thread(self, thread: ChannelExtractionThread):
        # if we find this is too slow, can switch to the method employed by
        # prediction/service, but may be overkill initially
        if thread and thread.isRunning():
            thread.requestInterruption()
            thread.wait()

    def _start_channel_extraction_thread(
        self, folder: Path, channel_callback: Callable, error_handler: Callable
    ) -> ChannelExtractionThread:
        img_path: Path = FileUtils.get_img_path_from_folder(folder)
        new_thread = ChannelExtractionThread(img_path)
        new_thread.channels_ready.connect(channel_callback)
        new_thread.task_failed.connect(error_handler)
        new_thread.start()
        return new_thread

    def _handle_thread_error(
        self,
        thread: ChannelExtractionThread,
        err_event: Event,
        err: Exception = None,
    ):
        if err:
            show_info(
                "Selected directory does not contain images that are able to be curated. Please select directory of only supported images"
            )
            self._stop_channel_extraction_thread(thread)
            self._curation_model.dispatch(err_event)

    def select_directory_raw(self, path: Path):
        """
        Select a raw directory

        path(Path): path to raw directory
        """
        self._curation_model.set_raw_directory(path)
        self._stop_channel_extraction_thread(self._raw_thread)
        self._raw_thread = self._start_channel_extraction_thread(
            path,
            self._curation_model.set_raw_image_channel_count,
            lambda err: self._handle_thread_error(
                self._raw_thread, Event.ACTION_CURATION_RAW_THREAD_ERROR, err
            ),
        )

    def select_directory_seg1(self, path: Path):
        """
        Select a seg1 directory

        path(Path): path to seg1 directory
        """
        self._curation_model.set_seg1_directory(path)
        self._stop_channel_extraction_thread(self._seg1_thread)
        self._seg1_thread = self._start_channel_extraction_thread(
            path,
            self._curation_model.set_seg1_image_channel_count,
            lambda err: self._handle_thread_error(
                self._seg1_thread, Event.ACTION_CURATION_SEG1_THREAD_ERROR, err
            ),
        )

    def select_directory_seg2(self, path: Path):
        """
        Select a seg2 directory

        path(Path): path to seg2 directory
        """
        self._curation_model.set_seg2_directory(path)
        self._stop_channel_extraction_thread(self._seg2_thread)
        self._seg2_thread = self._start_channel_extraction_thread(
            path,
            self._curation_model.set_seg2_image_channel_count,
            lambda err: self._handle_thread_error(
                self._seg2_thread, Event.ACTION_CURATION_SEG2_THREAD_ERROR, err
            ),
        )

    def save_excluding_mask(self) -> None:
        """
        Save the current excluding mask to disk and update napari
        """
        if not self._curation_model.is_user_experiment_selected():
            show_info("Please select an experiment to save masks.")
            return False
        
        folder_path: Path = (
            self._curation_model.get_save_masks_path() / "excluding_masks"
        )
        # TODO: make this long self._curation_model.get_image... into a method on curation_model itself
        save_path_mask_file: Path = (
            folder_path
            / f"excluding_mask_{self._curation_model.get_image_loader().get_raw_image_data().path.stem}.npy"
        )

        # Checking to see if there is already a merging mask saved.
        if save_path_mask_file.exists():
            # There is already a merging mask saved. Ask if user wants to overwrite
            overwrite_merging_mask_dialog = DialogBox(
                "There is already an excluding mask saved. Would you like to overwrite?"
            )
            overwrite_merging_mask_dialog.exec()
            if not overwrite_merging_mask_dialog.selection:
                return False
        else:
            folder_path.mkdir(parents=True, exist_ok=True)

        mask_to_save: Shapes = self._get_layer_by_name(EXCLUDING_MASK_LAYER_NAME)
        if mask_to_save is None:
            show_info("Please create mask before saving.")
            return False
        
        np.save(
            save_path_mask_file, np.asarray(mask_to_save, dtype=object)
        )
        self._curation_model.dispatch(
            Event.ACTION_CURATION_SAVE_EXCLUDING_MASK
        )
        return True
    
    # TODO: refactor these methods to make more DRY
    def save_merging_mask(self, base_image: str) -> bool:
        """
        Save the current merging mask to disk and update napari
        """
        if not self._curation_model.is_user_experiment_selected():
            show_info("Please select an experiment to save masks.")
            return False
        
        folder_path: Path = (
            self._curation_model.get_save_masks_path() / "merging_masks"
        )
        # TODO: make this long self._curation_model.get_image... into a method on curation_model itself
        save_path_mask_file: Path = (
            folder_path
            / f"merging_mask_{self._curation_model.get_image_loader().get_raw_image_data().path.stem}.npy"
        )

        # Checking to see if there is already a merging mask saved.
        if save_path_mask_file.exists():
            # There is already a merging mask saved. Ask if user wants to overwrite
            overwrite_merging_mask_dialog = DialogBox(
                "There is already a merging mask saved. Would you like to overwrite?"
            )
            overwrite_merging_mask_dialog.exec()
            if not overwrite_merging_mask_dialog.selection:
                return False
        else:
            folder_path.mkdir(parents=True, exist_ok=True)

        mask_to_save: Shapes = self._get_layer_by_name(MERGING_MASK_LAYER_NAME)
        if mask_to_save is None:
            show_info("Please create mask before saving.")
            return False

        np.save(
            save_path_mask_file, np.asarray(mask_to_save, dtype=object)
        )
        self._curation_model.set_merging_mask_base_layer(base_image)
        self._curation_model.dispatch(
            Event.ACTION_CURATION_SAVED_MERGING_MASK
        )
        return True

    def update_curation_record(self, use_image: bool) -> None:
        """
        Update the curation record with the users selection for the current image
        """
        # DEAL WITH EXCLUDING MASKS
        excluding_mask_path: Union[Path, str] = (
            self._curation_model.get_current_excluding_mask_path_and_reset_mask()
        )
        if excluding_mask_path is not None:
            excluding_mask_path = str(excluding_mask_path)
        else:
            excluding_mask_path = ""

        # DEAL WITH MERGING MASKS
        merging_mask_path: Union[Path, str] = (
            self._curation_model.get_current_merging_mask_path()
        )
        if merging_mask_path is not None:
            # user has drawn and saved merging masks.
            merging_mask_path = str(merging_mask_path)
            base_image_name: str = (
                self._curation_model.get_merging_mask_base_layer()
            )
        else:
            # there is no merging mask so dont write anything to the CurationRecord\
            merging_mask_path = ""
            base_image_name = ""
        # Save this curation record.
        loader: ICurationImageLoader = self._curation_model.get_image_loader()
        self._curation_model.append_curation_record(
            CurationRecord(
                loader.get_raw_image_data().path,
                loader.get_seg1_image_data().path,
                (
                    loader.get_seg2_image_data().path
                    if loader.get_seg2_image_data()
                    else None
                ),
                excluding_mask_path,
                merging_mask_path,
                base_image_name,
                use_image,
            )
        )

    def next_image(self, use_image: bool) -> None:
        """
        Load the next image in the curation image stack. Updates curation record and progress bar accordingly.
        """
        _ = show_info("Loading the next image...")
        self.update_curation_record(use_image)
        loader: ICurationImageLoader = self._curation_model.get_image_loader()
        # load next image
        if loader.has_next():
            self.remove_all_images_from_viewer_layers()
            self._curation_model.set_current_merging_mask_path(None)
            self._curation_model.set_current_excluding_mask_path(None)
            loader.next()
            raw_img_data: ImageData = loader.get_raw_image_data()
            seg1_img_data: ImageData = loader.get_seg1_image_data()
            seg2_img_data: Optional[ImageData] = loader.get_seg2_image_data()
            self.add_image_to_viewer(
                raw_img_data, f"[raw] {raw_img_data.path.name}"
            )
            self.add_image_to_viewer(
                seg1_img_data, f"[seg1] {seg1_img_data.path.name}"
            )

            if seg2_img_data:
                self.add_image_to_viewer(
                    seg2_img_data, f"[seg2] {seg2_img_data.path.name}"
                )

        else:
            # No more images to load - curation is complete
            _ = show_info("No more image to load")

            self.write_curation_record(
                self._curation_model.get_curation_record(),
                path=self._curation_model.experiments_model.get_user_experiments_path()
                / self._curation_model.experiments_model.get_experiment_name()
                / "data"
                / "train.csv",
            )

    def curation_setup(self) -> None:
        """
        Set up curation workflow, called once
        """
        # build list of raw images, ignore .DS_Store files
        raw: List[Path] = self.build_raw_images_list()
        # build list of seg1 images, ignore .DS_Store files
        seg1: List[Path] = self.build_seg1_images_list()

        # If seg 2 is selected, build list of seg2 images
        if self._curation_model.get_seg2_directory() is not None:
            seg2 = self.build_seg2_images_list()
        else:
            seg2 = None

        loader: ICurationImageLoader = self._img_loader_factory.create(
            raw, seg1, seg2
        )
        self._curation_model.set_image_loader(loader)
        # reset
        self.remove_all_images_from_viewer_layers()

        raw_img_data: ImageData = loader.get_raw_image_data()
        seg1_img_data: ImageData = loader.get_seg1_image_data()
        seg2_img_data: Optional[ImageData] = loader.get_seg2_image_data()

        self.add_image_to_viewer(
            raw_img_data, f"[raw] {raw_img_data.path.name}"
        )
        self.add_image_to_viewer(
            seg1_img_data, f"[seg1] {seg1_img_data.path.name}"
        )
        if seg2_img_data is not None:
            self.add_image_to_viewer(
                seg2_img_data, f"[seg2] {seg2_img_data.path.name}"
            )
