import aicsimageio.exceptions
from PyQt5.QtCore import QThread

from allencell_ml_segmenter.core.csv_writer_thread import CSVWriterThread, CSVWriterMode
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


class SelectionMode(Enum):
    EXCLUDING = "excluding"
    MERGING = "merging"


class CurationService(Subscriber):
    """ """

    def __init__(
        self,
        curation_model: CurationModel,
        viewer: IViewer,
        img_loader_factory: Optional[ICurationImageLoaderFactory],
    ) -> None:
        super().__init__()
        self._curation_model: CurationModel = curation_model
        self._viewer: Viewer = viewer
        self._img_loader_factory = img_loader_factory
        self._raw_thread: Optional[ChannelExtractionThread] = None
        self._seg1_thread: Optional[ChannelExtractionThread] = None
        self._seg2_thread: Optional[ChannelExtractionThread] = None
        self._csv_write_thread: Optional[CSVWriterThread] = None

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
        self
    ) -> None:
        """
        Save the curation record as a csv at the specified path. will create parent directories to path as needed.

        curation_record (List[CurationRecord]): record to save to csv
        path (Path): path to save csv
        """
        # only if we have something to write
        curation_record = self._curation_model.get_curation_record()
        if len(curation_record) > 0:
            self._stop_thread(self._csv_write_thread)
            # Start csv writing thread
            self._csv_write_thread = self._start_csv_writer_thread(self._curation_model.experiments_model.get_user_experiments_path()
        / self._curation_model.experiments_model.get_experiment_name()
        / "data"
        / "train.csv", curation_record)


    def remove_all_images_from_viewer_layers(self) -> None:
        """
        Remove all images from the napari viewer
        """
        self._viewer.clear_layers()
        # if all images are removed, need to reset excluding mask and merging mask state as well
        self._curation_model.set_excluding_mask_shape_layers([])
        self._curation_model.set_merging_mask_shape_layers([])

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

    def enable_shape_selection_viewer(self, mode: SelectionMode) -> None:
        """
        Enable shape selection in napari

        mode(SelectionMode): EXCLUDING or MERGING
        """
        _ = show_info("Draw excluding area")
        if mode == SelectionMode.EXCLUDING:
            # append points layer to excluding mask shapes list
            points_layer: Shapes = self._viewer.add_shapes(
                name="Excluding Mask"
            )
            points_layer.mode = "add_polygon"
            # todo combine maybe
            self._curation_model.append_excluding_mask_shape_layer(
                points_layer
            )
            self._curation_model.dispatch(Event.ACTION_CURATION_DRAW_EXCLUDING)
        elif mode == SelectionMode.MERGING:
            points_layer: Shapes = self._viewer.add_shapes(name="Merging Mask")
            points_layer.face_color = "royal_blue"
            points_layer.mode = "add_polygon"
            self._curation_model.append_merging_mask_shape_layer(points_layer)
            self._curation_model.dispatch(Event.ACTION_CURATION_DRAW_MERGING)

    def _stop_thread(self, thread: QThread):
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

    def _start_csv_writer_thread(
        self, csv_path: Path, curation_record: List[CurationRecord]
    ) -> CSVWriterThread:
        new_thread = CSVWriterThread(csv_path, CSVWriterMode.curation, curation_record)
        # connect write finished
        new_thread.write_finished.connect(self._csv_write_finished)
        # connect error handling
        # new_thread.task_failed.connect(error_handler)
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
            self._stop_thread(thread)
            self._curation_model.dispatch(err_event)

    def select_directory_raw(self, path: Path):
        """
        Select a raw directory

        path(Path): path to raw directory
        """
        self._curation_model.set_raw_directory(path)
        self._stop_thread(self._raw_thread)
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
        self._stop_thread(self._seg1_thread)
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
        self._stop_thread(self._seg2_thread)
        self._seg2_thread = self._start_channel_extraction_thread(
            path,
            self._curation_model.set_seg2_image_channel_count,
            lambda err: self._handle_thread_error(
                self._seg2_thread, Event.ACTION_CURATION_SEG2_THREAD_ERROR, err
            ),
        )

    def finished_shape_selection(self, selection_mode: SelectionMode) -> None:
        """
        Called when done drawing to enable default PAN_ZOOM mode for normal interactivity in napari

        selection_mode(SelectionMode): MERGING or EXCLUDING for mode that we are finishing
        """
        if selection_mode == selection_mode.EXCLUDING:
            current_points_layer: Shapes = (
                self._curation_model.get_excluding_mask_shape_layers()[-1]
            )
            current_points_layer.mode = "PAN_ZOOM"  # default mode which allows for normal interactivity with napari canvas
        elif selection_mode == selection_mode.MERGING:
            current_points_layer = (
                self._curation_model.get_merging_mask_shape_layers()[-1]
            )
            current_points_layer.mode = "PAN_ZOOM"

    def save_excluding_mask(self) -> None:
        """
        Save the current excluding mask to disk and update napari
        """
        continue_save: bool = True
        # Checking to see if user has experiment selected
        if not self._curation_model.is_user_experiment_selected():
            # User does not have experiment selected
            continue_save = False
            # show information to user that experiment not selected
            show_info("Please select an experiment to save masks.")

        # Checking to see if there is already a merging mask saved.
        if (
            self._curation_model.get_current_excluding_mask_path_and_reset_mask()
        ):
            # There is already a merging mask saved. Ask if user wants to overwrite
            overwrite_excluding_mask_dialog = DialogBox(
                "There is already a excluding mask saved. Would you like to overwrite?"
            )
            overwrite_excluding_mask_dialog.exec()
            continue_save = (
                overwrite_excluding_mask_dialog.selection
            )  # True if overwrite selected, false if not.
        if continue_save:
            mask_to_save = (
                self._curation_model.get_excluding_mask_shape_layers()[-1].data
            )

            folder_path: Path = (
                self._curation_model.get_save_masks_path() / "excluding_masks"
            )
            # create excluding mask folder if one doesnt exist
            folder_path.mkdir(parents=True, exist_ok=True)
            # save mask and keep record of path
            save_path_mask_file: Path = (
                folder_path
                / f"excluding_mask_{self._curation_model.get_image_loader().get_raw_image_data().path.stem}.npy"
            )
            np.save(save_path_mask_file, np.asarray(mask_to_save))
            # if current mask path is set, we know that we've saved an excluding mask for the curationrecord.
            self._curation_model.set_current_excluding_mask_path(
                save_path_mask_file
            )
            self.clear_excluding_mask_layers_all()
            new_merging_shapes_layer: Shapes = self._viewer.viewer.add_shapes(
                mask_to_save,
                shape_type="polygon",
                face_color="coral",
                name="Saved Excluding Mask",
            )
            self._curation_model.append_excluding_mask_shape_layer(
                new_merging_shapes_layer
            )
            self._curation_model.dispatch(
                Event.ACTION_CURATION_SAVE_EXCLUDING_MASK
            )

    def save_merging_mask(self, base_image: str) -> bool:
        """
        Save the current merging mask to disk and update napari
        """
        continue_save: bool = True
        # Checking to see if user has experiment selected
        if not self._curation_model.is_user_experiment_selected():
            # User does not have experiment selected
            continue_save = False
            # show information to user that experiment not selected
            show_info("Please select an experiment to save masks.")

        # Checking to see if there is already a merging mask saved.
        if self._curation_model.get_current_merging_mask_path():
            # There is already a merging mask saved. Ask if user wants to overwrite
            overwrite_merging_mask_dialog = DialogBox(
                "There is already a merging mask saved. Would you like to overwrite?"
            )
            overwrite_merging_mask_dialog.exec()
            continue_save = (
                overwrite_merging_mask_dialog.selection
            )  # True if overwrite selected, false if not.

        if continue_save:
            # get all merging masks, can be in the same layer or in different layers
            mask_to_save = (
                self._curation_model.get_merging_mask_shape_layers()[-1].data
            )

            folder_path: Path = (
                self._curation_model.get_save_masks_path() / "merging_masks"
            )
            # create excluding mask folder if one doesnt exist
            folder_path.mkdir(parents=True, exist_ok=True)
            # save mask with same name as original raw file and keep record of path
            save_path_mask_file: Path = (
                folder_path
                / f"merging_mask_{self._curation_model.get_image_loader().get_raw_image_data().path.stem}.npy"
            )
            np.save(
                save_path_mask_file, np.asarray(mask_to_save, dtype=object)
            )
            # if current mask path is set, we know that we've saved an excluding mask for the curationrecord.
            self._curation_model.set_current_merging_mask_path(
                save_path_mask_file
            )
            self._curation_model.set_merging_mask_base_layer(base_image)
            self._curation_model.dispatch(
                Event.ACTION_CURATION_SAVED_MERGING_MASK
            )
            self.clear_merging_mask_layers_all()
            new_merging_shapes_layer: Shapes = self._viewer.viewer.add_shapes(
                mask_to_save,
                shape_type="polygon",
                face_color="royalblue",
                name="Saved Merging Mask",
            )
            self._curation_model.append_merging_mask_shape_layer(
                new_merging_shapes_layer
            )
        return continue_save

    def clear_merging_mask_layers_all(self) -> None:
        """
        Clear all merging mask layers in napari for one image
        """
        self._viewer.clear_mask_layers(
            self._curation_model.get_merging_mask_shape_layers()
        )
        self._curation_model.set_merging_mask_shape_layers([])

    def clear_excluding_mask_layers_all(self) -> None:
        """
        Clear all excluding mask layers in napari for one image
        """
        self._viewer.clear_mask_layers(
            self._curation_model.get_excluding_mask_shape_layers()
        )
        self._curation_model.set_excluding_mask_shape_layers([])

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

    def _csv_write_finished(self, saved_path) -> None:
        show_info(f"CSV Written to: {saved_path}")