from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.main.viewer import Viewer

from pathlib import Path
from typing import List, Union
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

    def __init__(self, curation_model: CurationModel, viewer: IViewer) -> None:
        super().__init__()
        self._curation_model: CurationModel = curation_model
        self._viewer: Viewer = viewer

    def build_raw_images_list(self) -> List[Path]:
        """
        Return all raw images in the raw images path as a list of Paths
        """
        raw_path: Path = self._curation_model.get_raw_directory()
        if raw_path is None:
            raise ValueError(
                "Raw directory not set. Please set raw directory."
            )
        return self._get_files_list_from_path(raw_path)

    def build_seg1_images_list(self) -> List[Path]:
        """
        Return all raw images in the raw images path as a list of Paths
        """
        seg1_path: Path = self._curation_model.get_seg1_directory()
        if seg1_path is None:
            raise ValueError(
                "Raw directory not set. Please set raw directory."
            )
        return self._get_files_list_from_path(seg1_path)

    def build_seg2_images_list(self) -> List[Path]:
        """
        Return all raw images in the raw images path as a list of Paths
        """
        seg2_path: Path = self._curation_model.get_seg2_directory()
        if seg2_path is None:
            raise ValueError(
                "Raw directory not set. Please set raw directory."
            )
        return self._get_files_list_from_path(seg2_path)

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
                            str(record.seg2),
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
        # if all images are removed, need to reset excluding mask and merging mask state as well
        self._curation_model.set_excluding_mask_shape_layers([])
        self._curation_model.set_merging_mask_shape_layers([])

    def add_image_to_viewer(
        self, image_data: np.ndarray, title: str = ""
    ) -> None:
        """
        Add an image to the napari viewer as its own layer

        image_data(np.ndarray): image to display in napari
        title(str): title of layer that will be displayed in napari
        """
        self._viewer.add_image(image_data, name=title)

    def add_image_to_viewer_from_path(
        self, path: Path, title: str = ""
    ) -> None:
        """
        Add an image to the napari viewer from a path

        path(Path): path to image to display in napari
        title(str): title of layer that will be displayed in napari
        """
        image: AICSImage = self.open_image_from_path(path)
        self._curation_model.set_curation_image_dims(
            (
                image.dims.X,
                image.dims.Y,
                image.dims.Z,
            )
        )
        self.add_image_to_viewer(image.data, title)

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

    def _get_files_list_from_path(self, path: Path) -> List[Path]:
        """
        Return all files in the path as a list of Paths
        """
        return [
            file
            for file in sorted(path.iterdir())
            if not file.name.endswith(".DS_Store")
        ]

    def get_total_num_channels_of_images_in_path(self, path: Path) -> int:
        """
        Determine total number of channels for image in a set folder
        """
        # we expect user to have the same number of channels for all images in their folders
        # and that only images are stored in those folders
        first_image: Path = self._get_files_list_from_path(path)[0]
        print(first_image.resolve())
        print(str(first_image.resolve()))
        img: AICSImage = AICSImage(str(first_image.resolve()))
        # return num channel
        return img.dims.C

    def select_directory_raw(self, path: Path):
        """
        Select a raw directory

        path(Path): path to raw directory
        """
        self._curation_model.set_raw_directory(path)
        self._curation_model.set_raw_image_channel_count(
            self.get_total_num_channels_of_images_in_path(path)
        )
        self._curation_model.dispatch(Event.ACTION_CURATION_RAW_SELECTED)

    def select_directory_seg1(self, path: Path):
        """
        Select a seg1 directory

        path(Path): path to seg1 directory
        """
        self._curation_model.set_seg1_directory(path)
        self._curation_model.set_seg1_image_channel_count(
            self.get_total_num_channels_of_images_in_path(path)
        )
        self._curation_model.dispatch(Event.ACTION_CURATION_SEG1_SELECTED)

    def select_directory_seg2(self, path: Path):
        """
        Select a seg2 directory

        path(Path): path to seg2 directory
        """
        self._curation_model.set_seg2_directory(path)
        self._curation_model.set_seg2_image_channel_count(
            self.get_total_num_channels_of_images_in_path(path)
        )
        self._curation_model.dispatch(Event.ACTION_CURATION_SEG2_SELECTED)

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
                / f"excluding_mask_{self._curation_model.get_current_loaded_images()[0].stem}.npy"
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
                / f"merging_mask_{self._curation_model.get_current_loaded_images()[0].stem}.npy"
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
        self._curation_model.append_curation_record(
            CurationRecord(
                self._curation_model.get_current_raw_image(),
                self._curation_model.get_current_seg1_image(),
                self._curation_model.get_current_seg2_image(),
                excluding_mask_path,
                merging_mask_path,
                base_image_name,
                use_image,
            )
        )

        # increment curation index
        self._curation_model.set_curation_index(
            self._curation_model.get_curation_index() + 1
        )

    def next_image(self, use_image: bool) -> None:
        """
        Load the next image in the curation image stack. Updates curation record and progress bar accordingly.
        """
        _ = show_info("Loading the next image...")
        self.update_curation_record(use_image)

        # load next image
        if self._curation_model.image_available():
            self.remove_all_images_from_viewer_layers()
            self._curation_model.set_current_merging_mask_path(None)
            self._curation_model.set_current_excluding_mask_path(None)

            raw_to_view: Path = self._curation_model.get_current_raw_image()
            # Add image with [raw] prepended to layer name
            self.add_image_to_viewer_from_path(
                raw_to_view, title=f"[raw] {raw_to_view.name}"
            )

            seg1_to_view: Path = self._curation_model.get_current_seg1_image()
            # Add image with [seg] prepended to layer name
            self.add_image_to_viewer_from_path(
                seg1_to_view, title=f"[seg 1] {seg1_to_view.name}"
            )
            if self._curation_model.get_seg2_images() is not None:
                seg2_to_view: Path = (
                    self._curation_model.get_current_seg2_image()
                )
                # Add image with [seg] prepended to layer name
                self.add_image_to_viewer_from_path(
                    seg2_to_view, title=f"[seg 2] {seg2_to_view.name}"
                )
            else:
                seg2_to_view = None

            self._curation_model.set_current_loaded_images(
                (raw_to_view, seg1_to_view, seg2_to_view)
            )
            self._curation_model.dispatch(Event.PROCESS_CURATION_NEXT_IMAGE)
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
        self._curation_model.set_raw_images(self.build_seg2_images_list())
        # build list of seg1 images, ignore .DS_Store files
        self._curation_model.set_seg1_images(self.build_seg1_images_list())

        # If seg 2 is selected, build list of seg2 images
        if self._curation_model.get_seg2_directory() is not None:
            self._curation_model.set_seg2_images(self.build_seg2_images_list())
        else:
            self._curation_model.set_seg2_images(None)

        # reset
        self.remove_all_images_from_viewer_layers()

        first_raw: Path = self._curation_model.get_raw_images()[0]
        self.add_image_to_viewer_from_path(
            first_raw, title=f"[raw] {first_raw.name}"
        )

        first_seg1: Path = self._curation_model.get_seg1_images()[0]
        self.add_image_to_viewer_from_path(
            first_seg1, title=f"[Seg 1] {first_seg1.name}"
        )
        if self._curation_model.get_seg2_directory() is not None:
            first_seg2: Path = self._curation_model.get_seg2_images()[0]
            self.add_image_to_viewer_from_path(
                first_seg2, title=f"[Seg 2] {first_seg2.name}"
            )
        else:
            first_seg2 = None
        self._curation_model.set_current_loaded_images(
            (first_raw, first_seg1, first_seg2)
        )
