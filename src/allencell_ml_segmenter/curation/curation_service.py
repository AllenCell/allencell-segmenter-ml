import csv
import shutil

import numpy as np

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from pathlib import Path
from typing import List, Tuple

from aicsimageio import AICSImage
import napari
from napari.utils.notifications import show_info
from napari.layers.shapes.shapes import Shapes
from enum import Enum
from skimage import draw

from allencell_ml_segmenter.main.viewer import Viewer
from napari.utils.notifications import show_info


class SelectionMode(Enum):
    EXCLUDING = "excluding"
    MERGING = "merging"


class CurationService(Subscriber):
    """ """

    def __init__(self, curation_model: CurationModel, viewer: Viewer) -> None:
        super().__init__()
        self._curation_model = curation_model
        self._viewer = viewer

        self._curation_model.subscribe(
            Event.ACTION_CURATION_SAVE_EXCLUDING_MASK,
            self,
            self.save_excluding_mask,
        )

    def get_raw_images_list(self):
        """
        Return all raw images in the raw images path as a list of Paths
        """
        raw_path: Path = self._curation_model.get_raw_directory()
        if raw_path is None:
            raise ValueError(
                "Raw directory not set. Please set raw directory."
            )
        return self._get_files_list_from_path(raw_path)

    def get_seg1_images_list(self):
        """
        Return all raw images in the raw images path as a list of Paths
        """
        seg1_path: Path = self._curation_model.get_seg1_directory()
        if seg1_path is None:
            raise ValueError(
                "Raw directory not set. Please set raw directory."
            )
        return self._get_files_list_from_path(seg1_path)

    def get_seg2_images_list(self):
        """
        Return all raw images in the raw images path as a list of Paths
        """
        seg2_path: Path = self._curation_model.get_seg2_directory()
        if seg2_path is None:
            raise ValueError(
                "Raw directory not set. Please set raw directory."
            )
        return self._get_files_list_from_path(seg2_path)

    def get_image_data_from_path(self, path: Path) -> AICSImage:
        return AICSImage(str(path))

    def write_curation_record(
        self, curation_record: List[CurationRecord], path: Path
    ) -> None:
        """
        Save the curation record as a csv at the specified path
        """
        parent_path: Path = path.parents[0]
        if not parent_path.is_dir():
            parent_path.mkdir(parents=True)

        with open(path, "w") as f:
            # need file header
            writer: csv.writer = csv.writer(f, delimiter=",")
            writer.writerow(["", "raw", "seg", "mask"])
            for idx, record in enumerate(curation_record):
                if record.to_use:
                    writer.writerow(
                        [
                            str(idx),
                            str(record.raw_file),
                            str(record.seg1),
                            str(record.excluding_mask),
                        ]
                    )
                f.flush()

        # TODO: WRITE ACTUAL VALIDATION AND TEST SETS
        # shutil.copy(path, parent_path / "valid.csv")
        # shutil.copy(path, parent_path / "test.csv")

    def remove_all_images_from_viewer_layers(self):
        self._viewer.clear_layers()

    def add_image_to_viewer(self, image_data: np.ndarray, title: str = ""):
        self._viewer.add_image(image_data, name=title)

    def add_image_to_viewer_from_path(self, path: Path, title: str = ""):
        image = self.get_image_data_from_path(path)
        self._curation_model.curation_image_dims = (
            image.dims.X,
            image.dims.Y,
            image.dims.Z,
        )
        self.add_image_to_viewer(image.data, title)

    def enable_shape_selection_viewer(self, mode: SelectionMode):
        _ = show_info("Draw excluding area")
        if mode == SelectionMode.EXCLUDING:
            # append points layer to excluding mask shapes list
            points_layer: Shapes = self._viewer.add_shapes(
                name="Excluding Mask"
            )
            points_layer.mode = "add_polygon"
            # todo combine maybe
            self._curation_model.excluding_mask_shape_layers.append(
                points_layer
            )
            self._curation_model.dispatch(Event.ACTION_CURATION_DRAW_EXCLUDING)
        elif mode == SelectionMode.MERGING:
            points_layer: Shapes = self._viewer.add_shapes(name="Merging Mask")
            points_layer.mode = "add_polygon"
            self._curation_model.merging_mask_shape_layers.append(points_layer)

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
        first_image: Path = path.iterdir().__next__()
        print(first_image.resolve())
        print(str(first_image.resolve()))
        img: AICSImage = AICSImage(str(first_image.resolve()))
        # return num channel
        return img.dims.C

    def select_directory_raw(self, path: Path):
        self._curation_model.set_raw_directory(path)
        self._curation_model.set_total_num_channels_raw(
            self.get_total_num_channels_of_images_in_path(path)
        )
        self._curation_model.dispatch(Event.ACTION_CURATION_RAW_SELECTED)

    def select_directory_seg1(self, path: Path):
        self._curation_model.set_seg1_directory(path)
        self._curation_model.set_total_num_channels_seg1(
            self.get_total_num_channels_of_images_in_path(path)
        )
        self._curation_model.dispatch(Event.ACTION_CURATION_SEG1_SELECTED)

    def select_directory_seg2(self, path: Path):
        self._curation_model.set_seg2_directory(path)
        self._curation_model.set_total_num_channels_seg2(
            self.get_total_num_channels_of_images_in_path(path)
        )
        self._curation_model.dispatch(Event.ACTION_CURATION_SEG2_SELECTED)

    def finished_shape_selection(self, selection_mode: SelectionMode) -> None:
        if selection_mode == selection_mode.EXCLUDING:

            current_points_layer = (
                self._curation_model.excluding_mask_shape_layers[-1]
            )
            current_points_layer.mode = "PAN_ZOOM"  # default mode which allows for normal interactivity with napari canvas
        elif selection_mode == selection_mode.MERGING:
            current_points_layer = (
                self._curation_model.merging_mask_shape_layers[-1]
            )
            current_points_layer.mode = "PAN_ZOOM"

    def convert_shape_layer_to_mask(
        self, image_shape: Tuple[int, int], shape_layer
    ) -> np.ndarray:
        return draw.polygon2mask(image_shape, shape_layer.data[0])

    def save_excluding_mask(self, event: Event = None) -> None:
        current_excluding_mask_layer = (
            self._curation_model.excluding_mask_shape_layers[-1]
        )
        # first two dims are x, y
        mask: np.ndarray = self.convert_shape_layer_to_mask(
            self._curation_model.curation_image_dims[:2],
            current_excluding_mask_layer,
        )

        mask = self.extend_mask_in_z(
            self._curation_model.curation_image_dims, mask
        )
        folder_path = (
            self._curation_model.get_save_masks_path() / "excluding_masks"
        )
        # create excluding mask folder if one doesnt exist
        folder_path.mkdir(parents=True, exist_ok=True)
        # save mask and keep record of path
        save_path_mask_file: Path = (
            folder_path
            / f"excluding_mask_{self._curation_model.get_current_loaded_images()[0].stem}.npy"
        )
        np.save(save_path_mask_file, mask)
        # if current mask path is set, we know that we've saved an excluding mask for the curationrecord.
        self._curation_model.set_current_excluding_mask_path(save_path_mask_file)

    def extend_mask_in_z(
        self, shape: Tuple, mask_to_extend: np.ndarray
    ) -> np.ndarray:
        three_dim_mask = np.ndarray(shape, dtype=np.bool)
        for z in range(shape[2]):
            three_dim_mask[:, :, z] = mask_to_extend
        return three_dim_mask

    def save_merging_mask(self, base_image: str):
        if self._curation_model.get_user_experiment_selected():
            # get all merging masks, can be in the same layer or in different layers
            merging_masks = []
            for layer in self._curation_model.merging_mask_shape_layers:
                for shape in layer.data:
                    merging_masks.append(shape)

            folder_path = (
                    self._curation_model.get_save_masks_path() / "merging_masks"
            )
            # create excluding mask folder if one doesnt exist
            folder_path.mkdir(parents=True, exist_ok=True)
            # save mask and keep record of path
            save_path_mask_file: Path = (
                    folder_path
                    / f"merging_mask_{self._curation_model.get_current_loaded_images()[0].stem}.npy"
            )
            np.save(save_path_mask_file, np.asarray(merging_masks))
            # if current mask path is set, we know that we've saved an excluding mask for the curationrecord.
            self._curation_model.set_current_merging_mask_path(save_path_mask_file)
            show_info("Merging mask saved.")
        else:
            show_info("Please select an experiment to save masks.")
