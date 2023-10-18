import csv
import shutil

import numpy as np

from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from pathlib import Path
from typing import List

from aicsimageio import AICSImage
import napari
from napari.utils.notifications import show_info
from napari.layers.shapes.shapes import Shapes

class CurationService():
    """
    """

    def __init__(self, curation_model: CurationModel, viewer: napari.Viewer) -> None:
        self._curation_model = curation_model
        self._viewer = viewer

    def get_raw_images_list(self):
        """
        Return all raw images in the raw images path as a list of Paths
        """
        raw_path: Path = self._curation_model.get_raw_directory()
        if raw_path is None:
            raise ValueError("Raw directory not set. Please set raw directory.")
        return self._get_files_list_from_path(raw_path)

    def get_seg1_images_list(self):
        """
        Return all raw images in the raw images path as a list of Paths
        """
        seg1_path: Path = self._curation_model.get_seg1_directory()
        if seg1_path is None:
            raise ValueError("Raw directory not set. Please set raw directory.")
        return self._get_files_list_from_path(seg1_path)

    def get_seg2_images_list(self):
        """
        Return all raw images in the raw images path as a list of Paths
        """
        seg2_path: Path = self._curation_model.get_seg1_directory()
        if seg2_path is None:
            raise ValueError("Raw directory not set. Please set raw directory.")
        return self._get_files_list_from_path(seg2_path)

    def get_image_data_from_path(self, path: Path) -> np.ndarray:
        return AICSImage(str(path)).data

    def write_curation_record(self, curation_record: List[CurationRecord], path: Path) -> None:
        """
        Save the curation record as a csv at the specified path
        """
        parent_path: Path = path.parents[0]
        if not parent_path.is_dir():
            parent_path.mkdir(parents=True)

        with open(path, "w") as f:
            # need file header
            writer: csv.writer = csv.writer(f, delimiter=",")
            writer.writerow(["", "raw", "seg"])
            for idx, record in enumerate(curation_record):
                if record.to_use:
                    writer.writerow(
                        [str(idx), str(record.raw_file), str(record.seg1)]
                    )
                f.flush()

        # TODO: WRITE ACTUAL VALIDATION AND TEST SETS
        shutil.copy(path, parent_path / "valid.csv")
        shutil.copy(path, parent_path / "test.csv")

    def remove_all_images_from_viewer_layers(self):
        self._viewer.layers.clear()

    def add_image_to_viewer(self, image_data: np.ndarray, title: str = ""):
        self._viewer.add_image(image_data, name=title)

    def enable_shape_selection_viewer(self):
        _ = show_info("Draw excluding area")
        points_layer: Shapes = self._viewer.add_shapes(None)
        points_layer.mode = "add_polygon"


    def _get_files_list_from_path(self, path: Path) -> List[Path]:
        """
        Return all files in the path as a list of Paths
        """
        return [file for file in path.iterdir() if not file.name.endswith(".DS_Store")]


