import aicsimageio.exceptions

from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.image_data_extractor import ImageData
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.core.image_data_extractor import IImageDataExtractor, AICSImageDataExtractor
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.main.viewer import Viewer
from allencell_ml_segmenter.utils.file_utils import FileUtils

from pathlib import Path
from typing import List, Union, Optional, Callable, Tuple
import csv
import numpy as np
from aicsimageio import AICSImage
from enum import Enum
from napari.utils.notifications import show_info
from napari.layers.shapes.shapes import Shapes
from napari.qt.threading import thread_worker, FunctionWorker


MERGING_MASK_LAYER_NAME: str = "Merging Mask"
EXCLUDING_MASK_LAYER_NAME: str = "Excluding Mask"


class CurationService(Subscriber):
    """ """

    def __init__(
        self,
        curation_model: CurationModel,
        img_data_extractor: IImageDataExtractor = AICSImageDataExtractor.global_instance(),
    ) -> None:
        super().__init__()
        self._curation_model: CurationModel = curation_model
        self._img_data_extractor = img_data_extractor

        self._curation_model.raw_directory_set.connect(self._on_raw_dir_set)
        self._curation_model.seg1_directory_set.connect(self._on_seg1_dir_set)
        self._curation_model.seg2_directory_set.connect(self._on_seg2_dir_set)

    @thread_worker
    def _get_dir_data(self, dir: Path) -> Tuple[List[Path], int]:
        files: List[Path] = FileUtils.get_all_files_in_dir_ignore_hidden(dir)
        img_data: ImageData = self._img_data_extractor.extract_image_data(files[0], np_data=False)
        return files, img_data.channels

    def _on_raw_dir_data_extracted(self, dir_data: Tuple[List[Path], int]) -> None:
        self._curation_model.set_raw_directory_paths(dir_data[0])
        self._curation_model.set_raw_image_channel_count(dir_data[1])

    def _on_raw_dir_set(self) -> None:
        raw_dir: Path = self._curation_model.get_raw_directory()
        dir_extractor: FunctionWorker = self._get_dir_data(raw_dir)
        dir_extractor.returned.connect(self._on_raw_dir_data_extracted)
        dir_extractor.start()
    
    def _on_seg1_dir_data_extracted(self, dir_data: Tuple[List[Path], int]) -> None:
        self._curation_model.set_seg1_directory_paths(dir_data[0])
        self._curation_model.set_seg1_image_channel_count(dir_data[1])

    def _on_seg1_dir_set(self) -> None:
        seg1_dir: Path = self._curation_model.get_seg1_directory()
        dir_extractor: FunctionWorker = self._get_dir_data(seg1_dir)
        dir_extractor.returned.connect(self._on_seg1_dir_data_extracted)
        dir_extractor.start()
    
    def _on_seg2_dir_data_extracted(self, dir_data: Tuple[List[Path], int]) -> None:
        self._curation_model.set_seg2_directory_paths(dir_data[0])
        self._curation_model.set_seg2_image_channel_count(dir_data[1])

    def _on_seg2_dir_set(self) -> None:
        seg2_dir: Path = self._curation_model.get_seg2_directory()
        dir_extractor: FunctionWorker = self._get_dir_data(seg2_dir)
        dir_extractor.returned.connect(self._on_seg2_dir_data_extracted)
        dir_extractor.start()

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

