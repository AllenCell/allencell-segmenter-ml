from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.curation.curation_model import (
    CurationModel,
    CurationRecord,
    ImageType,
    CurationView,
)
from allencell_ml_segmenter.core.image_data_extractor import (
    IImageDataExtractor,
    AICSImageDataExtractor,
    ImageData,
)
from allencell_ml_segmenter.core.task_executor import (
    ITaskExecutor,
    NapariThreadTaskExecutor,
)
from allencell_ml_segmenter.utils.file_utils import FileUtils
from allencell_ml_segmenter.utils.file_writer import IFileWriter, FileWriter
from allencell_ml_segmenter.main.main_model import MIN_DATASET_SIZE

from pathlib import Path
from qtpy.QtCore import QObject
from typing import List, Optional, Callable
from copy import deepcopy
from collections import namedtuple


DirectoryData = namedtuple("DirectoryData", ["fpaths", "channels"])


# Important note: we do not want to access the model in any of the threads because model state may change
# while thread is executing. So, opt to copy/pass in all relevant model state to threads
class CurationService(QObject):

    def __init__(
        self,
        curation_model: CurationModel,
        experiments_model: IExperimentsModel,
        img_data_extractor: IImageDataExtractor = AICSImageDataExtractor.global_instance(),
        task_executor: ITaskExecutor = NapariThreadTaskExecutor.global_instance(),
        file_writer: IFileWriter = FileWriter.global_instance(),
    ) -> None:
        super().__init__()
        self._curation_model: CurationModel = curation_model
        self._experiments_model: IExperimentsModel = experiments_model
        self._img_data_extractor: IImageDataExtractor = img_data_extractor
        self._task_executor: ITaskExecutor = task_executor
        self._file_writer: IFileWriter = file_writer
        self._file_utils: FileUtils = FileUtils(file_writer)

        self._curation_model.image_directory_set.connect(
            self._on_image_dir_set
        )
        self._curation_model.cursor_moved.connect(self._on_cursor_moved)
        self._curation_model.save_to_disk_requested.connect(
            self._on_save_to_disk
        )

    def _get_dir_data(self, dir: Path) -> DirectoryData:
        files: List[Path] = (
            self._file_utils.get_all_files_in_dir_ignore_hidden(dir)
        )

        if len(files) < MIN_DATASET_SIZE:
            raise RuntimeError(
                f"Curation requires at least {MIN_DATASET_SIZE} images and their segmentations"
            )

        img_data: ImageData = self._img_data_extractor.extract_image_data(
            files[0], np_data=False
        )
        return DirectoryData(files, img_data.channels)

    def _on_dir_data_extracted(
        self, img_type: ImageType, dir_data: DirectoryData
    ) -> None:
        self._curation_model.set_image_directory_paths(
            img_type, dir_data.fpaths
        )
        self._curation_model.set_channel_count(img_type, dir_data.channels)

    def _on_dir_data_errored(self, img_type: ImageType, e: Exception) -> None:
        self._curation_model.set_channel_count(img_type, 0)
        raise e

    def _on_image_dir_set(self, img_type: ImageType) -> None:
        # paths are immutable, so no need to explicitly copy
        dir: Path = self._curation_model.get_image_directory(img_type)
        self._task_executor.exec(
            lambda: self._get_dir_data(dir),
            on_return=lambda dir_data: self._on_dir_data_extracted(
                img_type, dir_data
            ),
            on_error=lambda e: self._on_dir_data_errored(img_type, e),
        )

    def _extract_images(
        self,
        img_idx: int,
        setter_fn: Callable[[ImageType, ImageData], None],
        err_str: str,
    ) -> None:
        """
        Uses TaskExecutor to extract data for images at :param img_idx:, saves extracted data using
        :param setter_fn: to set model state. Provides :param err_str: ('curr' or 'next') to the error
        handler for additional debugging info.
        """
        raw_paths: List[Path] = self._curation_model.get_image_directory_paths(
            ImageType.RAW
        )
        seg1_paths: List[Path] = (
            self._curation_model.get_image_directory_paths(ImageType.SEG1)
        )
        seg2_paths: Optional[List[Path]] = (
            self._curation_model.get_image_directory_paths(ImageType.SEG2)
        )

        raw_channel: int = self._curation_model.get_selected_channel(
            ImageType.RAW
        )
        seg1_channel: int = self._curation_model.get_selected_channel(
            ImageType.SEG1
        )
        seg2_channel: Optional[int] = (
            self._curation_model.get_selected_channel(ImageType.SEG2)
        )

        self._task_executor.exec(
            lambda: self._img_data_extractor.extract_image_data(
                raw_paths[img_idx],
                channel=raw_channel,
            ),
            on_return=lambda img_data: setter_fn(ImageType.RAW, img_data),
            on_error=lambda e: self._on_cursor_moved_error(
                ImageType.RAW, err_str, e
            ),
        )
        self._task_executor.exec(
            lambda: self._img_data_extractor.extract_image_data(
                seg1_paths[img_idx], channel=seg1_channel, seg=1
            ),
            on_return=lambda img_data: setter_fn(ImageType.SEG1, img_data),
            on_error=lambda e: self._on_cursor_moved_error(
                ImageType.SEG1, err_str, e
            ),
        )
        if seg2_paths is not None:
            self._task_executor.exec(
                lambda: self._img_data_extractor.extract_image_data(
                    seg2_paths[img_idx], channel=seg2_channel, seg=2
                ),
                on_return=lambda img_data: setter_fn(ImageType.SEG2, img_data),
                on_error=lambda e: self._on_cursor_moved_error(
                    ImageType.SEG2, err_str, e
                ),
            )

    def _on_cursor_moved_error(
        self, img_type: ImageType, curr_or_next: str, e: Exception
    ) -> None:
        raise RuntimeError(
            f"There was a problem loading {img_type} for the {curr_or_next} images"
        )

    def _on_cursor_moved(self) -> None:
        cursor: int = self._curation_model.get_curr_image_index()

        if self._curation_model.is_waiting_for_curr_images():
            # start extraction tasks for curr images (paths at cursor)
            self._extract_images(
                cursor, self._curation_model.set_curr_image_data, "curr"
            )

        if self._curation_model.is_waiting_for_next_images():
            # start extraction tasks for next images (paths at cursor + 1)
            self._extract_images(
                cursor + 1, self._curation_model.set_next_image_data, "next"
            )

    def _on_save_to_disk_error(self, err: Exception) -> None:
        self._curation_model.set_curation_record_saved_to_disk(False)
        raise err

    def _on_save_to_disk(self) -> None:
        record: List[CurationRecord] = deepcopy(
            self._curation_model.get_curation_record()
        )
        csv_path: Path = self._curation_model.get_csv_path()
        save_path: Path = self._curation_model.get_save_masks_path()
        self._task_executor.exec(
            lambda: self._file_utils.write_curation_record(
                record, csv_path, save_path
            ),
            on_finish=lambda: self._curation_model.set_curation_record_saved_to_disk(
                True
            ),
            on_error=self._on_save_to_disk_error,
        )
