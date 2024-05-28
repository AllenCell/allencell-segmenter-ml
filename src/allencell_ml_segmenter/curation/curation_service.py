from allencell_ml_segmenter.core.image_data_extractor import ImageData
from allencell_ml_segmenter.curation.curation_model import (
    CurationModel,
    CurationRecord,
)
from allencell_ml_segmenter.core.image_data_extractor import (
    IImageDataExtractor,
    AICSImageDataExtractor,
)
from allencell_ml_segmenter.core.task_executor import (
    ITaskExecutor,
    NapariThreadTaskExecutor,
)
from allencell_ml_segmenter.utils.file_utils import FileUtils
from allencell_ml_segmenter.utils.file_writer import IFileWriter, FileWriter

from pathlib import Path
from qtpy.QtCore import QObject
from typing import List, Tuple
from copy import deepcopy


# Important note: we do not want to access the model in any of the threads because model state may change
# while thread is executing. So, opt to copy/pass in all relevant model state to threads
class CurationService(QObject):

    def __init__(
        self,
        curation_model: CurationModel,
        img_data_extractor: IImageDataExtractor = AICSImageDataExtractor.global_instance(),
        task_executor: ITaskExecutor = NapariThreadTaskExecutor.global_instance(),
        file_writer: IFileWriter = FileWriter.global_instance(),
    ) -> None:
        super().__init__()
        self._curation_model: CurationModel = curation_model
        self._img_data_extractor: IImageDataExtractor = img_data_extractor
        self._task_executor: ITaskExecutor = task_executor
        self._file_utils: FileUtils = FileUtils(file_writer)

        self._curation_model.raw_directory_set.connect(self._on_raw_dir_set)
        self._curation_model.seg1_directory_set.connect(self._on_seg1_dir_set)
        self._curation_model.seg2_directory_set.connect(self._on_seg2_dir_set)
        self._curation_model.save_to_disk_requested.connect(
            self._on_save_to_disk
        )

    def _get_dir_data(self, dir: Path) -> Tuple[List[Path], int]:
        files: List[Path] = self._file_utils.get_all_files_in_dir_ignore_hidden(dir)
        img_data: ImageData = self._img_data_extractor.extract_image_data(
            files[0], np_data=False
        )
        return files, img_data.channels

    def _on_raw_dir_data_extracted(
        self, dir_data: Tuple[List[Path], int]
    ) -> None:
        self._curation_model.set_raw_directory_paths(dir_data[0])
        self._curation_model.set_raw_image_channel_count(dir_data[1])

    def _on_raw_dir_errored(self, e: Exception) -> None:
        self._curation_model.set_raw_image_channel_count(0)
        raise e

    def _on_raw_dir_set(self) -> None:
        # paths are immutable, so no need to explicitly copy
        raw_dir: Path = self._curation_model.get_raw_directory()
        self._task_executor.exec(
            lambda: self._get_dir_data(raw_dir),
            on_return=self._on_raw_dir_data_extracted,
            on_error=self._on_raw_dir_errored,
        )

    def _on_seg1_dir_data_extracted(
        self, dir_data: Tuple[List[Path], int]
    ) -> None:
        self._curation_model.set_seg1_directory_paths(dir_data[0])
        self._curation_model.set_seg1_image_channel_count(dir_data[1])

    def _on_seg1_dir_errored(self, e: Exception) -> None:
        self._curation_model.set_seg1_image_channel_count(0)
        raise e

    def _on_seg1_dir_set(self) -> None:
        seg1_dir: Path = self._curation_model.get_seg1_directory()
        self._task_executor.exec(
            lambda: self._get_dir_data(seg1_dir),
            on_return=self._on_seg1_dir_data_extracted,
            on_error=self._on_seg1_dir_errored,
        )

    def _on_seg2_dir_data_extracted(
        self, dir_data: Tuple[List[Path], int]
    ) -> None:
        self._curation_model.set_seg2_directory_paths(dir_data[0])
        self._curation_model.set_seg2_image_channel_count(dir_data[1])

    def _on_seg2_dir_errored(self, e: Exception) -> None:
        self._curation_model.set_seg2_image_channel_count(0)
        raise e

    def _on_seg2_dir_set(self) -> None:
        seg2_dir: Path = self._curation_model.get_seg2_directory()
        self._task_executor.exec(
            lambda: self._get_dir_data(seg2_dir),
            on_return=self._on_seg2_dir_data_extracted,
            on_error=self._on_seg2_dir_errored,
        )

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
        )
