from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.image_data_extractor import ImageData
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.core.image_data_extractor import IImageDataExtractor, AICSImageDataExtractor
from allencell_ml_segmenter.utils.file_utils import FileUtils

from pathlib import Path
from typing import List, Tuple
from napari.qt.threading import thread_worker, FunctionWorker


MERGING_MASK_LAYER_NAME: str = "Merging Mask"
EXCLUDING_MASK_LAYER_NAME: str = "Excluding Mask"

# Important note: we do not want to access the model in any of the threads because model state may change
# while thread is executing. So, opt to copy/pass in all relevant model state to threads
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
        self._curation_model.save_to_disk_requested.connect(self._on_save_to_disk)

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

    @thread_worker
    def _save_curation_record(self, curation_record: CurationRecord, csv_path: Path, mask_dir_path: Path) -> None:
        FileUtils.write_curation_record(curation_record, csv_path, mask_dir_path)

    def _on_save_to_disk(self) -> None:
        curation_record: List[CurationRecord] = self._curation_model.get_curation_record()
        record_saver: FunctionWorker = self._save_curation_record(curation_record, self._curation_model.get_csv_path(), self._curation_model.get_save_masks_path())
        record_saver.finished.connect(lambda: self._curation_model.set_curation_record_saved_to_disk(True))
        record_saver.start()


