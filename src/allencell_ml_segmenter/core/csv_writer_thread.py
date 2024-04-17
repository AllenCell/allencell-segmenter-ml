from enum import Enum
from pathlib import Path
from typing import List

from qtpy.QtCore import QObject, QThread, Signal

from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.utils.csv_utils import CSVUtils


class CSVWriterMode(Enum):
    curation = "curation"


class CSVWriterThread(QThread):
    """
    Thread to handle csv file writing.
    """
    write_finished: Signal(bool)
    error: Signal(Exception)

    def __init__(self, write_path: Path, write_mode: CSVWriterMode, curation_record: List[CurationRecord] = None):
        """
        :param write_path: Path of csv to be written
        :param write_mode: CSVWriterMode, curation for curated data csv.
        :param curation_record: CurationRecord to write to csv, None if not using CSVWriterMode.Curation
        """
        super().__init__()
        self._path: Path = write_path
        self._mode: CSVWriterMode = write_mode
        self._curation_record: List[CurationRecord] = curation_record

    def run(self):
        # call the corresponding csv writer method based on mode
        if self._mode == CSVWriterMode.curation:
            # If writing curation csv, check to see if a curation_record to write was provided
            if self._curation_record is None:
                self.error.emit(ValueError("CSVWriterThread needs to be initialized with CurationRecord if being used to write a curation csv."))
                return
            else:
                CSVUtils.write_curation_csv(self._curation_record, self._path)

        if not QThread.currentThread().isInterruptionRequested():
            self.write_finished(True)




