from PyQt5.QtWidgets import QPushButton
from qtpy.QtWidgets import QDialog, QWidget, QComboBox, QVBoxLayout
from typing import Optional

from allencell_ml_segmenter.core.info_dialog_box import InfoDialogBox
from allencell_ml_segmenter.main.experiments_model import IExperimentsModel
from allencell_ml_segmenter.utils.s3_model_downloader import S3ModelDownloader


class ModelDownloadDialog(QDialog):
    def __init__(self, parent: QWidget, experiments_model: IExperimentsModel):
        super().__init__(parent)
        self._experiments_model = experiments_model
        self._model_downloader = S3ModelDownloader(staging=True)
        self.setLayout(QVBoxLayout())

        self._model_select_dropdown: QComboBox = QComboBox()
        self._model_select_dropdown.setCurrentIndex(-1)
        self._model_select_dropdown.addItems(self._model_downloader.get_available_models())
        self.layout().addWidget(self._model_select_dropdown)

        self._download_button: QPushButton = QPushButton("Download")
        self._download_button.clicked.connect(self._download_button_handler)
        self.layout().addWidget(self._download_button)

    def _download_button_handler(self) -> None:
        selected_model_name: str = str(self._model_select_dropdown.currentText())
        self._model_downloader.download_model_to(selected_model_name,
                                                 self._experiments_model.get_user_experiments_path())
        download_complete_message = InfoDialogBox(f"Downloadaded {selected_model_name} to {self._experiments_model.get_user_experiments_path() / selected_model_name}")
        download_complete_message.exec()






