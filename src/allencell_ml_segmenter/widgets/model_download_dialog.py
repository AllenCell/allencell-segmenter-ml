from typing import Optional

from qtpy.QtWidgets import (
    QDialog,
    QWidget,
    QComboBox,
    QVBoxLayout,
    QPushButton,
)

from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter.core.info_dialog_box import InfoDialogBox
from allencell_ml_segmenter.main.experiments_model import IExperimentsModel
from allencell_ml_segmenter.utils.s3.s3_model_bucket import (
    S3ModelBucket,
)
from allencell_ml_segmenter.utils.s3.s3_bucket_constants import PROD_BUCKET


class ModelDownloadDialog(QDialog):
    def __init__(
        self, parent: Optional[QWidget], experiments_model: IExperimentsModel
    ):
        super().__init__(parent)
        self._experiments_model = experiments_model
        self._available_models = S3ModelBucket(
            PROD_BUCKET, self._experiments_model.get_user_experiments_path()
        ).get_available_models()
        self.setLayout(QVBoxLayout())

        self._model_select_dropdown: QComboBox = QComboBox()
        self._model_select_dropdown.setCurrentIndex(-1)
        self._model_select_dropdown.addItems(self._available_models.keys())
        self.layout().addWidget(self._model_select_dropdown)

        self._download_button: QPushButton = QPushButton("Download")
        self._download_button.clicked.connect(self._download_button_handler)
        self.layout().addWidget(self._download_button)

    def _download_button_handler(self) -> None:
        selected_model_name: str = str(
            self._model_select_dropdown.currentText()
        )

        self._available_models[selected_model_name].download_model_and_unzip()
        download_complete_message = InfoDialogBox(
            f"Downloaded {selected_model_name} to {self._experiments_model.get_user_experiments_path() / selected_model_name}"
        )
        download_complete_message.exec()
        self._experiments_model.refresh_experiments()  # prevents repeat downloads of model before exiting download dialog
