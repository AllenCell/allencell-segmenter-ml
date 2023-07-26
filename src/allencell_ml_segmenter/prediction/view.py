from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel

from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.file_input_widget import (
    PredictionFileInput,
)
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.prediction.service import ModelFileService
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.prediction.model_input_widget import (
    ModelInputWidget,
)
from qtpy.QtWidgets import QVBoxLayout, QSizePolicy, QPushButton


class PredictionView(View):
    """
    Holds the image and model input widgets for prediction.
    """

    def __init__(self, main_model: MainModel):
        super().__init__()

        self._main_model = main_model
        self._prediction_model: PredictionModel = PredictionModel()

        self._service: ModelFileService = ModelFileService(
            self._prediction_model
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.setLayout(QVBoxLayout())

        self._title: QLabel = QLabel("SEGMENTATION PREDICTION", self)
        self._title.setObjectName("title")
        self.layout().addWidget(self._title, alignment=Qt.AlignCenter)

        self._file_input_widget: PredictionFileInput = PredictionFileInput(
            self._prediction_model
        )
        self._file_input_widget.setObjectName("fileInput")

        self._model_input_widget: ModelInputWidget = ModelInputWidget(
            self._prediction_model
        )
        self._model_input_widget.setObjectName("modelInput")

        # Dummy divs allow for easy alignment
        top_container, top_dummy = QVBoxLayout(), QFrame()
        bottom_container, bottom_dummy = QVBoxLayout(), QFrame()

        top_container.addWidget(self._file_input_widget)
        top_dummy.setLayout(top_container)
        self.layout().addWidget(top_dummy)

        bottom_container.addWidget(self._model_input_widget)
        bottom_dummy.setLayout(bottom_container)
        self.layout().addWidget(bottom_dummy)

        self._run_btn: QPushButton = QPushButton("Run")
        self._run_btn.setObjectName("run")
        self.layout().addWidget(self._run_btn)

        self.setStyleSheet(Style.get_stylesheet("prediction_view.qss"))

        self._main_model.subscribe(
            Event.PROCESS_TRAINING_COMPLETE,
            self,
            lambda e: self._main_model.set_current_view(self),
        )
