from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
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


class PredictionView(View, Subscriber):
    """
    Holds the image and model input widgets for prediction.
    """

    def __init__(self, main_model: MainModel):
        super().__init__()

        self._main_model: MainModel = main_model
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

        self.setStyleSheet("color: #F0F1F2")

        self._title: QLabel = QLabel("SEGMENTATION PREDICTION", self)
        self._title.setStyleSheet("font-weight: bold; margin: 10px 0px")
        self.layout().addWidget(self._title, alignment=Qt.AlignCenter)

        self._file_input_widget: PredictionFileInput = PredictionFileInput(
            self._prediction_model
        )

        self._model_input_widget: ModelInputWidget = ModelInputWidget(
            self._prediction_model
        )

        # Dummy divs allow for easy alignment
        top_container, top_dummy = QVBoxLayout(), QFrame()
        bottom_container, bottom_dummy = QVBoxLayout(), QFrame()

        top_container.addWidget(self._file_input_widget)
        top_dummy.setLayout(top_container)
        top_dummy.setObjectName("top")
        top_dummy.setStyleSheet("#top {margin: 0px 0px 20px}")
        self.layout().addWidget(top_dummy)

        bottom_container.addWidget(self._model_input_widget)
        bottom_dummy.setLayout(bottom_container)
        self.layout().addWidget(bottom_dummy)

        self._return_btn: QPushButton = QPushButton("Return")
        self._return_btn.clicked.connect(
            lambda: self._main_model.dispatch(Event.VIEW_SELECTION_MAIN)
        )
        self._return_btn.setStyleSheet("margin-top: 40px")
        self.layout().addWidget(self._return_btn)

        # TODO: replace return button (?) with run button

        self._main_model.subscribe(
            Event.VIEW_SELECTION_PREDICTION,
            self,
            lambda e: self._main_model.set_current_view(self),
        )

    def handle_event(self, event: Event) -> None:
        pass
