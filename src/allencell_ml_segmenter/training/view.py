from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy, QLabel, QPushButton

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.training.image_selection_widget import (
    ImageSelectionWidget,
)
from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)


class TrainingView(View, Subscriber):
    """
    Holds widgets pertinent to training processes.
    """

    def __init__(self, main_model: MainModel):
        super().__init__()

        self._main_model: MainModel = main_model
        # self._training_model: TrainingModel = TrainingModel()

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self._title: QLabel = QLabel("SEGMENTATION MODEL TRAINING", self)
        self._title.setObjectName("title")
        self.layout().addWidget(self._title, alignment=Qt.AlignCenter)

        image_selection_widget: ImageSelectionWidget = ImageSelectionWidget()
        image_selection_widget.setObjectName("imageSelection")
        self.layout().addWidget(image_selection_widget)

        model_selection_widget: ModelSelectionWidget = ModelSelectionWidget()
        model_selection_widget.setObjectName("modelSelection")
        self.layout().addWidget(model_selection_widget)

        self._train_btn: QPushButton = QPushButton("Start training")
        self._train_btn.setObjectName("trainBtn")
        self.layout().addWidget(self._train_btn)

        self._main_model.subscribe(
            Event.VIEW_SELECTION_TRAINING,
            self,
            lambda e: self._main_model.set_current_view(self),
        )

        # apply styling
        self.setStyleSheet("training_view.qss")
