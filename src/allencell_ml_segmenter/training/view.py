from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy

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

        self.layout().addWidget(ImageSelectionWidget())
        self.layout().addWidget(ModelSelectionWidget())

        self._main_model.subscribe(
            Event.VIEW_SELECTION_TRAINING,
            self,
            lambda e: self._main_model.set_current_view(self),
        )

        # apply styling
        self.setStyleSheet("prediction_view.qss")  # TODO: revisit styling
