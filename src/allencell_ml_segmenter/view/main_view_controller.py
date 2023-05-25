from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.view._main_template import MainTemplate
from qtpy.QtWidgets import QVBoxLayout
from allencell_ml_segmenter.model.main_model import MainModel, MainEvent
from allencell_ml_segmenter.model.publisher import Subscriber
from allencell_ml_segmenter.controller.ui_controller import UiController
from allencell_ml_segmenter.controller.training_controller import TrainingController
from allencell_ml_segmenter.model.training_model import TrainingModel


class MainViewController(Subscriber):
    def __init__(self, model: MainModel):
        self._main_model = model

        self._active_model = None
        self._active_controller = None
        model.subscribe(self)

    @property
    def model(self):
        return self._main_model

    def handle_event(self, event: MainEvent):
        # define this
        pass

