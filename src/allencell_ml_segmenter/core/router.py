from allencell_ml_segmenter.view.sample_view_controller import (
    SampleViewController,
)
from allencell_ml_segmenter.controller.ui_controller import UiController
from allencell_ml_segmenter.controller.training_controller import (
    TrainingController,
)
from allencell_ml_segmenter.model.training_model import TrainingModel


class Router:
    _controller = None

    def __init__(self, application):
        if application is None:
            raise ValueError("application")
        self._application = application
        self._model = None
        self._controller = None

    def _handle_navigation(self, controller):
        if self._controller:
            self._controller.cleanup()
        self._controller = controller
        self._controller.index()

    def navigate_to_test_view(self):
        model = TrainingModel()
        controller = UiController(self._application, model)
        TrainingController(
            self._application, model
        )  # not referenced, but subscribes to model.
        self._handle_navigation(controller)
