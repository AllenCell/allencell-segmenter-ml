from allencell_ml_segmenter.view.test_view import TestView
from allencell_ml_segmenter.controller.example_controller import UiController
from allencell_ml_segmenter.model.test_model import TestModel



class Router():
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
        model = TestModel()
        controller = UiController(self._application, model)
        self._handle_navigation(controller)
