from allencell_ml_segmenter.view.test_view import TestView
from allencell_ml_segmenter.controller.example_controller import TestController



class Router():
    _controller = None

    def __init__(self, application):
        if application is None:
            raise ValueError("application")
        self._application = application
        self._controller = None

    def _handle_navigation(self, controller):
        if self._controller:
            self._controller.cleanup()
        self._controller = controller
        self._controller.index()

    def navigate_to_test_view(self):
        controller = TestController(self._application)
        self._handle_navigation(controller)
