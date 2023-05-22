from allencell_ml_segmenter.view.sample_view import SampleViewController
from allencell_ml_segmenter.controller.example_controller import UiController
from allencell_ml_segmenter.controller.im2im_contoller import Im2imContoller
from allencell_ml_segmenter.model.sample_model import SampleModel



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
        model = SampleModel()
        controller = UiController(self._application, model)
        Im2imContoller(self._application, model) # not referenced, but subscribes to model.
        self._handle_navigation(controller)
