from allencell_ml_segmenter.view.test_view import TestView



class Router():
    _controller = None

    def __init__(self, application):
        if application is None:
            raise ValueError("application")
        self._application = application
        # TODO do some proper dependency injection in the future if the project grows
        self._controller = None

    def _handle_navigation(self, controller):
        if self._controller:
            self._controller.cleanup()
        self._controller = controller
        self._controller.index()

    def show_test_view(self):
        view = TestView()
        # do the following in controller
        self._application._view_manager.load_view(view)