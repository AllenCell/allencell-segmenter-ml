from allencell_ml_segmenter.view.test_view import TestView
from allencell_ml_segmenter.model.pub_sub import Subscriber, Event
from allencell_ml_segmenter.model.test_model import TestModel


class UiController(Subscriber):
    def __init__(self, application, model: TestModel) -> None:
        super().__init__()
        # add all ui elements here
        self.application = application
        self._view: TestView = TestView()
        self._model: TestModel = model
        self._model.subscribe(self)

    def handle_event(self, event: Event):
        #TODO change to switch
        if event == Event.TRAINING:
            self._view.widget.label.setText(f"training is running {self._model.get_model_training()}")
            # set training label text method in view
            # mock view and see if method called

    def index(self):
        # called when loading new controller
        self.load_view()
        self._connect_slots()

    def _on_click(self) -> None:
        print("napari has", len(self.viewer.layers), "layers")

    def _connect_slots(self):
        self._view.widget.btn.clicked.connect(self.change_label)

    def load_view(self):
        """
        Loads the given view
        :param: view: the View to load
        """
        return self.application.view_manager.load_view(self._view)

    def change_label(self):
        self._model.set_model_training(not self._model.get_model_training())







