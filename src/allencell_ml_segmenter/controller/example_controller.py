from allencell_ml_segmenter.view.sample_view import SampleViewController
from allencell_ml_segmenter.model.pub_sub import Subscriber, Event
from allencell_ml_segmenter.model.sample_model import SampleModel

# higher level ui controller
class UiController(Subscriber):
    def __init__(self, application, model: SampleModel) -> None:
        super().__init__()
        # add all ui elements here
        self.application = application
        self._model: SampleModel = model
        self._view: SampleViewController = SampleViewController(self._model)
        self._model.subscribe(self)

    @property
    def view(self):
        return self._view

    def handle_event(self, event: Event):
        pass
        #TODO change to switch
        # if event == Event.TRAINING:

    def index(self):
        # called when loading new controller
        self.load_view()
        self._view.connect_slots()

    def load_view(self):
        """
        Loads the given view
        :param: view: the View to load
        """
        return self.application.view_manager.load_view(self._view)






