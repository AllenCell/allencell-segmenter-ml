from allencell_ml_segmenter.view.sample_view_controller import (
    SampleViewController,
)
from allencell_ml_segmenter.model.publisher import Subscriber, Event
from allencell_ml_segmenter.model.training_model import TrainingModel


class SampleController(Subscriber):
    """
    Controller for the SampleView
    """

    def __init__(self, model: TrainingModel) -> None:
        super().__init__()
        self._model: TrainingModel = model
        self._view: SampleViewController = SampleViewController(self._model)
        self._model.subscribe(self)

    @property
    def view(self) -> SampleViewController:
        return self._view

    def handle_event(self, event: Event) -> None:
        pass

    def index(self):
        """
        Index function called when loading this controller
        """
        # called when loading new controller
        self.load_view()
        self._view.connect_slots()

    def load_view(self) -> SampleViewController:
        """
        Loads the SampleView
        """
        return self.application.view_manager.load_view(self._view)
