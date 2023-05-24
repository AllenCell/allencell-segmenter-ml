from allencell_ml_segmenter.model.publisher import Subscriber, Event
from allencell_ml_segmenter.model.training_model import TrainingModel
# from allencell_ml_segmenter.services.training_service import TrainingService


class TrainingController(Subscriber):
    def __init__(
        self,
        application,
        model: TrainingModel,
        # training_service: TrainingService,
    ):
        super().__init__()
        self._model = model
        self._model.subscribe(self)
        self._application = application
        # self._training_service = training_service

    def handle_event(self, event: Event):
        # TODO change to switch
        if event == Event.TRAINING:
            if self._model.get_model_training():
                pass
