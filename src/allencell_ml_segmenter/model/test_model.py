from allencell_ml_segmenter.model.pub_sub import Publisher, Event

class TestModel(Publisher):
    def __init__(self):
        super().__init__()
        self._model_training: bool = False

    def get_model_training(self) -> bool:
        return self._model_training

    def set_model_training(self, running: bool):
        self._model_training = running
        self.dispatch(Event.TRAINING)


