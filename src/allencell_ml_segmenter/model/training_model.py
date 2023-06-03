from allencell_ml_segmenter.core.publisher import Publisher, Event


class TrainingModel(Publisher):
    """
    Model used for training activities
    """

    def __init__(self):
        super().__init__()
        # Current state of model training
        self._model_training: bool = False

    def get_model_training(self) -> bool:
        """
        Getter to get the current state of the model training
        """
        return self._model_training

    def set_model_training(self, running: bool) -> None:
        """
        Setter to set the current state of the model training
        """
        self._model_training = running
        self.dispatch(Event.TRAINING)
