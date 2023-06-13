from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.core.event import Event


class ExampleModel(Publisher):
    """
    Model used for the example exercise.
    """

    def __init__(self):
        super().__init__()
        # Current state of example model usage
        self.saved: bool = False
        self._model_in_use: bool = False

        # Set fields related to widget
        # TODO: edit in future commit when I implement the model (and add typing)
        self.text, self.option, self.choice, self.slider = (
            None,
            None,
            None,
            None,
        )

        # TODO: read in test.yaml and figure out the true index
        self.index = 0

    def get_model_usage_state(self) -> bool:
        """
        Getter to get the current state of the model's usage
        """
        return self._model_in_use

    def set_example_model(self, running: bool) -> None:
        """
        Setter to set the current state of the model.
        """
        self._model_in_use = running
        self.dispatch(Event.EXAMPLE)

    def save(self, save_status: bool) -> None:
        self.saved = save_status
        self.dispatch(Event.SAVE)
