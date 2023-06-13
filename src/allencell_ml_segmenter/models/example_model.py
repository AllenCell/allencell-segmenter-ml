from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.core.event import Event
import os, yaml
from typing import Dict, Any


class ExampleModel(Publisher):
    """
    Model used for the example exercise.
    """

    def __init__(self):
        super().__init__()

        # Current save state
        self.saved: bool = False

        # Set default values for fields related to widget
        self.text: str = ""
        self.option: str = "One"
        self.choice: int = None
        self.slider: int = 0

        # Read in test.yaml and figure out the proper index
        if not os.path.isfile("./test.yaml"):
            self.index: int = 0
        else:
            with open("./test.yaml", "r") as file:
                previous_entries: Dict[int, Dict[str, Any]] = yaml.safe_load(
                    file
                )
                self.index: int = len(previous_entries)

    def get_model_save_state(self) -> bool:
        """
        Getter to get the current save state of the model.
        """
        return self.saved

    def save(self, save_status: bool) -> None:
        """
        Setter to set the current save state and dispatch a save event to subs.
        """
        self.saved = save_status
        self.dispatch(Event.SAVE)
