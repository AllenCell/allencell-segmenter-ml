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

        # Set default values for fields related to widget
        self.text: str = ""
        self.option: int = None
        self.choice: str = "One"
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

    def save(self) -> None:
        """
        Setter to set the current save state and dispatch a save event to subs.
        """
        self.dispatch(Event.SAVE)
