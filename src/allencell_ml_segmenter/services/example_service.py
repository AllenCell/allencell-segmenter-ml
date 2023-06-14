from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
import os, yaml
from allencell_ml_segmenter.models.example_model import ExampleModel
from typing import Dict, Any


class ExampleService(Subscriber):
    def __init__(self, example_model):
        super().__init__()

        # models
        self._example_model: ExampleModel = example_model
        self._example_model.subscribe(Event.SAVE, self)

    def handle_event(self, event: Event) -> None:
        """
        Gathers field values from the example model instance and writes to test.yaml.
        """
        if event == Event.SAVE:
            field_to_value: Dict[str, Any] = {
                "text": self._example_model.text,
                "option": self._example_model.option,
                "choice": self._example_model.choice,
                "slider": self._example_model.slider,
            }

            # Write status to test.yaml; currently "./" refers to desktop
            if not os.path.isfile("./test.yaml"):
                mode: str = "w"  # write
            else:
                mode: str = "a"  # append

            with open("./test.yaml", mode) as file:
                yaml.dump({self._example_model.index: field_to_value}, file)

            self._example_model.index += 1
