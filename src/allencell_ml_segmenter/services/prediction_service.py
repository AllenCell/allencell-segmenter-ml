
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
import sys
from allencell_ml_segmenter.prediction.model import PredictionModel
from pathlib import Path
from typing import List, Any
from cyto_dl.eval import main as cyto_predict

# static method
def _list_to_string(list_to_convert: List[Any]) -> str:
    """
    Converts a list of ints to a string

    list (List[int]): list of ints to convert
    """
    # fastest python implementation of list to string
    ints_to_strings: str = ", ".join([str(i) for i in list_to_convert])
    return f"[{ints_to_strings}]"


class PredictionService(Subscriber):
    """
    Interface for training a model or predicting using a model.
    Uses cyto-dl for model training and inference.
    #TODO create an ABC for cyto-service and have prediction_service inherit it
    """

    def __init__(self, prediction_model: PredictionModel):
        super().__init__()
        self._prediction_model: PredictionModel = prediction_model
        self._prediction_model.subscribe(
            Event.PROCESS_PREDICTION,
            self,
            self.predict_model,
        )

    def predict_model(self) -> None:
        """
        Predict segmentations using model according to spec
        """
        self._prediction_model.set_config_name("config.yaml")
        self._prediction_model.set_config_dir(
            "/Users/brian.kim/Desktop/data"
        )

        # config needs to be called first
        self._set_config_dir()
        self._set_config_name()
        cyto_predict()


    def _set_config_dir(self) -> None:
        """
        Sets the config_dir hydra runtime variable using sys.argv
        Used for both
        """
        # This hydra runtime variable needs to be set in separate calls to sys.argv
        config_dir: Path = self._prediction_model.get_config_dir()
        if config_dir is None:
            raise ValueError(
                "Config directory not set. Please set config directory."
            )
        sys.argv.append("--config-dir")
        sys.argv.append(str(config_dir))

    def _set_config_name(self) -> None:
        """
        Sets the config_name hydra runtime variable using sys.argv
        Used for both
        """
        config_name: str = self._prediction_model.get_config_name()
        # set config name for predictions, or set custom config for training
        # This hydra runtime variable needs to be set in separate calls to sys.argv
        sys.argv.append("--config-name")
        sys.argv.append(str(config_name))


if __name__ == "__main__":
    model = PredictionModel()
    serv = PredictionService(model)
    serv.predict_model()
