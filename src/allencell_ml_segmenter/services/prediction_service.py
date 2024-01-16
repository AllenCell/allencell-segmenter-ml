import asyncio

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
import sys

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.prediction.model import PredictionModel
from pathlib import Path
from typing import Union, Dict

# from cyto_dl.api.model import CytoDLModel
from napari.utils.notifications import show_warning


class PredictionService(Subscriber):
    """
    Interface for training a model or predicting using a model.
    Uses cyto-dl for model training and inference.
    #TODO create an ABC for cyto-service and have prediction_service inherit it
    """

    def __init__(
        self,
        prediction_model: PredictionModel,
        experiments_model: ExperimentsModel,
    ):
        super().__init__()
        self._prediction_model: PredictionModel = prediction_model
        self._experiments_model: ExperimentsModel = experiments_model
        self._overrides: Dict[Union[str, int, float, bool]] = dict()

        self._prediction_model.subscribe(
            Event.PROCESS_PREDICTION,
            self,
            self._predict_model,
        )

    def _predict_model(self, _: Event) -> None:
        """
        Predict segmentations using model according to spec
        """
        continue_prediction: bool = True

        # Check to see if experiment selected
        experiment_name: str = self._experiments_model.get_experiment_name()
        if experiment_name is None:
            show_warning(
                "Please select an experiment before running prediction."
            )
            continue_prediction = False

        # Check to see if training has occurred with the selected experiment.
        training_config: Path = self._experiments_model.get_train_config_path(
            experiment_name
        )
        if continue_prediction and not training_config.exists():
            show_warning(
                f"Please train with the experiment: {experiment_name} before running a prediction."
            )
            continue_prediction = False

        # Check to see the user has specified a ckpt to use.
        checkpoint_selected: str = self._experiments_model.get_checkpoint()
        if (
            continue_prediction
            and self._experiments_model.get_checkpoint() is None
        ):
            show_warning(
                f"Please select a checkpoint to run predictions with."
            )
            continue_prediction = False

        if continue_prediction:
            cyto_api: CytoDLModel = CytoDLModel()
            cyto_api.load_config_from_file(training_config)
            # We must override the config to set up predictions correctly
            cyto_api.override_config(
                self._build_overrides(experiment_name, checkpoint_selected)
            )
            asyncio.run(cyto_api.predict())

    def _build_overrides(self, experiment_name: str, checkpoint: str) -> None:
        """
        Build an overrides list for the cyto-dl API containing the
        overrides requried to run predictions, formatted as cyto-dl expects.
        """
        # Default overrides needed for prediction
        self._overrides["test"] = False
        self._overrides["train"] = False
        self._overrides["mode"] = "predict"
        self._overrides["task_name"] = "predict_task_from_app"
        # passing the experiment_name and checkpoint as params to this function ensures we have a model before
        # attempting to build the overrides dict for predictions
        self._overrides["ckpt_path"] = str(
            self._experiments_model.get_model_checkpoints_path(
                experiment_name=experiment_name, checkpoint=checkpoint
            )
        )

        # overrides from model
        # if output_dir is not set, will default to saving in the experiment folder
        output_dir: Path = self._prediction_model.get_output_directory()
        if output_dir:
            self._overrides["paths.output_dir"] = str(output_dir)

        # if channel is not set, will default to same channel used to train
        channel: int = self._prediction_model.get_image_input_channel_index()
        if channel:
            self._overrides[
                "data.transforms.predict.transforms[0].reader[0].C"
            ] = channel
