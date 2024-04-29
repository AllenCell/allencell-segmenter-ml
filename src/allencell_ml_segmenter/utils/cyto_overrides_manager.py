from typing import Dict, Union, Optional

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    Hardware,
)


class CytoDLOverridesManager:
    """
    Class to generate overrides for cyto-dl based on user selections in app state
    """

    # TODO use a similar pattern for Predictions- willdo in another ticket
    # https://github.com/AllenCell/allencell-ml-segmenter/issues/234
    def __init__(
        self,
        experiments_model: ExperimentsModel,
        training_model: TrainingModel = None,
    ) -> None:
        self._experiments_model: ExperimentsModel = experiments_model
        self._training_model: Optional[TrainingModel] = training_model

    def get_training_overrides(
        self,
    ) -> Dict[str, Union[str, int, float, bool, Dict]]:
        # check to see if CytoOverridesManager was constructed with a training model
        if self._training_model is None:
            raise ValueError(
                "CytoOverridesManager must be constructed with a training model in order to get training overrides."
            )

        overrides_dict: Dict[str, Union[str, int, float, bool, Dict]] = dict()

        # Hardware override (required)
        overrides_dict["trainer.accelerator"] = "cpu"
        if self._training_model.get_hardware_type() == Hardware.GPU:
            overrides_dict["trainer.accelerator"] = "gpu"

        # Spatial Dims (required)
        overrides_dict["spatial_dims"] = (
            self._training_model.get_spatial_dims()
        )

        # Max Run
        # define max run (in epochs, required)
        current_epoch: int = self._experiments_model.get_current_epoch()
        current_epoch = current_epoch + 1 if current_epoch is not None else 0
        overrides_dict["trainer.max_epochs"] = (
            self._training_model.get_num_epochs() + current_epoch
        )
        # max run in time is also defined (optional)
        if self._training_model.use_max_time():
            # define max runtime (in hours)
            overrides_dict["trainer.max_time"] = {
                "minutes": self._training_model.get_max_time()
            }

        # Training input path (required)
        overrides_dict["data.path"] = str(
            self._training_model.get_images_directory()
        )

        # Patch shape (required)
        overrides_dict["data._aux.patch_shape"] = (
            self._training_model.get_patch_size().value
        )

        # Checkpoint (optional)
        if self._experiments_model.get_checkpoint() is not None:
            # We are going to continue training on an existing model
            overrides_dict["ckpt_path"] = str(
                self._experiments_model.get_model_checkpoints_path(
                    self._experiments_model.get_experiment_name(),
                    self._experiments_model.get_checkpoint(),
                )
            )

        # Filters/Model Size (required)
        overrides_dict["model._aux.filters"] = self._training_model.get_model_size().value

        return overrides_dict
