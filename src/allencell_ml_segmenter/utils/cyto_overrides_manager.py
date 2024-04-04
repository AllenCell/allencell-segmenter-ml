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

        # Hardware override
        overrides_dict["trainer.accelerator"] = "cpu"
        if self._training_model.get_hardware_type() == Hardware.GPU:
            overrides_dict["trainer.accelerator"] = "gpu"

        # Spatial Dims
        overrides_dict["spatial_dims"] = (
            self._training_model.get_spatial_dims()
        )

        # Max Run
        # define max run (in epochs, required)
        overrides_dict["trainer.max_epochs"] = (
            self._training_model.get_max_epoch()
        )
        # max run in time is also defined (optional)
        if self._training_model.use_max_time():
            # define max runtime (in hours)
            overrides_dict["trainer.max_time"] = {
                "minutes": self._training_model.get_max_time()
            }

        # Training input path
        overrides_dict["data.path"] = str(
            self._training_model.get_images_directory()
        )

        # Patch shape
        overrides_dict["data._aux.patch_shape"] = (
            self._training_model.get_patch_size().value
        )

        # Checkpoint
        if self._experiments_model.get_checkpoint() is not None:
            # We are going to continue training on an existing model
            overrides_dict["ckpt_path"] = str(
                self._experiments_model.get_model_checkpoints_path(
                    self._experiments_model.get_experiment_name(),
                    self._experiments_model.get_checkpoint(),
                )
            )

        return overrides_dict
