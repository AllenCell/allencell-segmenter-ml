from typing import Dict, Union, Optional, List

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    ImageType,
    ModelSize,
)
from allencell_ml_segmenter.utils.cuda_util import CUDAUtils


class CytoDLOverridesManager:
    """
    Class to generate overrides for cyto-dl based on user selections in app state
    """

    # TODO use a similar pattern for Predictions- willdo in another ticket
    # https://github.com/AllenCell/allencell-ml-segmenter/issues/234
    def __init__(
        self,
        experiments_model: ExperimentsModel,
        training_model: Optional[TrainingModel] = None,
    ) -> None:
        self._experiments_model: ExperimentsModel = experiments_model
        self._training_model: Optional[TrainingModel] = training_model

    def get_training_overrides(
        self,
    ) -> Dict[str, Union[str, int, float, bool, Dict, List]]:
        # check to see if CytoOverridesManager was constructed with a training model
        if self._training_model is None:
            raise ValueError(
                "CytoOverridesManager must be constructed with a training model in order to get training overrides."
            )

        overrides_dict: Dict[str, Union[str, int, float, bool, Dict, List]] = (
            dict()
        )

        # Hardware override (required)
        if CUDAUtils.cuda_available():
            overrides_dict["trainer.accelerator"] = "gpu"
        else:
            overrides_dict["trainer.accelerator"] = "cpu"

        overrides_dict["data.num_workers"] = CUDAUtils.get_num_workers()

        # Spatial Dims (required)
        dims: Optional[int] = self._training_model.get_spatial_dims()
        if dims is not None:
            overrides_dict["spatial_dims"] = dims

        # Patch shape (required)
        patch_size: Optional[list[int]] = self._training_model.get_patch_size()
        if patch_size is not None:
            overrides_dict["data._aux.patch_shape"] = patch_size

        # Max Run
        # define max run (in epochs, required)
        current_epoch: Optional[int] = self._experiments_model.get_current_epoch()
        num_epochs: Optional[int] = self._training_model.get_num_epochs()
        if num_epochs is not None:
            adjusted_epoch: int = current_epoch + 1 if current_epoch is not None else 0
            overrides_dict["trainer.max_epochs"] = (
                num_epochs + adjusted_epoch
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

        # Commented out 5/23 brian.kim
        # We no longer support training from a old checkpoint- leaving this in if we want to re-enable this in the
        # future to continue training from an existing checkpoint.
        # if self._experiments_model.get_checkpoint() is not None:
        #     # We are going to continue training on an existing model
        #     overrides_dict["ckpt_path"] = str(
        #         self._experiments_model.get_model_checkpoints_path(
        #             self._experiments_model.get_experiment_name(),
        #             self._experiments_model.get_checkpoint(),
        #         )
        #     )

        # Filters/Model Size (required)
        model_size: Optional[ModelSize] = self._training_model.get_model_size()
        if model_size is None:
            raise RuntimeError("Cannot generate overrides when model size is undefined")
        
        overrides_dict["model._aux.filters"] = (
            model_size.value
        )

        # Channel Override
        raw_channel: Optional[int] = self._training_model.get_selected_channel(ImageType.RAW)
        seg1_channel: Optional[int] = self._training_model.get_selected_channel(ImageType.SEG1)
        seg2_channel: Optional[int] = self._training_model.get_selected_channel(ImageType.SEG2)

        if raw_channel is not None:
            overrides_dict["input_channel"] = raw_channel
        if seg1_channel is not None:
            overrides_dict["target_col1_channel"] = seg1_channel
        if seg2_channel is not None:
            overrides_dict["target_col2_channel"] = seg2_channel

        return overrides_dict
