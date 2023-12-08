import asyncio

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event

# from lightning.pytorch.callbacks import Callback

# disabled for tests (cant import in ci yet)
# from cyto_dl.train import main as cyto_train
from cyto_dl.api.model import CytoDLModel


import sys
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.training.training_model import (
    TrainingType,
    Hardware,
    PatchSize,
)
from allencell_ml_segmenter.training.training_model import TrainingModel
from pathlib import Path
from typing import List, Any


# static method
def _list_to_string(list_to_convert: List[Any]) -> str:
    """
    Converts a list of ints to a string

    list (List[int]): list of ints to convert
    """
    # fastest python implementation of list to string
    ints_to_strings: str = ", ".join([str(i) for i in list_to_convert])
    return f"[{ints_to_strings}]"


class TrainingService(Subscriber):
    """
    Interface for training a model. Uses cyto-dl to train model according to spec
    """

    def __init__(
        self,
        training_model: TrainingModel,
        experiments_model: ExperimentsModel,
    ):
        super().__init__()
        self._training_model: TrainingModel = training_model
        self._experiments_model: ExperimentsModel = experiments_model
        self._training_model.subscribe(
            Event.PROCESS_TRAINING,
            self,
            self.train_model_handler,
        )

    def train_model_handler(self, _: Event) -> None:
        """
        Trains the model according to the spec
        """
        if self._training_model.is_training_running():
            # Only supporting segmentation config for now
            self._training_model.set_experiment_type("segmentation")

            # Only supporting MACOS and CPU use for now
            self._training_model.set_hardware_type("cpu")

            # UI is not activated yet.  Is 9 special?
            self._training_model.set_channel_index(9)

            # This field is not supported for now (maybe cancel button is sufficient?)
            self._training_model.set_max_time(9992)

            # Source of configs relative to user's home.  We need a dynamic solution in prod.
            self._training_model.set_config_dir(
                f"{self._experiments_model.get_cyto_dl_config().get_cyto_dl_home_path()}/configs"
            )
            #############################################
            sys.argv.append(
                # This is meant to be a string as is - not a string template.  In cyto-dl, it will be treated as a string template
                "hydra.run.dir=${paths.log_dir}/${task_name}/runs/${experiment_name}"
            )
            if self._experiments_model.get_checkpoint() is not None:
                sys.argv.append(
                    f"ckpt_path={self._experiments_model.get_model_checkpoints_path(self._experiments_model.get_experiment_name(), self._experiments_model.get_checkpoint())}"
                )
            model = CytoDLModel()
            model.download_example_data()
            model.load_default_experiment('segmentation',
                                          output_dir=f"{self._experiments_model.get_user_experiments_path()}/{self._experiments_model.get_experiment_name()}",
                                          overrides=[self._get_hardware_override(),
                                                     self._get_image_dims_override(),
                                                     self._get_experiment_name_override(),
                                                     self._get_max_epoch_override(),
                                                     self._get_images_directory_override(),
                                                     self._get_patch_shape_override()
                                                     ])
            #model.print_config()
            asyncio.run(model.train())

    def _get_hardware_override(self) -> str:
        """
        Get the hardware override for the CytoDLModel
        """
        hardware_type: Hardware = self._training_model.get_hardware_type()
        return f"trainer={hardware_type.value}"

    def _get_spatial_dims_override(self) -> str:
        """
        Get the spatial_dims override for the CytoDlModel
        """
        return(f"spatial_dims={self._training_model.get_spatial_dims()}")

    def _get_experiment_name_override(self) -> str:
        """
        Get the experiment name override for the CytoDlModel
        """
        return f"experiment_name={self._experiments_model.get_experiment_name()}"

    def _get_max_epoch_override(self) -> str:
        """
        Get the max epoch override for the CytoDlModel
        """
        return f"trainer.max_epochs={self._training_model.get_max_epoch()}"

    def _get_images_directory_override(self) -> str:
        """
        Get the data path override for the CytoDlModel
        Cyto dl expects a train.csv, valid.csv, and a test.csv in this folder for training.
        """
        return f"data.path={str(self._training_model.get_images_directory())}

    def _get_patch_shape_override(self) -> str:
        """
        Sets the data._aux.patch_shape argument variable for hydra override using sys.argv
        """
        patch_size: PatchSize = self._training_model.get_patch_size()
        return f"data._aux.patch_shape={_list_to_string(patch_size.value)}"