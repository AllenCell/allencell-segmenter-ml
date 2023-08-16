import time
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event

from lightning.pytorch.callbacks import Callback

from cyto_dl.train import main as cyto_train
from napari.qt.threading import thread_worker

import sys
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

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print(
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Training is starting@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        )

    def on_train_epoch_start(self, trainer, pl_module):
        print(
            f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Epoch {trainer.current_epoch} is starting@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        print(
            f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Training {trainer.current_epoch} is ending@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        )

    def on_train_end(self, trainer, pl_module):
        print(
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Training is ending@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        )


class TrainingService(Subscriber):
    """
    Interface for training a model. Uses cyto-dl to train model according to spec
    """

    def __init__(self, training_model: TrainingModel):
        super().__init__()
        self._training_model: TrainingModel = training_model
        self._training_model.subscribe(
            Event.PROCESS_TRAINING,
            self,
            lambda e: self.train_model_handler(),
        )

    def train_model_handler(self) -> None:
        """
        Trains the model according to the spec
        """
        if self._training_model.is_training_running():
            self._training_model.set_experiment_type("segmentation")
            self._training_model.set_hardware_type("cpu")
            self._training_model.set_image_dims(2)
            self._training_model.set_images_directory(
                "/Users/chrishu/dev/code/cyto-dl/data/example_experiment_data/segmentation"
            )
            self._training_model.set_channel_index(9)
            self._training_model.set_max_time(9992)
            self._training_model.set_config_dir(
                "/Users/chrishu/dev/code/cyto-dl/configs"
            )
            self._training_model.set_patch_size("small")
            self._training_model.set_max_epoch(100)
            self._set_config_dir()
            self._set_experiment()
            self._set_hardware()
            self._set_images_directory()

            sys.argv.append(
                "+callbacks.print_progress._target_=allencell_ml_segmenter.services.training_service.MyPrintingCallback"
            )

            cyto_train()

    def _set_experiment(self) -> None:
        """
        Sets the experiment argument variable for hydra using sys.argv
        """
        experiment_type: TrainingType = (
            self._training_model.get_experiment_type()
        )
        sys.argv.append(f"experiment=im2im/{experiment_type.value}.yaml")

    def _set_hardware(self) -> None:
        """
        Sets the hardware argument variable for hydra using sys.argv
        """
        hardware_type: Hardware = self._training_model.get_hardware_type()
        sys.argv.append(f"trainer={hardware_type.value}")

    def _set_image_dims(self) -> None:
        """
        Sets the spatial_dims argument variable for hydra override using sys.argv
        """
        image_dims: int = self._training_model.get_image_dims()
        sys.argv.append(f"++spatial_dims=[{image_dims}]")

    def _set_max_epoch(self) -> None:
        """
        Sets the trainer.max_epochs argument variable for hydra override using sys.argv
        """
        max_epoch: int = self._training_model.get_max_epoch()
        sys.argv.append(f"++trainer.max_epochs={max_epoch}")

    def _set_images_directory(self) -> None:
        """
        Sets the data.path argument variable for hydra override using sys.argv
        """
        images_directory: Path = self._training_model.get_images_directory()
        sys.argv.append(f"++data.path={str(images_directory)}")

    def _set_patch_shape_from_size(self) -> None:
        """
        Sets the data._aux.patch_shape argument variable for hydra override using sys.argv
        """
        patch_size: PatchSize = self._training_model.get_patch_size()
        sys.argv.append(
            f"++data._aux.patch_shape={_list_to_string(patch_size.value)}"
        )

    def _set_config_dir(self) -> None:
        """
        Sets the config_dir hydra runtime variable using sys.argv
        """
        # This hydra runtime variable needs to be set in separate calls to sys.argv
        sys.argv.append("--config-dir")
        sys.argv.append(str(self._training_model.get_config_dir()))
