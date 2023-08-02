from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
from cyto_dl.train import main as train_model
import sys
from allencell_ml_segmenter.training.training_model import (
    TrainingType,
    Hardware,
    PatchSize,
)
from allencell_ml_segmenter.training.training_model import TrainingModel
from pathlib import Path
from typing import List


# static method
def _list_of_int_to_string(list_to_convert: List[int]) -> str:
    """
    Converts a list of ints to a string

    list (List[int]): list of ints to convert
    """
    # fastest python implementation of list to string
    return "".join([str(i) for i in list_to_convert])


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
            lambda e: self.train_model(),
        )

    def train_model(self) -> None:
        """
        Trains the model according to the spec
        """
        self._set_experiment()
        self._set_hardware()

        # Call to cyto-dl's train.py
        train_model()

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
        sys.argv.append(f"++spatial_dims={image_dims}")

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
            f"++data._aux.patch_shape={_list_of_int_to_string(patch_size.value)}"
        )
