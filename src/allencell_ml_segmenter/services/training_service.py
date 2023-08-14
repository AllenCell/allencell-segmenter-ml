from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event

# TODO include when on artifactory from cyto_dl.train import main as cyto_train
import sys
from allencell_ml_segmenter.training.training_model import (
    TrainingType,
    Hardware,
    PatchSize,
)
from allencell_ml_segmenter.core.publisher import Publisher
from pathlib import Path
from typing import List, Any
from enum import Enum
from cyto_dl.train import main as cyto_train
from cyto_dl.eval import main as cyto_predict

class CytodlMode(Enum):
    """
    Different cyto-dl modes
    """
    TRAIN = "train"
    PREDICT = "predict"


# static method
def _list_to_string(list_to_convert: List[Any]) -> str:
    """
    Converts a list of ints to a string

    list (List[int]): list of ints to convert
    """
    # fastest python implementation of list to string
    ints_to_strings: str = ", ".join([str(i) for i in list_to_convert])
    return f"[{ints_to_strings}]"


class CytoService(Subscriber):
    """
    Interface for training a model or predicting using a model.
    Uses cyto-dl for model training and inference.
    """

    def __init__(self, model: Publisher, mode: CytodlMode):
        super().__init__()
        self._model: Publisher = model
        self._mode: CytodlMode = mode

        if mode == CytodlMode.TRAIN:
            # Training Service
            self._model.subscribe(
                Event.PROCESS_TRAINING,
                self,
                self.train_model,
            )
        elif mode == CytodlMode.PREDICT:
            # Prediction Service
            self._model.subscribe(
                Event.PROCESS_PREDICTION,
                self,
                self.predict_model,
            )

    def train_model(self) -> None:
        """
        Trains the model according to the spec
        """
        # needs to be called first
        self._set_config_dir()
        self._set_config_name() # cyto_dl.train as train.yaml hardcoded

        # cyto-dl args
        self._set_experiment()
        self._set_hardware()

        # hydra overrides
        self._set_image_dims()
        self._set_max_epoch()
        self._set_images_directory()
        self._set_patch_shape_from_size()

        # Call to cyto-dl's train.py
        # TODO include when on artifactory cyto_train()
        cyto_train()

    def predict_model(self) -> None:
        """
        Predict segmentations using model according to spec
        """

        # config needs to be called first
        self._set_config_dir()
        self._set_config_name()
        cyto_predict()

    def _set_experiment(self) -> None:
        """
        Sets the experiment argument variable for hydra using sys.argv
        Used mainly for Training
        """
        experiment_type: TrainingType = (
            self._model.get_experiment_type()
        )
        if experiment_type is None:
            raise ValueError(
                "Experiment type not set. Please set experiment type."
            )
        sys.argv.append(f"experiment=im2im/{experiment_type.value}.yaml")

    def _set_hardware(self) -> None:
        """
        Sets the hardware argument variable for hydra using sys.argv
        Used mainly for Training
        """
        hardware_type: Hardware = self._model.get_hardware_type()
        if hardware_type is None:
            raise ValueError(
                "Hardware type not set. Please set hardware type."
            )
        sys.argv.append(f"trainer={hardware_type.value}")

    def _set_image_dims(self) -> None:
        """
        Sets the spatial_dims argument variable for hydra override using sys.argv
        Used mainly for Training
        """
        image_dims: int = self._model.get_image_dims()
        if image_dims is not None:
            sys.argv.append(f"++spatial_dims=[{image_dims}]")

    def _set_max_epoch(self) -> None:
        """
        Sets the trainer.max_epochs argument variable for hydra override using sys.argv
        Used mainly for Training
        """
        max_epoch: int = self._model.get_max_epoch()
        if max_epoch is not None:
            sys.argv.append(f"++trainer.max_epochs={max_epoch}")

    def _set_images_directory(self) -> None:
        """
        Sets the data.path argument variable for hydra override using sys.argv
        """
        images_directory: Path = self._model.get_images_directory()
        if images_directory is not None:
            sys.argv.append(f"++data.path={str(images_directory)}")

    def _set_patch_shape_from_size(self) -> None:
        """
        Sets the data._aux.patch_shape argument variable for hydra override using sys.argv
        Used mainly for Training
        """
        patch_size: PatchSize = self._model.get_patch_size()
        if patch_size is not None:
            sys.argv.append(
                f"++data._aux.patch_shape={_list_to_string(patch_size.value)}"
            )

    def _set_config_dir(self) -> None:
        """
        Sets the config_dir hydra runtime variable using sys.argv
        Used for both
        """
        # This hydra runtime variable needs to be set in separate calls to sys.argv
        config_dir: Path = self._model.get_config_dir()
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
        config_name: str = self._model.get_config_name()

        if config_name is None:
            # prediction requires config name specified
            if self._mode == CytodlMode.PREDICT:
                raise ValueError(
                    "Config name not set for predictions. Please set config name."
                )
            # training does not require config name to be set- uses default train.yaml
            else:
                return

        # set config name for predictions, or set custom config for training
        # This hydra runtime variable needs to be set in separate calls to sys.argv
        sys.argv.append("--config-name")
        sys.argv.append(str(self._model.get_config_name()))



