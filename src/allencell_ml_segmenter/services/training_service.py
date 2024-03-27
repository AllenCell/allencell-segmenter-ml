import asyncio

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event

from cyto_dl.api.model import CytoDLModel

# from lightning.pytorch.callbacks import Callback

# disabled for tests (cant import in ci yet)
# from cyto_dl.train import main as cyto_train
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.training.training_model import (
    Hardware,
    PatchSize,
)
from allencell_ml_segmenter.training.training_model import TrainingModel
from typing import List, Any, Dict, Union, Optional
from napari.utils.notifications import show_warning


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
        self._overrides: Optional[Dict[str, Union[str, int, float, bool, Dict]]] = None

    def train_model_handler(self, _: Event) -> None:
        """
        Trains the model according to the spec
        """
        # Only supporting segmentation config for now
        self._training_model.set_experiment_type("segmentation")
        # TODO make set_images_directory and get_images_directory less brittle.
        #  https://github.com/AllenCell/allencell-ml-segmenter/issues/156
        if self._able_to_continue_training():
            model = CytoDLModel()
            # model.download_example_data()
            model.load_default_experiment(
                self._training_model.get_experiment_type().value,
                output_dir=f"{self._experiments_model.get_user_experiments_path()}/{self._experiments_model.get_experiment_name()}"
            )
            self._build_overrides()
            model.override_config(
                self._overrides
            )
            model.print_config()
            asyncio.run(model._train_async())

    def _able_to_continue_training(self) -> bool:
        if self._experiments_model.get_experiment_name() is None:
            show_warning(
                "Please select an experiment before running prediction."
            )
            return False

        if self._training_model.get_spatial_dims() is None:
            show_warning(
                "Please select spatial dims for training dataset. 2-D or 3-D."
            )
            return False

        if self._training_model.get_images_directory() is None:
            show_warning(
                "User has not selected input images for training"
            )
            return False

        if self._training_model.get_max_epoch() is None:
            if self._training_model.use_max_time() and self._training_model.get_max_time() is None:
                show_warning(
                    "Please define max epoch(s) to run, or max runtime for trainer."
                )
                return False
        return True

    def _hardware_override(self) -> None:
        """
        Get the hardware override for the CytoDLModel
        """
        # V1 defaults to CPU
        self._overrides["trainer.accelerator"] = "cpu"
        if self._training_model.get_hardware_type() == Hardware.GPU:
            self._overrides["trainer.accelerator"] = "gpu"

    def _spatial_dims_override(self) -> None:
        """
        Get the spatial_dims override for the CytoDlModel
        """
        self._overrides["spatial_dims"] = self._training_model.get_spatial_dims()

    def _experiment_name_override(self) -> None:
        """
        Get the experiment name override for the CytoDlModel
        """
        self._overrides["experiment_name"] = (
            self._experiments_model.get_experiment_name()
        )

    def _max_run_override(self) -> None:
        """
        Get the max epoch or time override for the CytoDlModel
        """
        # max run in time or epochs
        if self._training_model.use_max_time():
            # define max runtime (in hours)
            self._overrides["trainer.max_time"] = {"hours": self._training_model.get_max_time()}
        else:
            # define max run (in epochs)
            self._overrides["trainer.max_epochs"] = self._training_model.get_max_epoch()

    def _images_directory_override(self) -> None:
        """
        Get the data path override for the CytoDlModel
        Cyto dl expects a train.csv, valid.csv, and a test.csv in this folder for training.
        """
        self._overrides["data.path"] = str(self._training_model.get_images_directory())
    #
    def _patch_shape_override(self) -> None:
        """
        get the patch shape override for the CytoDLModel
        """
        self._overrides["data._aux.patch_shape"] = _list_to_string(
            self._training_model.get_patch_size().value
        )

    def _checkpoint_override(self) -> None:
        """
        Get the checkpoint path override for the CytoDLModel
        """
        if self._experiments_model.get_checkpoint() is not None:
            # We are going to continue training on an existing model
            self._overrides["ckpt_path"] = str(
                self._experiments_model.get_model_checkpoints_path(
                    self._experiments_model.get_experiment_name(),
                    self._experiments_model.get_checkpoint(),
                )
            )
    def _build_overrides(self):
        """
        Build a list of overrides for the CytoDLModel from plugin state.
        """
        # TODO: Add channel index selection from UI
        # TODO: Add max time from UI
        self.overrides = dict()

        # do overrides based on user selections
        self._hardware_override()
        self._spatial_dims_override()
        self._max_run_override()
        self._images_directory_override()
        self._patch_shape_override()
        self._checkpoint_override()
