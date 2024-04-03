import asyncio
from pathlib import Path

from allencell_ml_segmenter.core.channel_extraction import ChannelExtractionThread, get_img_path_from_csv
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
from typing import List, Any, Optional, Callable


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
        self._channel_extraction_thread: Optional[ChannelExtractionThread] = None

        self._training_model.subscribe(
            Event.ACTION_TRAINING_DATASET_SELECTED,
            self,
            self._training_image_directory_selected
        )

    def train_model_handler(self, _: Event) -> None:
        """
        Trains the model according to the spec
        """
        # Only supporting segmentation config for now
        self._training_model.set_experiment_type("segmentation")
        # TODO make set_images_directory and get_images_directory less brittle.
        #  https://github.com/AllenCell/allencell-ml-segmenter/issues/156
        # this is just to test for now.
        # self._training_model.set_images_directory(Path("/Users/brian.kim/work/cyto-dl/data/example_experiment_data/segmentation"))

        # Following lines does nothing currently, need to implement
        # self._training_model.set_channel_index(9)
        # self._training_model.set_max_time(9992)
        #############################################
        # sys.argv.append(
        #     # This is meant to be a string as is - not a string template.  In cyto-dl, it will be treated as a string template
        #     "hydra.run.dir=${paths.log_dir}/${task_name}/runs/${experiment_name}"
        # )
        model = CytoDLModel()
        model.download_example_data()
        model.load_default_experiment(
            self._training_model.get_experiment_type().value,
            output_dir=f"{self._experiments_model.get_user_experiments_path()}/{self._experiments_model.get_experiment_name()}",
            overrides=self._build_overrides(),
        )
        model.print_config()
        asyncio.run(model._train_async())

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
        return f"spatial_dims={self._training_model.get_spatial_dims()}"

    def _get_experiment_name_override(self) -> str:
        """
        Get the experiment name override for the CytoDlModel
        """
        return (
            f"experiment_name={self._experiments_model.get_experiment_name()}"
        )

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
        return f"data.path={str(self._training_model.get_images_directory())}"

    def _get_patch_shape_override(self) -> str:
        """
        get the patch shape override for the CytoDLModel
        """
        patch_size: PatchSize = self._training_model.get_patch_size()
        return f"data._aux.patch_shape={_list_to_string(patch_size.value)}"

    def _get_checkpoint_override(self) -> str:
        """
        Get the checkpoint path override for the CytoDLModel
        """
        return f"ckpt_path={self._experiments_model.get_model_checkpoints_path(self._experiments_model.get_experiment_name(), self._experiments_model.get_checkpoint())}"

    def _build_overrides(self) -> List[str]:
        """
        Build a list of overrides for the CytoDLModel from plugin state.
        """
        # TODO: Add channel index selection from UI
        # TODO: Add max time from UI
        overrides: List = []

        # REQUIRED OVERRIDES for cyto-dl run
        if self._training_model.get_hardware_type() is None:
            # v1 defaults to cpu
            self._training_model.set_hardware_type("cpu")
        overrides.append(self._get_hardware_override())

        if self._training_model.get_spatial_dims() is None:
            raise ValueError(
                "Must define spatial dims 2-d or 3-d to run training."
            )
        overrides.append(self._get_spatial_dims_override())

        if self._experiments_model.get_experiment_name() is None:
            raise ValueError(
                "User has not selected experiment to save model into"
            )
        overrides.append(self._get_experiment_name_override())

        if self._training_model.get_images_directory() is None:
            raise ValueError("User has not selected input images for training")
        overrides.append(self._get_images_directory_override())

        # OPTIONAL OVERRIDES for cyto-dl run
        if self._training_model.get_max_epoch() is not None:
            # TODO: ask benji- what happens if a user does not define max_epoch?
            # For now use the default coded values in cyto-dl's experiment config files
            overrides.append(self._get_max_epoch_override())

        if self._training_model.get_patch_size() is not None:
            # If for some reason patch size is not selected, default to the patch size defined in cyto-dl's experiment
            # config files.
            overrides.append(self._get_patch_shape_override())

        if self._experiments_model.get_checkpoint() is not None:
            # If checkpoint path is selected, use
            overrides.append(self._get_checkpoint_override())

        return overrides

    def _start_channel_extraction(self, to_extract: Path, channel_callback: Callable):
        self._channel_extraction_thread = ChannelExtractionThread(get_img_path_from_csv(to_extract / "train.csv"))
        self._channel_extraction_thread.channels_ready.connect(channel_callback)
        self._channel_extraction_thread.start()

    def _stop_channel_extraction(self) -> None:
        if self._channel_extraction_thread and self._channel_extraction_thread.isRunning():
            self._channel_extraction_thread.requestInterruption()
            self._channel_extraction_thread.wait()

    def _training_image_directory_selected(self, _: Event) -> None:
        self._stop_channel_extraction() # stop if already running
        self._start_channel_extraction(self._training_model.get_images_directory(), self._training_model.set_max_channel)


