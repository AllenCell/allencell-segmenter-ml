import asyncio

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event

from cyto_dl.api.model import CytoDLModel
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.training.training_model import (
    Hardware,
)
from allencell_ml_segmenter.training.training_model import TrainingModel
from typing import Dict, Union, Optional, List, Any
from napari.utils.notifications import show_warning
from allencell_ml_segmenter.utils.cuda_util import CUDAUtils
from allencell_ml_segmenter.utils.cyto_overrides_manager import (
    CytoDLOverridesManager,
)


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
            self._train_model_handler,
        )

    def _train_model_handler(self, _: Event) -> None:
        """
        Trains the model according to the spec
        """
        # Only supporting segmentation config for now, in the future this will be an option in the UI
        self._training_model.set_experiment_type("segmentation_plugin")
        # TODO make set_images_directory and get_images_directory less brittle.
        #  https://github.com/AllenCell/allencell-ml-segmenter/issues/156
        if self._able_to_continue_training():
            model = CytoDLModel()
            model.load_default_experiment(
                self._training_model.get_experiment_type(),
                output_dir=f"{self._experiments_model.get_user_experiments_path()}/{self._experiments_model.get_experiment_name()}",
            )
            cyto_overrides_manager: CytoDLOverridesManager = (
                CytoDLOverridesManager(
                    self._experiments_model, self._training_model
                )
            )
            model.override_config(
                cyto_overrides_manager.get_training_overrides()
            )
            model.print_config()
            asyncio.run(model.train(run_async=True))

    def _able_to_continue_training(self) -> bool:
        if self._experiments_model.get_experiment_name() is None:
            show_warning(
                "Please select an experiment before running prediction."
            )
            return False

        if self._training_model.get_experiment_type() is None:
            return False

        if self._training_model.get_spatial_dims() is None:
            show_warning(
                "Please select spatial dims for training dataset. 2-D or 3-D."
            )
            return False

        if self._training_model.get_images_directory() is None:
            show_warning("User has not selected input images for training")
            return False

        if self._training_model.get_patch_size() is None:
            show_warning("User has not selected a patch size for training")
            return False

        if self._training_model.get_max_epoch() is None:
            show_warning("Please define max epoch(s) to run for")
            return False
        if (
            self._training_model.get_max_channels() > 0
            and self._training_model.get_channel_index() is None
        ):
            show_warning(
                "Your raw images have multiple channels, please select a channel to train on."
            )
            return False
        return True
