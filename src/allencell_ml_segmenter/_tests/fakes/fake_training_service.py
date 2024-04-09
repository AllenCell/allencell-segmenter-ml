from pathlib import Path
from typing import Callable, Optional

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.training.training_model import TrainingModel


class FakeTrainingService(Subscriber):
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
            Event.ACTION_TRAINING_DATASET_SELECTED,
            self,
            self._training_image_directory_selected,
        )

        self.channel_callback_set: Optional[Callable] = None
        self.extraction_path_set: Optional[Path] = None
        self.channel_extraction_started: bool = False

    def _training_image_directory_selected(self, _: Event):
        self._start_channel_extraction(
            self._training_model.get_images_directory(),
            self._training_model.set_max_channel,
        )

    def _start_channel_extraction(
        self, to_extract: Path, channel_callback: Callable
    ):
        self.extraction_path_set = to_extract
        self.channel_callback_set = channel_callback
        self.channel_extraction_started = True
