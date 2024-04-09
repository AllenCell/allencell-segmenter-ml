from typing import Optional

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.training.training_model import TrainingModel


class FakeImageSelectionWidget:
    """
    A widget for training image selection.
    """

    TITLE_TEXT: str = "Training images"

    def __init__(
        self, model: TrainingModel, experiments_model: ExperimentsModel
    ):
        self._model: TrainingModel = model
        self._experiments_model: ExperimentsModel = experiments_model
        self.channels_updated_with_max: Optional[int] = None

    def _update_channels(self) -> None:
        self.channels_updated_with_max = self._model.get_max_channel()

