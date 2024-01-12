from pathlib import Path

from aicsimageio import AICSImage

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.prediction.model import PredictionModel


class ModelFileService(Subscriber):
    """
    Parses the chosen model file to extract the preprocessing method.
    """

    def __init__(self, model: PredictionModel):
        super().__init__()
        self._model: PredictionModel = model

        self._model.subscribe(
            Event.ACTION_PREDICTION_MODEL_FILE,
            self,
            lambda e: self.extract_preprocessing_method(),
        )

        self._model.subscribe(
            Event.ACTION_PREDICTION_EXTRACT_CHANNELS,
            self,
            lambda e: self._model.set_max_channels(self.exctact_num_channels())
        )

    def handle_event(self, event: Event) -> None:
        pass

    def extract_preprocessing_method(self) -> None:
        """
        Calls the prediction model's setter for the preprocessing method. Currently set up with a dummy value.
        """
        # TODO: replace dummy implementation
        self._model.set_preprocessing_method("foo")

    def exctact_num_channels(self) -> int:
        """
        Determine total number of channels for image in a set folder
        """
        # we expect user to have the same number of channels for all images in their folders
        # and that only images are stored in those folders
        # Get first image path
        first_image: Path = next(self._model.get_input_image_dir().glob('*'))
        img: AICSImage = AICSImage(str(first_image.resolve()))
        return img.dims.C

