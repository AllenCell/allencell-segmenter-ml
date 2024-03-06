from pathlib import Path
from typing import Generator
import csv

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
            lambda e: self._model.set_max_channels(
                self._determine_input_selection_type(
                    self._model.get_input_image_path()
                )
            ),
        )

    def handle_event(self, event: Event) -> None:
        pass

    def extract_preprocessing_method(self) -> None:
        """
        Calls the prediction model's setter for the preprocessing method. Currently set up with a dummy value.
        """
        # TODO: replace dummy implementation
        self._model.set_preprocessing_method("foo")

    def extract_num_channels_in_folder(self, path: Path) -> int:
        """
        Determine total number of channels for image in a set folder
        """
        # we expect user to have the same number of channels for all images in their folders
        # and that only images are stored in those folders
        # Get first image path
        path_generator: Generator[Path] = path.glob("*")
        first_image: Path = next(path_generator)
        # ignore hidden files
        while str(first_image.name).startswith("."):
            first_image = next(path_generator)
        return extract_num_channels_from_image(str(first_image.resolve()))



    def _determine_input_selection_type(self, path: Path):
        if path.is_dir():
            return self.extract_num_channels_in_folder(path)
        elif path.suffix == ".csv":
            return extract_num_channels_from_csv(path)


def extract_num_channels_from_image(path: Path):
    img: AICSImage = AICSImage(str(path))
    return img.dims.C


def extract_num_channels_from_csv(path: Path):
    with open(path) as file:
        reader: csv.reader = csv.DictReader(file)
        # first column contrains files of interest (zeroth column is index)
        line_data_path: str = next(reader)["raw"]
        return extract_num_channels_from_image(line_data_path)
