from enum import Enum
from pathlib import Path
from typing import List, Optional

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.core.FileInputModel import FileInputModel

class PredictionInputMode(Enum):
    FROM_PATH = "from_path"
    FROM_NAPARI_LAYERS = "from_napari_layers"


class PredictionModel(FileInputModel):
    """
    Stores state relevant to prediction processes.
    """

    def __init__(self) -> None:
        super().__init__()

        # state related to PredictionFileInput
        self.config_name: Optional[str] = None
        self.config_dir: Optional[Path] = None
        self._model_path: Optional[Path] = None

        # state related to ModelInputWidget
        self._preprocessing_method: Optional[str] = None
        self._postprocessing_method: Optional[str] = None
        self._postprocessing_simple_threshold: Optional[float] = None
        self._postprocessing_auto_threshold: Optional[str] = None

        # This is initialized as None, and set when the input data is processed during pre-processing
        # If it is none after Event.ACTION_PREDICTION_SETUP is dispatched, the csv for
        # prediction was not generated and prediction cannot continue.
        self.total_num_images: Optional[int] = None

    def get_output_seg_directory(self) -> Optional[Path]:
        """
        Gets path to where segmentation predictions are stored.
        """
        return (
            self._output_directory / "target"
            if self._output_directory is not None
            else None
        )

    def get_model_path(self) -> Optional[Path]:
        """
        Gets path to model.
        """
        return self._model_path

    def set_model_path(self, path: Optional[Path]) -> None:
        """
        Sets path to model.
        """
        self._model_path = path
        self.dispatch(Event.ACTION_PREDICTION_MODEL_FILE)

    def get_preprocessing_method(self) -> Optional[str]:
        """
        Gets preprocessing method associated with currently-selected model.
        """
        return self._preprocessing_method

    def set_preprocessing_method(self, method: Optional[str]) -> None:
        """
        Sets preprocessing method associated with model after the service parses the file.
        """
        self._preprocessing_method = method
        self.dispatch(Event.ACTION_PREDICTION_PREPROCESSING_METHOD)

    def get_postprocessing_method(self) -> Optional[str]:
        """
        Gets postprocessing method selected by user.
        """
        return self._postprocessing_method

    def set_postprocessing_method(self, method: Optional[str]) -> None:
        """
        Sets postprocessing method selected by user.
        """
        self._postprocessing_method = method
        self.dispatch(Event.ACTION_PREDICTION_POSTPROCESSING_METHOD)

    def get_postprocessing_simple_threshold(self) -> Optional[float]:
        """
        Gets simple threshold selected by user.
        """
        return self._postprocessing_simple_threshold

    def set_postprocessing_simple_threshold(
        self, threshold: Optional[float]
    ) -> None:
        """
        Sets simple threshold selected by user from the slider.
        """
        self._postprocessing_simple_threshold = threshold
        self.dispatch(Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD)

    def get_postprocessing_auto_threshold(self) -> Optional[str]:
        """
        Gets auto threshold selected by user.
        """
        return self._postprocessing_auto_threshold

    def set_postprocessing_auto_threshold(
        self, threshold: Optional[str]
    ) -> None:
        """
        Sets auto threshold selected by user.
        """
        self._postprocessing_auto_threshold = threshold
        self.dispatch(Event.ACTION_PREDICTION_POSTPROCESSING_AUTO_THRESHOLD)

    def dispatch_prediction(self) -> None:
        """
        Dispatches an event to start prediction
        """
        # Shoots off a prediction run
        self.dispatch(Event.PROCESS_PREDICTION)

    def dispatch_prediction_get_image_paths_from_napari(self) -> None:
        # Does some pre-configuring if needed for prediction runs
        self.dispatch(Event.ACTION_PREDICTION_GET_IMAGE_PATHS_FROM_NAPARI)

    def dispatch_prediction_setup(self) -> None:
        # Does some pre-configuring if needed for prediction runs
        self.dispatch(Event.ACTION_PREDICTION_SETUP)

    def set_total_num_images(self, total: Optional[int]) -> None:
        self.total_num_images = total

    def get_total_num_images(self) -> Optional[int]:
        return self.total_num_images
