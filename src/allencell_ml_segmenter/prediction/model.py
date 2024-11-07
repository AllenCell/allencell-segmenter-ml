from enum import Enum
from pathlib import Path
from typing import List, Optional

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.core.file_input_model import FileInputModel


class PredictionModel(Publisher):
    """
    Stores state relevant to prediction processes.
    """

    def __init__(self) -> None:
        super().__init__()
        # state related to ModelInputWidget
        self._preprocessing_method: Optional[str] = None
        self._postprocessing_method: Optional[str] = None
        self._postprocessing_simple_threshold: Optional[float] = None
        self._postprocessing_auto_threshold: Optional[str] = None

        # This is initialized as None, and set when the input data is processed during pre-processing
        # If it is none after Event.ACTION_PREDICTION_SETUP is dispatched, the csv for
        # prediction was not generated and prediction cannot continue.
        self.total_num_images: Optional[int] = None

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
