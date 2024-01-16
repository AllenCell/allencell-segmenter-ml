from pathlib import Path
from typing import List

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class PredictionModel(Publisher):
    """
    Stores state relevant to prediction processes.
    """

    def __init__(self):
        super().__init__()

        # state related to PredictionFileInput
        self.config_name: str = None
        self.config_dir: Path = None
        self._input_image_paths: List[Path] = []
        self._image_input_channel_index: int = None
        self._output_directory: Path = None

        # state related to ModelInputWidget
        self._preprocessing_method: str = None
        self._postprocessing_method: str = None
        self._postprocessing_simple_threshold: float = None
        self._postprocessing_auto_threshold: str = None

    def get_input_image_paths(self) -> List[Path]:
        """
        Gets list of paths to input images.
        """
        return self._input_image_paths

    def set_input_image_paths(self, paths: List[Path]) -> None:
        """
        Sets list of paths to input images.
        """
        self._input_image_paths = paths
        # TODO: make new event if a service is used

    def get_image_input_channel_index(self) -> int:
        """
        Gets path to output directory.
        """
        return self._image_input_channel_index

    def set_image_input_channel_index(self, idx: int) -> None:
        """
        Sets path to output directory.
        """
        self._image_input_channel_index = idx

    def get_output_directory(self) -> Path:
        """
        Gets path to output directory.
        """
        return self._output_directory

    def set_output_directory(self, dir: Path) -> None:
        """
        Sets path to output directory.
        """
        self._output_directory = dir

    def get_model_path(self) -> Path:
        """
        Gets path to model.
        """
        return self._model_path

    def set_model_path(self, path: Path) -> None:
        """
        Sets path to model.
        """
        self._model_path = path
        self.dispatch(Event.ACTION_PREDICTION_MODEL_FILE)

    def get_preprocessing_method(self) -> str:
        """
        Gets preprocessing method associated with currently-selected model.
        """
        return self._preprocessing_method

    def set_preprocessing_method(self, method: str) -> None:
        """
        Sets preprocessing method associated with model after the service parses the file.
        """
        self._preprocessing_method = method
        self.dispatch(Event.ACTION_PREDICTION_PREPROCESSING_METHOD)

    def get_postprocessing_method(self) -> str:
        """
        Gets postprocessing method selected by user.
        """
        return self._postprocessing_method

    def set_postprocessing_method(self, method: str) -> None:
        """
        Sets postprocessing method selected by user.
        """
        self._postprocessing_method = method
        self.dispatch(Event.ACTION_PREDICTION_POSTPROCESSING_METHOD)

    def get_postprocessing_simple_threshold(self) -> float:
        """
        Gets simple threshold selected by user.
        """
        return self._postprocessing_simple_threshold

    def set_postprocessing_simple_threshold(self, threshold: float) -> None:
        """
        Sets simple threshold selected by user from the slider.
        """
        self._postprocessing_simple_threshold = threshold
        self.dispatch(Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD)

    def get_postprocessing_auto_threshold(self) -> str:
        """
        Gets auto threshold selected by user.
        """
        return self._postprocessing_auto_threshold

    def set_postprocessing_auto_threshold(self, threshold: str) -> None:
        """
        Sets auto threshold selected by user.
        """
        self._postprocessing_auto_threshold = threshold
        self.dispatch(Event.ACTION_PREDICTION_POSTPROCESSING_AUTO_THRESHOLD)

    def set_config_dir(self, config_dir: Path) -> None:
        self.config_dir = config_dir

    def get_config_dir(self) -> Path:
        return self.config_dir

    def set_config_name(self, config_name: str) -> None:
        self.config_name = config_name

    def get_config_name(self) -> str:
        return self.config_name
