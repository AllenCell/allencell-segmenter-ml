from enum import Enum
from pathlib import Path
from typing import List, Optional

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class PredictionInputMode(Enum):
    FROM_PATH = "from_path"
    FROM_NAPARI_LAYERS = "from_napari_layers"


class PredictionModel(Publisher):
    """
    Stores state relevant to prediction processes.
    """

    def __init__(self) -> None:
        super().__init__()

        # state related to PredictionFileInput
        self.config_name: Optional[str] = None
        self.config_dir: Optional[Path] = None
        self._input_image_path: Optional[Path] = None
        self._image_input_channel_index: Optional[int] = None
        self._input_mode: Optional[PredictionInputMode] = None
        self._output_directory: Optional[Path] = None
        self._selected_paths: Optional[list[Path]] = None
        self._max_channels: Optional[int] = None
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

    def get_input_image_path(self) -> Optional[Path]:
        """
        Gets list of paths to input images.
        """
        return self._input_image_path

    def set_input_image_path(
        self, path: Optional[Path], extract_channels: bool = False
    ) -> None:
        """
        Sets list of paths to input images.
        """
        self._input_image_path = path
        if extract_channels and path is not None:
            # This will extract and set number of channels
            self.dispatch(Event.ACTION_PREDICTION_EXTRACT_CHANNELS)

    def get_image_input_channel_index(self) -> Optional[int]:
        """
        Gets path to output directory.
        """
        return self._image_input_channel_index

    def set_image_input_channel_index(self, idx: Optional[int]) -> None:
        """
        Sets path to output directory.
        """
        self._image_input_channel_index = idx

    def get_output_directory(self) -> Optional[Path]:
        """
        Gets path to output directory.
        """
        return self._output_directory

    def get_output_seg_directory(self) -> Optional[Path]:
        """
        Gets path to where segmentation predictions are stored.
        """
        return self._output_directory / "target" if self._output_directory is not None else None

    def set_output_directory(self, dir: Optional[Path]) -> None:
        """
        Sets path to output directory.
        """
        self._output_directory = dir

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

    def set_postprocessing_simple_threshold(self, threshold: Optional[float]) -> None:
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

    def set_postprocessing_auto_threshold(self, threshold: Optional[str]) -> None:
        """
        Sets auto threshold selected by user.
        """
        self._postprocessing_auto_threshold = threshold
        self.dispatch(Event.ACTION_PREDICTION_POSTPROCESSING_AUTO_THRESHOLD)

    def set_prediction_input_mode(self, mode: Optional[PredictionInputMode]) -> None:
        self._input_mode = mode

    def get_prediction_input_mode(self) -> Optional[PredictionInputMode]:
        return self._input_mode

    def set_selected_paths(
        self, paths: Optional[list[Path]], extract_channels: bool = False
    ) -> None:
        self._selected_paths = paths
        if extract_channels and paths is not None:
            self.dispatch(Event.ACTION_PREDICTION_EXTRACT_CHANNELS)

    def get_selected_paths(self) -> Optional[list[Path]]:
        return self._selected_paths

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

    def set_max_channels(self, max: Optional[int]) -> None:
        self._max_channels = max
        # this will enable the combobox
        if max is not None:
            self.dispatch(Event.ACTION_PREDICTION_MAX_CHANNELS_SET)

    def get_max_channels(self) -> Optional[int]:
        return self._max_channels

    def set_total_num_images(self, total: Optional[int]) -> None:
        self.total_num_images = total

    def get_total_num_images(self) -> Optional[int]:
        return self.total_num_images
