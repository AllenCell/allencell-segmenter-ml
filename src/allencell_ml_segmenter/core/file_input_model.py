from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from enum import Enum
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class InputMode(Enum):
    FROM_PATH = "from_path"
    FROM_NAPARI_LAYERS = "from_napari_layers"


class FileInputModel(Publisher, ABC):
    """
    Abstract base class for PredictionModel and PostprocessModel.
    Defines the common methods and attributes both models should implement.
    """

    def __init__(self) -> None:
        super().__init__()
        self._input_image_path: Optional[Path] = None
        self._image_input_channel_index: Optional[int] = None
        self._output_directory: Optional[Path] = None
        self._input_mode: Optional[InputMode] = None
        self._selected_paths: Optional[list[Path]] = None
        self._max_channels: Optional[int] = None

    def set_input_image_path(
        self, path: Optional[Path], extract_channels: bool = False
    ) -> None:
        self._input_image_path = path
        if extract_channels and path is not None:
            self.dispatch(Event.ACTION_FILEINPUT_EXTRACT_CHANNELS)

    def get_input_image_path(self) -> Optional[Path]:
        return self._input_image_path

    def set_image_input_channel_index(self, idx: Optional[int]) -> None:
        self._image_input_channel_index = idx

    def get_image_input_channel_index(self) -> Optional[int]:
        return self._image_input_channel_index

    def set_output_directory(self, dir: Optional[Path]) -> None:
        self._output_directory = dir

    def get_output_directory(self) -> Optional[Path]:
        return self._output_directory

    def set_input_mode(self, mode: Optional[InputMode]) -> None:
        self._input_mode = mode

    def get_input_mode(self) -> Optional[InputMode]:
        return self._input_mode

    def set_selected_paths(
        self, paths: Optional[List[Path]], extract_channels: bool = False
    ) -> None:
        self._selected_paths = paths
        if extract_channels and paths is not None:
            self.dispatch(Event.ACTION_FILEINPUT_EXTRACT_CHANNELS)

    def get_max_channels(self) -> Optional[int]:
        return self._max_channels

    def set_max_channels(self, max: Optional[int]) -> None:
        self._max_channels = max
        if max is not None:
            self.dispatch(Event.ACTION_FILEINPUT_MAX_CHANNELS_SET)

    def get_selected_paths(self) -> Optional[list[Path]]:
        return self._selected_paths
