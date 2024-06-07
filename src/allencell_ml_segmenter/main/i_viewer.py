from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from allencell_ml_segmenter.main.segmenter_layer import (
    ShapesLayer,
    ImageLayer,
    LabelsLayer,
)
import numpy as np
from napari.layers import Layer
from napari.utils.events import Event as NapariEvent


class IViewer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def add_image(self, image: np.ndarray, name: str) -> None:
        pass

    @abstractmethod
    def get_image(self, name: str) -> Optional[ImageLayer]:
        pass

    @abstractmethod
    def get_all_images(self) -> List[ImageLayer]:
        pass

    @abstractmethod
    def add_shapes(self, name: str, face_color: str, mode: str) -> None:
        pass

    @abstractmethod
    def get_shapes(self, name: str) -> Optional[ShapesLayer]:
        pass

    @abstractmethod
    def get_all_shapes(self) -> List[ShapesLayer]:
        pass

    @abstractmethod
    def add_labels(self, data: np.ndarray, name: str) -> None:
        pass

    @abstractmethod
    def get_labels(self, name: str) -> Optional[LabelsLayer]:
        pass

    @abstractmethod
    def get_all_labels(self) -> List[LabelsLayer]:
        pass

    @abstractmethod
    def clear_layers(self) -> None:
        pass

    @abstractmethod
    def remove_layer(self, name: str) -> bool:
        pass

    @abstractmethod
    def contains_layer(self, name: str) -> bool:
        pass

    # TODO: refactor prediction/file_input_widget.py to not use this and use get_all_images instead
    @abstractmethod
    def get_layers(self) -> List[Layer]:
        pass

    @abstractmethod
    def subscribe_layers_change_event(
        self, function: Callable[[NapariEvent], None]
    ) -> None:
        pass
