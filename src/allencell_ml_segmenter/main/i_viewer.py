from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable, Any
from allencell_ml_segmenter.main.segmenter_layer import (
    ShapesLayer,
    ImageLayer,
    LabelsLayer,
)
import numpy as np
from napari.layers import Layer, Labels  # type: ignore
from napari.utils.events import Event as NapariEvent  # type: ignore


class IViewer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def add_image(self, image: np.ndarray, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def get_image(self, name: str) -> Optional[ImageLayer]:
        pass

    @abstractmethod
    def get_all_images(self) -> list[ImageLayer]:
        pass

    @abstractmethod
    def add_shapes(self, name: str, face_color: str, mode: str) -> None:
        pass

    @abstractmethod
    def get_shapes(self, name: str) -> Optional[ShapesLayer]:
        pass

    @abstractmethod
    def get_all_shapes(self) -> list[ShapesLayer]:
        pass

    @abstractmethod
    def add_labels(self, data: np.ndarray, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def get_labels(self, name: str) -> Optional[LabelsLayer]:
        pass

    @abstractmethod
    def get_all_labels(self) -> list[LabelsLayer]:
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
    def get_layers(self) -> list[Layer]:
        pass

    @abstractmethod
    def subscribe_layers_change_event(
        self, function: Callable[[NapariEvent], None]
    ) -> None:
        pass

    @abstractmethod
    def get_seg_layers(self) -> list[Layer]:
        """
        Get only segmentation layers (which should be probability mappings) from the viewer.
        These are the layers that start with [seg].
        """
        pass

    @abstractmethod
    def insert_binary_map_into_layer(
        self, layer_name: str, img: np.ndarray, seg_layers: bool = False
    ) -> None:
        """
        Insert a thresholded image into the viewer.
        If a layer for this thresholded image already exists, the new image will replace the old one and refresh the viewer.
        If the layer does not exist, it will be added to the viewer in the correct place (on top of the original segmentation image:
        index_of_segmentation + 1 in the LayerList)
        """
        pass

    @abstractmethod
    def get_layers_nonthreshold(self) -> list[Layer]:
        """
        Get only layers which are not segmentation layers from the viewer.
        These are the layers that do not start with [threshold].
        """
        pass

    @abstractmethod
    def get_source_path(self, layer: Layer) -> Optional[Path]:
        pass

    @abstractmethod
    def get_all_layers_containing_prob_map(self) -> list[Layer]:
        pass

    @abstractmethod
    def clear_binary_map_from_layer(self, layer: Layer) -> None:
        pass
