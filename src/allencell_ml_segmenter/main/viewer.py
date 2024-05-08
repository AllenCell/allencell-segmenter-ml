from pathlib import Path

from napari.layers import Layer, Shapes, Image
from napari.layers.shapes.shapes import Mode
from napari.utils.events import Event as NapariEvent

from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.main.segmenter_layer import ShapesLayer, ImageLayer
import napari
from typing import List, Callable, Optional
import numpy as np
from napari.components import LayerList


class Viewer(IViewer):
    def __init__(
        self,
        viewer: napari.Viewer,
    ):
        super().__init__()
        self.viewer: napari.Viewer = viewer

    def add_image(self, image: np.ndarray, name: str) -> None:
        self.viewer.add_image(image, name=name)

    def get_image(self, name: str) -> Optional[ImageLayer]:
        for img in self.get_all_images():
            if img.name == name:
                return img
        return None

    def get_all_images(self) -> List[ImageLayer]:
        imgs: List[ImageLayer] = []
        for l in self.viewer.layers:
            if type(l) == Image and l.source.path:
                imgs.append(ImageLayer(l.name, Path(l.source.path)))
            elif type(l) == Image:
                imgs.append(ImageLayer(l.name, None))
        return imgs

    def add_shapes(self, name: str, face_color: str, mode: Mode) -> None:
        shapes: Shapes = self.viewer.add_shapes(
            None, name=name, face_color=face_color
        )
        shapes.mode = mode

    def get_shapes(self, name: str) -> Optional[ShapesLayer]:
        for img in self.get_all_shapes():
            if img.name == name:
                return img
        return None

    def get_all_shapes(self) -> List[ShapesLayer]:
        return [
            ShapesLayer(l.name, np.asarray(l.data, dtype=object))
            for l in self.viewer.layers
            if type(l) == Shapes
        ]

    def clear_layers(self) -> None:
        self.viewer.layers.clear()

    def remove_layer(self, name: str) -> bool:
        layer: Optional[Layer] = self._get_layer_by_name(name)
        if layer is not None:
            self.viewer.layers.remove(layer)
            return True
        return False

    def contains_layer(self, name: str) -> bool:
        return self._get_layer_by_name() is not None

    def get_layers(self) -> List[Layer]:
        return [l for l in self.viewer.layers]

    def subscribe_layers_change_event(
        self, function: Callable[[NapariEvent], None]
    ) -> None:
        self.viewer.events.layers_change.connect(function)

    def _get_layer_by_name(self, name: str) -> Optional[Layer]:
        layers: List[Layer] = self.get_layers()
        for l in layers:
            if l.name == name:
                return l
        return None
