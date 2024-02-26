from pathlib import Path

from napari.components import LayerList
from napari.layers.shapes.shapes import Shapes
from napari.utils.events.evented_model import EventedModel as NapariEventModel

from allencell_ml_segmenter.main.i_viewer import IViewer
import napari
from typing import Tuple, List, Callable


class Viewer(IViewer):
    def __init__(
        self,
        viewer: napari.Viewer,
    ):
        super().__init__()
        self.viewer: napari.Viewer = viewer

    def add_image(self, image, name: str) -> None:
        self.viewer.add_image(image, name=name)

    def add_shapes(self, name: str):
        return self.viewer.add_shapes(None, name=name)

    def clear_layers(self) -> None:
        self.viewer.layers.clear()

    def clear_mask_layers(self, layers_to_remove: List[Shapes]) -> None:
        for layer in layers_to_remove:
            self.viewer.layers.remove(layer)

    def get_layers(self) -> LayerList:
        return self.viewer.layers

    def get_paths_of_image_layers(self) -> List[Path]:
        return [layer.source.path for layer in self.viewer.layers]

    def subscribe_layers_change_event(self, function: Callable):
        self.viewer.events.layers_change.connect(function)

    def get_image_dims(self) -> Tuple:
        # just return x_y dims
        return self.viewer.layers[0].data.shape
