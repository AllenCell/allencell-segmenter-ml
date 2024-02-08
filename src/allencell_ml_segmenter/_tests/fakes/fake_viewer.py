from pathlib import Path

from allencell_ml_segmenter.main.i_viewer import IViewer

import napari
from typing import List, Callable
from napari.layers import Layer
from unittest.mock import Mock
from napari.layers.shapes.shapes import Shapes


class FakeLayer:
    def __init__(self):
        self.removed: List[Layer] = []
        self.connected: List[Callable] = []

    def remove(self, layer_remove: Layer):
        self.removed.append(layer_remove)

    def is_removed(self, layer: Layer):
        return layer in self.removed

    def connect(self, callable: Callable):
        self.connected.append(callable)

class FakeNapariEvent():
    def __init__(self):
        self.layers_change: FakeLayer = FakeLayer()


class FakeViewer(IViewer):
    def __init__(self):
        self._viewer = None
        self.layers = FakeLayer()
        self.layers_cleared_count = 0
        self.images_added = dict()
        self.shapes_layers_added = []
        self.shapes_layers_removed = []
        self.layers_change_event = None
        self.events = FakeNapariEvent()

    def add_image(self, image, name=None):
        self.images_added[name] = image

    def clear_layers(self) -> None:
        self.layers_cleared_count = self.layers_cleared_count + 1

    def add_shapes(self, name) -> Shapes:
        self.shapes_layers_added.append(name)
        mock_shapes_return = Mock(Shapes)
        mock_shapes_return.name = name
        return mock_shapes_return

    def clear_mask_layers(self, layers_to_remove: List[Shapes]) -> None:
        for layer in layers_to_remove:
            self.layers.remove(layer)

    def is_layer_removed(self, layer: Layer):
        return self.layers.is_removed(layer)

    def get_paths_of_image_layers(self) -> List:
        return list(self.images_added.keys())

    def subscribe_layers_change_event(self, function):
        self.layers_change_event = function
