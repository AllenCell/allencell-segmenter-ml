from allencell_ml_segmenter.main.i_viewer import IViewer

import napari
from typing import List
from napari.layers import Layer
from unittest.mock import Mock
from napari.layers.shapes.shapes import Shapes


class FakeLayer:
    def __init__(self):
        self.removed: List[Layer] = []

    def remove(self, layer_remove: Layer):
        self.removed.append(layer_remove)

    def is_removed(self, layer: Layer):
        return layer in self.removed


class FakeViewer(IViewer):
    def __init__(self):
        self._viewer = None
        self.layers = FakeLayer()
        self.layers_cleared_count = 0
        self.images_added = dict()
        self.shapes_layers_added = []

    def add_image(self, image, name=None):
        self.images_added[name] = image

    def clear_layers(self) -> None:
        self.layers_cleared_count = self.layers_cleared_count + 1

    def add_shapes(self, name) -> Shapes:
        self.shapes_layers_added.append(name)
        mock_shapes_return = Mock(Shapes)
        mock_shapes_return.name = name
        return mock_shapes_return
