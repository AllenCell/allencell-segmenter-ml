from allencell_ml_segmenter.main.i_viewer import IViewer

import napari
from typing import List
from napari.layers import Layer


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

    def add_image(image, name):
        pass

    def clear_layers(self) -> None:
        self.layers_cleared_count = self.layers_cleared_count + 1
