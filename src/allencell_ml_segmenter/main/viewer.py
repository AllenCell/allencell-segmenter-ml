from napari.layers.shapes.shapes import Shapes

from allencell_ml_segmenter.main.i_viewer import IViewer
import napari
from typing import Tuple, List


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

    def get_image_dims(self) -> Tuple:
        # just return x_y dims
        return self.viewer.layers[0].data.shape

    def clear_mask_layers(self, layers_to_remove: List[Shapes]) -> None:
        for layer in layers_to_remove:
            self.viewer.layers.remove(layer)