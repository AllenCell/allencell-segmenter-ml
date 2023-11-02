from allencell_ml_segmenter.main.i_viewer import IViewer
import napari
from typing import Tuple


class Viewer(IViewer):
    def __init__(
        self,
        viewer: napari.Viewer,
    ):
        super().__init__()
        self.viewer: napari.Viewer = viewer

    def add_image(self, image, name: str):
        self.viewer.add_image(image, name=name)

    def add_shapes(self, name: str):
        return self.viewer.add_shapes(None, name=name)

    def clear_layers(self):
        self.viewer.layers.clear()

    def get_image_dims(self) -> Tuple[int, int]:
        # just return x_y dims
        return self.viewer.layers[0].data.shape[:2]
