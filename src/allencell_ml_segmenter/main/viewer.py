from allencell_ml_segmenter.main.i_viewer import IViewer
import napari


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
