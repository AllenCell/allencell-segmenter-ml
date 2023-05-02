from qtpy.QtWidgets import QWidget
from allencell_ml_segmenter.view.exmaple_view import ExampleView
import napari

class ExampleController(QWidget):
    def __init__(self, napari_viewer: napari.Viewer) -> None:
        # add all ui elements here
        self.viewer = napari_viewer
        self.view: ExampleView = ExampleView(napari_viewer)
        # connect buttons from within controller and delegate actions
        self.view._connect_slots()
        self.view.show()

    def _on_click(self) -> None:
        print("napari has", len(self.viewer.layers), "layers")

    def _connect_slots(self):
        self.view.btn.clicked.connect(self._on_click)

