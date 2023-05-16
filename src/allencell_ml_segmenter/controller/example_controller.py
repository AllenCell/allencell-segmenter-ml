from qtpy.QtWidgets import QWidget
import napari

class ExampleController(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        # add all ui elements here
        self.viewer = viewer
        # connect buttons from within controller and delegate actions
        self._connect_slots()
        self.view.show()

    def _on_click(self) -> None:
        print("napari has", len(self.viewer.layers), "layers")

    def _connect_slots(self):
        self.view.btn.clicked.connect(self._on_click)

