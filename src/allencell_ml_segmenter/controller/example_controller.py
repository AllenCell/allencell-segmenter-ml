from allencell_ml_segmenter.view.exmaple_view import ExampleView
import napari

class ExampleController():
    def __init__(self, viewer: napari.Viewer) -> None:
        # add all ui elements here
        self.viewer = viewer
        self.view: ExampleView = ExampleView(viewer)
        # connect buttons from within controller and delegate actions
        self._connect_slots()
        self.view.show()

    def _on_click(self) -> None:
        print("napari has", len(self.viewer.layers), "layers")

    def _connect_slots(self):
        self.view.btn.clicked.connect(self._on_click)

