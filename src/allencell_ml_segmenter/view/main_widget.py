import napari

from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)


class MainWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.btn: QPushButton = QPushButton("Click me!")
        self.btn.clicked.connect(self._on_click)
        self.layout().addWidget(self.btn)

    def _on_click(self) -> None:
        print("napari has", len(self.viewer.layers), "layers")