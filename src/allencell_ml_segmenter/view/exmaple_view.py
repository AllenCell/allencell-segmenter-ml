import napari

from qtpy.QtWidgets import QFrame

from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout
)


class ExampleView(QFrame):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.btn: QPushButton = QPushButton("Click me!")
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.btn)

