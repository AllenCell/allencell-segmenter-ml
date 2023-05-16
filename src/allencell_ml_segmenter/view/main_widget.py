import napari

from allencell_ml_segmenter.core.application_manager import ApplicationManager
from allencell_ml_segmenter.view.test_view import TestView

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
        self._application = ApplicationManager(viewer, self.layout())
        self.btn: QPushButton = QPushButton("Click me!")
        self.btn.clicked.connect(self._on_click)
        self.layout().addWidget(self.btn)

    def _on_click(self) -> None:
        view = TestView()
        self._application.view_manager.load_view(view)