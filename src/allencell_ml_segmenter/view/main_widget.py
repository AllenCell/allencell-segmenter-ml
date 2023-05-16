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
        self.button = QPushButton("click to add a widget")
        self.button.clicked.connect(self._application._router.show_test_view)
        self.layout().addWidget(self.button)
