import napari

from allencell_ml_segmenter.core.application_manager import ApplicationManager
from allencell_ml_segmenter.view.sample_view import SampleViewController

from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)


class MainWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        #basic styling
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # init app
        self._application = ApplicationManager(viewer, self.layout())

        # Controller
        self.button = QPushButton("click to add a widget")
        self.button.clicked.connect(self._application._router.navigate_to_test_view)
        self.layout().addWidget(self.button)
